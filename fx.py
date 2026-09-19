"""Post-render DJ effects + loudness pass.

Runs after the plain mix is compiled (assemble -> fade_edges -> finalize), inside
build_mix. Two stages, in order:
  1. fx()        - sparse, phrase-locked DJ effects via pedalboard, all high-passed.
  2. loudness()  - LUFS-based gain match + true peak limiter, run LAST.

Design rules (from DJ-practice research):
  - Sparse + phrase-locked: effects land on bar boundaries, never sustained.
  - Rate-based (probability per phrase boundary), NOT "N per mix".
  - No clumping: never on consecutive transitions.
  - Skip gracefully: a marker that can't host an effect is left plain + logged.
  - Bass-protected: every FX goes through a 180 Hz high-pass send.
"""
from __future__ import annotations

import random

import numpy as np

import mix
from pedalboard import Pedalboard, HighpassFilter, Delay, Reverb, Phaser

SR = mix.SR
CROSSOVER_HZ = mix.CROSSOVER_HZ
BEATS_PER_BAR = mix.BEATS_PER_BAR

DEFAULTS = {
    "p_echo": 0.25,
    "p_reverb": 0.15,
    "reverb_before_key_change": True,
    "phaser_prob": 0.02,
    "echo_out_bars": 1.0,
    "rhythm_pool": {1.0: 3, 0.75: 2, 3.0: 1, 1.5: 2},
    "echo_wet": 0.55,
    "echo_feedback": 0.5,
    "reverb_wet": 0.28,
    "reverb_input_bars": 2.0,
    "reverb_room": 0.5,
    "phaser_wet": 0.30,
    "phaser_rate": 0.4,
    "target_lufs": -14.0,
    "limiter_ceiling": 0.92,
}


def _bar_samples(bpm):
    return int(mix.sec_per_bar(bpm) * SR)


def compute_markers(loops, bpms, slots, xfade_bars):
    """Phrase-locked markers on the compiled timeline."""
    n = len(loops)
    taper = [mix.bars_to_samples(xfade_bars, bpms[i + 1]) for i in range(n - 1)]
    offsets = [0]
    for i in range(n - 1):
        offsets.append(offsets[-1] + len(loops[i]) - taper[i])
    transitions = [offsets[i + 1] - taper[i] for i in range(n - 1)]
    key_changes = [offsets[i] for i in range(1, n)
                   if i < len(slots) and slots[i].camelot != slots[i - 1].camelot]
    return {
        "offsets": offsets, "taper": taper, "transitions": transitions,
        "key_changes": key_changes,
    }


def _highpass_seg(seg):
    return np.asarray(
        Pedalboard([HighpassFilter(cutoff_frequency_hz=CROSSOVER_HZ)])(seg, SR),
        dtype=np.float32)


def _ring_tail(effect, send, sr=SR, block=22050, noise_floor=1e-4):
    """Send library-style: feed `send` into `effect` with reset=False (state
    preserved), then keep flushing the effect with silence so its internal
    feedback/decay keeps ringing out until it falls below the noise floor.
    Returns the full wet output (dry-pass + tail)."""
    wet = []
    # feed the dry send in place (preserving effect state)
    n = send.shape[0]
    for j in range(0, n, block):
        chunk = np.ascontiguousarray(send[j:j + block])
        wet.append(effect.process(chunk, sr, reset=False))
    # flush the ringing tail with silence
    silence = np.zeros((block, send.shape[1]), dtype=np.float32)
    for _ in range(64):
        c = effect.process(silence, sr, reset=False)
        wet.append(c)
        if float(np.abs(c).max()) < noise_floor:
            break
    return np.concatenate(wet, axis=0)


def fx(y, loops, bpms, slots, xfade_bars, cfg=None, seed=None, log=print):
    """Apply the FX program to the compiled mix y (samples, 2). Returns y_fx.

    Args:
      y          : compiled mix (after assemble+fade).
      loops      : the prepared loop arrays used by assemble (for markers).
      bpms       : per-loop bpm, same order as loops.
      slots      : Slot objects (for camelot key-change markers).
      xfade_bars : blend width in bars.
    """
    cfg = cfg if cfg is not None else DEFAULTS
    rng = random.Random(seed) if seed is not None else random.Random()
    # Tempo is fixed across the mix; bar length in samples at that tempo.
    mix_bpm = bpms[0] if bpms else 120.0
    bar = _bar_samples(mix_bpm)
    m = compute_markers(loops, bpms, slots, xfade_bars)
    total = y.shape[0]
    events = _schedule_events(m, cfg, rng, total, bar)
    out = y.copy()
    for kind, s in events:
        res = _apply(out, kind, s, cfg, rng, bar, mix_bpm)
        if res is not None:
            out = res
            log(f"fx: {kind} @ {s / SR:.1f}s")
    return out


def _schedule_events(m, cfg, rng, total, bar):
    events = []
    last = -10 ** 9

    def far(s):
        return abs(s - last) > bar

    # echo-out at a fraction of transitions (sparser)
    for s in m["transitions"]:
        if far(s) and rng.random() < cfg["p_echo"]:
            events.append(("echo", s)); last = s
    # reverb before every key change + at a fraction of transitions
    for s in m["key_changes"]:
        if far(s):
            events.append(("reverb", s)); last = s
    for s in m["transitions"]:
        if far(s) and rng.random() < cfg["p_reverb"]:
            events.append(("reverb", s)); last = s
    # phaser, rare
    for s in m["offsets"]:
        if far(s) and rng.random() < cfg["phaser_prob"]:
            events.append(("phaser", s)); last = s
    events.sort(key=lambda e: e[1])
    return [e for e in events if 0 <= e[1] < total]


def _apply(y, kind, s, cfg, rng, bar, bpm):
    if kind == "echo": return _echo(y, s, cfg, rng, bpm)
    if kind == "reverb": return _reverb(y, s, cfg, bar)
    if kind == "phaser": return _phaser(y, s, cfg)
    return None


def _echo(y, s, cfg, rng, bpm):
    bar = _bar_samples(bpm)
    start = max(0, s - int(cfg["echo_out_bars"] * bar))
    seg = y[start:s]
    if seg.shape[0] < SR * 0.1:
        return None
    beat = 60.0 / bpm
    mults = [m for m, _ in cfg["rhythm_pool"].items()]
    wts = [w for _, w in cfg["rhythm_pool"].items()]
    mult = rng.choices(mults, weights=wts, k=1)[0]
    delay_sec = max(0.02, min(mult * beat, 2.0))
    hp = _highpass_seg(seg)  # bass-protected send
    effect = Delay(delay_seconds=delay_sec, feedback=cfg["echo_feedback"], mix=1.0)
    wet = _ring_tail(effect, hp)  # send + flush -> decaying taps
    out = y.copy()
    nseg = seg.shape[0]
    # cap wet to what remains in the mix after start
    wet = wet[: y.shape[0] - start]
    # dry window: blend wet echo over the first nseg samples
    m = min(nseg, wet.shape[0])
    out[start:start + m] += cfg["echo_wet"] * wet[:m]
    # trailing echo taps (feedback ring) extend past the dry window
    if wet.shape[0] > nseg:
        n_tail = min(wet.shape[0] - nseg, y.shape[0] - (start + nseg))
        out[start + nseg:start + nseg + n_tail] += \
            cfg["echo_wet"] * wet[nseg:nseg + n_tail]
    return out


def _reverb(y, s, cfg, bar):
    # Library send/return: send the material in the window right before s into
    # the reverb (reset=False so state persists), then flush with silence so
    # it rings out naturally. The wet signal is faded in over the first bar to
    # smooth the onset and mixed back at wet_level as the return.
    in_bars = max(0.5, cfg.get("reverb_input_bars", 2.0))
    seg = y[max(0, s - int(in_bars * bar)):s]
    if seg.shape[0] < SR * 0.1:
        return None
    hp = _highpass_seg(seg)
    effect = Reverb(room_size=cfg["reverb_room"], wet_level=1.0, dry_level=0.0)
    wet = _ring_tail(effect, hp)  # send + flush -> ringing tail
    wet = wet[: y.shape[0] - s]
    if wet.shape[0] == 0:
        return None
    # fade the wet in over the first bar to avoid an abrupt slam
    fade_n = min(int(1.0 * bar), wet.shape[0])
    if fade_n > 0:
        wet[:fade_n] *= np.linspace(0.0, 1.0, fade_n)[:, None]
    out = y.copy()
    end = min(y.shape[0], s + wet.shape[0])
    out[s:end] += cfg["reverb_wet"] * wet[: end - s]
    return out


def _phaser(y, s, cfg):
    seg = y[s:]
    if seg.shape[0] < SR * 0.5:
        return None
    hp = _highpass_seg(seg)
    board = Pedalboard([Phaser(rate_hz=cfg["phaser_rate"], depth=0.6)])
    wet = np.asarray(board(hp, SR), dtype=np.float32)
    wet = wet[:seg.shape[0]]
    out = y.copy()
    out[s:] = seg + cfg["phaser_wet"] * wet
    return out


# -------------------------------------------------------------- loudness ---

def loudness(y, cfg=None):
    """LUFS-style gain match + true peak limiter. Run LAST (post-fx)."""
    cfg = cfg if cfg is not None else DEFAULTS
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-9)
    target_rms_db = cfg["target_lufs"] - 4.0
    gain = 10 ** ((target_rms_db - rms_db) / 20.0)
    y = y * gain
    peak = float(np.abs(y).max())
    if peak > cfg["limiter_ceiling"]:
        y = y * (cfg["limiter_ceiling"] / peak)
    return np.asarray(y, dtype=np.float32)
