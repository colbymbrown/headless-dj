"""Smallest checks that fail if the mixer breaks.

    python -m pytest test_mix.py    # or just: python test_mix.py
"""
import numpy as np

import mix


def test_bar_math():
    assert mix.bars_to_samples(16, 120.0) == 1_411_200   # 32.000s exactly
    assert mix.bars_to_samples(16, 128.0) == 1_323_000
    assert mix.bars_to_samples(1, 120.0) == 88_200       # 2s per bar


def test_assemble_length():
    bpm, n = 128.0, 5
    loop_len = mix.bars_to_samples(16, bpm)
    taper = mix.bars_to_samples(4, bpm)
    loops = [np.zeros((loop_len, 2), np.float32) for _ in range(n)]
    out = mix.assemble(loops, [bpm] * n, xfade_bars=4)
    assert len(out) == n * loop_len - (n - 1) * taper


def test_high_band_level_is_flat_through_blend():
    """Equal-power crossfade: uncorrelated material must not dip in level."""
    from scipy.signal import butter, sosfilt

    bpm = 128.0
    L = mix.bars_to_samples(16, bpm)
    tp = mix.bars_to_samples(4, bpm)
    sos = butter(4, 400 / (mix.SR / 2), btype="high", output="sos")
    rng = np.random.default_rng(0)
    n = rng.normal(0, 0.2, (L, 2)).astype(np.float32)
    highs = [sosfilt(sos, n, axis=0).astype(np.float32),
             sosfilt(sos, rng.normal(0, 0.2, (L, 2)).astype(np.float32), axis=0).astype(np.float32)]
    out = mix.assemble(highs, [bpm, bpm], xfade_bars=4)
    overlap = out[L - tp : L, 0]

    win = mix.SR // 2
    rms = np.array([np.sqrt((overlap[i : i + win] ** 2).mean())
                    for i in range(0, len(overlap) - win, win)])
    assert rms.max() / rms.min() < 1.15, rms.max() / rms.min()


def test_low_band_never_drops_out_during_blend():
    """Complementary bass curves: low end is always exactly one loop's worth."""
    bpm = 128.0
    L = mix.bars_to_samples(16, bpm)
    tp = mix.bars_to_samples(4, bpm)
    dc = np.full((L, 2), 0.5, np.float32)
    out = mix.assemble([dc, dc], [bpm, bpm], xfade_bars=4)
    overlap = out[L - tp : L]
    assert np.abs(overlap - 0.5).max() < 0.02, np.abs(overlap - 0.5).max()


def test_bass_swap_moves_low_end():
    """Low band of the outgoing loop must actually be gone by the blend end."""
    bpm = 128.0
    L = mix.bars_to_samples(16, bpm)
    out = mix.assemble(
        [np.full((L, 2), 0.5, np.float32), np.zeros((L, 2), np.float32)],
        [bpm, bpm], xfade_bars=4)
    end = out[len(out) - 1]
    assert np.abs(end).max() < 1e-3      # faded to nothing


def test_edge_fades_start_and_end_at_silence():
    bpm = 128.0
    y = np.full((mix.bars_to_samples(16, bpm), 2), 0.5, np.float32)
    mix.fade_edges(y, mix.SR, bpm, fade_in_bars=4, fade_out_bars=4)
    assert abs(y[0]).max() < 1e-6
    assert abs(y[-1]).max() < 1e-6
    assert abs(y[len(y) // 2]).max() > 0.4


def test_tempo_match_and_beat_alignment():
    """A 126 BPM click track asked for 128 must come back at 128, on the grid,
    and exactly 16 bars long."""
    try:
        import librosa  # noqa: F401
    except ImportError:
        print("skip test_tempo_match_and_beat_alignment (no librosa)")
        return
    src_bpm, target_bpm, bars = 126.0, 128.0, 16
    raw = _click_loop(src_bpm, 20)  # headroom before the downbeat

    loop, info = mix.prepare_loop(raw, mix.SR, target_bpm, bars)
    assert len(loop) == mix.bars_to_samples(bars, target_bpm)
    assert info["padded_samples"] == 0, info
    assert info["warped"], info
    assert abs(info["detected_bpm"] - src_bpm) < 0.2, info

    period = mix.sec_per_bar(target_bpm) * mix.SR / 4
    phase = (int(np.argmax(np.abs(loop[:, 0]))) % period) / period
    assert min(phase, 1 - phase) < 0.02, phase
    # no silent tail from over-trimming: the loop ends on real audio
    tail = loop[-int(0.5 * mix.SR):, 0]
    assert np.abs(tail).max() > 0.05 * np.abs(loop[:, 0]).max(), "silent tail"


def test_downbeat_vote_picks_accented_phase():
    """With accented downbeats, the loop must start on an accent."""
    try:
        import librosa  # noqa: F401
    except ImportError:
        print("skip test_downbeat_vote (no librosa)")
        return
    raw = _click_loop(126.0, 20, accent=3.0)
    mono = raw.mean(axis=1)
    tempo, beats, oenv, hop = mix.analyze_timing(mono, mix.SR, 126.0)
    beats = mix.refine_beats(beats, oenv, hop)
    d = mix.pick_downbeat(beats, oenv)
    assert d % 4 == 0, d
    loop, _ = mix.prepare_loop(raw, mix.SR, 126.0, 16)
    # first kick is an accent (3x) and sits at the very start of the loop
    assert abs(loop[:2000, 0]).max() > 2.0 * abs(loop[2000:4000, 0]).max()


def _click_loop(bpm, bars, accent=1.0):
    """Stereo click track, one kick per beat, optionally accented downbeats."""
    beat = mix.sec_per_bar(bpm) / 4
    n = int(bars * mix.sec_per_bar(bpm) * mix.SR)
    y = np.zeros(n, np.float32)
    for k in range(int(bars * 4)):
        i = int(k * beat * mix.SR)
        y[i : i + 2000] = np.hanning(2000).astype(np.float32) * accent * (3.0 if k % 4 == 0 else 1.0)
    return np.stack([y, y], axis=1)


def test_end_to_end_mix_from_synthetic_loops():
    """Whole chain: drifting-tempo loops in, one faded-out mix out."""
    rng = np.random.default_rng(1)
    want_bpms = [124.0, 125.0, 126.0, 127.0, 128.0]
    prepared = []
    for want in want_bpms:
        src = want + rng.uniform(-4, 4)        # tempo the model "actually" produced
        beat = mix.sec_per_bar(src) / 4
        n = int(18 * mix.sec_per_bar(src) * mix.SR)
        t = np.arange(n) / mix.SR
        kick = np.zeros(n, np.float32)
        for k in range(int(n / (beat * mix.SR))):
            j = int(k * beat * mix.SR)
            kick[j : j + 2000] += np.hanning(2000).astype(np.float32) * 0.8
        voice = (kick + 0.3 * np.sin(2 * np.pi * 55 * t)
                 + rng.normal(0, 0.02, n)).astype(np.float32)
        raw = np.stack([voice, voice * 0.9], axis=1)
        loop, info = mix.prepare_loop(raw, mix.SR, want, 16)
        assert info["padded_samples"] == 0, info
        prepared.append(mix.rms_normalize(loop))

    y = mix.assemble(prepared, want_bpms, xfade_bars=4)
    y = mix.fade_edges(y, mix.SR, want_bpms[0], 8, 8)
    y = mix.finalize(y)

    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 1.0
    assert abs(y[0]).max() < 1e-6 and abs(y[-1]).max() < 1e-6
    assert len(y) / mix.SR > 60.0
    # no dropouts in the body (skip the deliberate intro/outro fades)
    win = mix.SR
    body = y[int(0.15 * len(y)) : int(0.85 * len(y))]
    rms = [np.sqrt((body[i : i + win] ** 2).mean())
           for i in range(0, len(body) - win, win)]
    assert min(rms) > 0.02, min(rms)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")