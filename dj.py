#!/usr/bin/env python3
"""headless-dj: a daily Stable Audio Open DJ mix bot.

Picks a style, tempo and key; generates 16-bar loops; matches each loop's tempo
to the mix tempo, beat-aligns it, and blends it into the running mix with a
bar-length bass-swap crossfade. Fades in from and out to silence.

  ./dj.py --plan-only                 # print the plan, no GPU needed
  ./dj.py --minutes 15                # generate + blend + write mixes/*.flac
  ./dj.py --minutes 15 --mix-only     # re-render from cached loops, no GPU
"""
import argparse
import datetime as dt
import json
import math
import warnings
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import soundfile as sf

import mix
import genes

# Ignore the benign float64-epsilon time-boundary warnings from
# torchsde's Brownian sampler (tb=0.29999998 vs t0=0.3 at the
# final diffusion step) -- purely cosmetic, not an error.
warnings.filterwarnings(
    "ignore",
    message=r"Should have t\S*=t0",
    category=UserWarning,
)

SR = mix.SR
MODEL_ID = "stabilityai/stable-audio-open-1.0"
MAX_GEN_S = 47.0  # Stable Audio Open's hard limit
NEGATIVE = ("vocals, singing, speech, voice, acapella, low quality, muffled, "
            "noisy, distorted, silence, intro, outro, drone, breakdown")

# style -> (bpm range, prompt variants). Seed population for the gene pool
# (genes.py): used only to create gene_pool.json on first run. After that the
# pool evolves on its own via web.py votes; one mix = one gene trial.
STYLES = {
    "deep house": ((118, 125), [
        "warm analog bassline, lush pads, shuffling hi-hats, Rhodes chords",
        "deep sub bass, filtered disco strings, groovy bassline, soft claps",
        "dubby chords, swung percussion, organic house groove, congas",
    ]),
    "tech house": ((122, 128), [
        "rolling tech house bassline, crisp hats, vocal chops, tight kick",
        "punchy kick, tribal percussion, filtered stab, jacking groove",
        "acid bassline, dry claps, tight shaker, warehouse groove",
    ]),
    "melodic techno": ((122, 128), [
        "arpeggiated analog synth, deep rolling bass, atmospheric pads",
        "driving bassline, emotive minor chords, wide reverb, hypnotic arp",
        "modulated saw lead, tom groove, dark pad, rolling sub bass",
    ]),
    "techno": ((128, 138), [
        "relentless four on the floor, industrial percussion, hypnotic loop",
        "hard kick, offbeat open hat, metallic stab, tunnel groove",
        "driving kick, distorted bass, rave stab, relentless groove",
    ]),
    "minimal": ((122, 128), [
        "sparse percussion, deep sub bass, subtle clicks, dubby chords",
        "tight minimal groove, rimshots, warm sub, micro-house textures",
    ]),
    "progressive house": ((122, 128), [
        "chugging bassline, layered pads, plucked synth, wide stereo field",
        "rolling bass, gated pad, dreamy lead, hypnotic groove",
    ]),
    "disco house": ((118, 124), [
        "live disco bassline, string stabs, wah guitar, congas, tight kit",
        "filtered disco loop, funky guitar, lush strings, punchy drums",
    ]),
    "afro house": ((118, 126), [
        "organic afro percussion, deep bass, marimba melody, warm pads",
        "tribal drums, rolling bassline, kalimba lead, hypnotic chant-free groove",
    ]),
    "trance": ((132, 140), [
        "rolling bassline, supersaw chords, uplifting arpeggio, gated pads",
        "driving kick, plucked arp, euphoric pad, offbeat bass",
    ]),
    "uk garage": ((128, 136), [
        "swung 2-step garage drums, sub bass, chopped vocal-free chords",
        "shuffled garage beat, deep sub, organ stab, skippy percussion",
    ]),
    "drum and bass": ((172, 176), [
        "rolling breakbeat, reese bass, atmospheric pad, amen chops",
        "liquid dnb drums, deep sub bass, soulful chords, tight breaks",
    ]),
    "electro": ((126, 134), [
        "electro breakbeat, 808 bass, robotic vocoder-free synth, drum machine",
        "808 drums, syncopated bass, detuned lead, electro groove",
    ]),
}

# Camelot wheel -> key name for the prompt. Neighbours mix cleanly, which is
# the only reason to bother: the model does not reliably honour the key.
CAMELOT = {
    ("A", 1): "Ab minor", ("A", 2): "Eb minor", ("A", 3): "Bb minor",
    ("A", 4): "F minor",  ("A", 5): "C minor",  ("A", 6): "G minor",
    ("A", 7): "D minor",  ("A", 8): "A minor",  ("A", 9): "E minor",
    ("A", 10): "B minor", ("A", 11): "F# minor", ("A", 12): "C# minor",
    ("B", 1): "B major",  ("B", 2): "F# major", ("B", 3): "C# major",
    ("B", 4): "Ab major", ("B", 5): "Eb major", ("B", 6): "Bb major",
    ("B", 7): "F major",  ("B", 8): "C major",  ("B", 9): "G major",
    ("B", 10): "D major", ("B", 11): "A major", ("B", 12): "E major",
}


@dataclass
class Slot:
    index: int
    gene_id: int
    gene_text: str
    bpm: float
    key: str
    camelot: str
    seed: int
    gen_seconds: float
    loop_bars: int
    prev_seed: int | None = None
    strength: float | None = None
    path: str = ""

    @property
    def prompt(self):
        return (f"{int(round(self.bpm))} BPM loop, {self.gene_text}, "
                f"stereo club mix, instrumental")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def make_plan(args):
    """This mix is a trial of a randomly picked gene. Tempo drifts slowly
    across the mix within the gene's bpm range; keys move every few loops."""
    rng = random.Random(args.seed)
    pool = genes.load()
    trial = genes.pick_trial(pool["genes"], rng, contains=args.style)
    bpm_lo, bpm_hi = trial["bpm_lo"], trial["bpm_hi"]

    # Fixed tempo for the whole mix, drawn at random from the gene's range.
    # A single constant BPM (rather than progressive drift) is what produces
    # the cleanest blends.
    bpm = rng.uniform(bpm_lo, bpm_hi) if args.bpm is None else args.bpm
    keys = _key_iter(rng, args.loops_per_key)

    target_s = args.minutes * 60.0
    preroll = 6.0  # headroom to find the downbeat and still get a full loop
    slots, total = [], 0.0
    i = 0
    while total < target_s:
        letter, num = next(keys)
        bar_s = mix.sec_per_bar(bpm)
        want = args.loop_bars * bar_s + preroll
        slots.append(Slot(
            index=i,
            gene_id=trial["id"],
            gene_text=trial["text"],
            bpm=bpm,
            key=CAMELOT[(letter, num)],
            camelot=f"{num}{letter}",
            seed=args.seed + i,
            gen_seconds=round(min(want, MAX_GEN_S), 2),
            loop_bars=args.loop_bars,
        ))
        if want > MAX_GEN_S:
            print(f"  ! {bpm:.1f} BPM: {args.loop_bars} bars + preroll exceeds "
                  f"{MAX_GEN_S:.0f}s, generating {MAX_GEN_S:.0f}s", file=sys.stderr)
        total += (args.loop_bars - args.xfade_bars) * bar_s
        i += 1
    # Slow LFO on the img2img transform strength: a sine with a 12-loop period
    # and +-0.2 depth around --transform-strength. High points stay close to the
    # previous loop (coherent), low points wander (fresh material), so the mix
    # breathes instead of degenerating into one looping cell.
    if args.transform_strength > 0:
        period, depth = 12, 0.1
        for sl in slots:
            sl.strength = round(min(0.85, max(0.25, args.transform_strength +
                depth * math.sin(2 * math.pi * sl.index / period))), 3)
    for i in range(1, len(slots)):
        slots[i].prev_seed = slots[i - 1].seed
    return slots, bpm, pool, trial


def _key_iter(rng, loops_per_key):
    """Walk the Camelot wheel: same / +-1 number, or relative major-minor."""
    num, letter = rng.randint(1, 12), rng.choice("AB")
    while True:
        for _ in range(loops_per_key):
            yield letter, num
        if rng.random() < 0.25:
            letter = "B" if letter == "A" else "A"
        else:
            num = (num - 1 + rng.choice([1, -1])) % 12 + 1


# ------------------------------------------------------------- generation ---

def load_pipe():
    import torch
    from diffusers import StableAudioPipeline

    print(f"loading {MODEL_ID} (first run downloads ~5GB)...", flush=True)
    pipe = StableAudioPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    sr = int(pipe.vae.sampling_rate)
    if sr != SR:
        raise SystemExit(f"model sampling rate {sr} != mixer's {SR}; update mix.SR")
    return pipe, torch


def loop_path(slot, loops_dir, args):
    """Cache name for a slot's raw (unstretched) generation.

    Includes the gene id (gene text is immutable per id) plus --steps,
    --guidance and the transform strength + previous loop's seed: they all
    change the audio, so they are part of the cache key.
    """
    ts = slot.strength if slot.strength is not None else args.transform_strength
    xf = (f"_x{ts:g}"
          + (f"p{slot.prev_seed}" if slot.prev_seed is not None else "")
          if args.transform_strength > 0 else "")
    return Path(loops_dir) / (
        f"gene{slot.gene_id}_{slug(slot.gene_text.split(',')[0])}_{int(round(slot.bpm))}"
        f"_{slot.seed}_s{args.steps}_g{args.guidance:g}{xf}.flac")


def img2img_loop(pipe, torch, slot, args, cond_loop):
    """Whole-loop transform: denoise from enc(cond_loop) + sigma_k*noise, starting
    at the intermediate timestep k = strength*(steps-1). Strength 0 is the
    pipeline's default start; strength 1 keeps almost all of the source loop.

    This is a real img2img: the diffusers port's built-in initial_audio_waveforms
    is inert (it adds the audio at ~1/500th the noise scale, so the model washes
    it out -- measured). Starting at an intermediate timestep keeps the content
    on the trained manifold so it survives.
    """
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    device = pipe._execution_device
    gen = torch.Generator("cuda").manual_seed(slot.seed)
    x = torch.from_numpy(cond_loop.T.copy()).float().cuda().half().unsqueeze(0)
    with torch.no_grad():
        enc = pipe.vae.encode(x).latent_dist.sample(gen)
    enc = enc.cpu(); del x
    n_frames = int(pipe.transformer.config.sample_size)
    enc_pad = torch.zeros(1, enc.shape[1], n_frames, dtype=torch.float16)
    enc_pad[:, :, : enc.shape[2]] = enc.half()

    do_cfg = args.guidance > 1.0
    prompt_embeds = pipe.encode_prompt(slot.prompt, device, do_cfg, NEGATIVE,
                                       None, None, None, None)
    start = torch.tensor([0.0], device=device)
    end = torch.tensor([slot.gen_seconds], device=device)
    ss, se = pipe.encode_duration(start, end, device, do_cfg, 1)
    ted = torch.cat([prompt_embeds, ss, se], dim=1)
    ade = torch.cat([ss, se], dim=2)

    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    ts = slot.strength if slot.strength is not None else args.transform_strength
    k = int(round(ts * (len(timesteps) - 1)))
    k = min(max(k, 0), len(timesteps) - 1)
    sigma_k = float(pipe.scheduler.sigmas[k])
    eps = torch.randn(enc_pad.shape, device=device, dtype=torch.float16, generator=gen)
    latents = enc_pad.half().cuda() + sigma_k * eps

    rotary = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim, latents.shape[2] + ade.shape[1],
        use_real=True, repeat_interleave_real=False)
    with torch.no_grad():
        for t in timesteps[k:]:
            lmi = torch.cat([latents] * 2) if do_cfg else latents
            lmi = pipe.scheduler.scale_model_input(lmi, t)
            np_ = pipe.transformer(lmi, t.unsqueeze(0), encoder_hidden_states=ted,
                                   global_hidden_states=ade, rotary_embedding=rotary,
                                   return_dict=False)[0]
            if do_cfg:
                nu, nt = np_.chunk(2)
                np_ = nu + args.guidance * (nt - nu)
            latents = pipe.scheduler.step(np_, t, latents).prev_sample
        audio = pipe.vae.decode(latents).sample
    return audio[0, :, : int(slot.gen_seconds * SR)].T.float().cpu().numpy()


def generate_slot(pipe, torch, slot, args, loops_dir, prev_slot):
    """Generate (or reuse) a slot's raw loop. -> (path, reused).

    With --transform-strength and a previous slot, the loop is a whole-loop
    transform of the previous prepared loop instead of a fresh text-to-audio
    generation.

    Silence gate: if more than --max-silence of the prepared (heard) loop is
    below -45 dBFS, or the groove starts later than --max-intro (kickless
    intro: the model sometimes ignores the NEGATIVE prompt), the loop is
    rejected and regenerated with a bumped seed, up to --silence-attempts
    tries; the least-bad try wins as fallback.
    """
    best = None  # ((intro, frac), seed): non-intro loops preferred, then least silent
    for attempt in range(max(1, args.silence_attempts)):
        path = loop_path(slot, loops_dir, args)
        slot.path = str(path)
        fresh = not (path.exists() and not args.no_cache)
        if fresh:
            gen = torch.Generator("cuda").manual_seed(slot.seed)
            if args.transform_strength > 0 and prev_slot is not None:
                raw_prev, sr = sf.read(prev_slot.path, dtype="float32", always_2d=True)
                cond, _ = mix.prepare_loop(raw_prev, sr, prev_slot.bpm, args.loop_bars,
                                           stretch=not args.no_stretch,
                                           align="onset" if args.onset_trim
                                           else not args.no_align,
                                           max_silence=args.max_silence)
                cond = mix.rms_normalize(cond)
                wav = img2img_loop(pipe, torch, slot, args, cond)
            else:
                audio = pipe(slot.prompt, negative_prompt=NEGATIVE,
                             num_inference_steps=args.steps,
                             audio_end_in_s=slot.gen_seconds,
                             guidance_scale=args.guidance, generator=gen).audios
                wav = audio[0].T.float().cpu().numpy()
            sf.write(path, wav, SR)
        raw, sr = sf.read(path, dtype="float32", always_2d=True)
        loop, info = mix.prepare_loop(raw, sr, slot.bpm, args.loop_bars,
                                      stretch=not args.no_stretch,
                                      max_silence=args.max_silence)
        frac = mix.silence_fraction(loop, SR)
        intro = info.get("first_kick_s") or 0.0
        score = (intro > args.max_intro, frac)
        if best is None or score < best[0]:
            best = (score, slot.seed)
        if frac <= args.max_silence and intro <= args.max_intro:
            return path, not fresh
        if intro > args.max_intro:
            print(f"  ! [{slot.index}] rejected: {intro:.1f}s kickless intro "
                  f"(> {args.max_intro:.0f}s), retrying with seed {slot.seed + 100000}",
                  flush=True)
        else:
            print(f"  ! [{slot.index}] rejected: {frac:.0%} silent (> {args.max_silence:.0%}), "
                  f"retrying with seed {slot.seed + 100000}", flush=True)
        slot.seed += 100000
    # exhausted attempts: fall back to the least-bad version
    slot.seed = best[1]
    slot.path = str(loop_path(slot, loops_dir, args))
    print(f"  ! [{slot.index}] all {args.silence_attempts} attempts rejected; "
          f"keeping best", flush=True)
    return slot.path, False


# ------------------------------------------------------------------- mixin ---

def build_mix(slots, args, out_path):
    loops, bpms, report = [], [], []
    for s in slots:
        raw, sr = sf.read(s.path, dtype="float32", always_2d=True)
        info = {}
        if sr != SR:
            raise SystemExit(f"{s.path}: expected {SR}Hz, got {sr}Hz")
        loop, info = mix.prepare_loop(raw, sr, s.bpm, args.loop_bars,
                                      stretch=not args.no_stretch,
                                      align="onset" if args.onset_trim
                                      else not args.no_align,
                                      max_silence=args.max_silence)
        loops.append(mix.rms_normalize(loop, args.loop_dbfs))
        bpms.append(s.bpm)
        report.append({**asdict(s), **{k: (None if v is None else round(float(v), 4))
                                       for k, v in info.items()}})
        det = info["detected_bpm"]
        print(f"  [{s.index:3d}] {s.camelot:>3} {s.bpm:6.2f} BPM  "
              f"detected {round(det, 2) if det else 'n/a':>6}  "
              f"rate {info['stretch_rate']:.4f}", flush=True)

    y = mix.assemble(loops, bpms, args.xfade_bars)
    y = mix.fade_edges(y, SR, bpms[0], args.fade_bars, args.fade_bars)
    y = mix.finalize(y)
    if not getattr(args, "no_fx", False):
        import fx
        y = fx.fx(y, loops, bpms, slots, args.xfade_bars)
        y = fx.loudness(y)
    if args.lowcut > 0 or args.highcut > 0:
        print(f"band limiting: low cut {args.lowcut or 'off'} Hz, "
              f"high cut {args.highcut or 'off'} Hz", flush=True)
        y = mix.band_limit(y, SR, args.lowcut or None, args.highcut or None)
    sf.write(out_path, y, SR)
    return y, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--style", default=None,
                    help="only try genes whose prompt contains this substring")
    ap.add_argument("--bpm", type=float, default=None, help="fixed tempo (no drift)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loop-bars", type=int, default=16)
    ap.add_argument("--xfade-bars", type=int, default=4)
    ap.add_argument("--fade-bars", type=int, default=8)
    ap.add_argument("--loops-per-key", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100, help="diffusion steps (25 fast, 100 default)")
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--loop-dbfs", type=float, default=-18.0)
    ap.add_argument("--lowcut", type=float, default=32.0,
                    help="high-pass the mix at this Hz to remove boomy sub "
                         "bass (0 disables)")
    ap.add_argument("--highcut", type=float, default=16000.0,
                    help="low-pass the mix at this Hz to remove sibilance/"
                         "distortion artifacts (0 disables)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--loops-dir", default="loops")
    ap.add_argument("--mix-only", action="store_true", help="reuse cached loops, no GPU")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-silence", type=float, default=0.2,
                    help="reject + regenerate a loop when more than this "
                         "fraction of the heard loop is silent (0.2 = 20%%)")
    ap.add_argument("--max-intro", type=float, default=6.0,
                    help="reject + regenerate a loop whose groove starts "
                         "later than this many seconds in (kickless intro; "
                         "0 disables)")
    ap.add_argument("--silence-attempts", type=int, default=3)
    ap.add_argument("--transform-strength", type=float, default=0.6,
                    help="0..1: each loop is a whole-loop transform of the previous "
                         "one instead of a fresh generation (0 = off). ~0.5-0.7 is a "
                         "good starting point.")
    ap.add_argument("--no-stretch", action="store_true",
                    help="trust the model's tempo instead of matching it (for A/B)")
    ap.add_argument("--onset-trim", action="store_true",
                    help="trim each loop at its first strong onset instead of "
                         "the voted downbeat (normalizes Stable Audio's ~0.1s "
                         "start offset across the chain)")
    ap.add_argument("--no-align", action="store_true",
                    help="disable downbeat detection: trim at t=0 and trust the "
                         "transform chain for phase (for A/B; implies raw cuts)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--no-fx", action="store_true",
                    help="disable the post-render DJ effects + loudness pass "
                         "(fx is ON by default; see fx.py)")
    args = ap.parse_args()

    if args.seed is None:
        # No fixed seed: stamp the run with the current wallclock time so every
        # generation differs (variety over reproducibility). Pass --seed to
        # override and reproduce a specific mix.
        args.seed = int(dt.datetime.now().timestamp() * 1_000_000)
    if args.style and args.style not in STYLES:
        raise SystemExit(f"unknown style {args.style!r}; try: {', '.join(sorted(STYLES))}")
    if args.xfade_bars >= args.loop_bars:
        raise SystemExit("--xfade-bars must be smaller than --loop-bars")

    slots, bpm, pool, trial = make_plan(args)
    print(f"plan: {len(slots)} loops, gene #{trial['id']} (score {genes.score(trial):+d}, "
          f"{trial['plays']} plays): {trial['text']}")
    print(f"{bpm:.1f} BPM fixed, {args.loop_bars}-bar loops, "
          f"{args.xfade_bars}-bar blends")

    if args.plan_only:
        for s in slots:
            print(f"  [{s.index:3d}] {s.camelot:>3} {s.bpm:6.2f} BPM {s.gen_seconds:5.1f}s  {s.prompt}")
        print(f"\n-> plan only, nothing generated. {len(slots)} generations "
              f"~{len(slots) * 0.28 * args.steps / 60:.0f} min on an RTX 3060.")
        return
    else:
        for s in slots:
            print(f"  gene #{s.gene_id}: {s.prompt}")

    loops_dir = Path(args.loops_dir)
    loops_dir.mkdir(exist_ok=True)
    out_dir = Path("mixes")
    out_dir.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = Path(args.out) if args.out else out_dir / f"mix_{stamp}_{args.minutes:g}min.flac"

    t0 = time.time()
    if not args.mix_only:
        pipe, torch = load_pipe()
        prev_slot = None
        for s in slots:
            _, reused = generate_slot(pipe, torch, s, args, loops_dir, prev_slot)
            print(f"  [{s.index:3d}] {'reused' if reused else 'generated'} {s.path} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
            prev_slot = s
        trial["plays"] += 1  # a real render counts as a trial
        genes.save(pool)
    else:
        for s in slots:
            s.path = str(loop_path(s, loops_dir, args))
            if not Path(s.path).exists():
                raise SystemExit(f"--mix-only but {s.path} is missing; run without it first")

    print("blending...")
    y, report = build_mix(slots, args, out_path)
    dur = len(y) / SR
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "path": str(out_path),
        "minutes": args.minutes,
        "actual_seconds": round(dur, 2),
        "gene": trial,
        "bpm": bpm,
        "loop_bars": args.loop_bars,
        "xfade_bars": args.xfade_bars,
        "steps": args.steps,
        "seed": args.seed,
        "slots": report,
    }, indent=2))
    print(f"wrote {out_path} ({dur / 60:.2f} min, {out_path.stat().st_size / 1e6:.0f} MB) "
          f"+ {sidecar.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
