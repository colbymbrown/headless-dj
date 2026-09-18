# headless-dj

A bot that builds a DJ-style dance mix with Stable Audio Open. It picks a style,
a tempo and a key, generates 16-bar loops, matches each loop's tempo to the mix
tempo, and blends loop into loop with bar-length crossfades that swap the bass
once, mid-blend. The mix fades in from and out to silence.

Loops are cached on disk, so re-rendering a mix with different blend settings
takes seconds and needs no GPU.

## Setup

Stable Audio Open is behind a gated HF repo, so this one step needs your account:

1. Accept the licence at https://huggingface.co/stabilityai/stable-audio-open-1.0
2. `hf auth login` (a read token is enough)

Then:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python test_mix.py          # 8 checks, no GPU or network needed
```

First real run downloads ~5 GB of weights.

## Use

```bash
# See the plan (style, tempo drift, key walk, prompts) without touching the GPU.
.venv/bin/python dj.py --plan-only --minutes 15

# Render a 15 minute mix. Writes mixes/mix_<date>_15min.flac + a .json plan.
.venv/bin/python dj.py --minutes 15

# Re-blend from cached loops: no model, ~seconds.
.venv/bin/python dj.py --minutes 15 --mix-only

# Ready for the hour once you trust it.
.venv/bin/python dj.py --minutes 60
```

Useful knobs: `--style`, `--bpm` (fixes tempo, no drift), `--seed`,
`--steps` (25 for a fast preview, 100 default), `--xfade-bars`, `--fade-bars`,
`--loop-bars`, `--no-stretch` (trust the model's tempo, for A/B).

`--transform-strength 0..1` chains loops as *transforms*: each loop is
re-imagined from the previous one's audio instead of generated fresh, so
consecutive loops share timbre and structure (measured: adjacent-loop
MFCC distance ~37 vs ~57 unconditional). ~0.5-0.7 is a good starting point;
0.9 is nearly a copy, 0.2 barely steers. The first loop is always generated
fresh.

Styles live in the `STYLES` dict in `dj.py`; add your own by copying a line.

### Daily

```cron
30 4 * * *  cd /home/colby/MediaSynthesis/headless-dj && .venv/bin/python dj.py --minutes 60 >> mixes/cron.log 2>&1
```

## How it works

`mix.py` is pure DSP with no torch, which is the point: blends can be re-rendered
in seconds without the model.

- **Tempo & timing.** Stable Audio Open takes no BPM control, so its output is
  measured, not trusted. Each loop is **warped onto a constant-BPM beat grid**
  (the Mixxx/Ableton approach): beats are tracked with the requested tempo as
  the prior, refined to sub-frame onset peaks, and the audio is resampled so
  every beat lands exactly on the grid. This matters because SAO loops drift
  off the grid internally — measured up to 258 ms (half a beat) by the loop's
  tail, which is exactly the region a blend uses. Warping fixes it: worst
  kick-to-grid error after warp is ~37 ms. Tempo still drifts across the mix
  (~3 BPM per 15 min), so neighbours differ by < 0.1 BPM and never need
  audible re-adjustment.
- **Key.** The model does not reliably honour key either, so the mix walks the
  Camelot wheel instead and holds each key for `--loops-per-key` loops. Every
  transition is same-key, ±1, or relative major/minor.
- **Downbeat.** Loops start on a detected downbeat: the 4/4 phase that carries
  the most onset energy across the whole loop wins (robust to breakdowns and
  syncopation).
- **Transform (optional).** `--transform-strength` implements real audio
  img2img. The diffusers port's built-in `initial_audio_waveforms` is inert:
  it adds the source audio to the initial latents at ~1/500th the noise scale
  (measured: conditioned output indistinguishable from unconditional). We
  instead encode the previous loop, start the denoising loop at an
  intermediate timestep `x = enc(loop) + sigma_k*noise` (on the model's
  trained manifold), and denoise from there with the new loop's prompt.
- **Blend.** Over `--xfade-bars`: the high band crossfades equal-power (sin/cos,
  no level dip on uncorrelated material) while the low end hands over once, in a
  short window mid-blend, so two kicks or basslines never fight.
- **Level.** Each loop is RMS-matched before assembly and the finished mix is
  peak-normalised to -1 dBFS.

## Known ceilings

- **Kick timing accuracy ~37 ms worst case.** The warp anchors on detected beats;
  residual error is onset-envelope frame resolution plus SAO's own soft kick
  timing. Audibly fine for dance music; a finer hop or a proper transient
  detector would shave it further.
- **Key is a prompt hint.** If the model ignores it, two loops can still clash;
  the crossfade is what hides it.
- **RMS, not LUFS.** Per-loop level matching is RMS, not perceived loudness.
  Swap in `pyloudnorm` if level jumps ever bother you.
- **Everything is held in RAM.** A 60 minute mix peaks around 3 GB. Fine here;
  stream to disk if this ever runs somewhere smaller.
## Post-render FX + loudness

See **docs_effect_plan.md** (on the  branch) for the plan to add DJ-style effects
(filter sweep, echo-out, reverb, phaser) and LUFS/limiter loudness after the mix is rendered.
