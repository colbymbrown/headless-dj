# Post-Render DJ Effects + Loudness — Implementation Plan

Branch: `effects` · Integration point: inside `build_mix`, after `mix.finalize(y)`,
before `sf.write(out_path, y, SR)`.

## Pipeline (order matters)
1. Render plain mix (existing): `assemble` -> `fade_edges` -> `finalize`.
2. **FX pass** -> `mixes/*_fx.flac`
3. **Loudness/mastering pass** (LUFS match + peak limiter) on the FX output -> final render.

Loudness runs LAST so it sees the true final signal (echo/reverb tails change peaks).

## Guiding principles (from DJ practice research)
- Effects are **sparse** and **phrase-locked** — hit bar boundaries, never sustained.
- **Felt, not noticed** — but defaults are set LOUD/AUDIBLE initially so we can judge them,
  then dialed back later.
- **Rate/frequency-based**, not count-based: a 60-min mix gets ~3x the FX of a 20-min one
  via per-phrase-boundary probability, not "N per mix".
- Guard against clumping: no FX on back-to-back transitions.
- **Skip gracefully**: if a marker can't host an effect, render plain for that moment + log.
- **Bass-protected**: all FX sends high-passed at 180 Hz (reuse crossover) so the kick/bass stays clean.

## Effect palette (Pedalboard)
| Effect | Plugin | Role |
|---|---|---|
| Filter build sweep | `LowpassFilter` | tension into a drop |
| Echo/repeats-out | `Delay` (tempo-linked rhythm) | transition outro |
| Reverb wash | `Reverb` | breakdown / key-change tail |
| Phaser | `Phaser` | rare mid-mix texture |

## Marker model (precomputed in dj.py's plan)
- Transition points: blend offsets (bar index + sample offset).
- Key-change points: where Camelot key changes between loops.
- Loudness ledger: per-loop RMS (or LUFS) to flag quiet sections for breakdowns and
  quiet->loud boundaries for drops.

## Effect program (defaults; all tunable in a config)
1. **Echo-out**: fires with probability p_echo (~every 3rd-4th transition, sparser than
   original). Rhythm chosen per-use, tempo-linked via beat-duration multiplier pool:
   **1/4, dotted 1/8, dotted half, 3/16 offbeat** (dotted half weighted to sometimes star).
   On outgoing loop's last bar. Wet ~70-80% (audible default).
2. **Reverb wash**: fires with p_reverb (~1/4 transitions) AND before every key change.
   High-passed send, ring 1-2 bars into incoming loop. Wet ~40% (audible default).
3. **Filter build sweep**: only on a loud-drop boundary (quiet->loud ledger edge).
   Low-pass cutoff ramps ~400 Hz -> open over the last 8 bars into the drop.
   Clearly-audible range by default.
4. **Phaser**: ~once per mix (probability per phrase, no hard count), mid-mix quiet section.
   Wet ~25-40% (audible default).

## Rhythm pool for echo-out (tempo-linked, via beat_duration * multiplier)
- 1/4  (mult 1.0)
- dotted 1/8 (mult 0.75)
- dotted half (mult 3.0, weighted to sometimes star)
- 3/16 offbeat (mult 1.5)
All computed from the mix's fixed BPM.

## Config knobs (JSON, e.g. fx_config.json)
- p_echo, p_reverb, phaser_prob, filter_requires_drop(bool)
- wet_dry defaults (audible) + later a "subtle" preset
- rhythm pool weights
- max_consecutive_fx (guard, e.g. 1 = no back-to-back)
- loudness: target LUFS, limiter ceiling, headroom

## Loudness stage (separate concern, sequenced last)
- Replace per-loop fixed-RMS target (`rms_normalize` to fixed -18 dBFs caused the
  intensity jumps) with **LUFS-based matching** to a target loudness.
- True **peak limiter** at the end to kill clipping (crossfade summation can exceed
  per-loop clamps).
- Runs after FX so it matches the final signal.
