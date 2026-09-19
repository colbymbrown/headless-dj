"""Pure DSP for the DJ mixer. No torch here on purpose, so blends can be
re-rendered from cached loops in seconds without touching the GPU."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 44100
BEATS_PER_BAR = 4
CROSSOVER_HZ = 180.0  # bass-swap crossover
BASS_SWAP_BEATS = 2.0  # how fast the low end hands over, once, mid-blend

# ponytail: one 180Hz 4th-order crossover. Upgrade: 3-band EQ (bass/low-mid/high)
# if blends still feel muddy, or a mid-band duck to stop two basslines fighting.
_SOS = butter(4, CROSSOVER_HZ / (SR / 2), btype="low", output="sos")

# Max stretch factor: beyond this we suspect a beat-tracker error and fall back
# to simple resampling (pitch shifts are acceptable there since it's a last resort).
_MAX_STRETCH = 1.5   # e.g. detected 90 BPM when asked for 128 → ~43% stretch
_MIN_STRETCH = 1.0 / _MAX_STRETCH  # ~0.67


def sec_per_bar(bpm, beats_per_bar=BEATS_PER_BAR):
    return beats_per_bar * 60.0 / bpm


def bars_to_samples(bars, bpm, sr=SR):
    return int(round(bars * sec_per_bar(bpm) * sr))


# ---------------------------------------------------------------- analysis ---

def refine_tempo(oenv, sr, hop, tempo0, beats, span=0.03, steps=241):
    """Recover sub-bin tempo accuracy.

    The tempogram only resolves tempo to 60*fps/integer_lag, which near 126 BPM
    is a ~3 BPM quantization step. Two loops that far apart end a 4-bar blend
    with the kicks visibly (and audibly) flammed. So instead fit the tempo whose
    beat grid stays on the onsets for the whole loop.

    ponytail: brute-force scan around the coarse estimate; a few ms on a 35s
    loop, and tightening `span` only risks missing truly wild model output.
    """
    if len(oenv) == 0:
        return tempo0
    fps = sr / hop
    first = float(beats[0]) if len(beats) else 0.0
    remaining = len(oenv) - first
    if remaining <= 0:
        return tempo0
    frames = np.arange(len(oenv))

    def score(tempo):
        period = 60.0 / tempo * fps
        grid = first + np.arange(0.0, remaining / period) * period
        return np.interp(grid, frames, oenv).mean() if len(grid) else -np.inf

    return max(tempo0 * (1.0 + np.linspace(-span, span, steps)), key=score)


def analyze_timing(mono, sr, target_bpm, hop=512):
    """(tempo, beat_frames, onset_envelope, hop). The tempo prior is seeded with
    the requested BPM, which is the biggest robustness win: the model was asked
    for that tempo, so half/double errors become unlikely. tempo=None means
    nothing plausible was found and the caller should trust the prompt."""
    import librosa

    oenv = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=hop, units="frames",
        start_bpm=float(target_bpm),
    )
    tempo = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(tempo) or tempo <= 0:
        return None, np.array([]), oenv, hop
    while tempo < target_bpm / 1.5:
        tempo *= 2.0
    while tempo > target_bpm * 1.5:
        tempo /= 2.0
    beats = np.asarray(beats, dtype=float)
    return refine_tempo(oenv, sr, hop, tempo, beats), beats, oenv, hop


def refine_beats(beats, oenv, hop):
    """Snap each beat frame to the nearest onset-envelope peak.

    beat_track returns frames on the hop grid (11.6 ms at hop 512), which is
    visible as anchor jitter when warping. Parabolic peak on the envelope
    removes the grid quantization.
    """
    out = []
    n = len(oenv)
    for b in beats:
        i = int(round(b))
        lo, hi = max(0, i - 4), min(n, i + 5)
        w = oenv[lo:hi]
        if len(w) < 3:
            out.append(b)
            continue
        k = lo + int(np.argmax(w))
        if 1 <= k - lo < len(w) - 1:  # parabolic sub-frame peak
            a, c0, c1 = w[k - lo - 1], w[k - lo], w[k - lo + 1]
            denom = a - 2 * c0 + c1
            if denom != 0:
                k += 0.5 * (a - c1) / denom
        out.append(k)
    return np.asarray(out, dtype=float)


def pick_downbeat(beats, oenv):
    """Index of the first beat on the strongest 4/4 phase.

    Real downbeats are detected by voting: sum the onset energy of every beat
    at each beat-of-the-bar position and take the strongest. This is robust to
    syncopated or breakdown sections that fool single-onset heuristics.
    """
    if len(beats) < 4:
        return 0
    i = np.clip(beats.astype(int), 0, len(oenv) - 1)
    e = oenv[i]
    phases = np.arange(len(beats)) % 4
    scores = [e[phases == p].sum() for p in range(4)]
    p = int(np.argmax(scores))
    return int(np.nonzero(phases == p)[0][0])


def resample_time(y, rate):
    """Constant-rate tempo change via linear resampling (pitch preserved).
    Used only as a fallback when beat tracking finds nothing."""
    if abs(rate - 1.0) < 0.005:
        return y
    n = int(len(y) / rate)
    t = np.arange(n, dtype=np.float64) / rate
    x = np.arange(len(y))
    return np.column_stack(
        [np.interp(t, x, y[:, c]) for c in range(y.shape[1])]
    ).astype(np.float32)


def prepare_loop(raw, sr, target_bpm, loop_bars, stretch=True):
    """Raw generation -> exactly `loop_bars` of beat-locked audio.

    A single global tempo is fit to the whole loop, then the audio is stretched
    at that one constant rate via RubberBand (pitch-preserving, transient-aware)
    to the mix tempo. Unlike the old per-beat varispeed warp this preserves pitch
    throughout and avoids erratic beat-anchor jumps; unlike naive resampling it
    does not shift pitch.

    Trim is by actual audio length: the loop is cut exactly at `loop_bars`, so
    no silent tail can appear. If stretch is absurd (beat-tracker error) or
    RubberBand fails, we fall back to cheap linear resampling and zero-pad only
    the shortfall.
    """
    import pyrubberband as prb

    mono = raw.mean(axis=1)
    det_bpm, beats, oenv, hop = analyze_timing(mono, sr, target_bpm)
    desired_len = bars_to_samples(loop_bars, target_bpm, sr)
    rate = target_bpm / det_bpm if (det_bpm and stretch) else 1.0
    padded = 0
    used_stretch = False

    # Start the loop on the detected downbeat so blends phase-lock at the bar line.
    start = int(pick_downbeat(beats, oenv) * hop) if len(beats) else 0
    body = raw[start:]  # source audio from the downbeat on

    # Constant-tempo stretch, only when the rate is sane (else it's a tracker error)
    # and the clip is long enough that stretching won't run past its end.
    if (det_bpm is not None and len(beats) >= 8 and stretch
            and _MIN_STRETCH <= rate <= _MAX_STRETCH
            and len(body) >= int(desired_len / _MAX_STRETCH)):
        try:
            # pyrubberband takes (n, channels) directly — no transpose needed.
            stretched = prb.time_stretch(body, sr, rate)
        except Exception:
            stretched = None
        if stretched is not None and len(stretched) >= desired_len:
            out = stretched[:desired_len].astype(np.float32)
            used_stretch = True
        else:
            stretched = None
    else:
        stretched = None

    if stretched is None:  # fallback: cheap resample, trim, zero-pad shortfall
        resampled = resample_time(body, rate)
        out = resampled[:desired_len]
        padded = max(0, desired_len - len(out))
        if padded:
            out = np.pad(out, ((0, padded), (0, 0)))

    return out.astype(np.float32), {
        "detected_bpm": det_bpm,
        "stretch_rate": rate,
        "stretched": used_stretch,
        "warped": used_stretch,  # alias for back-compat; we no longer per-beat warp
        "padded_samples": padded,
        "padded_seconds": round(padded / sr, 3),
    }


def silence_fraction(loop, sr, thresh_dbfs=-45.0, win_s=0.05):
    """Fraction of short windows of the prepared (actually-heard) loop whose
    RMS sits below thresh_dbfs. Loops with long dead stretches sound broken
    when beat-locked into a mix, so dj.py rejects them."""
    win = max(1, int(win_s * sr))
    n = len(loop) // win * win
    if n == 0:
        return 1.0
    frames = loop[:n].reshape(-1, win, loop.shape[1])
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=(1, 2)))
    return float((rms < 10 ** (thresh_dbfs / 20.0)).mean())


# ------------------------------------------------------------------ mixing ---

def rms_normalize(y, target_dbfs=-18.0, peak_ceiling=0.95):
    """Match loop loudness so the mix does not jump level at each blend."""
    y = y * (10 ** (target_dbfs / 20.0) / (np.sqrt(np.mean(y**2)) + 1e-9))
    peak = np.abs(y).max()
    if peak > peak_ceiling:
        y *= peak_ceiling / peak
    return y


def _bass_curve(t, swap):
    """Outgoing low-band gain over an overlap of `t` samples: full bass until the
    middle, then hand over across `swap` samples. Complement is 1 - this, so the
    low end is always exactly one loop's bass instead of both or neither."""
    swap = max(1.0, float(swap))
    j = np.arange(t, dtype=np.float32)
    return np.clip((t / 2 + swap / 2 - j) / swap, 0.0, 1.0)


def _place(out, seg, offset, fade_in_len, fade_out_len, swap_len):
    """Overlap-add one loop, split into two bands.

    The high band crossfades equal-power (sin/cos) so uncorrelated material
    sums to constant power with no level dip. The low band hands over in a
    short window mid-blend so two kicks/basslines never fight.
    """
    L = len(seg)
    low = sosfiltfilt(_SOS, seg, axis=0)
    high = seg - low
    gl = np.ones(L, dtype=np.float32)
    gh = np.ones(L, dtype=np.float32)

    if fade_in_len:
        t = np.linspace(0, np.pi / 2, fade_in_len, endpoint=False, dtype=np.float32)
        gh[:fade_in_len] = np.sin(t)
        gl[:fade_in_len] = 1.0 - _bass_curve(fade_in_len, swap_len)
    if fade_out_len:
        t = np.linspace(0, np.pi / 2, fade_out_len, endpoint=False, dtype=np.float32)
        gh[L - fade_out_len:] = np.cos(t)
        gl[L - fade_out_len:] = _bass_curve(fade_out_len, swap_len)

    out[offset : offset + L] += low * gl[:, None] + high * gh[:, None]


def assemble(loops, bpms, xfade_bars, sr=SR, swap_beats=BASS_SWAP_BEATS):
    """Chain loops with bar-length crossfades. Returns (samples, 2)."""
    n = len(loops)
    if n == 1:
        return loops[0].copy()
    # Overlap between i and i+1 is measured at the incoming tempo.
    taper = [bars_to_samples(xfade_bars, bpms[i + 1], sr) for i in range(n - 1)]
    swap = [int(swap_beats * 60.0 / bpms[i + 1] * sr) for i in range(n - 1)]
    for i, (t, seg) in enumerate(zip(taper, loops)):
        if t >= len(seg):
            raise ValueError(f"crossfade {t} samples >= loop {i} length {len(seg)}; "
                             f"lower --xfade-bars or raise --loop-bars")

    offsets = [0]
    for i in range(n - 1):
        offsets.append(offsets[-1] + len(loops[i]) - taper[i])

    out = np.zeros((offsets[-1] + len(loops[-1]), 2), dtype=np.float32)
    for i, seg in enumerate(loops):
        _place(
            out,
            seg,
            offsets[i],
            fade_in_len=taper[i - 1] if i > 0 else 0,
            fade_out_len=taper[i] if i < n - 1 else 0,
            swap_len=swap[i - 1] if i > 0 else swap[0],
        )
    return out


def fade_edges(y, sr, bpm, fade_in_bars, fade_out_bars):
    """Raised-sine fade in from silence and out to silence (exactly 0 at both ends)."""
    n_in = min(bars_to_samples(fade_in_bars, bpm, sr), len(y))
    n_out = min(bars_to_samples(fade_out_bars, bpm, sr), len(y))
    if n_in:
        t = np.linspace(0, np.pi / 2, n_in, dtype=np.float32)
        y[:n_in] *= np.sin(t)[:, None]
    if n_out:
        t = np.linspace(0, np.pi / 2, n_out, dtype=np.float32)
        y[-n_out:] *= np.cos(t)[:, None]
    return y


def band_limit(y, sr, low_hz=32.0, high_hz=16000.0):
    """High-pass the boomy, non-musical sub and low-pass harsh sibilance /
    distortion artifacts. 4th-order Butterworth, zero-phase (sosfiltfilt).
    Set low_hz/high_hz to 0 (or None) to skip that edge."""
    ny = sr / 2
    if low_hz and low_hz > 0:
        y = sosfiltfilt(butter(4, low_hz / ny, btype="high", output="sos"),
                        y, axis=0).astype(np.float32)
    if high_hz and high_hz > 0 and high_hz < ny:
        y = sosfiltfilt(butter(4, high_hz / ny, btype="low", output="sos"),
                        y, axis=0).astype(np.float32)
    return y


def finalize(y, peak=0.89):
    """Peak-normalize the finished mix to about -1 dBFS."""
    p = np.abs(y).max()
    return y * (peak / p) if p > 0 else y
