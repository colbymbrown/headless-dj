"""Pure DSP for the DJ mixer. No torch here on purpose, so blends can be
re-rendered from cached loops in seconds without touching the GPU."""
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve

from pedalboard import Compressor

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


def sec_per_bar(bpm):
    return BEATS_PER_BAR * 60.0 / bpm


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


def first_onset_time(oenv, hop, sr, thresh_frac=0.1, max_search_s=2.0):
    """Sample offset of the first strong onset near t=0.

    Diagnostic finding: Stable Audio starts its beat ~0.1s after t=0 and the
    img2img chain preserves that offset — but a few generations start early.
    Cutting every loop at ITS first onset (low threshold, so a soft filtered
    kick still counts) normalizes phase across the chain.
    """
    n = min(len(oenv), int(max_search_s * sr / hop))
    if n < 3:
        return 0
    seg = oenv[:n]
    mx = seg.max()
    if mx <= 0:
        return 0
    peaks = [i for i in range(n) if (
        (i == 0 or seg[i] >= seg[i - 1]) and (i == n - 1 or seg[i] > seg[i + 1]))]
    for i in peaks:
        if seg[i] >= thresh_frac * mx:
            return int(i * hop)
    return 0


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


def _mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def _chain_start(parent, child, sr, bpm, loop_bars):
    """Trim sample offset for a chain child: where the parent's downbeat
    landed in this generation.

    Each img2img transform shifts the whole groove by its own constant offset
    (measured 0..0.7 beat, constant over the loop), so the only reliable
    anchor is the parent itself: correlate its prepared downbeat against the
    child's onset envelope, then nudge until the blend window (parent's last
    4 bars vs child's first 4) is aligned -- a beatmatch-and-nudge, accepted
    only when it lands within 60 ms of perfect.

    ponytail: 2-beat reference / 3-beat search; revisit if intro-heavy
    generations grow beyond that.
    """
    import librosa
    hop = 512
    fps = sr / hop
    ce = librosa.onset.onset_strength(y=_mono(child), sr=sr, hop_length=hop)
    if parent is None:
        _, beats = librosa.beat.beat_track(onset_envelope=ce, sr=sr,
                                           hop_length=hop, units="frames",
                                           start_bpm=bpm)
        beats = np.atleast_1d(beats)
        return int(beats[0]) * hop if len(beats) else 0
    pe = librosa.onset.onset_strength(y=_mono(parent), sr=sr, hop_length=hop)
    k = min(len(pe), int(2 * 60 / bpm * fps))
    seg = ce[:min(len(ce), int(3 * 60 / bpm * fps))].astype(np.float64)
    re = pe[:k].astype(np.float64)
    re -= re.mean()
    if len(seg) <= k:
        return 0
    c = fftconvolve(seg, re[::-1], mode="full")[k - 1:][:len(seg) - k + 1]
    csum = np.convolve(seg * seg, np.ones(k), "valid")[:len(c)]
    start = int(np.argmax(c / np.sqrt(np.maximum(csum, 1e-9) * (re * re).sum()))) * hop

    # blend-window nudge: parent's last 4 bars vs candidate child's first 4
    w = np.hanning(int(0.02 * sr))
    w /= w.sum()
    sos = butter(4, [300 / (sr / 2), 8000 / (sr / 2)], btype="band", output="sos")
    def pulse(x):
        d = np.convolve(np.maximum(np.diff(np.abs(sosfilt(sos, x[:, 0]))), 0), w, "same")
        return d - d.mean()
    def blend_lag(cand):
        beat = 60.0 / bpm
        a, b = pulse(parent), pulse(cand)
        m = int(4 * beat * sr)
        aa, bb = a[len(a) - m:], b[:m]
        maxlag = int(0.75 * beat * sr)
        cc = fftconvolve(aa, bb[::-1], mode="full")[m - 1 - maxlag:m - 1 + maxlag]
        lag = (int(np.argmax(cc)) - maxlag) / sr
        return ((lag / beat + 0.5) % 1 - 0.5) * beat
    def trim(st):
        want = bars_to_samples(loop_bars, bpm, sr)
        body = child[st:st + want]
        if len(body) < want:
            body = np.pad(body, ((0, want - len(body)), (0, 0)))
        return body.astype(np.float32)
    cand = trim(start)
    for _ in range(2):
        lag = blend_lag(cand)
        if abs(lag) <= 0.06:
            break
        st2 = max(0, start - int(lag * sr))
        cand2 = trim(st2)
        lag2 = blend_lag(cand2)
        if abs(lag2) < abs(lag) and abs(lag2) <= 0.06:
            start, cand = st2, cand2
        else:
            break
    return start


def resample_time(y, rate):
    """Constant-rate tempo change via linear resampling (varispeed: pitch
    moves with tempo, vinyl-style). Used for per-loop stretch and as a
    fallback when beat tracking finds nothing."""
    if abs(rate - 1.0) < 0.005:
        return y
    n = int(len(y) / rate)
    t = np.arange(n, dtype=np.float64) / rate
    x = np.arange(len(y))
    return np.column_stack(
        [np.interp(t, x, y[:, c]) for c in range(y.shape[1])]
    ).astype(np.float32)


def varispeed(y, sr, r0, r1):
    """Time-varying varispeed: playhead rate ramps linearly from r0 to r1
    over the input. Pitch + tempo move together (vinyl deck feel).
    Applied post-fx so constant-tempo beatmatching is never disturbed."""
    if abs(r0 - 1.0) < 1e-4 and abs(r1 - 1.0) < 1e-4:
        return y
    n_in = y.shape[0]
    # input position at output sample o: r0*o + (r1-r0)*o^2/(2*L_out)
    # with L_in = L_out*(r0+r1)/2  =>  L_out = 2*L_in/(r0+r1)
    n_out = int(2 * n_in / (r0 + r1))
    o = np.arange(n_out, dtype=np.float64)
    in_pos = r0 * o + (r1 - r0) * o * o / (2.0 * n_out)
    x = np.arange(n_in, dtype=np.float64)
    return np.column_stack(
        [np.interp(in_pos, x, y[:, c]) for c in range(y.shape[1])]
    ).astype(np.float32)


def prepare_loop(raw, sr, target_bpm, loop_bars, stretch=True, align=True,
                 max_silence=None, align_ref=None):
    """Raw generation -> exactly `loop_bars` of beat-locked audio.

    align: True = trim at the first beat of the broadband pulse, "onset" =
    first strong onset, False = t=0.

    align_ref: the parent loop's PREPARED audio. Each img2img transform shifts
    the whole groove by its own constant offset (0..0.7 beat), so for chain
    children the trim is found by locating the parent's downbeat in this
    generation (see _chain_start) -- per-loop beat grids cannot do it, and
    low-band kick grids don't even exist on some genres (melodic techno).

    Silence fallback (max_silence): if the trimmed loop is more than this
    fraction silent, tile the longest silence-free power-of-2 prefix instead
    (16 -> 8 -> 4 -> 2 -> 1 bars, repeated to full length); info["tiled_bars"]
    records what was used. If even 1 bar is silent the loop is returned as-is
    and the caller's gate rejects it.

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
    desired_len = bars_to_samples(loop_bars, target_bpm, sr)
    if not align:
        # Chain-trust mode: no downbeat detection, no tempo correction. The
        # raw generation starts on its parent's downbeat by construction, so
        # trim at t=0 and cut to length; phase comes entirely from the chain.
        out = raw[:desired_len]
        padded = max(0, desired_len - len(out))
        if padded:
            out = np.pad(out, ((0, padded), (0, 0)))
        return out.astype(np.float32), {
            "detected_bpm": None, "stretch_rate": 1.0, "stretched": False,
            "stretched": False, "padded_samples": padded,
            "padded_seconds": round(padded / sr, 3),
            "first_beat_s": 0.0,
        }
    det_bpm, beats, oenv, hop = analyze_timing(mono, sr, target_bpm)
    rate = target_bpm / det_bpm if (det_bpm and stretch) else 1.0
    padded = 0
    used_stretch = False

    # Trim at the first beat of the broadband pulse. The broadband onset
    # envelope has a strong 1-beat periodicity on all corpus genres, while
    # the LOW band alone does not (melodic techno's rolling bass has no kick
    # grid to find -- measured: low-band autocorrelation ~ 0). beat_track's
    # global phase optimization is therefore the reliable anchor; the trim
    # must land on the pulse, anything else flams.
    if align == "onset":
        start = first_onset_time(oenv, hop, sr)
    elif align_ref is not None:
        start = _chain_start(align_ref, raw, sr, target_bpm, loop_bars)
    else:
        if not len(beats):
            raise SystemExit("no pulse found in loop; cannot beat-align "
                             "(rerun with a bumped seed)")
        beats = refine_beats(beats, oenv, hop)
        start = int(beats[0] * hop)
    body = raw[start:]  # source audio from the downbeat on

    # If the detected tempo is outside the sane stretch band, treat it as a
    # beat-tracker octave error and trust the prompt tempo (no stretch).
    # Otherwise a 2x-detected tempo yields rate=0.5, the rubberband guard
    # skips it, but the resample fallback below still applies it -> half-speed.
    if stretch and not (_MIN_STRETCH <= rate <= _MAX_STRETCH):
        rate = 1.0

    # Constant-tempo stretch, only when the rate is sane (else it's a tracker error)
    # and the clip is long enough that stretching won't run past its end.
    full = None
    if (det_bpm is not None and len(beats) >= 8 and stretch
            and _MIN_STRETCH <= rate <= _MAX_STRETCH
            and len(body) >= int(desired_len / _MAX_STRETCH)):
        try:
            # pyrubberband takes (n, channels) directly — no transpose needed.
            full = prb.time_stretch(body, sr, rate)
        except Exception:
            full = None
        if full is not None and len(full) >= desired_len:
            out = full[:desired_len].astype(np.float32)
            used_stretch = True
        else:
            full = None

    if full is None:  # fallback: cheap resample, trim, zero-pad shortfall
        full = resample_time(body, rate)
        out = full[:desired_len]
    padded = max(0, desired_len - len(out))
    if padded:
        out = np.pad(out, ((0, padded), (0, 0)))

    # Silence fallback: tile the longest silence-free power-of-2 prefix
    # (16 -> 8 -> 4 -> 2 -> 1 bars) to full length rather than ship a loop
    # with a dead stretch.
    tiled = None
    if max_silence is not None and silence_fraction(out, sr) > max_silence:
        bars = loop_bars
        while bars > 1:
            bars //= 2
            n = bars_to_samples(bars, target_bpm, sr)
            seg = full[:n]
            if len(seg) == n and silence_fraction(seg, sr) <= max_silence:
                out = np.tile(seg, (loop_bars // bars, 1)).astype(np.float32)
                padded = 0
                tiled = bars
                break

    return out.astype(np.float32), {
        "detected_bpm": det_bpm,
        "stretch_rate": rate,
        "stretched": used_stretch,
        "first_beat_s": round(start / sr, 3),
        "tiled_bars": tiled,
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


def groove_balance(loop, sr, bpm, bars=16):
    """Ratio of per-bar onset energy in the first half of the loop to the
    second half. A sustained groove sits near 1.0; a kickless intro ~0.5;
    a groove that drops out into a breakdown/pad after 8 bars -> infinity.
    Detects the "beat for only half" failure that silence_fraction misses
    (a pad at -30 dBFS slips past RMS but has no transient energy)."""
    import librosa
    hop = 512
    mono = _mono(loop)
    oenv = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    bar_frames = max(1, int(sec_per_bar(bpm) * sr / hop))
    nbars = len(oenv) // bar_frames
    if nbars < bars:
        return 1.0  # too short to judge; leave to other gates
    pe = oenv[:nbars * bar_frames].reshape(nbars, bar_frames).sum(1)
    h = nbars // 2
    a = float(pe[:h].mean())
    b = float(pe[h:].mean())
    if b <= 1e-9:
        return float("inf") if a > 1e-9 else 1.0
    return a / b


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


def multiband_compress(y, sr, cfg=None):
    """3-band compressor: tame per-band dynamics so no single frequency
    range jumps out unnaturally. Split at 200 Hz / 3 kHz (Linkwitz-Riley
    crossovers, zero-phase), compress each band, sum. Applied per-loop before
    normalization so hot bass or harsh highs are controlled before mixing."""
    if cfg is None:
        cfg = _MBC_DEFAULTS
    ny = sr / 2
    lo_lp = butter(4, cfg["xover_lo"] / ny, btype="low", output="sos")
    mid_hp = butter(4, cfg["xover_lo"] / ny, btype="high", output="sos")
    mid_lp = butter(4, cfg["xover_hi"] / ny, btype="low", output="sos")
    hi_hp = butter(4, cfg["xover_hi"] / ny, btype="high", output="sos")
    lo = sosfiltfilt(lo_lp, y, axis=0)
    mid = sosfiltfilt(mid_lp, sosfiltfilt(mid_hp, y, axis=0), axis=0)
    hi = sosfiltfilt(hi_hp, y, axis=0)
    lo = np.asarray(Compressor(**cfg["lo"])(lo, sr), dtype=np.float32)
    mid = np.asarray(Compressor(**cfg["mid"])(mid, sr), dtype=np.float32)
    hi = np.asarray(Compressor(**cfg["hi"])(hi, sr), dtype=np.float32)
    return (lo + mid + hi).astype(np.float32)


_MBC_DEFAULTS = {
    "xover_lo": 200.0,
    "xover_hi": 3000.0,
    "lo":  {"threshold_db": -18.0, "ratio": 3.0, "attack_ms": 10.0, "release_ms": 150.0},
    "mid": {"threshold_db": -20.0, "ratio": 2.0, "attack_ms": 8.0,  "release_ms": 120.0},
    "hi":  {"threshold_db": -22.0, "ratio": 2.5, "attack_ms": 5.0,  "release_ms": 80.0},
}


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
