# SPDX-License-Identifier: Apache-2.0
"""Tempo estimation for the MIDI writer (finding M-4).

The writer used to place every note on a fixed 120 BPM grid. That is fine for
absolute timing (the tick conversion is exact either way) but it makes the
exported file *look* like it was played at 120 BPM: a DAW's bar/beat grid lands
in the wrong place, quantisation does the wrong thing, and a 90 BPM recording
reads as syncopated.

``estimate_tempo_bpm`` derives the tempo from the signal itself with the
standard onset-envelope + autocorrelation method:

1. magnitude spectrogram (numpy rFFT, Hann window),
2. positive log-magnitude flux summed over bins -> onset strength per frame,
3. remove the DC term, autocorrelate, and search the lags that correspond to
   ``min_bpm``..``max_bpm``,
4. pick the strongest lag, slightly favouring 70-180 BPM to avoid the classic
   octave error, and refine it with a parabolic fit.

The estimate is a *guess*: it is reported next to the notes (``tempo_source``)
so a caller can tell a detected tempo from the default one. Nothing downstream
depends on it being right - it only changes the beat grid, never the absolute
note times (the writer converts frames with the same tempo it writes).
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEMPO_US = 500_000  # 120 BPM, the writer's historical grid
DEFAULT_MIN_BPM = 40.0
DEFAULT_MAX_BPM = 240.0
# Range that gets a small prior boost: most music the model sees is here, and
# autocorrelation otherwise reports half-tempo for strong backbeats.
_PRIOR_LO = 70.0
_PRIOR_HI = 180.0
_PRIOR_BOOST = 1.15
# Minimum autocorrelation peak height above the searched-lag median.
_MIN_PROMINENCE = 0.2
# Minimum normalised autocorrelation peak accepted as a tempo.
_MIN_CONFIDENCE = 0.35
# Fewer onsets than this and there is no pulse to measure.
_MIN_ONSETS = 4
# Onset envelopes of real music are sparse (CV > 1); a drone's is flat.
_MIN_ENVELOPE_CV = 1.0


def _onset_envelope(audio: np.ndarray, sr: int, hop_length: int, n_fft: int = 2048) -> np.ndarray:
    """Positive log-magnitude flux, one value per hop frame."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size < n_fft:
        return np.zeros(0, dtype=np.float64)

    n_frames = 1 + (audio.size - n_fft) // hop_length
    window = np.hanning(n_fft)
    # Strided view: (n_frames, n_fft) without copying the whole signal.
    idx = np.arange(n_frames)[:, None] * hop_length + np.arange(n_fft)[None, :]
    frames = audio[idx] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    logmag = np.log1p(mag)
    flux = np.diff(logmag, axis=0)
    np.maximum(flux, 0.0, out=flux)
    env = flux.sum(axis=1)
    if env.size == 0:
        return env
    peak = float(env.max())
    if peak > 0:
        env = env / peak
    return env


def estimate_tempo_bpm(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> tuple[float | None, float]:
    """Estimate tempo from audio.

    Returns ``(bpm, confidence)``. ``bpm`` is ``None`` when there is no usable
    periodicity (silence, a single click, a drone), in which case the caller
    should keep the fixed grid. ``confidence`` is the normalised
    autocorrelation peak in ``[0, 1]``.
    """
    if sr <= 0 or hop_length <= 0:
        raise ValueError(f"sr and hop_length must be positive (got {sr}, {hop_length})")
    if not 0 < min_bpm < max_bpm:
        raise ValueError(f"need 0 < min_bpm < max_bpm, got {min_bpm}, {max_bpm}")

    env = _onset_envelope(audio, sr, hop_length)
    if env.size < 8:
        return None, 0.0
    # A drone or a sustained chord has an (almost) constant envelope: with no
    # onsets there is no tempo to find, and autocorrelating the numerical
    # residue would happily report one.
    n_onsets = int(np.count_nonzero(env > 0.25 * float(env.max()))) if env.max() > 0 else 0
    cv = float(env.std() / (env.mean() + 1e-12)) if env.size else 0.0
    if n_onsets < _MIN_ONSETS or cv < _MIN_ENVELOPE_CV:
        return None, 0.0
    env = env - env.mean()
    if not np.any(env):
        return None, 0.0

    frames_per_second = sr / hop_length
    min_lag = max(1, int(np.floor(frames_per_second * 60.0 / max_bpm)))
    max_lag = int(np.ceil(frames_per_second * 60.0 / min_bpm))
    max_lag = min(max_lag, env.size - 1)
    if max_lag <= min_lag:
        return None, 0.0

    # Autocorrelation via FFT (the envelope is short; O(n log n)).
    n = int(2 ** np.ceil(np.log2(2 * env.size)))
    spectrum = np.fft.rfft(env, n=n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=n)[: env.size]
    if acf[0] <= 0:
        return None, 0.0
    acf = acf / acf[0]

    lags = np.arange(min_lag, max_lag + 1)
    bpms = frames_per_second * 60.0 / lags
    scores = acf[lags].copy()
    scores *= np.where((bpms >= _PRIOR_LO) & (bpms <= _PRIOR_HI), _PRIOR_BOOST, 1.0)

    best = int(np.argmax(scores))
    score = float(scores[best])
    # Octave tie-break: a click train correlates just as well at a multiple of
    # its period, so a 175 BPM groove ties with 87.5. Prefer the FASTEST lag
    # that is nearly as strong - musical pulses are what the beat grid wants.
    near_best = np.nonzero(scores >= 0.85 * score)[0]
    lag = int(lags[int(near_best[0])])
    if score <= 0:
        return None, 0.0

    # Parabolic refinement on the raw (unweighted) ACF around the peak.
    bpm = frames_per_second * 60.0 / lag
    if 1 <= lag < acf.size - 1:
        left, centre, right = acf[lag - 1], acf[lag], acf[lag + 1]
        denom = left - 2.0 * centre + right
        if denom != 0:
            offset = 0.5 * (left - right) / denom
            offset = float(np.clip(offset, -0.5, 0.5))
            bpm = frames_per_second * 60.0 / (lag + offset)

    confidence = float(np.clip(score / _PRIOR_BOOST, 0.0, 1.0))
    # Prominence over the median of the searched lags: periodicity has to stand
    # out of the autocorrelation's own noise floor. White noise scores ~0.05
    # here, a click track ~0.7, so this is what separates "a tempo" from "a
    # random peak in a flat ACF".
    prominence = float(acf[lag] - np.median(acf[lags]))
    if (
        not (min_bpm <= bpm <= max_bpm)
        or confidence < _MIN_CONFIDENCE
        or (prominence < _MIN_PROMINENCE)
    ):
        # Below this the "tempo" is noise; the fixed grid is the honest answer.
        return None, confidence
    return float(bpm), confidence


def tempo_us_from_bpm(bpm: float) -> int:
    """Microseconds per beat, the unit of a MIDI ``set_tempo`` event."""
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    return round(60_000_000.0 / bpm)
