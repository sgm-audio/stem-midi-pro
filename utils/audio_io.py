# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Audio I/O utilities for Stem+MIDI Pro."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".flac", ".mp3"}
SUPPORTED_SAMPLE_RATES = {44100, 48000}
MAX_DURATION_SECONDS = 600


def load_audio(path: str, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio file. Returns (audio, sample_rate)."""
    try:
        audio, sr = sf.read(path)
    except Exception:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)

    if mono and audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if target_sr is not None and sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Save audio as WAV file."""
    sf.write(path, audio, sr, format="WAV")


def validate_audio(
    path: str,
    max_duration: float = MAX_DURATION_SECONDS,
    supported_srs: set[int] | None = None,
) -> dict:
    """Validate audio file properties. Returns info dict or raises ValueError."""
    if supported_srs is None:
        supported_srs = SUPPORTED_SAMPLE_RATES

    file_ext = Path(path).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{file_ext}'. Supported: {SUPPORTED_FORMATS}")

    try:
        with sf.SoundFile(path) as f:
            info = {
                "duration": len(f) / f.samplerate,
                "sample_rate": f.samplerate,
                "channels": f.channels,
                "format": str(f.format),
                "subtype": str(f.subtype),
            }
    except Exception as e:
        raise ValueError(f"Invalid audio file: {e}")

    if info["duration"] > max_duration:
        raise ValueError(
            f"Duration {info['duration']:.2f}s exceeds maximum {max_duration}s"
        )

    if info["sample_rate"] not in supported_srs:
        raise ValueError(
            f"Sample rate {info['sample_rate']}Hz not supported. Supported: {supported_srs}"
        )

    return info
