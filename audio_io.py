# SPDX-License-Identifier: Apache-2.0
"""
Audio I/O helpers for Stem+MIDI Pro (RF-3.2.4).

Single source of truth for audio loading/validation used by api.py.
Validation and loading share ONE `sf.SoundFile` open (API-6.6), and loading
forces native sample rate with a manual mean downmix (API-6.15) — no
librosa (which may silently resample on sr=None in some versions).
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
import soundfile as sf

SUPPORTED_SAMPLE_RATES = {44100, 48000}
SUPPORTED_BIT_DEPTHS = {16, 24}
MAX_DURATION_SECONDS = 600  # 10 minutes


class AudioValidationError(ValueError):
    """Raised when an audio file fails validation."""


def _infer_bit_depth(subtype: str) -> int | None:
    if "FLOAT" in subtype or "DOUBLE" in subtype:
        return 32
    if "PCM_16" in subtype:
        return 16
    if "PCM_24" in subtype:
        return 24
    if "PCM_32" in subtype:
        return 32
    return None


@contextmanager
def open_and_validate(file_path: str, max_duration_seconds: int = MAX_DURATION_SECONDS):
    """
    Open `file_path` in a single sf.SoundFile handle, validate it, and yield
    (soundfile_handle, info_dict). The handle is closed on exit.

    Raises AudioValidationError on bad duration / sample rate / unreadable file.
    """
    try:
        snd = sf.SoundFile(file_path)
    except Exception as e:
        raise AudioValidationError(f"Invalid audio file: {e}") from e

    info = {
        "duration": snd.frames / snd.samplerate,
        "sample_rate": snd.samplerate,
        "channels": snd.channels,
        "format": snd.format,
        "subtype": snd.subtype,
        "frames": snd.frames,
        "bit_depth": _infer_bit_depth(snd.subtype),
    }

    if info["duration"] > max_duration_seconds:
        snd.close()
        raise AudioValidationError(
            f"Audio duration {info['duration']:.2f}s exceeds maximum {max_duration_seconds}s"
        )

    if info["sample_rate"] not in SUPPORTED_SAMPLE_RATES:
        snd.close()
        raise AudioValidationError(
            f"Sample rate {info['sample_rate']}Hz not supported. "
            f"Supported: {SUPPORTED_SAMPLE_RATES}"
        )

    try:
        yield snd, info
    finally:
        snd.close()


def load_native_mono(snd: sf.SoundFile) -> tuple[np.ndarray, int]:
    """
    Read from an open SoundFile handle at NATIVE sample rate (sr=None — no
    resample, no librosa), then mean-downmix to mono (API-6.15).
    """
    audio = snd.read(dtype="float32", always_2d=True)
    sr = snd.samplerate
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    return np.asarray(audio, dtype=np.float32), sr


def load_audio(
    file_path: str, max_duration_seconds: int = MAX_DURATION_SECONDS
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """
    Validate and load an audio file in one open (API-6.6).

    Returns (mono_float32_audio, sample_rate, info_dict).
    """
    with open_and_validate(file_path, max_duration_seconds) as (snd, info):
        audio, sr = load_native_mono(snd)
    return audio, sr, info
