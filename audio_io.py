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
# Integer PCM only. Float WAV/FLAC and compressed formats are accepted (they are
# decoded to float32 anyway) and their subtype is reported in ``info``; 8-bit and
# 32-bit integer PCM are rejected because they are almost always an export
# mistake and the model has no headroom handling for them (audit M-5: the
# constant existed but was never enforced).
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

    if info["frames"] <= 0 or info["duration"] <= 0:
        snd.close()
        raise AudioValidationError(
            f"Audio file {file_path!r} contains no frames ({info['frames']} frames, "
            f"{info['duration']:.3f}s). A zero-length file is a client error: there is "
            "nothing to separate or transcribe."
        )

    if info["duration"] > max_duration_seconds:
        snd.close()
        raise AudioValidationError(
            f"Audio duration {info['duration']:.2f}s exceeds maximum {max_duration_seconds}s"
        )

    subtype = str(info["format"] and info["subtype"] and snd.subtype)
    if subtype.startswith("PCM") and info["bit_depth"] not in SUPPORTED_BIT_DEPTHS:
        snd.close()
        raise AudioValidationError(
            f"Unsupported PCM bit depth {info['bit_depth']} (subtype {snd.subtype}). "
            f"Supported integer depths: {sorted(SUPPORTED_BIT_DEPTHS)}. "
            "Export as 16- or 24-bit PCM, or as float."
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


def check_finite(file_path: str, chunk_frames: int = 1 << 20) -> None:
    """Stream ``file_path`` and raise if any sample is NaN or Inf (IO-3).

    Headers cannot tell you a file is corrupt; the samples can. This walks the
    file in ``chunk_frames`` blocks so the check costs O(1) memory even for the
    600 s maximum, and stops at the first bad sample.
    """
    with sf.SoundFile(file_path) as snd:
        while True:
            block = snd.read(frames=chunk_frames, dtype="float32", always_2d=True)
            if block.size == 0:
                return
            if not np.isfinite(block).all():
                bad = int(np.count_nonzero(~np.isfinite(block)))
                raise AudioValidationError(
                    f"Audio contains {bad} non-finite sample(s) (NaN/Inf) near frame "
                    f"{int(snd.tell())}; refusing to process. Check the file for corrupt "
                    "frames or a broken decoder."
                )


def load_native_mono(snd: sf.SoundFile) -> tuple[np.ndarray, int]:
    """
    Read from an open SoundFile handle at NATIVE sample rate (sr=None — no
    resample, no librosa), then mean-downmix to mono (API-6.15).
    """
    audio = snd.read(dtype="float32", always_2d=True)
    sr = snd.samplerate
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)
    # NaN/Inf samples survive every decoder and then poison the STFT, the masks
    # and finally the JSON report (IO-3). Refuse them at the boundary.
    if not np.isfinite(audio).all():
        bad = int(np.count_nonzero(~np.isfinite(audio)))
        raise AudioValidationError(
            f"Audio contains {bad} non-finite sample(s) (NaN/Inf); refusing to process. "
            "Check the file for corrupt frames or a broken decoder."
        )
    return audio, sr


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
