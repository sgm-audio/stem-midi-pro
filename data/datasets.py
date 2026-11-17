# SPDX-License-Identifier: Apache-2.0
"""
Dataset loaders for Stem+MIDI Pro.
Supports Slakh2100-YourMT3-16k and MUSDB18-HQ with synthetic fallback.

Label policy (C-4 / C-5):
  * Real Slakh tracks with MIDI files get REAL onset/pitch/velocity targets
    via ``utils.midi_targets.midi_to_targets`` (the periodic A2 impulse train
    is only used as a last-resort fallback, loudly logged and counted).
  * MUSDB18-HQ ships no MIDI, so transcription targets are NOT available:
    the mixture is the true sum of stems and ``target_bass`` is the real
    bass stem, but onset/pitch targets fall back to the synthetic periodic
    pattern unless ``strict=True`` — strict mode refuses instead.
  * ``train.py`` refuses to save checkpoints trained on any synthetic-fallback
    split unless ``--allow-synthetic`` is passed explicitly.
"""

import glob
import logging
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from utils.midi_targets import midi_to_targets

logger = logging.getLogger(__name__)

# Frame grid used by the dataset targets (matches the model hop_length).
TARGET_HOP_LENGTH = 512
SYNTHETIC_ONSET_VELOCITY = 0.8


class AudioDataset(Dataset):
    """
    Base dataset for audio data (Slakh2100-YourMT3-16k / MUSDB18-HQ).
    Returns mixtures and separated stems (guitar/bass) for training.
    """

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 44100,
        segment_length: float = 6.0,  # seconds
        instruments: list[str] | None = None,
        augment: bool = True,
        strict: bool = False,
    ):
        """
        Args:
            root_dir: Path to dataset root
            sample_rate: Target sample rate (default 44.1kHz)
            segment_length: Length of audio segments in seconds
            instruments: List of stem instruments to separate
            augment: Whether to apply data augmentation
            strict: Refuse (raise) instead of falling back when labels are
                missing; used by train.py --strict-data (C-5).
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.instruments = instruments if instruments is not None else ["guitar", "bass"]
        self.augment = augment
        self.strict = strict

        # Validate dataset existence; sets self.synthetic only (C-2.3).
        self._validate_dataset()

        # Build file list (short-circuits with synthetic names when needed)
        self.file_list = self._build_file_list()

        # Count of real items that had to fall back to synthetic labels (C-4).
        self.items_missing_labels = 0

        # Augmentation parameters
        self.time_stretch_range = (0.9, 1.1)
        self.pitch_shift_range = (-2, 2)  # semitones
        self.noise_level = 0.005

    def _validate_dataset(self):
        """Check if dataset exists; falls back to synthetic mode for development.

        Only sets ``self.synthetic`` — the file list is owned solely by
        ``_build_file_list`` (C-2.3: previously set twice).
        """
        self.synthetic = not os.path.exists(self.root_dir)
        if self.synthetic:
            logger.warning(f"Dataset not found at {self.root_dir}")
            logger.warning(
                "Falling back to SYNTHETIC data for this split. Training on "
                "synthetic data requires an explicit --allow-synthetic in "
                "train.py; the resulting checkpoint is not usable for real music."
            )

    def _build_file_list(self) -> list[str]:
        """Build list of audio files in the dataset (synthetic short-circuit)."""
        if self.synthetic:
            return [f"synthetic_{i}" for i in range(100)]

        # Walk through directory and find track directories (containing mix.wav)
        file_list = []

        for entry in os.listdir(self.root_dir):
            track_dir = os.path.join(self.root_dir, entry)
            if os.path.isdir(track_dir) and os.path.exists(os.path.join(track_dir, "mix.wav")):
                file_list.append(track_dir)

        if not file_list:
            raise ValueError(
                f"No Slakh2100 track directories found in {self.root_dir}. "
                "Expected subdirectories each containing mix.wav."
            )

        return sorted(file_list)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample."""
        if self.synthetic:
            return self._get_synthetic_item(idx)
        return self._get_real_item(idx)

    # ------------------------------------------------------------------
    # Synthetic items and target fallbacks
    # ------------------------------------------------------------------

    def _periodic_targets(self, n_frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Synthetic periodic onset/pitch/velocity targets (A2 every 0.5 s).

        FABRICATED LABELS — used only for the synthetic generator and, loudly
        logged, as a last-resort fallback for real tracks without MIDI
        (counted in ``self.items_missing_labels``). Never silent (C-4).
        """
        target_onsets = torch.zeros((1, n_frames, 1))
        target_pitch = torch.zeros((1, n_frames, 128))
        target_velocity = torch.zeros((1, n_frames, 1))
        onset_interval_frames = max(1, int(0.5 * self.sample_rate / TARGET_HOP_LENGTH))
        for f in range(0, n_frames, onset_interval_frames):
            target_onsets[0, f, 0] = 1.0
            target_pitch[0, f, 45] = 1.0  # MIDI note 45 (A2 ~ 110Hz)
            target_velocity[0, f, 0] = SYNTHETIC_ONSET_VELOCITY
        return target_onsets, target_pitch, target_velocity

    def _handle_missing_labels(self, track_dir: str, n_frames: int):
        """Strict raises; otherwise loud per-track fallback + count (C-4/C-5)."""
        self.items_missing_labels += 1
        if self.strict:
            raise FileNotFoundError(
                f"--strict-data: track {track_dir!r} provides no MIDI metadata, so no "
                "real transcription targets can be built. Add the MIDI files, drop "
                "the track, or run without --strict-data."
            )
        logger.warning(
            "Track %s has no MIDI metadata: using FABRICATED periodic "
            "onset/pitch/velocity targets (A2 @ 0.5s). This split trains on "
            "synthetic labels; train.py will refuse to checkpoint without "
            "--allow-synthetic. (%d tracks affected so far)",
            track_dir,
            self.items_missing_labels,
        )
        return self._periodic_targets(n_frames)

    def _find_midi_files(self, track_dir: str) -> list[str]:
        """Collect per-track MIDI candidates (Slakh layouts vary; be explicit)."""
        paths: list[str] = []
        for sub in ("midi", "MIDI", "metadata"):
            d = os.path.join(track_dir, sub)
            if os.path.isdir(d):
                paths.extend(sorted(glob.glob(os.path.join(d, "**", "*.mid"), recursive=True)))
        paths.extend(sorted(glob.glob(os.path.join(track_dir, "*.mid"))))
        return paths

    def _get_synthetic_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate synthetic audio for development/testing."""
        # Create a simple guitar-like and bass-like signal
        duration = self.segment_length
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Guitar: harmonics with decay
        guitar = (
            0.3 * np.sin(2 * np.pi * 110 * t) * np.exp(-t * 0.5)  # Fundamental
            + 0.2 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 0.3)  # 2nd harmonic
            + 0.1 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 0.2)  # 3rd harmonic
        )

        # Bass: lower frequencies with longer decay
        bass = 0.4 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 0.2) + 0.2 * np.sin(  # A1
            2 * np.pi * 110 * t
        ) * np.exp(
            -t * 0.3
        )  # A2

        # Add some noise
        noise_level = 0.02
        guitar += np.random.normal(0, noise_level, len(t))
        bass += np.random.normal(0, noise_level, len(t))

        # Create mixture (simple sum)
        mixture = guitar + bass

        # Apply random augmentation if enabled
        if self.augment:
            mixture, guitar, bass = self._augment_audio(mixture, guitar, bass)

        # Convert to torch tensors and ensure correct shape (C, T)
        mixture = torch.from_numpy(mixture).float().unsqueeze(0)
        guitar = torch.from_numpy(guitar).float().unsqueeze(0)
        bass = torch.from_numpy(bass).float().unsqueeze(0)

        # Compute frame dimension for transcription targets (matches model hop_length)
        n_frames = mixture.shape[-1] // TARGET_HOP_LENGTH

        target_onsets, target_pitch, target_velocity = self._periodic_targets(n_frames)

        return {
            "audio": mixture,
            "target_guitar": guitar,
            "target_bass": bass,
            "target_onsets": target_onsets,
            "target_pitch": target_pitch,
            "target_velocity": target_velocity,
        }

    def _get_real_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load real audio file and extract stems."""
        # This would be implemented based on actual dataset structure
        # For now, we'll use a placeholder that mimics the synthetic approach
        # In practice, you would:
        # 1. Load the mixture audio file
        # 2. Load the separated stem files for guitar and bass
        # 3. Resample to target sample rate if needed
        # 4. Extract a random segment
        # 5. Apply augmentation

        # Placeholder: return synthetic data with a warning on first call
        if idx == 0:
            logger.warning(
                "Using synthetic data placeholder. "
                "Implement _get_real_item for your specific dataset structure."
            )
        return self._get_synthetic_item(idx)

    def _augment_audio(
        self, mixture: np.ndarray, guitar: np.ndarray, bass: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation to audio signals.

        NOTE: time-stretch / pitch-shift invalidate frame-aligned MIDI
        targets, so subclasses use real-target items WITHOUT augmentation.
        """
        # Time stretching
        if random.random() < 0.5:
            orig_len = len(mixture)
            rate = random.uniform(*self.time_stretch_range)
            mixture = librosa.effects.time_stretch(mixture, rate=rate)
            guitar = librosa.effects.time_stretch(guitar, rate=rate)
            bass = librosa.effects.time_stretch(bass, rate=rate)

            def _trim_pad(a, length):
                return a[:length] if len(a) > length else np.pad(a, (0, length - len(a)))

            mixture = _trim_pad(mixture, orig_len)
            guitar = _trim_pad(guitar, orig_len)
            bass = _trim_pad(bass, orig_len)

        # Pitch shifting
        if random.random() < 0.3:
            n_steps = random.uniform(*self.pitch_shift_range)
            mixture = librosa.effects.pitch_shift(mixture, sr=self.sample_rate, n_steps=n_steps)
            guitar = librosa.effects.pitch_shift(guitar, sr=self.sample_rate, n_steps=n_steps)
            bass = librosa.effects.pitch_shift(bass, sr=self.sample_rate, n_steps=n_steps)

        # Add noise
        if random.random() < 0.4:
            noise = np.random.normal(0, self.noise_level, len(mixture))
            mixture += noise
            guitar += noise * 0.5  # Less noise on stems
            bass += noise * 0.5

        # Random gain
        if random.random() < 0.3:
            gain = random.uniform(0.8, 1.2)
            mixture *= gain
            guitar *= gain
            bass *= gain

        return mixture, guitar, bass

    @staticmethod
    def _slice_targets(onset, pitch, velocity, frame_start: int, n_frames: int):
        """Crop/pad frame-aligned targets to [frame_start, frame_start + n_frames)."""
        out = []
        for t in (onset, pitch, velocity):
            t = t[:, frame_start : frame_start + n_frames]
            if t.shape[1] < n_frames:
                t = torch.nn.functional.pad(t, (0, n_frames - t.shape[1]))
            out.append(t)
        return out


def get_data_loaders(
    data_config: dict, strict: bool = False, allow_synthetic: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for the configured datasets.

    Args:
        data_config: Dictionary containing:
            - dataset_type: 'slakh2100_yourmt3' or 'musdb18hq' or 'synthetic'
            - root_dir: Path to dataset
            - batch_size: Batch size for training
            - num_workers: Number of DataLoader workers
            - sample_rate: Target sample rate
            - segment_length: Segment length in seconds
        strict: Raise instead of falling back when a split directory is
            missing (C-5). MUSDB18-HQ has no 'val' split upstream — strict
            mode surfaces that instead of silently training on synthetic data.
        allow_synthetic: Whether missing split directories may fall back to
            synthetic data. train.py passes False by default so a silent
            synthetic split cannot produce a "real-looking" checkpoint.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_type = data_config.get("dataset_type", "synthetic")
    root_dir = data_config["root_dir"]
    batch_size = data_config.get("batch_size", 16)
    num_workers = data_config.get("num_workers", 4)
    sample_rate = data_config.get("sample_rate", 44100)
    segment_length = data_config.get("segment_length", 6.0)

    dataset_subdir = {
        "slakh2100_yourmt3": "slakh2100_yourmt3",
        "musdb18hq": "musdb18hq",
    }.get(dataset_type, "")

    def _split_path(split: str) -> str:
        return os.path.join(root_dir, dataset_subdir, split)

    def _build_split(split: str, augment: bool, cls):
        split_dir = _split_path(split)
        if dataset_type != "synthetic" and not os.path.isdir(split_dir):
            if strict:
                raise FileNotFoundError(
                    f"--strict-data: {dataset_type} split directory not found: "
                    f"{split_dir!r}. Note: MUSDB18-HQ ships only train/ and "
                    "test/ — there is no upstream 'val' split (M-13)."
                )
            logger.warning(
                "Split directory %s not found — falling back to SYNTHETIC data "
                "for the '%s' split (C-5). Pass strict=True to refuse instead.",
                split_dir,
                split,
            )
        return cls(
            root_dir=split_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=augment,
            strict=strict,
        )

    if dataset_type == "slakh2100_yourmt3":
        train_dataset = _build_split("train", True, Slakh2100YourMT3Dataset)
        val_dataset = _build_split("val", False, Slakh2100YourMT3Dataset)
        test_dataset = _build_split("test", False, Slakh2100YourMT3Dataset)
    elif dataset_type == "musdb18hq":
        train_dataset = _build_split("train", True, MUSDB18HQDataset)
        val_dataset = _build_split("val", False, MUSDB18HQDataset)
        test_dataset = _build_split("test", False, MUSDB18HQDataset)
    else:  # synthetic
        train_dataset = AudioDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=True,
            strict=strict,
        )
        val_dataset = AudioDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False,
            strict=strict,
        )
        test_dataset = AudioDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False,
            strict=strict,
        )

    # C-5: refuse synthetic fallback unless it is explicitly allowed.
    if not allow_synthetic:
        offenders = [
            (name, ds)
            for name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset))
            if getattr(ds, "synthetic", False)
        ]
        if offenders:
            details = ", ".join(f"{name}={ds.root_dir}" for name, ds in offenders)
            raise FileNotFoundError(
                "Refusing to train on SYNTHETIC fallback data (allow_synthetic=False): "
                f"{details}. Provide the real dataset paths or pass "
                "--allow-synthetic to proceed knowingly."
            )

    for name, ds in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        logger.info(
            "Dataset[%s]: synthetic=%s tracks=%d missing_labels=%d",
            name,
            getattr(ds, "synthetic", "?"),
            len(ds),
            getattr(ds, "items_missing_labels", 0),
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class Slakh2100YourMT3Dataset(AudioDataset):
    """Slakh2100-YourMT3-16k dataset loader.

    Structure:
        root/
            Track00001/
                mix.wav             # mixture audio
                stems/
                    bass.wav        # individual stem audio
                    drums.wav
                    guitar.wav
                    vocals.wav
                    ...
                midi/  (or metadata/)   # per-track MIDI used for REAL targets
            Track00002/
                ...
    """

    def _build_file_list(self) -> list[str]:
        """Slakh layout lives in the base class; short-circuit synthetic here (C-2.4)."""
        if self.synthetic:
            return [f"synthetic_{i}" for i in range(100)]
        return super()._build_file_list()

    def _get_real_item(self, idx: int) -> dict[str, torch.Tensor]:
        track_dir = self.file_list[idx]

        # Load mix
        mix_path = os.path.join(track_dir, "mix.wav")
        if not os.path.exists(mix_path):
            return self._get_synthetic_item(idx)

        mix, sr = sf.read(mix_path)

        # Load guitar and bass stems from stems/ subdirectory
        stems_dir = os.path.join(track_dir, "stems")
        guitar = None
        bass = None

        guitar_path = os.path.join(stems_dir, "guitar.wav")
        bass_path = os.path.join(stems_dir, "bass.wav")

        if os.path.exists(guitar_path):
            guitar, _ = sf.read(guitar_path)
        if os.path.exists(bass_path):
            bass, _ = sf.read(bass_path)

        # Fall back to empty arrays if stems not found
        if guitar is None:
            guitar = np.zeros_like(mix)
        if bass is None:
            bass = np.zeros_like(mix)

        # Resample to target sample rate if needed
        if sr != self.sample_rate:
            mix = librosa.resample(mix, orig_sr=sr, target_sr=self.sample_rate)
            guitar = librosa.resample(guitar, orig_sr=sr, target_sr=self.sample_rate)
            bass = librosa.resample(bass, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        # Ensure mono (mix from sf.read may be (T, C) — mean-downmix)
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
        if guitar.ndim > 1:
            guitar = guitar.mean(axis=1)
        if bass.ndim > 1:
            bass = bass.mean(axis=1)

        # ------------------------------------------------------------------
        # Real transcription targets from the track's MIDI (C-4).
        # ------------------------------------------------------------------
        midi_paths = self._find_midi_files(track_dir)
        if midi_paths:
            # Time-stretch/pitch-shift augmentation invalidates frame-aligned
            # MIDI targets — augmentation is disabled on real-target items.
            augment = False
            n_frames_total = len(mix) // TARGET_HOP_LENGTH
            onset_full, pitch_full, velocity_full = midi_to_targets(
                midi_paths,
                n_frames_total,
                hop_length=TARGET_HOP_LENGTH,
                sample_rate=self.sample_rate,
            )
        else:
            augment = self.augment
            onset_full = pitch_full = velocity_full = None

        # Random segment crop
        seg_len = int(self.segment_length * sr)
        if len(mix) > seg_len:
            start = random.randint(0, len(mix) - seg_len)
            end = start + seg_len
            mix = mix[start:end]
            guitar = guitar[start:end]
            bass = bass[start:end]
            frame_start = start // TARGET_HOP_LENGTH
        else:
            mix = np.pad(mix, (0, seg_len - len(mix)))
            guitar = np.pad(guitar, (0, seg_len - len(guitar)))
            bass = np.pad(bass, (0, seg_len - len(bass)))
            frame_start = 0

        # Apply augmentation (only when targets are not MIDI-aligned)
        if augment:
            mix, guitar, bass = self._augment_audio(mix, guitar, bass)

        # Re-crop/pad to segment length after augmentation (B6: pitch_shift /
        # time_stretch can change array length, breaking batch collation).
        seg_len = int(self.segment_length * sr)
        if len(mix) > seg_len:
            mix = mix[:seg_len]
            guitar = guitar[:seg_len]
            bass = bass[:seg_len]
        else:
            mix = np.pad(mix, (0, seg_len - len(mix)))
            guitar = np.pad(guitar, (0, seg_len - len(guitar)))
            bass = np.pad(bass, (0, seg_len - len(bass)))

        mix_t = torch.from_numpy(mix).float().unsqueeze(0)
        guitar_t = torch.from_numpy(guitar).float().unsqueeze(0)
        bass_t = torch.from_numpy(bass).float().unsqueeze(0)

        n_frames = seg_len // TARGET_HOP_LENGTH
        if onset_full is not None:
            target_onsets, target_pitch, target_velocity = self._slice_targets(
                onset_full, pitch_full, velocity_full, frame_start, n_frames
            )
        else:
            target_onsets, target_pitch, target_velocity = self._handle_missing_labels(
                track_dir, n_frames
            )

        return {
            "audio": mix_t,
            "target_guitar": guitar_t,
            "target_bass": bass_t,
            "target_onsets": target_onsets,
            "target_pitch": target_pitch,
            "target_velocity": target_velocity,
        }


class MUSDB18HQDataset(AudioDataset):
    """MUSDB18-HQ dataset loader.

    MUSDB18-HQ stems: bass.wav, drums.wav, vocals.wav, other.wav; the mixture
    is the SUM of the four stems (C-4: previously divided by 4, training on an
    attenuated mixture). ``guitar`` is mapped to ``other`` — MUSDB has no
    isolated guitar stem. No MIDI exists in MUSDB, so transcription targets
    fall back to the (loudly logged) periodic pattern; strict mode refuses.
    """

    def _build_file_list(self) -> list[str]:
        """Override: MUSDB18-HQ uses stem .wav files (bass.wav, drums.wav, etc.) in track dirs."""
        if self.synthetic:
            return [f"synthetic_{i}" for i in range(100)]
        file_list = []
        for entry in os.listdir(self.root_dir):
            track_dir = os.path.join(self.root_dir, entry)
            if os.path.isdir(track_dir) and os.path.exists(os.path.join(track_dir, "bass.wav")):
                file_list.append(track_dir)
        if not file_list:
            raise ValueError(
                f"No MUSDB18-HQ track directories found in {self.root_dir}. "
                "Expected subdirectories each containing bass.wav, drums.wav, "
                "vocals.wav, other.wav."
            )
        return sorted(file_list)

    def _get_real_item(self, idx: int) -> dict[str, torch.Tensor]:
        track_dir = self.file_list[idx]
        stem_map = {
            "bass": "bass",
            "guitar": "other",
            "drums": "drums",
            "vocals": "vocals",
            "other": "other",
        }

        stem_wavs = {}
        sr_orig = None
        for stem_name in ["bass", "drums", "vocals", "other"]:
            path = os.path.join(track_dir, f"{stem_name}.wav")
            if os.path.exists(path):
                wav, sr_orig = sf.read(path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                stem_wavs[stem_name] = wav.astype(np.float32)

        if not stem_wavs:
            return self._get_synthetic_item(idx)

        max_len = max(len(w) for w in stem_wavs.values())
        for k in stem_wavs:
            if len(stem_wavs[k]) < max_len:
                stem_wavs[k] = np.pad(stem_wavs[k], (0, max_len - len(stem_wavs[k])))

        # Explicit sum (C-2.7 / C-4): the MUSDB mixture is the sum of its
        # stems. Dividing by the count trains on an attenuated mixture whose
        # statistics never match full-scale inference input.
        stem_list = list(stem_wavs.values())
        total = stem_list[0].copy()
        for wav in stem_list[1:]:
            total = total + wav
        mixture = total
        if mixture.ndim != 1:
            raise ValueError(f"mixture must be 1-D, got shape {mixture.shape}")

        target_name = self.instruments[0]
        guitar_stem = stem_wavs.get(stem_map.get(target_name, target_name))
        if guitar_stem is None:
            guitar_stem = np.zeros_like(mixture)
        bass_stem = stem_wavs.get("bass")
        if bass_stem is None:
            bass_stem = np.zeros_like(mixture)

        if sr_orig != self.sample_rate:
            mixture = librosa.resample(mixture, orig_sr=sr_orig, target_sr=self.sample_rate)
            guitar_stem = librosa.resample(guitar_stem, orig_sr=sr_orig, target_sr=self.sample_rate)
            bass_stem = librosa.resample(bass_stem, orig_sr=sr_orig, target_sr=self.sample_rate)
            sr = self.sample_rate
        else:
            sr = sr_orig

        seg_len = int(self.segment_length * sr)
        if len(mixture) > seg_len:
            start = random.randint(0, len(mixture) - seg_len)
            mixture = mixture[start : start + seg_len]
            guitar_stem = guitar_stem[start : start + seg_len]
            bass_stem = bass_stem[start : start + seg_len]
        else:
            mixture = np.pad(mixture, (0, seg_len - len(mixture)))
            guitar_stem = np.pad(guitar_stem, (0, seg_len - len(guitar_stem)))
            bass_stem = np.pad(bass_stem, (0, seg_len - len(bass_stem)))

        n_frames = seg_len // TARGET_HOP_LENGTH
        target_onsets, target_pitch, target_velocity = self._handle_missing_labels(
            track_dir, n_frames
        )

        if self.augment:
            mixture, guitar_stem, bass_stem = self._augment_audio(mixture, guitar_stem, bass_stem)

        # Re-crop/pad to segment length after augmentation (B6: pitch_shift /
        # time_stretch can change array length, breaking batch collation).
        if len(mixture) > seg_len:
            mixture = mixture[:seg_len]
            guitar_stem = guitar_stem[:seg_len]
            bass_stem = bass_stem[:seg_len]
        else:
            mixture = np.pad(mixture, (0, seg_len - len(mixture)))
            guitar_stem = np.pad(guitar_stem, (0, seg_len - len(guitar_stem)))
            bass_stem = np.pad(bass_stem, (0, seg_len - len(bass_stem)))

        return {
            "audio": torch.from_numpy(mixture).float().unsqueeze(0),
            "target_guitar": torch.from_numpy(guitar_stem).float().unsqueeze(0),
            "target_bass": torch.from_numpy(bass_stem).float().unsqueeze(0),
            "target_onsets": target_onsets,
            "target_pitch": target_pitch,
            "target_velocity": target_velocity,
        }
