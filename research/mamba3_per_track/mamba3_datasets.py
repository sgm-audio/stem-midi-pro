"""
Mamba-3 dataset loaders — relocated from data/datasets.py.

Used only by the Mamba-3 per-track experimental path under
research/mamba3_per_track/. The production v1 pipeline uses
data/datasets.py (Slakh2100YourMT3Dataset / MUSDB18HQDataset).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class StemDataset(Dataset):
    """MUSDB18-HQ StemDataset: loads tracks with per-stem .wav + .crepe.npy files.

    Each track directory should contain:
        bass.wav, drums.wav, vocals.wav, other.wav
        bass.crepe.npy, bass.pitched.npy, drums.crepe.npy, ...

    Returns:
        target: (T,) waveform of chosen stem
        other_stems: (K, T) waveforms of all OTHER stems (for cross-track conditioning)
        target_crepe: (T_frames, 360) CREPE pitch probability matrix
        other_crepe: (K, T_frames, 360) CREPE for other stems
        target_pitched: float, fraction of pitched frames
        other_pitched: (K,) pitched fractions
        stem_name: str, name of the target stem
        valid: bool, whether data was successfully loaded
    """
    STEM_NAMES = ['bass', 'drums', 'vocals', 'other', 'guitar', 'piano', 'keys']

    def __init__(
        self,
        root: str,
        cfg,
        clip_seconds: float = 4.0,
        augment: bool = True,
        target_stems: Optional[list[str]] = None,
        cache_in_ram: bool = False,
    ):
        self.root = Path(root)
        self.sample_rate = cfg.sample_rate
        self.hop_length = getattr(cfg, 'hop_length', 512)
        self.clip_len = int(clip_seconds * self.sample_rate)
        self.augment = augment
        self.stems = target_stems or ['bass', 'drums', 'vocals', 'other']
        self.cache_in_ram = cache_in_ram
        self._ram_cache = {}

        try:
            self.tracks = sorted([
                d for d in self.root.iterdir()
                if d.is_dir() and any((d / f"{s}.wav").exists() for s in self.stems)
            ])
        except (FileNotFoundError, NotADirectoryError):
            self.tracks = []
        if not self.tracks:
            logger.warning(f"No MUSDB18-HQ tracks found in {root}, using synthetic fallback")
            self._synthetic = True
        else:
            self._synthetic = False

        if self.cache_in_ram and not self._synthetic:
            self._preload_all_stems()

    def _preload_all_stems(self):
        logger.info(f"Pre-loading {len(self.tracks)} tracks into RAM cache...")
        import tqdm
        total_stems = 0
        total_size_gb = 0.0
        for track_dir in tqdm.tqdm(self.tracks, desc="Loading stems"):
            for stem_name in self.stems:
                wav = self._load_stem_uncached(track_dir, stem_name)
                if wav is not None:
                    key = f"{track_dir.name}/{stem_name}"
                    self._ram_cache[key] = wav
                    total_stems += 1
                    total_size_gb += wav.element_size() * wav.nelement()
        total_size_gb /= (1024 ** 3)
        logger.info(f"  Cached {total_stems} stems ({total_size_gb:.2f} GB)")

    def _load_stem_uncached(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        path = track_dir / f"{stem_name}.wav"
        if not path.exists():
            return None
        try:
            data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        except Exception as e:
            logger.warning(f"Could not read {path.name} ({e}), skipping")
            return None
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        wav = torch.from_numpy(data)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        return wav

    def _load_stem(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        if self.cache_in_ram and hasattr(self, '_ram_cache'):
            key = f"{track_dir.name}/{stem_name}"
            if key in self._ram_cache:
                return self._ram_cache[key].clone()
        return self._load_stem_uncached(track_dir, stem_name)

    def __len__(self) -> int:
        if self._synthetic:
            return 100
        return len(self.tracks) * len(self.stems) * 4

    def _load_crepe(self, track_dir: Path, stem_name: str, target_len: int) -> tuple[Tensor, float]:
        crepe_path = track_dir / f"{stem_name}.crepe.npy"
        pitched_path = track_dir / f"{stem_name}.pitched.npy"
        T_expected = 1 + target_len // self.hop_length
        if not crepe_path.exists() or not pitched_path.exists():
            return torch.zeros(T_expected, 360, dtype=torch.float32), 0.0
        probs = np.load(str(crepe_path)).astype(np.float32)
        pitched = float(np.load(str(pitched_path)))
        return torch.from_numpy(probs), pitched

    def _crop_with_crepe(self, wav: Tensor, crepe: Tensor) -> tuple[Tensor, Tensor]:
        L = wav.shape[-1]
        T_clip = 1 + self.clip_len // self.hop_length
        if L <= self.clip_len:
            wav_padded = F.pad(wav, (0, self.clip_len - L))
            T_have = crepe.shape[0]
            if T_have >= T_clip:
                crepe_out = crepe[:T_clip]
            else:
                crepe_out = F.pad(crepe, (0, 0, 0, T_clip - T_have))
            return wav_padded, crepe_out
        start = random.randint(0, L - self.clip_len)
        wav_crop = wav[start:start + self.clip_len]
        frame_start = start // self.hop_length
        frame_end = frame_start + T_clip
        T_have = crepe.shape[0]
        if frame_end <= T_have:
            crepe_crop = crepe[frame_start:frame_end]
        else:
            crepe_crop = crepe[frame_start:]
            crepe_crop = F.pad(crepe_crop, (0, 0, 0, T_clip - crepe_crop.shape[0]))
        return wav_crop, crepe_crop

    def _augment(self, wav: Tensor) -> Tensor:
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            wav = -wav
        return wav.clamp(-1.0, 1.0)

    def _get_synthetic(self) -> dict:
        T_clip = 1 + self.clip_len // self.hop_length
        return {
            'target': torch.zeros(self.clip_len),
            'other_stems': torch.zeros(len(self.stems) - 1, self.clip_len),
            'target_crepe': torch.zeros(T_clip, 360),
            'other_crepe': torch.zeros(len(self.stems) - 1, T_clip, 360),
            'target_pitched': 0.0,
            'other_pitched': torch.zeros(len(self.stems) - 1),
            'stem_name': 'synthetic',
            'valid': False,
        }

    def __getitem__(self, idx: int) -> dict:
        if self._synthetic:
            return self._get_synthetic()

        track_idx = (idx // (len(self.stems) * 4)) % len(self.tracks)
        stem_idx = (idx // 4) % len(self.stems)
        track_dir = self.tracks[track_idx]
        target_name = self.stems[stem_idx]

        target = self._load_stem(track_dir, target_name)
        if target is None:
            return self._get_synthetic()

        target_crepe, target_pitched = self._load_crepe(track_dir, target_name, target.shape[-1])
        target_trim, target_crepe = self._crop_with_crepe(target, target_crepe)

        other_wavs = []
        other_crepes = []
        other_pitched_l = []
        for name in self.stems:
            if name == target_name:
                continue
            wav = self._load_stem(track_dir, name)
            if wav is not None:
                cr, pit = self._load_crepe(track_dir, name, wav.shape[-1])
                w_crop, c_crop = self._crop_with_crepe(wav, cr)
                other_wavs.append(w_crop)
                other_crepes.append(c_crop)
                other_pitched_l.append(pit)
            else:
                other_wavs.append(torch.zeros(self.clip_len))
                other_crepes.append(torch.zeros(1 + self.clip_len // self.hop_length, 360))
                other_pitched_l.append(0.0)

        if self.augment:
            target_trim = self._augment(target_trim)
            other_wavs = [self._augment(w) for w in other_wavs]

        return {
            'target': target_trim,
            'other_stems': torch.stack(other_wavs, dim=0),
            'target_crepe': target_crepe,
            'other_crepe': torch.stack(other_crepes, dim=0),
            'target_pitched': target_pitched,
            'other_pitched': torch.tensor(other_pitched_l),
            'stem_name': target_name,
            'valid': True,
        }


def collate_valid(batch: list[dict]) -> Optional[dict]:
    """Filter out invalid batches and collate."""
    batch = [b for b in batch if b['valid']]
    if not batch:
        return None
    return {
        'target': torch.stack([b['target'] for b in batch]),
        'other_stems': torch.stack([b['other_stems'] for b in batch]),
        'target_crepe': torch.stack([b['target_crepe'] for b in batch]),
        'other_crepe': torch.stack([b['other_crepe'] for b in batch]),
        'target_pitched': torch.tensor([b['target_pitched'] for b in batch]),
        'other_pitched': torch.stack([b['other_pitched'] for b in batch]),
        'stem_names': [b['stem_name'] for b in batch],
    }


def get_musdb18hq_data_loader(
    data_root: str,
    cfg,
    batch_size: int = 16,
    num_workers: int = 8,
    clip_seconds: float = 4.0,
    augment: bool = True,
    cache_in_ram: bool = False,
) -> DataLoader:
    """Create a DataLoader for MUSDB18-HQ using the StemDataset format."""
    dataset = StemDataset(
        root=data_root, cfg=cfg, clip_seconds=clip_seconds,
        augment=augment, cache_in_ram=cache_in_ram,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=augment,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_valid, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )


class Slakh2100StemDataset(Dataset):
    """Slakh2100-YourMT3-16k dataset with StemDataset-compatible output format.

    Structure per track:
        Track00001/
            mix.wav
            stems/
                bass.wav drums.wav vocals.wav guitar.wav piano.wav ...

    Returns same dict keys as StemDataset (for drop-in compatibility with train_mamba3.py):
        target, other_stems, target_crepe, other_crepe, target_pitched, other_pitched, stem_name, valid
    """
    STEM_NAMES = ['bass', 'drums', 'vocals', 'other', 'guitar', 'piano', 'keys']

    def __init__(
        self,
        root: str,
        cfg,
        clip_seconds: float = 4.0,
        augment: bool = True,
        target_stems: Optional[list[str]] = None,
        cache_in_ram: bool = False,
    ):
        self.root = Path(root)
        self.sample_rate = cfg.sample_rate
        self.hop_length = getattr(cfg, 'hop_length', 512)
        self.clip_len = int(clip_seconds * self.sample_rate)
        self.augment = augment
        self.stems = target_stems or ['bass', 'drums', 'vocals', 'other', 'guitar']
        self.cache_in_ram = cache_in_ram
        self._ram_cache = {}

        try:
            self.tracks = sorted([
                d for d in self.root.iterdir()
                if d.is_dir() and (d / 'mix.wav').exists()
            ])
        except (FileNotFoundError, NotADirectoryError):
            self.tracks = []
        if not self.tracks:
            logger.warning(f"No Slakh2100 tracks found in {root}, using synthetic fallback")
            self._synthetic = True
        else:
            self._synthetic = False

        if self.cache_in_ram and not self._synthetic:
            self._preload_all_stems()

    def _preload_all_stems(self):
        logger.info(f"Pre-loading {len(self.tracks)} Slakh2100 tracks into RAM cache...")
        import tqdm
        total_stems = 0
        total_size_gb = 0.0
        for track_dir in tqdm.tqdm(self.tracks, desc="Loading Slakh2100 stems"):
            for stem_name in self.stems:
                wav = self._load_stem_uncached(track_dir, stem_name)
                if wav is not None:
                    key = f"{track_dir.name}/{stem_name}"
                    self._ram_cache[key] = wav
                    total_stems += 1
                    total_size_gb += wav.element_size() * wav.nelement()
        total_size_gb /= (1024 ** 3)
        logger.info(f"  Cached {total_stems} stems ({total_size_gb:.2f} GB)")

    def _load_stem_uncached(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        stem_path = track_dir / 'stems' / f"{stem_name}.wav"
        if not stem_path.exists():
            return None
        try:
            data, sr = sf.read(str(stem_path), dtype='float32', always_2d=True)
        except Exception as e:
            logger.warning(f"Could not read {stem_path.name} ({e}), skipping")
            return None
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]
        wav = torch.from_numpy(data)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        return wav

    def _load_stem(self, track_dir: Path, stem_name: str) -> Optional[Tensor]:
        if self.cache_in_ram and hasattr(self, '_ram_cache'):
            key = f"{track_dir.name}/{stem_name}"
            if key in self._ram_cache:
                return self._ram_cache[key].clone()
        return self._load_stem_uncached(track_dir, stem_name)

    def __len__(self) -> int:
        if self._synthetic:
            return 100
        return len(self.tracks) * len(self.stems) * 4

    def _make_synthetic_crepe(self, wav_len: int) -> Tensor:
        T = 1 + wav_len // self.hop_length
        return torch.zeros(T, 360, dtype=torch.float32)

    def _load_crepe(self, track_dir: Path, stem_name: str, target_len: int) -> tuple[Tensor, float]:
        return self._make_synthetic_crepe(target_len), 0.0

    def _crop_with_crepe(self, wav: Tensor, crepe: Tensor) -> tuple[Tensor, Tensor]:
        L = wav.shape[-1]
        T_clip = 1 + self.clip_len // self.hop_length
        if L <= self.clip_len:
            wav_padded = F.pad(wav, (0, self.clip_len - L))
            T_have = crepe.shape[0]
            if T_have >= T_clip:
                crepe_out = crepe[:T_clip]
            else:
                crepe_out = F.pad(crepe, (0, 0, 0, T_clip - T_have))
            return wav_padded, crepe_out
        start = random.randint(0, L - self.clip_len)
        wav_crop = wav[start:start + self.clip_len]
        frame_start = start // self.hop_length
        frame_end = frame_start + T_clip
        T_have = crepe.shape[0]
        if frame_end <= T_have:
            crepe_crop = crepe[frame_start:frame_end]
        else:
            crepe_crop = crepe[frame_start:]
            crepe_crop = F.pad(crepe_crop, (0, 0, 0, T_clip - crepe_crop.shape[0]))
        return wav_crop, crepe_crop

    def _augment(self, wav: Tensor) -> Tensor:
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            wav = -wav
        return wav.clamp(-1.0, 1.0)

    def _get_synthetic(self) -> dict:
        T_clip = 1 + self.clip_len // self.hop_length
        return {
            'target': torch.zeros(self.clip_len),
            'other_stems': torch.zeros(len(self.stems) - 1, self.clip_len),
            'target_crepe': torch.zeros(T_clip, 360),
            'other_crepe': torch.zeros(len(self.stems) - 1, T_clip, 360),
            'target_pitched': 0.0,
            'other_pitched': torch.zeros(len(self.stems) - 1),
            'stem_name': 'synthetic',
            'valid': False,
        }

    def __getitem__(self, idx: int) -> dict:
        if self._synthetic:
            return self._get_synthetic()

        track_idx = (idx // (len(self.stems) * 4)) % len(self.tracks)
        stem_idx = (idx // 4) % len(self.stems)
        track_dir = self.tracks[track_idx]
        target_name = self.stems[stem_idx]

        target = self._load_stem(track_dir, target_name)
        if target is None:
            return self._get_synthetic()

        target_crepe, target_pitched = self._load_crepe(track_dir, target_name, target.shape[-1])
        target_trim, target_crepe = self._crop_with_crepe(target, target_crepe)

        other_wavs = []
        other_crepes = []
        other_pitched_l = []
        for name in self.stems:
            if name == target_name:
                continue
            wav = self._load_stem(track_dir, name)
            if wav is not None:
                cr, pit = self._load_crepe(track_dir, name, wav.shape[-1])
                w_crop, c_crop = self._crop_with_crepe(wav, cr)
                other_wavs.append(w_crop)
                other_crepes.append(c_crop)
                other_pitched_l.append(pit)
            else:
                other_wavs.append(torch.zeros(self.clip_len))
                other_crepes.append(torch.zeros(1 + self.clip_len // self.hop_length, 360))
                other_pitched_l.append(0.0)

        if self.augment:
            target_trim = self._augment(target_trim)
            other_wavs = [self._augment(w) for w in other_wavs]

        return {
            'target': target_trim,
            'other_stems': torch.stack(other_wavs, dim=0),
            'target_crepe': target_crepe,
            'other_crepe': torch.stack(other_crepes, dim=0),
            'target_pitched': target_pitched,
            'other_pitched': torch.tensor(other_pitched_l),
            'stem_name': target_name,
            'valid': True,
        }


def get_slakh2100_loader(
    data_root: str,
    cfg,
    batch_size: int = 16,
    num_workers: int = 8,
    clip_seconds: float = 4.0,
    augment: bool = True,
    cache_in_ram: bool = False,
) -> DataLoader:
    """Create a DataLoader for Slakh2100-YourMT3-16k."""
    dataset = Slakh2100StemDataset(
        root=data_root, cfg=cfg, clip_seconds=clip_seconds,
        augment=augment, cache_in_ram=cache_in_ram,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=augment,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_valid, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )
