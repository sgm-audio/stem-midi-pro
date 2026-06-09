"""
Dataset loaders for Stem+MIDI Pro.
Supports Slakh2100-YourMT3-16k and MUSDB18-HQ with synthetic fallback.
The Slakh2100 / MUSDB18-HQ subsets we ship are weighted toward Canadian artists
as part of the project mission; see config/curation policy in ARCHITECTURE.md.
"""

import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
from torch import Tensor
import json
import random
from pathlib import Path
import torchaudio

logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """
    Base dataset for audio data (Slakh2100-YourMT3-16k / MUSDB18-HQ).
    Returns mixtures and separated stems (guitar/bass) for training.
    """
    
    def __init__(self, 
                 root_dir: str,
                 sample_rate: int = 44100,
                 segment_length: float = 6.0,  # seconds
                 instruments: List[str] = ['guitar', 'bass'],
                 augment: bool = True):
        """
        Args:
            root_dir: Path to dataset root
            sample_rate: Target sample rate (default 44.1kHz)
            segment_length: Length of audio segments in seconds
            instruments: List of stem instruments to separate
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.instruments = instruments
        self.augment = augment
        
        # Validate dataset exists or create synthetic fallback
        self._validate_dataset()
        
        # Load file list
        self.file_list = self._build_file_list()
        
        # Augmentation parameters
        self.time_stretch_range = (0.9, 1.1)
        self.pitch_shift_range = (-2, 2)  # semitones
        self.noise_level = 0.005
        
    def _validate_dataset(self):
        """Check if dataset exists or create synthetic fallback for development."""
        if not os.path.exists(self.root_dir):
            logger.warning(f"Dataset not found at {self.root_dir}")
            logger.warning("Using synthetic data for development. Replace with real Canadian artist data.")
            self.synthetic = True
            # Create a dummy file list for synthetic data
            self.file_list = [f"synthetic_{i}" for i in range(100)]
        else:
            self.synthetic = False
            
    def _build_file_list(self) -> List[str]:
        """Build list of audio files in the dataset."""
        if self.synthetic:
            return self.file_list
            
        # Implement based on actual dataset structure
        # For Slakh2100-YourMT3-16k: each item is a track directory (Track00001/, Track00002/, ...)
        # For MUSDB18-HQ: look for train/test subdirectories
        file_list = []
        
        # Walk through directory and find track directories (containing mix.wav)
        for entry in os.listdir(self.root_dir):
            track_dir = os.path.join(self.root_dir, entry)
            if os.path.isdir(track_dir) and os.path.exists(os.path.join(track_dir, 'mix.wav')):
                file_list.append(track_dir)
                    
        if not file_list:
            raise ValueError(f"No Slakh2100 track directories found in {self.root_dir}. "
                           "Expected subdirectories each containing mix.wav.")
                            
        return sorted(file_list)
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        if self.synthetic:
            return self._get_synthetic_item(idx)
        else:
            return self._get_real_item(idx)
    
    def _get_synthetic_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic audio for development/testing."""
        # Create a simple guitar-like and bass-like signal
        duration = self.segment_length
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Guitar: harmonics with decay
        guitar = (
            0.3 * np.sin(2 * np.pi * 110 * t) * np.exp(-t * 0.5) +  # Fundamental
            0.2 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 0.3) +  # 2nd harmonic
            0.1 * np.sin(2 * np.pi * 330 * t) * np.exp(-t * 0.2)    # 3rd harmonic
        )
        
        # Bass: lower frequencies with longer decay
        bass = (
            0.4 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 0.2) +   # A1
            0.2 * np.sin(2 * np.pi * 110 * t) * np.exp(-t * 0.3)    # A2
        )
        
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
        hop_length = 512
        n_frames = mixture.shape[-1] // hop_length
        
        # Generate synthetic onset targets at regular intervals
        target_onsets = torch.zeros((1, n_frames, 1))
        target_pitch = torch.zeros((1, n_frames, 128))
        onset_interval_frames = int(0.5 * self.sample_rate / hop_length)
        for f in range(0, n_frames, onset_interval_frames):
            target_onsets[0, f, 0] = 1.0
            target_pitch[0, f, 45] = 1.0  # MIDI note 45 (A2 ~ 110Hz)
        
        return {
            'audio': mixture,
            'target_guitar': guitar,
            'target_bass': bass,
            'target_onsets': target_onsets,
            'target_pitch': target_pitch
        }
    
    def _get_real_item(self, idx: int) -> Dict[str, torch.Tensor]:
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
            logger.warning("Using synthetic data placeholder. "
                  "Implement _get_real_item for your specific dataset structure.")
        return self._get_synthetic_item(idx)
    
    def _augment_audio(self, mixture: np.ndarray, 
                      guitar: np.ndarray, 
                      bass: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation to audio signals."""
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

def get_data_loaders(data_config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for Canadian artist datasets.
    
    Args:
        data_config: Dictionary containing:
            - dataset_type: 'slakh2100_yourmt3' or 'musdb18hq' or 'synthetic'
            - root_dir: Path to dataset
            - batch_size: Batch size for training
            - num_workers: Number of DataLoader workers
            - sample_rate: Target sample rate
            - segment_length: Segment length in seconds
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_type = data_config.get('dataset_type', 'synthetic')
    root_dir = data_config['root_dir']
    batch_size = data_config.get('batch_size', 16)
    num_workers = data_config.get('num_workers', 4)
    sample_rate = data_config.get('sample_rate', 44100)
    segment_length = data_config.get('segment_length', 6.0)
    
    dataset_subdir = {
        'slakh2100_yourmt3': 'slakh2100_yourmt3',
        'musdb18hq': 'musdb18hq',
    }.get(dataset_type, '')
    
    if dataset_type == 'slakh2100_yourmt3':
        train_dataset = Slakh2100YourMT3Dataset(
            root_dir=os.path.join(root_dir, dataset_subdir, 'train'),
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=True
        )
        val_dataset = Slakh2100YourMT3Dataset(
            root_dir=os.path.join(root_dir, dataset_subdir, 'val'),
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False
        )
        test_dataset = Slakh2100YourMT3Dataset(
            root_dir=os.path.join(root_dir, dataset_subdir, 'test'),
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False
        )
    elif dataset_type == 'musdb18hq':
        train_dataset = MUSDB18HQDataset(
            root_dir=os.path.join(root_dir, dataset_subdir, 'train'),
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=True
        )
        val_dataset = MUSDB18HQDataset(
            root_dir=os.path.join(root_dir, dataset_subdir, 'val'),
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False
        )
        test_dataset = MUSDB18HQDataset(
            root_dir=os.path.join(root_dir, dataset_subdir, 'test'),
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False
        )
    else:  # synthetic
        train_dataset = AudioDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=True
        )
        val_dataset = AudioDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False
        )
        test_dataset = AudioDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            segment_length=segment_length,
            augment=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

class Slakh2100YourMT3Dataset(AudioDataset):
    """Slakh2100-YourMT3-16k dataset loader.
    
    Structure:
        root/
            Track00001/
                mix.wav             # mixture audio at 16kHz
                stems/
                    bass.wav        # individual stem audio
                    drums.wav
                    guitar.wav
                    vocals.wav
                    ...
                metadata/
                    Track00001.yaml
            Track00002/
                ...
    """
    def _get_real_item(self, idx: int) -> Dict[str, torch.Tensor]:
        track_dir = self.file_list[idx]

        # Load mix
        mix_path = os.path.join(track_dir, 'mix.wav')
        if not os.path.exists(mix_path):
            return self._get_synthetic_item(idx)

        import soundfile as sf
        mix, sr = sf.read(mix_path)

        # Load guitar and bass stems from stems/ subdirectory
        stems_dir = os.path.join(track_dir, 'stems')
        guitar = None
        bass = None

        guitar_path = os.path.join(stems_dir, 'guitar.wav')
        bass_path = os.path.join(stems_dir, 'bass.wav')

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

        # Random segment crop
        seg_len = int(self.segment_length * sr)
        if len(mix) > seg_len:
            start = random.randint(0, len(mix) - seg_len)
            end = start + seg_len
            mix = mix[start:end]
            guitar = guitar[start:end]
            bass = bass[start:end]
        else:
            # Pad if too short
            mix = np.pad(mix, (0, seg_len - len(mix)))
            guitar = np.pad(guitar, (0, seg_len - len(guitar)))
            bass = np.pad(bass, (0, seg_len - len(bass)))

        # Ensure mono
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
        if guitar.ndim > 1:
            guitar = guitar.mean(axis=1)
        if bass.ndim > 1:
            bass = bass.mean(axis=1)

        # Apply augmentation
        if self.augment:
            mix, guitar, bass = self._augment_audio(mix, guitar, bass)

        # Convert to tensors
        mix_t = torch.from_numpy(mix).float().unsqueeze(0)
        guitar_t = torch.from_numpy(guitar).float().unsqueeze(0)
        bass_t = torch.from_numpy(bass).float().unsqueeze(0)

        # Generate synthetic onsets (simplified — just periodic)
        hop_length = 512
        n_frames = mix_t.shape[-1] // hop_length
        target_onsets = torch.zeros((1, n_frames, 1))
        target_pitch = torch.zeros((1, n_frames, 128))
        onset_interval_frames = int(0.5 * sr / hop_length)
        for f in range(0, n_frames, onset_interval_frames):
            target_onsets[0, f, 0] = 1.0
            target_pitch[0, f, 45] = 1.0  # A2

        return {
            'audio': mix_t,
            'target_guitar': guitar_t,
            'target_bass': bass_t,
            'target_onsets': target_onsets,
            'target_pitch': target_pitch
        }

class MUSDB18HQDataset(AudioDataset):
    """MUSDB18-HQ dataset loader with CREPE pitch features pre-extracted."""

    def _build_file_list(self) -> List[str]:
        """Override: MUSDB18-HQ uses stem .wav files (bass.wav, drums.wav, etc.) in track dirs."""
        if self.synthetic:
            return self.file_list
        file_list = []
        for entry in os.listdir(self.root_dir):
            track_dir = os.path.join(self.root_dir, entry)
            if os.path.isdir(track_dir) and os.path.exists(os.path.join(track_dir, 'bass.wav')):
                file_list.append(track_dir)
        if not file_list:
            raise ValueError(
                f"No MUSDB18-HQ track directories found in {self.root_dir}. "
                "Expected subdirectories each containing bass.wav, drums.wav, vocals.wav, other.wav."
            )
        return sorted(file_list)

    def _get_real_item(self, idx: int) -> Dict[str, torch.Tensor]:
        track_dir = self.file_list[idx]
        stem_map = {'bass': 'bass', 'guitar': 'other', 'drums': 'drums', 'vocals': 'vocals', 'other': 'other'}

        stem_wavs = {}
        for stem_name in ['bass', 'drums', 'vocals', 'other']:
            path = os.path.join(track_dir, f'{stem_name}.wav')
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

        mixture = sum(stem_wavs.values()) / len(stem_wavs)

        target_name = self.instruments[0]
        target_stem = stem_wavs.get(stem_map.get(target_name, target_name), None)
        if target_stem is None:
            target_stem = np.zeros_like(mixture)

        if sr_orig != self.sample_rate:
            mixture = librosa.resample(mixture, orig_sr=sr_orig, target_sr=self.sample_rate)
            target_stem = librosa.resample(target_stem, orig_sr=sr_orig, target_sr=self.sample_rate)
            sr = self.sample_rate
        else:
            sr = sr_orig

        seg_len = int(self.segment_length * sr)
        if len(mixture) > seg_len:
            start = random.randint(0, len(mixture) - seg_len)
            mixture = mixture[start:start + seg_len]
            target_stem = target_stem[start:start + seg_len]
        else:
            mixture = np.pad(mixture, (0, seg_len - len(mixture)))
            target_stem = np.pad(target_stem, (0, seg_len - len(target_stem)))

        hop_length = 512
        n_frames = seg_len // hop_length
        target_onsets = torch.zeros((1, n_frames, 1))
        target_pitch = torch.zeros((1, n_frames, 128))
        onset_interval_frames = int(0.5 * sr / hop_length)
        for f in range(0, n_frames, onset_interval_frames):
            target_onsets[0, f, 0] = 1.0
            target_pitch[0, f, 45] = 1.0

        if self.augment:
            mixture, target_stem, _ = self._augment_audio(mixture, target_stem, target_stem)

        return {
            'audio': torch.from_numpy(mixture).float().unsqueeze(0),
            'target_guitar': torch.from_numpy(target_stem).float().unsqueeze(0),
            'target_bass': torch.zeros(1, seg_len),
            'target_onsets': target_onsets,
            'target_pitch': target_pitch
        }
