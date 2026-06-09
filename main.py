#!/usr/bin/env python3
"""
NeMo ModelPT Wrapper for Stem+MIDI Pro
Production-ready implementation matching the specified architecture
"""

import torch
from nemo.core.classes import ModelPT, typecheck
from nemo.core.neural_types import AudioSignal, NeuralType, ChannelType, LogitsType, VoidType
from typing import Optional, Dict, Tuple, List
import numpy as np

# Import our custom components
from models.mamba_separator import MambaSeparator
from models.mamba_transcriber import MambaTranscriber
from models.confidence_injector import ConfidenceInjector
from models.losses import PerceptualAudioLoss
from utils.quality_gates import route_by_quality, ProcessingReport, QualityTier


class StemMidiModel(ModelPT):
    """
    NeMo ModelPT wrapper for Stem+MIDI Pro
    Integrates separation, transcription, confidence injection, and quality routing
    """
    
    def __init__(self, config: dict):
        super().__init__(config=config)
        self.cfg = config
        
        # Initialize core components
        self.separator = MambaSeparator(self.cfg)
        self.transcriber = MambaTranscriber(self.cfg)
        self.confidence_injector = ConfidenceInjector(
            onset_threshold=self.cfg['transcriber'].get('onset_threshold', 0.5),
            min_confidence=self.cfg['quality_gates']['min_confidence']
        )
        self.loss_fn = PerceptualAudioLoss(self.cfg)
        
        # For training: store targets for loss computation
        
    @property
    def input_types(self):
        """Define NeMo neural types for model input"""
        return {
            "audio": NeuralType(('B', 'C', 'T'), AudioSignal()),
            # Optional targets for training (can be None during inference)
            "target_guitar": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "target_bass": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "target_onsets": NeuralType(('B', 'T', 1), LogitsType()),
            "target_pitch": NeuralType(('B', 'T', 'V'), LogitsType())
        }
    
    @property
    def output_types(self):
        """Define NeMo neural types for model output"""
        return {
            "guitar_stem": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "bass_stem": NeuralType(('B', 'C', 'T'), AudioSignal()),
            "midi_metadata": VoidType(),  # Complex metadata dict
            "processing_report": VoidType(),  # Quality metrics
            "routing_decision": VoidType()   # User-facing action
        }
    
    @typecheck()
    def forward(self, 
                audio: torch.Tensor,
                target_guitar: Optional[torch.Tensor] = None,
                target_bass: Optional[torch.Tensor] = None,
                target_onsets: Optional[torch.Tensor] = None,
                target_pitch: Optional[torch.Tensor] = None) -> Dict:
        """
        Full forward pass: audio → stems → MIDI + confidence + quality routing
        """
        B, C, T = audio.shape
        
        # 1. Separation with state caching (for streaming compatibility)
        guitar_stem, bass_stem, residual, _state_cache, sep_metrics = self.separator(audio)
        # _state_cache is available for streaming inference when implemented
        
        # 2. Transcription on guitar stem (could also do bass)
        onset_logits, pitch_logits, velocity, expression, confidence = self.transcriber(guitar_stem)
        
        # 3. Convert logits to MIDI events (simplified for MVP)
        midi_events = self._logits_to_midi(onset_logits, pitch_logits, velocity, confidence)
        
        # 4. Inject confidence metadata and compute alignment scores
        stem_energy = torch.mean(guitar_stem**2, dim=1).squeeze(-1)  # (B, T)
        enhanced_midi = self.confidence_injector.inject_midi_metadata(
            midi_events, confidence.squeeze(-1), stem_energy
        )
        
        # 5. Compute processing report for quality gating
        report = ProcessingReport(
            si_sdr=sep_metrics[0, 0].item(),  # Average over batch
            phase_coherence=sep_metrics[0, 1].item(),
            avg_confidence=confidence.mean().item(),
            artifact_flags=self._detect_artifacts(guitar_stem, bass_stem),
            low_confidence_notes=enhanced_midi['summary']['low_confidence_count'],
            config=self.cfg
        )
        
        # 6. Get routing decision based on quality
        routing = route_by_quality(report)
        
        # Prepare outputs
        outputs = {
            "guitar_stem": guitar_stem,
            "bass_stem": bass_stem,
            "midi_metadata": enhanced_midi,  # Contains MIDI events + summary
            "processing_report": report,
            "routing_decision": routing
        }
        
        # 7. Compute loss if targets provided (training mode)
        if target_guitar is not None and target_bass is not None:
            loss, loss_dict = self.loss_fn(
                pred_stems=[guitar_stem, bass_stem],
                target_stems=[target_guitar, target_bass],
                pred_onsets=onset_logits,
                target_onsets=target_onsets
            )
            outputs["loss"] = loss
            outputs["loss_dict"] = loss_dict
            
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step for NeMo PTL integration"""
        audio = batch["audio"]
        targets = {
            "target_guitar": batch["target_guitar"],
            "target_bass": batch["target_bass"],
            "target_onsets": batch["target_onsets"],
            "target_pitch": batch["target_pitch"]
        }
        
        outputs = self.forward(audio, **targets)
        loss = outputs["loss"]
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        for k, v in outputs["loss_dict"].items():
            self.log(f"train_{k}", v)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        audio = batch["audio"]
        targets = {
            "target_guitar": batch["target_guitar"],
            "target_bass": batch["target_bass"],
            "target_onsets": batch["target_onsets"],
            "target_pitch": batch["target_pitch"]
        }
        
        outputs = self.forward(audio, **targets)
        loss = outputs["loss"]
        
        self.log("val_loss", loss, prog_bar=True)
        for k, v in outputs["loss_dict"].items():
            self.log(f"val_{k}", v)
            
        return loss
    
    def _logits_to_midi(self, onset_logits, pitch_logits, velocity, confidence):
        """Convert network outputs to MIDI event list"""
        # Apply thresholds and get discrete predictions
        onsets = (torch.sigmoid(onset_logits) > self.cfg['transcriber'].get('onset_threshold', 0.5)).squeeze(-1)
        pitches = torch.argmax(pitch_logits, dim=-1)
        vel_values = (velocity * 127).long().squeeze(-1)
        
        events = []
        for b in range(onsets.shape[0]):  # Batch dimension
            batch_events = []
            for t in range(onsets.shape[1]):
                if onsets[b, t]:
                    batch_events.append({
                        'note': pitches[b, t].item(),
                        'onset_frame': t,
                        'velocity': vel_values[b, t].item(),
                        'confidence': confidence[b, t].item()
                    })
            events.append(batch_events)
        return events
    
    def _detect_artifacts(self, guitar: torch.Tensor, bass: torch.Tensor) -> List[str]:
        """Simple artifact detection for quality gating"""
        flags = []
        
        # Check for clipping
        if guitar.abs().max() > 0.99 or bass.abs().max() > 0.99:
            flags.append("clipping")
            
        # Check for excessive noise (simplified)
        guitar_noise = torch.mean(guitar**2, dim=[1,2])
        bass_noise = torch.mean(bass**2, dim=[1,2])
        if torch.any(guitar_noise > 0.1) or torch.any(bass_noise > 0.1):
            flags.append("noise_floor")
            
        # Check for phase cancellation (simplified)
        cross_corr = torch.sum(guitar * bass, dim=[1,2])
        energy_guitar = torch.sum(guitar**2, dim=[1,2])
        energy_bass = torch.sum(bass**2, dim=[1,2])
        phase_corr = cross_corr / (torch.sqrt(energy_guitar * energy_bass) + 1e-8)
        if torch.any(torch.abs(phase_corr) > 0.9):  # Near-perfect correlation = likely artifact
            flags.append("phase_cancellation")
            
        return flags if flags else ["none"]
    
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Inference method for production use
        Handles file I/O and returns user-ready package
        """
        # Load and preprocess audio (would use audio_io.py in full implementation)
        audio, sr = self._load_audio(audio_path)
        audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)  # Add batch/channels
        
        # Ensure correct sample rate
        if sr != self.cfg['audio']['sample_rate']:
            audio = torch.nn.functional.interpolate(
                audio, 
                size=int(audio.shape[-1] * self.cfg['audio']['sample_rate'] / sr),
                mode='linear',
                align_corners=False
            )
        
        # Run inference
        with torch.inference_mode():
            outputs = self.forward(audio)
        
        # Format for user delivery
        return {
            "stems": {
                "guitar": outputs["guitar_stem"].squeeze().cpu().numpy(),
                "bass": outputs["bass_stem"].squeeze().cpu().numpy()
            },
            "midi": outputs["midi_metadata"],
            "report": outputs["processing_report"],
            "routing": outputs["routing_decision"],
            "sample_rate": self.cfg['audio']['sample_rate']
        }

    def process_audio_streaming(self, audio_path: str, chunk_seconds: float = 2.0) -> Dict:
        """
        Streaming inference: process audio in chunks with state caching.

        Args:
            audio_path: Path to input audio file
            chunk_seconds: Duration of each chunk in seconds (default 2.0)

        Returns:
            Same format as process_audio_file (aggregated outputs)
        """
        audio, sr = self._load_audio(audio_path)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)

        target_sr = self.cfg['audio']['sample_rate']
        if sr != target_sr:
            audio_t = torch.nn.functional.interpolate(
                audio_t,
                size=int(audio_t.shape[-1] * target_sr / sr),
                mode='linear',
                align_corners=False,
            )
            sr = target_sr

        chunk_len = int(chunk_seconds * sr)
        overlap_len = int(chunk_len * self.cfg['audio'].get('overlap_ratio', 0.5))
        hop_len = chunk_len - overlap_len
        total_len = audio_t.shape[-1]

        all_guitar = []
        all_bass = []
        all_residual = []
        all_confidence = []
        all_onsets = []

        state_cache = None
        n_chunks = 0

        self.eval()
        with torch.inference_mode():
            pos = 0
            while pos < total_len:
                end = min(pos + chunk_len, total_len)
                chunk = audio_t[:, :, pos:end]

                if chunk.shape[-1] < chunk_len:
                    pad_len = chunk_len - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_len))

                guitar, bass, residual, new_cache, sep_metrics = self.separator(
                    chunk, state_cache=state_cache
                )

                trim_len = end - pos if pos + chunk_len > total_len else hop_len
                all_guitar.append(guitar[:, :, :trim_len])
                all_bass.append(bass[:, :, :trim_len])
                all_residual.append(residual[:, :, :trim_len])

                onsets, pitches, velocity, expression, confidence = self.transcriber(guitar[:, :, :trim_len])
                all_onsets.append(onsets)
                all_confidence.append(confidence)

                state_cache = new_cache
                pos += hop_len
                n_chunks += 1

        full_guitar = torch.cat(all_guitar, dim=-1)
        full_bass = torch.cat(all_bass, dim=-1)

        final_metrics = torch.tensor([0.0, 0.0])

        onset_logits = torch.cat(all_onsets, dim=1)
        pitch_logits = torch.zeros((1, onset_logits.shape[1], 128))
        velocity = torch.zeros((1, onset_logits.shape[1], 1))
        confidence = torch.cat(all_confidence, dim=1)

        midi_events = self._logits_to_midi(onset_logits, pitch_logits, velocity, confidence)
        stem_energy = torch.mean(full_guitar ** 2, dim=1).squeeze(-1)
        enhanced_midi = self.confidence_injector.inject_midi_metadata(
            midi_events, confidence.squeeze(-1), stem_energy
        )

        report = ProcessingReport(
            si_sdr=final_metrics[0].item(),
            phase_coherence=final_metrics[1].item(),
            avg_confidence=confidence.mean().item(),
            artifact_flags=self._detect_artifacts(full_guitar, full_bass),
            low_confidence_notes=enhanced_midi['summary']['low_confidence_count'],
            config=self.cfg,
        )

        routing = route_by_quality(report)

        return {
            "stems": {"guitar": full_guitar.squeeze().cpu().numpy(), "bass": full_bass.squeeze().cpu().numpy()},
            "midi": enhanced_midi,
            "report": report,
            "routing": routing,
            "sample_rate": sr,
            "chunks_processed": n_chunks,
        }

    def _load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio file using soundfile (or librosa as fallback)."""
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
        except Exception:
            import librosa
            audio, sr = librosa.load(path, sr=None, mono=True)
        if audio.ndim > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr


def load_from_checkpoint(checkpoint_path: str, config_path: Optional[str] = None) -> StemMidiModel:
    """
    Load model from NeMo checkpoint
    """
    if config_path:
        # Load config from file
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Try to extract config from checkpoint
        config = None  # NeMo handles this internally
    
    if config is None:
        raise ValueError(
            "config_path is required when loading from checkpoint "
            "(config cannot be extracted from .nemo checkpoint in this prototype)"
        )
    
    model = StemMidiModel(config=config)
    return model.load_from_checkpoint(checkpoint_path)


if __name__ == "__main__":
    # Example usage for testing
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--checkpoint", type=str, help="Path to .nemo checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.checkpoint:
        # Load from checkpoint
        model = load_from_checkpoint(args.checkpoint, args.config)
    else:
        # Initialize new model
        model = StemMidiModel(config)
    
    # Process audio
    result = model.process_audio_file(args.audio)
    
    # Print summary
    print(f"Processing complete!")
    print(f"Quality: {result['report'].quality_tier.value}")
    print(f"SI-SDR: {result['report'].si_sdr:.1f}dB")
    print(f"Avg Confidence: {result['report'].avg_confidence:.0%}")
    print(f"Routing: {result['routing']['action']}")