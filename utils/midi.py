# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""MIDI utilities: logits-to-events, events-to-bytes, MIDI file building."""
from __future__ import annotations

import io
from typing import Dict, List

import torch
from mido import MidiFile, MidiTrack, Message, MetaMessage


def logits_to_events(
    onset_logits: torch.Tensor,
    pitch_logits: torch.Tensor,
    velocity: torch.Tensor,
    confidence: torch.Tensor,
    onset_threshold: float = 0.5,
) -> List[List[Dict]]:
    """Convert network outputs to MIDI event list (vectorized).
    
    Args:
        onset_logits: (B, T, 1) raw logits
        pitch_logits: (B, T, 128) raw pitch logits
        velocity: (B, T, 1) normalized velocity [0,1]
        confidence: (B, T) confidence scores
        onset_threshold: sigmoid threshold for onset detection
    Returns:
        List of batch_events, where each batch_events is a list of event dicts
    """
    onsets = (torch.sigmoid(onset_logits) > onset_threshold).squeeze(-1)
    pitches = torch.argmax(pitch_logits, dim=-1)
    vel_values = (velocity * 127).long().squeeze(-1)

    events = []
    for b in range(onsets.shape[0]):
        onset_mask = onsets[b]
        indices = torch.nonzero(onset_mask, as_tuple=False).squeeze(-1)
        batch_events = []
        for t in indices:
            batch_events.append({
                "note": pitches[b, t].item(),
                "onset_frame": t.item(),
                "velocity": vel_values[b, t].item(),
                "confidence": confidence[b, t].item(),
            })
        events.append(batch_events)
    return events


def events_to_midi_bytes(
    midi_events: list,
    stem_type: str,
    sample_rate: int = 44100,
    hop_length: int = 512,
    ticks_per_beat: int = 480,
    tempo_bpm: int = 120,
) -> bytes:
    """Convert MIDI event dicts to a MIDI file (bytes).
    
    Args:
        midi_events: List of event dicts with keys: onset_frame, note, velocity, confidence
        stem_type: 'guitar' or 'bass'
        sample_rate: Audio sample rate
        hop_length: STFT hop length
        ticks_per_beat: MIDI ticks per quarter note
        tempo_bpm: Tempo in BPM
    Returns:
        MIDI file as bytes
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    tempo_microseconds = int(60_000_000 / tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo_microseconds))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4))
    track.append(MetaMessage("track_name", name=f"{stem_type} transcription"))
    track.append(
        MetaMessage(
            "instrument_name",
            name="Electric Guitar" if stem_type == "guitar" else "Electric Bass",
        )
    )
    track.append(
        MetaMessage("marker", text=f"{stem_type.capitalize()} - Stem+MIDI Pro")
    )

    ticks_per_second = ticks_per_beat * (tempo_bpm / 60)

    for ev in midi_events:
        tick = int(ev["onset_frame"] * hop_length / sample_rate * ticks_per_second)
        note = ev["note"]
        velocity = min(127, max(1, int(ev.get("velocity", 64))))
        track.append(Message("note_on", note=note, velocity=velocity, time=tick))

        dur_frames = ev.get("duration_frames", int(0.25 * sample_rate))
        dur_ticks = max(1, int(dur_frames * hop_length / sample_rate * ticks_per_second))
        track.append(Message("note_off", note=note, velocity=0, time=dur_ticks))

        cc_val = ev.get("cc_127_value", int(ev.get("confidence", 0.5) * 127))
        track.append(Message("control_change", control=127, value=cc_val, time=0))

    track.append(MetaMessage("end_of_track"))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def build_midi_from_events(
    midi_data: Dict,
    stem_type: str,
    sample_rate: int = 44100,
    hop_length: int = 512,
) -> bytes:
    """Convenience wrapper: build MIDI bytes from the enhanced_midi dict."""
    return events_to_midi_bytes(
        midi_events=midi_data.get("midi_events", []),
        stem_type=stem_type,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
