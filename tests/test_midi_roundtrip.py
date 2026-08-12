# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
"""Tests for MIDI round-trip (build MidiFile, save, re-parse)."""
from __future__ import annotations

import io

import pytest
from mido import MidiFile

from utils.midi import events_to_midi_bytes, logits_to_events
import torch


class TestMidiRoundtrip:
    """Test MIDI file construction and re-parsing."""

    def test_build_and_reparse_guitar(self):
        """Build a MIDI file and re-parse it."""
        events = [
            {"note": 60, "onset_frame": 100, "velocity": 100, "confidence": 0.9},
            {"note": 64, "onset_frame": 200, "velocity": 80, "confidence": 0.85},
            {"note": 67, "onset_frame": 300, "velocity": 64, "confidence": 0.75},
        ]

        midi_bytes = events_to_midi_bytes(events, "guitar")
        assert len(midi_bytes) > 0

        # Re-parse
        mid = MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) >= 1

    def test_build_and_reparse_bass(self):
        """Build a bass MIDI file and re-parse."""
        events = [
            {"note": 36, "onset_frame": 100, "velocity": 90, "confidence": 0.88},
            {"note": 43, "onset_frame": 300, "velocity": 70, "confidence": 0.82},
        ]

        midi_bytes = events_to_midi_bytes(events, "bass")
        mid = MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) >= 1

    def test_cc127_values_preserved(self):
        """CC#127 values should be present in the MIDI data."""
        events = [
            {"note": 60, "onset_frame": 100, "velocity": 100, "confidence": 0.9,
             "cc_127_value": 114},
        ]

        midi_bytes = events_to_midi_bytes(events, "guitar")
        mid = MidiFile(file=io.BytesIO(midi_bytes))

        # Check that control_change messages exist
        cc_messages = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == "control_change" and msg.control == 127:
                    cc_messages.append(msg)

        assert len(cc_messages) > 0, "No CC#127 messages found"

    def test_empty_events(self):
        """Empty events should produce valid minimal MIDI."""
        midi_bytes = events_to_midi_bytes([], "guitar")
        mid = MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) >= 1  # At least one track with header

    def test_logits_to_events(self):
        """logits_to_events should produce correct output structure."""
        onset = torch.randn(1, 20, 1)
        pitch = torch.randn(1, 20, 128)
        vel = torch.rand(1, 20, 1)
        conf = torch.rand(1, 20)

        events = logits_to_events(onset, pitch, vel, conf, onset_threshold=0.5)

        assert isinstance(events, list)
        assert isinstance(events[0], list)
        # Events will depend on random threshold; just verify structure
        for ev in events[0]:
            assert "note" in ev
            assert "onset_frame" in ev
            assert "velocity" in ev
            assert "confidence" in ev
