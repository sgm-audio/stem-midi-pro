# SPDX-License-Identifier: Apache-2.0
"""Deterministic signal builders and MIDI parsing for the test suite.

Why this exists: every pre-existing audio fixture in this suite is a windowed
sine wave (``conftest._sine_seconds``) or ``torch.randn``. Those two families
cannot fail an audio model in the ways real audio fails: they never clip, never
go silent, never carry a transient, never exceed 0.3 amplitude, and never have
a DC component. A separator that is simply "input x 0.6" passes all of them.

``Synth`` provides the adversarial/realistic corpus (silence, hard-clipped
square, DC, single-sample impulse, plucked-string transients, dense polyphony,
pink noise, anti-phase stereo, NaN, full-scale ramps) with exact, seed-fixed
numpy so tests stay reproducible on CPU.

``parse_notes`` decodes an SMF into absolute-tick note events, which is what
the MIDI tests need: counting messages proves nothing about timing (that is how
the 60x tempo-scaling bug in utils/midi.py survived).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np

SR = 44100
#: Guitar/bass practical register in MIDI note numbers (open E2=40 on guitar,
#: B0=23..C6=84 across both instruments with a little slack for harmonics).
GUITAR_BASS_LOWEST_USABLE = 28
GUITAR_BASS_HIGHEST_USABLE = 96


class Synth:
    """Signal builders. All return float32 mono ``(n,)`` unless stated."""

    SR = SR

    # -- degenerate / adversarial -------------------------------------------
    @staticmethod
    def silence(seconds: float = 1.0, sr: int = SR) -> np.ndarray:
        return np.zeros(int(seconds * sr), dtype=np.float32)

    @staticmethod
    def dc(level: float = 1.0, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
        """Constant offset — no onsets, no AC energy. Breaks naive RMS/peak logic."""
        return np.full(int(seconds * sr), level, dtype=np.float32)

    @staticmethod
    def impulse(seconds: float = 1.0, sr: int = SR, n: int = 1, amp: float = 1.0) -> np.ndarray:
        out = np.zeros(int(seconds * sr), dtype=np.float32)
        out[:n] = amp
        return out

    @staticmethod
    def clipped_square(freq: float = 220.0, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
        """A square wave: every sample sits at ±1.0 — genuine clipping."""
        t = np.arange(int(seconds * sr)) / sr
        return np.sign(np.sin(2 * math.pi * freq * t)).astype(np.float32)

    @staticmethod
    def hard_clip(x: np.ndarray, level: float = 1.0) -> np.ndarray:
        return np.clip(x, -level, level).astype(np.float32)

    @staticmethod
    def overflow(
        seconds: float = 1.0, freq: float = 220.0, gain: float = 8.0, sr: int = SR
    ) -> np.ndarray:
        """Far past full scale (as a float WAV may legitimately contain)."""
        t = np.arange(int(seconds * sr)) / sr
        return (gain * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    @staticmethod
    def with_nan(x: np.ndarray, at: int = 100) -> np.ndarray:
        out = np.array(x, copy=True)
        out[at] = np.nan
        return out

    @staticmethod
    def with_inf(x: np.ndarray, at: int = 100) -> np.ndarray:
        out = np.array(x, copy=True)
        out[at] = np.inf
        return out

    @staticmethod
    def denormal_quiet(seconds: float = 1.0, freq: float = 220.0, sr: int = SR) -> np.ndarray:
        """-140 dBFS tone: exercises log(x+eps) floors and threshold logic."""
        t = np.arange(int(seconds * sr)) / sr
        return (1e-7 * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    # -- musical-ish ---------------------------------------------------------
    @staticmethod
    def tone(
        freq: float = 220.0, seconds: float = 1.0, amp: float = 0.3, sr: int = SR
    ) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)

    @staticmethod
    def pluck(
        freq: float = 110.0,
        seconds: float = 1.0,
        amp: float = 0.35,
        n_partials: int = 6,
        decay: float = 3.5,
        attack_samples: int = 48,
        sr: int = SR,
    ) -> np.ndarray:
        """Karplus-Strong-adjacent plucked string: sharp attack + decaying harmonics.

        This is the closest cheap stand-in for real guitar material: unlike a
        windowed sine it has a broadband transient (the thing onset detectors
        must survive) and a decaying, slightly inharmonic spectrum.
        """
        n = int(seconds * sr)
        t = np.arange(n) / sr
        env = np.exp(-decay * t)
        if attack_samples > 1:  # linear ramp so the attack is not a pure step
            ramp = np.linspace(0.0, 1.0, attack_samples)
            env[:attack_samples] *= ramp
        x = np.zeros(n, dtype=np.float64)
        for k in range(1, n_partials + 1):
            # strings are mildly inharmonic; partials also decay faster
            x += (1.0 / k) * np.sin(2 * math.pi * freq * k * t) * np.exp(-decay * 1.4 * (k - 1) * t)
        return (amp * x * env).astype(np.float32)

    @staticmethod
    def note_sequence(
        notes: list[int] | None = None,
        seconds_per_note: float = 0.25,
        total_seconds: float = 4.0,
        amp: float = 0.35,
        sr: int = SR,
    ) -> np.ndarray:
        """A monophonic line of plucked notes on an exact frame grid."""
        if notes is None:
            notes = [40, 45, 47, 52, 55, 43, 48, 40]
        n = int(total_seconds * sr)
        out = np.zeros(n, dtype=np.float64)
        step = int(seconds_per_note * sr)
        for i, midi in enumerate(notes):
            start = i * step
            if start >= n:
                break
            freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            seg = Synth.pluck(freq, seconds=(n - start) / sr, amp=amp, sr=sr)
            out[start : start + len(seg)] += seg[: n - start]
        return out.astype(np.float32)

    @staticmethod
    def chord(
        n_voices: int = 8,
        seconds: float = 2.0,
        amp: float = 0.25,
        low_note: int = 40,
        sr: int = SR,
        spread: int = 12,
    ) -> np.ndarray:
        """Simultaneous plucked partials — extreme polyphony."""
        per_voice = amp / max(1, n_voices) ** 0.5
        out = np.zeros(int(seconds * sr), dtype=np.float64)
        for v in range(n_voices):
            midi = low_note + (v * spread) // max(1, n_voices)
            freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            out += Synth.pluck(freq, seconds=seconds, amp=per_voice, sr=sr).astype(np.float64)
        peak = float(np.max(np.abs(out))) or 1.0
        return (out / peak * min(0.95, amp * 2)).astype(np.float32)

    @staticmethod
    def noise(
        seconds: float = 1.0, seed: int = 0, color: str = "white", sr: int = SR
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(seconds * sr)
        x = rng.standard_normal(n)
        if color == "pink":
            # simple 1/f shaping by cumulative smoothing (deterministic, no scipy)
            k = 24
            x = np.convolve(x, np.ones(k) / k, mode="same")
            x = x / (np.max(np.abs(x)) or 1.0)
        x = x / (np.max(np.abs(x)) or 1.0) * 0.25
        return x.astype(np.float32)

    @staticmethod
    def anti_phase_stereo(mono: np.ndarray) -> np.ndarray:
        """(T, 2) with L = -R: any mean-downmix turns this into silence."""
        return np.stack([mono, -mono], axis=1).astype(np.float32)

    @staticmethod
    def mismatched_sr(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Length-preserving naive resample (no librosa) for sample-rate paths."""
        n = int(len(mono) * dst_sr / src_sr)
        idx = np.linspace(0, len(mono) - 1, n)
        lo = np.floor(idx).astype(int)
        hi = np.clip(lo + 1, 0, len(mono) - 1)
        frac = (idx - lo).astype(np.float32)
        return (mono[lo] * (1 - frac) + mono[hi] * frac).astype(np.float32)

    # -- tensor helpers ------------------------------------------------------
    @staticmethod
    def to_batch(x: np.ndarray) -> object:
        """(T,) -> torch (1, 1, T) float32."""
        import torch

        return torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# MIDI decoding helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvent:
    note: int
    velocity: int
    on_tick: int
    off_tick: int
    on_sec: float
    off_sec: float
    #: Polyphonic Key Pressure value in force at this note's onset, i.e. the
    # note-specific confidence value emitted by the product. None when no
    # pressure precedes the note. Off-by-one emission of confidence metadata is
    # exactly what tests/test_midi_semantics.py guards.
    cc_at_onset: int | None = None
    cc_after: int | None = None  # value written between this note_off and the next note_on

    @property
    def duration_sec(self) -> float:
        return self.off_sec - self.on_sec


def parse_midi(midi_bytes: bytes | io.BytesIO, track: int = 0, cc: int | None = 16) -> dict:
    """Decode an SMF into notes + absolute time in ticks and seconds.

    The product writes confidence as Polyphonic Key Pressure, a note-specific
    MIDI message. For compatibility with the audit suite's ``cc_values`` name,
    pressure values are collected when ``cc`` is the default 16; explicit
    ``cc`` values select actual control-change messages, and ``cc=None``
    collects both streams.

    Returns {"notes": [NoteEvent], "ticks_per_beat": int, "tempo_us": int,
             "all_messages": [mido.Message], "length_sec": float,
             "dangling_notes": {pitch: n_open_at_end}}
    """
    import mido

    src = io.BytesIO(midi_bytes) if isinstance(midi_bytes, bytes | bytearray) else midi_bytes
    mid = mido.MidiFile(file=src)
    trk = mid.tracks[track]

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    sec_per_tick = tempo_us / 1e6 / tpb

    tick = 0
    messages = []
    open_stack: dict[int, list] = {}
    notes: list[list] = []  # mutable rows, frozen into NoteEvent below
    cc_events: list[tuple[int, int]] = []  # (abs_tick, value)

    for msg in trk:
        tick += msg.time
        messages.append(msg)
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            sec_per_tick = tempo_us / 1e6 / tpb
            continue
        if msg.type == "control_change":
            if cc is None or msg.control == cc:
                cc_events.append((tick, msg.value))
            continue
        if msg.type == "polytouch":
            if cc is None or cc == 16:
                cc_events.append((tick, msg.value))
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            open_stack.setdefault(msg.note, []).append((tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = open_stack.get(msg.note)
            if stack:
                on_tick, vel = stack.pop(0)
                if not stack:
                    open_stack.pop(msg.note, None)
                notes.append([msg.note, vel, on_tick, tick])

    def cc_before(t: int) -> int | None:
        val = None
        for ct, cv in cc_events:
            if ct <= t:
                val = cv
            else:
                break
        return val

    def cc_after(off_t: int) -> int | None:
        for ct, cv in cc_events:
            if ct > off_t:
                return cv
        return None

    rows = [
        NoteEvent(
            note=n,
            velocity=v,
            on_tick=on_t,
            off_tick=off_t,
            on_sec=on_t * sec_per_tick,
            off_sec=off_t * sec_per_tick,
            cc_at_onset=cc_before(on_t),
            cc_after=cc_after(off_t),
        )
        for (n, v, on_t, off_t) in notes
    ]
    return {
        "notes": rows,
        "ticks_per_beat": tpb,
        "tempo_us": tempo_us,
        "all_messages": messages,
        "length_sec": tick * sec_per_tick,
        "dangling_notes": {n: len(v) for n, v in open_stack.items()},
        "cc_values": [cv for _ct, cv in cc_events],
    }
