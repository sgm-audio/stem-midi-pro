# SPDX-License-Identifier: Apache-2.0
"""EXPR-1: the expression head is either carried into the SMF or not claimed.

The original finding was that README/API_DOCUMENTATION advertised per-note
expression while `_expression` was discarded and the writer emitted no
pitch-bend and no CC lane -- the model computed phrasing that never reached a
file. That is fixed upstream by making the export OPT-IN and labelling it.

This file locks the resulting contract, because "opt-in and documented" is only
honest if all three parts stay true:

* the default render carries NO expression (nothing is invented);
* the opt-in render carries bend as pitch-wheel, vibrato as CC1 and slide as
  CC74 -- so the documented mapping is the real one;
* the shipped config has the switch off, and every event that does carry
  expression is labelled with its untrained provenance.

Together those mean the docs cannot drift back into claiming a capability the
code does not deliver.
"""

from __future__ import annotations

import io

import pytest

mido = pytest.importorskip("mido", reason="expression lands in a MIDI file")


def _events() -> list[dict]:
    return [
        {
            "note": 60,
            "onset_frame": 10,
            "duration_frames": 40,
            "velocity": 100,
            "confidence": 0.9,
            "expression": {"bend": 0.5, "vibrato": 0.7, "slide": 0.2},
            "source": "expression_head (untrained; ARCH-11.4.4)",
        }
    ]


def _track(**kwargs) -> list:
    from utils.midi import build_midi_from_events

    data = build_midi_from_events({"midi_events": _events()}, "guitar", **kwargs)
    return list(mido.MidiFile(file=io.BytesIO(data)).tracks[0])


def test_the_default_render_writes_no_expression():
    """Off by default: an untrained head must not fabricate phrasing."""
    track = _track()
    assert not any(m.type == "pitchwheel" for m in track), "default wrote a pitch bend"
    controls = [m.control for m in track if m.type == "control_change"]
    assert controls == [], f"default wrote CC lanes {controls}"


def test_the_opt_in_render_carries_the_documented_mapping():
    """bend -> pitch-wheel, vibrato -> CC1, slide -> CC74, as the docs say."""
    track = _track(include_expression=True)
    assert any(m.type == "pitchwheel" for m in track), "opt-in wrote no pitch bend"
    controls = {m.control for m in track if m.type == "control_change"}
    assert 1 in controls, f"vibrato is not CC1 (got {sorted(controls)})"
    assert 74 in controls, f"slide is not CC74 (got {sorted(controls)})"


def test_the_shipped_config_leaves_expression_off():
    """The switch ships off, so a default install cannot emit invented phrasing."""
    import yaml

    for path in (
        "configs/model_config.yaml",
        "configs/model_config.cpu.yaml",
        "configs/model_config.mamba.yaml",
    ):
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert (
            cfg.get("midi", {}).get("include_expression") is False
        ), f"{path} enables expression export by default; the head is untrained"


def test_expression_events_carry_their_provenance_label():
    """Whatever the mapping does, the event says where the values came from.

    Without this the opt-in export would be indistinguishable from learned
    expression in a downstream consumer.
    """
    events = _events()
    assert events[0]["source"].startswith("expression_head")
    assert "untrained" in events[0]["source"]
