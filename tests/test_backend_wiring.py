# SPDX-License-Identifier: Apache-2.0
"""Is the configured backend the backend that runs?

``models/backends/`` provides a factory (``fake`` / ``demucs`` / ``basic_pitch``
/ ``mamba``) and three of the four are never executed by any test — but more
importantly, none of them are reached by the production entry point either:

    main.StemMidiModel.__init__  ->  MambaSeparator(cfg) / MambaTranscriber(cfg)

It reads ``cfg["separator"]`` and ``cfg["transcriber"]`` directly and never
calls ``models.backends.factory``. Consequently:

* ``configs/model_config.cpu.yaml``, whose whole purpose is
  ``backends: {separator: demucs, transcriber: basic_pitch}``, produces a
  Mamba model when loaded through ``main``, ``api``, ``demo`` or the CLI.
* ``tests/test_backends.py::test_stem_midi_model_with_fake_backends`` does not
  build a fake-backed model (see ``test_fake_backend_is_not_reachable_...``).

These tests pin the *wiring* rather than the factory, which is already covered.
"""

import pytest

torch = pytest.importorskip("torch", reason="requires torch")


def test_factory_selects_the_configured_backend(mock_config):
    """Control: the factory itself does honour ``cfg['backends']`` (passes today)."""
    from models.backends.factory import build_separator, build_transcriber
    from models.backends.fake_backend import FakeSeparatorBackend, FakeTranscriberBackend

    assert isinstance(build_separator(mock_config), FakeSeparatorBackend)
    assert isinstance(build_transcriber(mock_config), FakeTranscriberBackend)


def test_stem_midi_model_uses_the_configured_backend(mock_config):
    """A model configured with ``backends.separator='fake'`` must use it.

    ``tests/test_backends.py::test_stem_midi_model_with_fake_backends`` passes a
    fake-backend config and then asserts only on shapes, so it would pass even
    if the fake backend did not exist. The model actually built there is the
    real 12-layer Mamba stack at test dims — the fixture's promise ("no
    mamba_ssm / demucs needed") is kept only because of the pure-torch fallback,
    not because the fake backend is in play.
    """
    from main import StemMidiModel
    from models.backends.fake_backend import FakeSeparatorBackend, FakeTranscriberBackend

    model = StemMidiModel(mock_config)
    # Configured backends live on ``*_backend`` (H-9); the built-in Mamba
    # modules live on ``separator`` / ``transcriber``.
    separator = getattr(model, "separator_backend", getattr(model, "separator", None))
    transcriber = getattr(model, "transcriber_backend", getattr(model, "transcriber", None))
    assert isinstance(
        separator, FakeSeparatorBackend
    ), f"separator is {type(separator).__name__} despite backends.separator='fake'"
    assert isinstance(transcriber, FakeTranscriberBackend)


def test_cpu_config_still_declares_external_backends():
    """Config rot guard: the CPU config must keep naming its intended backends."""
    import yaml

    with open("configs/model_config.cpu.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    backends = cfg.get("backends") or {}
    assert backends.get("separator") == "demucs"
    assert backends.get("transcriber") == "basic_pitch"


def test_report_identifies_the_backend_that_produced_it(mock_config):
    """The delivered report must name the separator/transcriber that ran.

    The fake backend returns a hard-coded ``si_sdr = 12.0`` and a single middle-C
    note; a Demucs run returns something else entirely. Without a backend field
    the two are indistinguishable downstream, which is how a fixture ends up in
    a customer ZIP.
    """
    from main import StemMidiModel

    with torch.inference_mode():
        report = StemMidiModel(mock_config).eval()(torch.zeros(1, 1, 4410))["processing_report"]
    assert getattr(report, "backend", None)
