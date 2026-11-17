# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for Stem+MIDI Pro (T-8.1.1, honesty layer T-8.1.3).

All heavy imports (torch, numpy, soundfile, mamba_ssm) happen INSIDE
fixtures/tests so that `pytest --collect-only` works in environments without
those dependencies installed.

Two things were added on top of the original fixture set:

* dependency gates go through ``tests.honesty.require`` instead of
  ``pytest.importorskip`` — under ``--strict-deps`` (CI) a missing torch is an
  ERROR, not a pass-by-skip;
* the session-finish honesty report (:func:`pytest_sessionfinish`) fails the run
  when a test that is supposed to cover the real model path did not run, which
  is how the ``mamba_ssm`` skip used to hide a checkpoint-format bug.
"""

import os

import pytest

from tests.honesty import mamba_ssm_available, require, set_mode
from tests.synth import SR, Synth, parse_midi

# ---------------------------------------------------------------------------
# CLI options + honesty enforcement
# ---------------------------------------------------------------------------


def pytest_addoption(parser) -> None:
    group = parser.getgroup("stem-midi-honesty")
    group.addoption(
        "--strict-deps",
        action="store_true",
        default=False,
        help="Missing declared dependencies fail instead of skipping (use in CI).",
    )
    group.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Fail the session if ANY test skipped (strictest audit mode, nightly).",
    )
    group.addoption(
        "--require-mamba-ssm",
        action="store_true",
        default=False,
        help="Fail if mamba_ssm is importable but the mamba-ssm tier did not execute "
        "(for the dedicated CUDA job; keeps the CPU skip story honest).",
    )


# nodeid -> set of marker names / outcome record, filled during collection+run
_TIER_OF: dict[str, set] = {}
_OUTCOME: dict[str, dict] = {}

#: Markers that declare "this test belongs to a tier that must execute".
_MUST_RUN_TIERS = {"real_mamba"}

#: Skip-reason prefixes that the honesty gate accepts (mirror of
#: tests/honesty.py::ALLOWED_SKIP_REASONS plus pytest's own marker wording).
_ALLOWED_PREFIXES = (
    "optional-dep:",
    "codec-unavailable:",
    "artifact-unavailable:",
    "Skipped:",
    "unneeded on this host:",
)


def pytest_configure(config) -> None:
    set_mode(
        strict_deps=bool(config.getoption("--strict-deps")),
        fail_on_skip=bool(config.getoption("--fail-on-skip")),
        require_mamba_ssm=bool(config.getoption("--require-mamba-ssm")),
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items) -> None:
    """Record tiers for the *selected* items only.

    trylast so the marker/keyword deselection (``-m``, ``-k``, explicit file
    args) has already happened: a test the user deliberately deselected is not a
    suite-honesty violation, an unrun must-run test is.
    """
    selected = {item.nodeid for item in items}
    for nodeid in list(_TIER_OF):
        if nodeid not in selected:
            _TIER_OF.pop(nodeid, None)
    for item in items:
        tiers = {m.name for m in item.iter_markers()}
        _TIER_OF[item.nodeid] = tiers
        # requires_mamba_ssm tests may skip on a CPU box — but if the package IS
        # importable and the test still does not run, that is a mislabel.
        if "requires_mamba_ssm" in tiers and not mamba_ssm_available():
            item.add_marker(pytest.mark.skip(reason="optional-dep: mamba_ssm not installed"))


def pytest_runtest_logreport(report) -> None:
    rec = _OUTCOME.setdefault(report.nodeid, {})
    longrepr = report.longrepr
    reason = str(longrepr[-1] if isinstance(longrepr, tuple) else longrepr)
    if report.when == "setup" and report.skipped:
        rec.setdefault("outcome", "skipped")
        rec["reason"] = reason
    elif report.failed:
        rec["outcome"] = "failed"
    elif report.when == "call":
        if getattr(report, "wasxfail", None) is not None:
            rec["outcome"] = "xfailed" if report.skipped else "xpassed"
        elif report.passed:
            rec["outcome"] = "passed"
        elif report.skipped:
            rec["outcome"] = "skipped"
            rec["reason"] = reason


def _honesty_violations(config) -> list[str]:
    """Violations of the tier contract, as human-readable lines."""
    out: list[str] = []
    fail_on_skip = config.getoption("--fail-on-skip")
    for nodeid, tiers in sorted(_TIER_OF.items()):
        rec = _OUTCOME.get(nodeid, {})
        outcome = rec.get("outcome", "not-run")
        reason = rec.get("reason", "")
        must = tiers & _MUST_RUN_TIERS
        if must and outcome in ("skipped", "not-run"):
            out.append(
                f"{nodeid}: must-run tier {sorted(must)} did not execute ({outcome}: {reason})"
            )
        if "requires_mamba_ssm" in tiers and mamba_ssm_available() and outcome == "skipped":
            out.append(f"{nodeid}: mamba_ssm IS importable but the tier skipped ({reason})")
        too_long = not any(pref in reason for pref in _ALLOWED_PREFIXES)
        if outcome == "skipped" and fail_on_skip and too_long:
            out.append(f"{nodeid}: unregistered skip under --fail-on-skip ({reason})")
    if os.getenv("STEM_KNOWN_BUGS", "xfail").strip().lower() == "fail":
        for nodeid, rec in sorted(_OUTCOME.items()):
            if rec.get("outcome") == "xfailed":
                out.append(
                    f"{nodeid}: known bug still open "
                    f"(STEM_KNOWN_BUGS=fail reports every xfail as a failure)"
                )
    if config.getoption("--require-mamba-ssm") and not mamba_ssm_available():
        out.append(
            "--require-mamba-ssm was passed but mamba_ssm is not importable in this environment"
        )
    return out


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Capability table: which tiers ran, per marker. Never silent."""
    tr = terminalreporter
    counts: dict[str, int] = {}
    for nodeid, tiers in _TIER_OF.items():
        outcome = _OUTCOME.get(nodeid, {}).get("outcome", "not-run")
        for tier in sorted(tiers) or ["unmarked"]:
            key = f"{tier}:{outcome}"
            counts[key] = counts.get(key, 0) + 1
    if counts:
        tr.write_sep("=", "test-suite capability report (honesty layer)")
        for key in sorted(counts):
            tr.write_line(f"  {key:36s} {counts[key]}")
        tr.write_line(f"  mamba_ssm importable: {mamba_ssm_available()}")
    violations = _honesty_violations(config)
    if violations:
        tr.write_line("")
        tr.write_line("SUITE-HONESTY VIOLATIONS (skips that hide coverage):")
        for v in violations[:40]:
            tr.write_line(f"  ! {v}")
        if len(violations) > 40:
            tr.write_line(f"  ... and {len(violations) - 40} more")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Flip the exit code when a must-run tier silently skipped.

    ``session.exitstatus`` is read by the CLI after this hook, so mutating it
    here is the supported way to fail a run whose skips would otherwise be
    reported as green.
    """
    if _honesty_violations(session.config):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# ---------------------------------------------------------------------------
# Audio tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio():
    """1-second 44.1 kHz mono audio tensor of shape (1, 1, 44100)."""
    torch = require("torch", "the audio tensor fixture")
    sr = 44100
    t = torch.linspace(0, 1, sr)
    mono = 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 110 * t)
    return mono.reshape(1, 1, sr).float()


@pytest.fixture
def stereo_audio():
    """1-second 44.1 kHz stereo audio tensor of shape (1, 2, 44100)."""
    torch = require("torch", "the stereo audio fixture")
    sr = 44100
    t = torch.linspace(0, 1, sr)
    left = 0.3 * torch.sin(2 * torch.pi * 220 * t)
    right = 0.3 * torch.sin(2 * torch.pi * 330 * t)
    return torch.stack([left, right]).unsqueeze(0).float()  # (1, 2, 44100)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Full config dict mirroring configs/model_config.yaml, downsized for CPU.

    Same keys as the real config, but tiny dims so tests run in seconds on
    CPU (d_model=32, n_layers=2, small n_fft so (1,1,8192) forwards are fast).
    """
    return {
        "name": "stem_midi_mamba_test",
        "restore_from_path": None,
        "audio": {
            "sample_rate": 44100,
            "n_fft": 512,
            "hop_length": 128,
            "n_mels": 32,
            "chunk_duration_sec": 2.0,
            "overlap_ratio": 0.5,
        },
        "separator": {
            "d_model": 32,
            "n_layers": 2,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
        },
        "transcriber": {
            "d_model": 32,
            "n_layers": 2,
            "d_state": 8,
            "onset_head_dim": 16,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend", "vibrato", "slide"],
            "onset_threshold": 0.5,
        },
        "training": {
            "precision": "fp32",
            "gradient_checkpointing": False,
        },
        "loss": {
            "mr_stft_weight": 1.0,
            "spectral_flatness_weight": 0.1,
            "crest_factor_weight": 0.05,
            "onset_f1_weight": 1.0,
            "pitch_ce_weight": 0.8,
            "velocity_mae_weight": 0.3,
            "cross_modal_alignment_weight": 0.5,
        },
        "quality_gates": {
            "studio_confidence_threshold": 0.85,
            "draft_confidence_threshold": 0.70,
            "min_confidence": 0.6,
            "min_si_sdr": 20.0,
        },
        # Default test backends: no mamba_ssm / demucs / basic-pitch required
        "backends": {
            "separator": "fake",
            "transcriber": "fake",
            "device": "cpu",
        },
    }


@pytest.fixture
def pipeline_config(mock_config):
    """Audio block from configs/model_config.yaml (n_fft 2048 / hop 512 / 80 mels).

    Two reasons pipeline tests must not use ``mock_config``:
      * ``mock_config`` has hop_length=128 while ``utils.midi.HOP_LENGTH`` is 512, so
        any frame→tick→frame round trip through MIDI silently mixes units (see
        MIDI-5); with this fixture the writer's default is the model's grid.
      * ``mock_config`` pairs n_fft=512 with hop=512, and torch.istft rejects a
        non-overlapping Hann hop ("window overlap add min"), so the real separator
        cannot run on it at all — see CFG-2 in test_audio_edge_cases.py.
    """
    audio = {**mock_config["audio"], "n_fft": 2048, "hop_length": 512, "n_mels": 80}
    return {**mock_config, "audio": audio}


# ---------------------------------------------------------------------------
# Temporary audio files
# ---------------------------------------------------------------------------


def _sine_seconds(seconds, sr=44100, freq=220.0):
    """Pure-numpy sine wave, float32 in [-1, 1)."""
    import math

    n = int(seconds * sr)
    return [0.3 * math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def _write_wav_wave_module(path, samples, sr):
    """Fallback WAV writer using the stdlib `wave` module (16-bit PCM)."""
    import struct
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        w.writeframes(frames)


@pytest.fixture
def temp_audio_file(tmp_path):
    """1-second 44.1 kHz mono WAV file (via soundfile, else stdlib wave)."""
    path = tmp_path / "test_audio.wav"
    sr = 44100
    samples = _sine_seconds(1.0, sr)
    try:
        import numpy as np
        import soundfile as sf

        sf.write(str(path), np.asarray(samples, dtype=np.float32), sr)
    except ImportError:
        _write_wav_wave_module(path, samples, sr)
    return str(path)


@pytest.fixture
def temp_stereo_audio_file(tmp_path):
    """1-second 44.1 kHz stereo WAV file."""
    sf = require("soundfile", "the stereo WAV fixture")
    np = require("numpy", "the stereo WAV fixture")
    path = tmp_path / "test_stereo.wav"
    sr = 44100
    t = np.linspace(0, 1, sr)
    stereo = np.stack(
        [0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)],
        axis=1,
    ).astype(np.float32)
    sf.write(str(path), stereo, sr)
    return str(path)


@pytest.fixture
def temp_flac_file(tmp_path):
    """1-second 44.1 kHz mono FLAC file (requires soundfile FLAC support)."""
    sf = require("soundfile", "the FLAC fixture")
    np = require("numpy", "the FLAC fixture")
    path = tmp_path / "test_audio.flac"
    sr = 44100
    samples = np.asarray(_sine_seconds(1.0, sr), dtype=np.float32)
    try:
        sf.write(str(path), samples, sr, format="FLAC")
    except Exception as exc:
        pytest.skip(f"codec-unavailable: soundfile lacks FLAC support: {exc}")
    return str(path)


@pytest.fixture
def temp_mp3_file(tmp_path):
    """1-second 44.1 kHz mono MP3 file; skipped if no MP3 encoder available."""
    sf = require("soundfile", "the MP3 fixture")
    np = require("numpy", "the MP3 fixture")
    path = tmp_path / "test_audio.mp3"
    sr = 44100
    samples = np.asarray(_sine_seconds(1.0, sr), dtype=np.float32)
    try:
        sf.write(str(path), samples, sr, format="MP3", subtype="MPEG_LAYER_III")
    except Exception as exc:
        pytest.skip(f"codec-unavailable: soundfile lacks MP3 encoder support: {exc}")
    return str(path)


# ---------------------------------------------------------------------------
# Adversarial / realistic audio corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synth():
    """Signal builders (silence, clipped, DC, impulse, plucks, polyphony, NaN...)."""
    return Synth


@pytest.fixture
def make_wav(tmp_path):
    """Write a numpy signal to a WAV file, return its path.

    ``make_wav(sig, name="silence.wav", sr=44100, subtype="FLOAT")``
    """
    sf = require("soundfile", "the make_wav fixture")
    np = require("numpy", "the make_wav fixture")

    def _make(signal, name="signal.wav", sr=SR, subtype=None):
        path = tmp_path / name
        arr = np.asarray(signal, dtype=np.float32)
        kwargs = {"subtype": subtype} if subtype else {}
        sf.write(str(path), arr, sr, format="WAV", **kwargs)
        return str(path)

    return _make


@pytest.fixture(scope="session")
def midi_reader():
    """SMF decoder returning absolute-tick note events (see tests/synth.py)."""
    return parse_midi


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model(mock_config):
    """StemMidiModel built from mock_config (downsized for CPU).

    The pure-PyTorch Mamba fallback in models/_mamba_compat.py means mamba_ssm
    and nemo are NOT required (both were dropped; see DEP-4.1.3 / DEP-4.1.5) —
    so this fixture has no excuse to skip when torch is installed: require()
    turns a missing torch into a failure under --strict-deps.
    """
    require("torch", "StemMidiModel")
    from main import StemMidiModel

    model = StemMidiModel(mock_config)
    model.eval()
    return model


@pytest.fixture
def real_model(pipeline_config):
    """StemMidiModel on the real (non-fake) model path, marked real_mamba.

    Deliberately NO skip guard around the import of models.mamba_*: if the
    import fails, that is a hard error, not a skip (a silent skip here is how
    the C-2.x class of bugs stayed invisible).
    """
    require("torch", "the real Mamba model path")
    from main import StemMidiModel

    torch = require("torch", "the real Mamba model path")
    model = StemMidiModel(pipeline_config)
    model.eval()
    torch.manual_seed(1234)
    return model
