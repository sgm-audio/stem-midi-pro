# SPDX-License-Identifier: Apache-2.0
"""Backend selection, protocol conformance and the mamba-ssm shim's honesty.

Two audit findings live here.

(1) ``models/backends/*`` is dead code as far as the product is concerned.
    ``StemMidiModel.__init__`` constructs ``MambaSeparator``/``MambaTranscriber``
    directly and never calls ``build_separator``/``build_transcriber``, so
    ``cfg["backends"]`` does nothing. The shipped ``configs/model_config.cpu.yaml``
    says ``separator: demucs / transcriber: basic_pitch`` — the "production" CPU
    path — and the model silently runs the untrained Mamba instead. The only tests
    that look like they cover this (``test_backends.py::test_stem_midi_model_with_fake_backends``,
    ``test_process_audio_file_fake``) assert key presence, so they pass while
    exercising the *real* modules and never touch the fake backends they are named
    after. ``test_pretrained_backends_are_unreachable_from_main`` below proves it.

(2) The CPU fallback in ``models/_mamba_compat.py`` swallows unknown kwargs
    (``**_ignored``), which is exactly how the C-2.1 invalid-kwargs bug stayed
    invisible on CPU: a config typo that hard-crashes on a CUDA box with real
    mamba-ssm is silently dropped here. ``test_shim_rejects_unknown_kwargs``
    closes that hole, and the mamba-ssm parity test keeps the two
    implementations numerically honest when the package IS present.
"""

from __future__ import annotations

import pytest

from tests.doubles import inject
from tests.honesty import known_bug, mamba_ssm_available, require

torch = require("torch", "backend contract tests", module_level=True)

SR = 44100
HOP = 512


# ---------------------------------------------------------------------------
# 1. factory + protocol conformance (must-run: no optional dep involved)
# ---------------------------------------------------------------------------


def test_factory_builds_protocol_conformant_fake_backends(mock_config):
    from models.backends.base import SeparatorBackend, TranscribeOut, TranscriberBackend
    from models.backends.factory import build_separator, build_transcriber

    sep = build_separator(mock_config)
    tr = build_transcriber(mock_config)
    assert isinstance(
        sep, SeparatorBackend
    ), "FakeSeparatorBackend does not satisfy SeparatorBackend"
    assert isinstance(tr, TranscriberBackend), "FakeTranscriberBackend is not protocol-conformant"

    out = sep.separate(torch.zeros(2, 1, 8192))
    assert set(out) == {"guitar", "bass", "residual", "metrics"}
    assert out["metrics"].shape == (2, 2)
    tout = tr.transcribe(torch.zeros(1, 1, SR))
    assert isinstance(tout, dict)
    for key in TranscribeOut.__annotations__:
        assert key in tout or not tout, f"TranscribeOut contract missing {key}"


def test_backend_names_are_validated_and_never_silently_downgraded(mock_config):
    """An unknown name must raise; a broken module must NOT be replaced by 'fake'."""
    from models.backends import factory

    with pytest.raises(ValueError, match="nope"):
        factory.build_separator({**mock_config, "backends": {"separator": "nope"}})

    # simulate `pip install demucs` missing: the import error must propagate
    broken = {**mock_config, "backends": {"separator": "demucs"}}
    real = factory._SEPARATORS["demucs"]
    factory._SEPARATORS["demucs"] = "definitely_not_a_module:Nope"
    try:
        with pytest.raises(Exception) as exc_info:
            factory.build_separator(broken)
        assert "definitely_not_a_module" in str(exc_info.value)
        assert not isinstance(exc_info.value, KeyError), "swallowed into a fake backend"
    finally:
        factory._SEPARATORS["demucs"] = real


def test_pretrained_backends_are_reachable_from_main(pipeline_config):
    """Regression guard (was BACK-1, fixed on main): cfg["backends"] must be honoured.

    ``StemMidiModel.__init__`` now builds through models/backends/factory when the
    key is present, and the pure-Mamba path stays the default when it is absent.
    """
    from main import StemMidiModel
    from models.backends.fake_backend import FakeSeparatorBackend, FakeTranscriberBackend

    cfg = {
        **pipeline_config,
        "backends": {"separator": "fake", "transcriber": "fake", "device": "cpu"},
    }
    model = StemMidiModel(cfg)
    assert isinstance(
        model.separator_backend, FakeSeparatorBackend
    ), f"backends.separator='fake' was ignored: got {type(model.separator_backend).__name__}"
    assert isinstance(
        model.transcriber_backend, FakeTranscriberBackend
    ), f"backends.transcriber='fake' was ignored: got {type(model.transcriber_backend).__name__}"

    # ...and the built-in path must still be the DEFAULT: an unset `backends` key may
    # not silently fall through to a fake, or every "real model" run would be theatre.
    plain = StemMidiModel(pipeline_config)
    assert not hasattr(
        plain, "separator_backend"
    ), "with no backends: key the model must use its own MambaSeparator, not a wrapper"
    assert type(plain.separator).__name__ == "MambaSeparator", type(plain.separator).__name__
    assert (
        "mamba" in plain.backend_name.lower()
    ), f"the model must report the backend it actually runs, got {plain.backend_name!r}"


def test_model_consumes_a_protocol_conformant_backend(pipeline_config):
    """Regression guard for BACK-2 (fixed by PR #11): the TypedDict contract is consumed.

    `models/backends/base.py` promises SeparateOut / TranscribeOut dicts with pre-built
    `events`; `_forward_impl` used to unpack a 5-tuple, which made a real backend
    impossible to wire in even after cfg["backends"] was honoured. This asserts the
    dict-shaped output (including a backend-supplied event list) reaches the MIDI, so a
    re-tupling of the contract fails here rather than in a downstream integration.
    """
    from main import StemMidiModel

    model = StemMidiModel(
        {**pipeline_config, "backends": {"separator": "fake", "transcriber": "fake"}}
    )
    model.eval()
    from models.backends.fake_backend import FakeSeparatorBackend, FakeTranscriberBackend

    audio = torch.full((1, 1, SR), 0.2)
    inject(model, "separator_backend", FakeSeparatorBackend(pipeline_config))
    inject(model, "transcriber_backend", FakeTranscriberBackend(pipeline_config))
    with torch.inference_mode():
        out = model(audio)
    # the fake backend pre-builds one note (60) with duration_frames=40; the
    # documented contract says main.py prefers `events` over the logits path
    md = out["midi_metadata"]["guitar"]
    got = [(e["note"], e.get("duration_frames")) for e in md["midi_events"]]
    assert (60, 4) in got, (
        f"backend-supplied note events (with duration_frames) never reached the MIDI: {got}. "
        f"_forward_impl unpacks a 5-tuple instead of consuming TranscribeOut"
    )


# ---------------------------------------------------------------------------
# 2. the mamba-ssm shim: capability reporting and strictness
# ---------------------------------------------------------------------------


def test_shim_reports_the_backend_it_actually_uses():
    """The shim's advertised backend must match reality (no 'mamba-ssm' on a CPU box)."""
    from models import _mamba_compat

    reported = _mamba_compat.MAMBA_BACKEND
    expected = "mamba-ssm" if mamba_ssm_available() else "pure-torch"
    assert reported == expected, (
        f"_mamba_compat advertises backend {reported!r} but mamba_ssm importable="
        f"{mamba_ssm_available()}: the capability string users/CI rely on is wrong"
    )


@known_bug(
    "SHIM-1",
    "the pure-torch fallback's signature ends in `**_ignored`, so CUDA-only kwargs "
    "(causal_conv1d_impl=, selective_scan_impl=, use_fast_path= — the exact C-2.1 bug class) "
    "are silently swallowed on CPU and only blow up on a CUDA box. Validate kwargs and raise "
    "TypeError, so the fallback rejects what real mamba-ssm rejects.",
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"causal_conv1d_impl": "triton"},
        {"selective_scan_impl": "cudnn"},
        {"use_fast_path": False},
        {"n_groups": 1},
        {"bogus_kwarg": 1},
    ],
    ids=["causal_conv1d_impl", "selective_scan_impl", "use_fast_path", "n_groups", "bogus"],
)
def test_shim_rejects_unknown_kwargs(kwargs):
    from models._mamba_compat import Mamba

    with pytest.raises(TypeError):
        Mamba(d_model=8, d_state=4, d_conv=4, expand=2, **kwargs)


def test_shim_accepts_the_documented_signature_only():
    """The four documented kwargs must work; this is the contract ARCHITECTURE.md cites."""
    from models._mamba_compat import Mamba

    blk = Mamba(d_model=16, d_state=4, d_conv=4, expand=2)
    x = torch.randn(1, 8, 16)
    with torch.inference_mode():
        y = blk(x)
    assert y.shape == x.shape
    assert blk.d_inner == 32 and blk.dt_rank == 1


def test_separator_and_transcriber_build_with_no_mamba_ssm_installed():
    """The CPU promise (D-STRAT-3): the real modules construct and run without mamba_ssm.

    Guards the *inverse* dishonesty too: nobody may re-add a hard mamba_ssm import
    and then have the tests skip around it.
    """
    from models.mamba_separator import MambaSeparator
    from models.mamba_transcriber import MambaTranscriber

    cfg = {
        "audio": {"sample_rate": SR, "n_fft": 2048, "hop_length": HOP, "n_mels": 80},
        "separator": {"d_model": 32, "n_layers": 1, "d_state": 8, "d_conv": 4, "expand": 2},
        "transcriber": {
            "d_model": 32,
            "n_layers": 1,
            "d_state": 8,
            "onset_head_dim": 8,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
    }
    sep, tr = MambaSeparator(cfg).eval(), MambaTranscriber(cfg).eval()
    x = torch.randn(1, 1, 8192) * 0.1
    with torch.inference_mode():
        g, _b, _r, cache, _metrics = sep(x)
        on, _pi, _ve, _ex, _co = tr(g)
    assert torch.isfinite(g).all() and torch.isfinite(on).all()
    assert cache is None


@pytest.mark.real_mamba
def test_mamba_block_is_causal():
    """Changing the future must not change the past (depthwise-conv padding/slicing bug)."""
    from models._mamba_compat import Mamba

    torch.manual_seed(3)
    blk = Mamba(d_model=16, d_state=4, d_conv=4, expand=2).eval()
    x = torch.randn(1, 48, 16)
    cut = 24
    with torch.inference_mode():
        y1 = blk(x)
        x2 = x.clone()
        x2[:, cut:] = torch.randn(1, x.shape[1] - cut, 16) * 5.0
        y2 = blk(x2)
    assert torch.equal(y1[:, :cut], y2[:, :cut]), (
        "output before the cut changed: the conv/scan is leaking future context "
        "(non-causal padding) — every onset would be anticipatory"
    )


@pytest.mark.real_mamba
def test_mamba_block_is_stateless_between_calls():
    """No hidden recurrent state may persist across forwards (C-2.5's cousin)."""
    from models._mamba_compat import Mamba

    torch.manual_seed(4)
    blk = Mamba(d_model=16, d_state=4, d_conv=4, expand=2).eval()
    x = torch.randn(1, 32, 16)
    with torch.inference_mode():
        a = blk(x)
        _ = blk(torch.randn(1, 32, 16) * 3)
        b = blk(x)
    assert torch.equal(a, b), "state leaked between forwards"


@pytest.mark.real_mamba
@known_bug(
    "REPRO-1",
    "inference is not reproducible: mc_dropout_eval() calls head.train() to keep Dropout live "
    "in eval mode, so two identical calls on the same file return different confidence "
    "(measured max |Δconf| = 0.0065 on an 8192-sample input). That moves CC#127, can flip "
    "needs_review at the threshold, and can change the quality tier between retries of the "
    "same paid request. Seed it per request or make the eval-time estimate deterministic.",
)
def test_inference_on_the_same_file_is_bitwise_reproducible(pipeline_config, make_wav, synth):
    from main import StemMidiModel

    path = make_wav(synth.note_sequence(total_seconds=0.5, sr=SR), name="rep.wav")
    torch.manual_seed(0)
    model = StemMidiModel(pipeline_config)
    model.eval()
    first = model.process_audio_file(path)
    second = model.process_audio_file(path)
    a = [e["confidence"] for e in first["midi"]["midi_events"]]
    b = [e["confidence"] for e in second["midi"]["midi_events"]]
    assert [e["note"] for e in first["midi"]["midi_events"]] == [
        e["note"] for e in second["midi"]["midi_events"]
    ], "the same file transcribed differently on a retry"
    assert a == b, (
        f"confidences differ between two runs of the same file "
        f"(max |Δ| = {max((abs(x - y) for x, y in zip(a, b, strict=False)), default=0):.5f})"
    )
    assert first["report"].quality_tier is second["report"].quality_tier


# ---------------------------------------------------------------------------
# 3. the tier that runs only where mamba_ssm exists
# ---------------------------------------------------------------------------


@pytest.mark.requires_mamba_ssm
@pytest.mark.cuda
def test_pure_torch_fallback_matches_mamba_ssm_within_tolerance():
    """The CPU reference must agree with the CUDA kernel, or CPU users get other results.

    Without this parity test the "pure-torch fallback" claim in
    models/_mamba_compat.py is unverified: a divergence in dt discretisation or the
    D-skip would change transcribed notes only on the hardware the product targets.
    """
    require("mamba_ssm", "mamba-ssm parity tier")
    if not torch.cuda.is_available():
        pytest.skip("unneeded on this host: mamba_ssm needs CUDA kernels to run")

    from mamba_ssm import Mamba as SsmMamba  # type: ignore

    from models._mamba_compat import Mamba as ShimMamba

    torch.manual_seed(0)
    d_model, L = 32, 64
    shim = ShimMamba(d_model=d_model, d_state=8, d_conv=4, expand=2).eval()
    ref = SsmMamba(d_model=d_model, d_state=8, d_conv=4, expand=2).to("cuda").eval()
    ref.load_state_dict({k: v.to("cuda") for k, v in shim.state_dict().items()})
    x = torch.randn(1, L, d_model, device="cuda")
    with torch.inference_mode():
        y_ref = ref(x)
        y_shim = shim(x.cpu().float())
    rel = (y_shim.cpu() - y_ref.cpu()).abs().max() / y_ref.abs().max().cpu()
    assert float(rel) < 5e-3, f"pure-torch vs mamba-ssm relative max diff = {float(rel):.2e}"


@pytest.mark.requires_mamba_ssm
def test_mamba_ssm_builds_the_separator_end_to_end(tmp_path):
    """Real mamba-ssm weights → real checkpoint → real forward (GPU/CUDA job tier)."""
    require("mamba_ssm", "mamba-ssm build tier")
    cfg = {
        "name": "parity",
        "audio": {"sample_rate": SR, "n_fft": 2048, "hop_length": HOP, "n_mels": 80},
        "separator": {"d_model": 64, "n_layers": 2, "d_state": 8, "d_conv": 4, "expand": 2},
        "transcriber": {
            "d_model": 64,
            "n_layers": 2,
            "d_state": 8,
            "onset_head_dim": 16,
            "pitch_vocab_size": 128,
            "velocity_bins": 128,
            "expression_heads": ["bend"],
            "onset_threshold": 0.5,
        },
        "quality_gates": {"min_confidence": 0.6},
        "backends": {"device": "cpu"},
    }
    import yaml

    from main import StemMidiModel, load_from_checkpoint

    model = StemMidiModel(cfg)
    ck = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict()}, ck)
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.dump(cfg), encoding="utf-8")
    loaded = load_from_checkpoint(str(ck), str(cp))
    x = torch.randn(1, 1, 8192) * 0.1
    with torch.inference_mode():
        a = model.separator(x)[0]
        b = loaded.separator(x)[0]
    assert torch.allclose(a, b, atol=1e-5), "mamba-ssm checkpoint round-trip changed the render"
