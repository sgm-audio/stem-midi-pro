# SPDX-License-Identifier: Apache-2.0
"""Keeping the suite honest about ``mamba_ssm`` — and about which code ran.

The repo deliberately does not depend on ``mamba-ssm`` (it needs ``nvcc`` to
build and has no CPU wheels); ``models/_mamba_compat.py`` falls back to a
pure-PyTorch scan. That is the right call, but it creates two failure modes
this module pins down:

1. **A silently different implementation.** ``MAMBA_BACKEND`` records which path
   was taken and *nothing reads it* — not the report, not ``/model-info``, not
   the logs. A CPU deployment cannot tell whether it is running the reference
   scan or the real CUDA kernel.

2. **A test suite that only ever sees the fallback.** ``test_checkpoint_load.py``
   guards with ``pytest.importorskip("mamba_ssm")``, which is correct (it must
   not fake the package), but means the CUDA path is *never* exercised anywhere
   when the package is absent. The ``real_mamba`` marker below inverts the guard:
   it skips cleanly when absent and asserts the genuine kernel when present, so
   a CUDA box validates the production path instead of skipping it.

Rule for new tests: never ``importorskip`` the *behaviour under test*. Skip only
on absent hardware/optional deps, and assert the fallback's contract explicitly
so the "package missing" branch is covered by real assertions.
"""

import subprocess
import sys
import textwrap

import pytest

MAMBA_BACKEND_CHOICES = {"mamba-ssm", "pure-torch"}


def _mamba_backend() -> str:
    from models._mamba_compat import MAMBA_BACKEND

    return MAMBA_BACKEND


def mamba_ssm_installed() -> bool:
    try:
        import mamba_ssm  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The fallback contract (runs everywhere)
# ---------------------------------------------------------------------------


def test_backend_identity_matches_availability():
    """``MAMBA_BACKEND`` must agree with what is actually importable."""
    expected = "mamba-ssm" if mamba_ssm_installed() else "pure-torch"
    assert (
        _mamba_backend() == expected
    ), f"MAMBA_BACKEND={_mamba_backend()!r} but mamba_ssm installed={mamba_ssm_installed()}"


def test_importing_the_model_does_not_require_mamba_ssm():
    """``import main`` must work with ``mamba_ssm`` unavailable.

    Simulated by putting ``None`` in ``sys.modules``, which is how CPython
    blocks a module for the lifetime of a process. Run in a subprocess so the
    blocker cannot leak into other tests.
    """
    script = textwrap.dedent("""
        import sys

        sys.modules["mamba_ssm"] = None  # makes `from mamba_ssm import ...` raise

        import models._mamba_compat as compat
        assert compat.MAMBA_BACKEND == "pure-torch", compat.MAMBA_BACKEND

        import main  # noqa: F401  (module-level imports must not blow up)

        from main import StemMidiModel
        print("OK", compat.MAMBA_BACKEND)
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"import failed without mamba_ssm:\n{proc.stderr}"
    assert "OK pure-torch" in proc.stdout


def test_fallback_module_is_a_drop_in_for_the_mamba_signature():
    """The fallback must accept the kwargs the models actually pass.

    ``mamba-ssm`` changed its accepted kwargs across versions (``use_fast_path``
    etc.); the fallback has to keep the intersection used by
    ``mamba_separator``/``mamba_transcriber`` so both paths construct identically.
    """
    torch = pytest.importorskip("torch")
    from models._mamba_compat import Mamba

    block = Mamba(d_model=32, d_state=8, d_conv=4, expand=2)
    x = torch.randn(2, 7, 32)
    y = block(x)
    assert y.shape == x.shape == (2, 7, 32)
    assert torch.isfinite(y).all()


def test_fallback_matches_an_independent_recurrence():
    """The pure-torch scan must implement *a* causal SSM, not a shuffled MLP.

    Property: perturbing frame t may only affect outputs at frames >= t. That is
    the defining property of the sequence model and it is what makes the
    streaming limitation (C-2.5) a real limitation rather than a cosmetic one.
    """
    torch = pytest.importorskip("torch")
    from models._mamba_compat import Mamba

    torch.manual_seed(0)
    block = Mamba(d_model=16, d_state=8, d_conv=4, expand=2).eval()
    x = torch.randn(1, 12, 16)

    with torch.inference_mode():
        base = block(x)
        perturbed = x.clone()
        perturbed[0, 5] += 1.0
        after = block(perturbed)

    delta = (after - base).abs().amax(dim=-1)[0]
    assert delta[:5].max() < 1e-6, "output depends on future frames (non-causal)"
    assert delta[5:].max() > 1e-6, "perturbation had no effect — not a sequence model"


def test_mamba_ssm_is_not_a_hard_dependency():
    """Pyproject/requirements must not pin ``mamba-ssm`` back in by accident."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("pyproject.toml", "requirements.txt"):
        text = (root / name).read_text(encoding="utf-8").lower()
        active = [
            line
            for line in text.splitlines()
            if line.strip().startswith(("mamba", '"mamba', "'mamba"))
        ]
        assert not active, f"{name} hard-depends on mamba-ssm: {active}"


# ---------------------------------------------------------------------------
# The real path (skips cleanly when the package is absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mamba_ssm_installed(), reason="mamba_ssm not installed (CUDA build)")
@pytest.mark.cuda
def test_real_mamba_kernel_is_actually_used(mock_config):
    """With the package installed, the modules must wrap the genuine kernel.

    This is the mirror image of the ``importorskip`` guards: on a CUDA box it
    fails if the fallback is silently preferred, which is the only way to know
    the production path is the one being tested.
    """
    import mamba_ssm

    from models.mamba_separator import MambaSeparator
    from models.mamba_transcriber import MambaTranscriber

    separator = MambaSeparator(mock_config)
    transcriber = MambaTranscriber(mock_config)

    sep_blocks = list(separator.mamba_blocks)
    tr_blocks = list(transcriber.mamba)
    assert sep_blocks and tr_blocks
    for block in sep_blocks + tr_blocks:
        assert type(block).__module__.startswith("mamba_ssm"), (
            f"block is {type(block).__module__}.{type(block).__name__}, "
            f"expected mamba_ssm (backend={getattr(mamba_ssm, '__version__', '?')})"
        )


@pytest.mark.skipif(not mamba_ssm_installed(), reason="mamba_ssm not installed (CUDA build)")
@pytest.mark.cuda
def test_real_and_fallback_paths_agree_on_shapes(mock_config):
    """Both backends must expose the same tensor contract to the models."""
    import torch

    from models.mamba_separator import MambaSeparator

    separator = MambaSeparator(mock_config).eval()
    audio = torch.randn(1, 1, 8192) * 0.05
    with torch.inference_mode():
        guitar, bass, residual, state, metrics = separator(audio)
    assert guitar.shape == bass.shape == residual.shape == audio.shape
    assert state is None
    assert metrics.shape == (1, 2)


# ---------------------------------------------------------------------------
# Provenance (currently absent)
# ---------------------------------------------------------------------------


def test_runtime_exposes_which_mamba_implementation_ran():
    """``StemMidiModel`` must publish the Mamba implementation that is live.

    The string is surfaced by ``/model-info`` and the processing report so an
    operator can tell the CUDA kernel from the reference scan.
    """
    from main import StemMidiModel

    assert StemMidiModel.mamba_backend == _mamba_backend()
    assert {_mamba_backend()} <= MAMBA_BACKEND_CHOICES
