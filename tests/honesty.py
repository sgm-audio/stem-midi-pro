# SPDX-License-Identifier: Apache-2.0
"""Suite-honesty layer (T-8.1.3 hardening).

The problem this module solves: an AI-generated audio suite stays green by
*skipping* its way around the parts that need real dependencies. `pytest.skip`
at import time is indistinguishable from "the real model path works", so a
suite can report "96 passed, 2 skipped" while the two skipped tests are exactly
the ones that would catch a checkpoint-format bug.

Rules enforced here (wired up in tests/conftest.py):

1. ``require(name, purpose)`` — a dependency gate. On a ``--strict-deps`` run
   (what CI uses) a missing dependency is an ERROR, not a skip. On a
   convenience run it is still a skip, but with a machine-readable
   ``optional-dep:`` prefix so the honesty report can count it.
2. Tiers are declared with markers (``real_mamba`` / ``requires_mamba_ssm``)
   and the report in conftest fails the session when a tier that must run did
   not run.
3. Skips must be registered through ``skip_optional`` / ``requires_*``
   fixtures so the allowlist in ``ALLOWED_SKIP_REASONS`` is the single source
   of truth for "this test is allowed to not run here".
"""

from __future__ import annotations

import importlib

# --- mode, set from CLI options by tests/conftest.py::pytest_configure ------
STRICT_DEPS = False
FAIL_ON_SKIP = False
REQUIRE_MAMBA_SSM = False


def set_mode(*, strict_deps: bool, fail_on_skip: bool, require_mamba_ssm: bool) -> None:
    global STRICT_DEPS, FAIL_ON_SKIP, REQUIRE_MAMBA_SSM
    STRICT_DEPS = strict_deps
    FAIL_ON_SKIP = fail_on_skip
    REQUIRE_MAMBA_SSM = require_mamba_ssm


#: Skips that are legitimate on a given host. A skip whose reason does not
#: start with one of these prefixes is a suite-honesty failure under
#: --fail-on-skip. Keep this list short: every entry is a hole in coverage.
ALLOWED_SKIP_REASONS = (
    "optional-dep:",  # gated by require() on a non-strict run
    "codec-unavailable:",  # soundfile lacks a FLAC/MP3 encoder
    "artifact-unavailable:",  # needs a trained checkpoint / real dataset (nightly tier)
)

_MAMBA_SSM_CHECK = None


def mamba_ssm_available() -> bool:
    """True when the CUDA-built ``mamba_ssm`` package can actually be imported."""
    global _MAMBA_SSM_CHECK
    if _MAMBA_SSM_CHECK is None:
        try:
            importlib.import_module("mamba_ssm")
        except Exception:  # ImportError, or a broken/incomplete CUDA build
            _MAMBA_SSM_CHECK = False
        else:
            _MAMBA_SSM_CHECK = True
    return _MAMBA_SSM_CHECK


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def require(name: str, purpose: str = "", *, module_level: bool = False):
    """Import ``name`` or fail (strict) / allowlist-skip (permissive).

    Use this instead of ``pytest.importorskip`` for anything that is a real
    project dependency (torch, numpy, mido, soundfile, fastapi): those are in
    requirements.txt / requirements-dev.txt, so a missing one means the
    environment is broken and the test MUST NOT be reported as a pass-by-skip.

    ``module_level=True`` is required when calling from a module body (pytest
    only honours a skip during collection with that flag set).
    """
    import pytest

    try:
        return importlib.import_module(name)
    except ImportError as exc:
        detail = f"could not import {name!r}"
        if purpose:
            detail = f"{detail} (needed for {purpose})"
        if STRICT_DEPS:
            pytest.fail(
                f"BROKEN TEST ENVIRONMENT: {detail}. This dependency is declared in "
                f"requirements*.txt, so skipping the test silently would hide real "
                f"failures. Install it, or drop --strict-deps for a local convenience run."
            )
        pytest.skip(
            f"optional-dep: {detail} — install it to enable this tier ({exc.__class__.__name__})",
            allow_module_level=module_level,
        )
    raise AssertionError("unreachable")


def skip_optional(reason: str) -> None:
    """Skip with a reason that the honesty report recognises."""
    import pytest

    assert reason.startswith(ALLOWED_SKIP_REASONS), (
        f"skip reason {reason!r} is not in ALLOWED_SKIP_REASONS; unregistered skips "
        f"are a coverage hole — use require() or 'artifact-unavailable: ...'"
    )
    pytest.skip(reason)


def require_artifact(when: bool, what: str) -> None:
    """Gate on a build artifact (trained checkpoint, real dataset), never on a pip dep."""
    if not when:
        skip_optional(f"artifact-unavailable: {what}")


# ---------------------------------------------------------------------------
# Known-bug markers
# ---------------------------------------------------------------------------

#: bug_id -> details. Surfaced in the capability report so the bug backlog is
#: visible on every run instead of buried in skip lines.
KNOWN_BUGS: dict[str, list[str]] = {}


def known_bug(bug_id: str, detail: str):
    """``xfail(strict=True)`` for a defect the suite has proven but not yet fixed.

    strict=True is non-negotiable: when somebody fixes the bug the test flips to
    XPASS and the run fails, forcing the marker to be deleted. That is the only
    way a "known bug" test cannot rot into a permanent green lie.

    ``detail`` must state what the test asserts and what the code does instead,
    because the reason string IS the bug report the CI log shows.
    """
    import pytest

    reason = f"TODO {bug_id}: {detail}"
    KNOWN_BUGS.setdefault(bug_id, []).append(detail)
    return pytest.mark.xfail(strict=True, reason=reason)
