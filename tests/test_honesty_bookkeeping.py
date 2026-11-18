# SPDX-License-Identifier: Apache-2.0
"""The anti-rot rules that keep the skip/xfail story honest.

This suite may skip exactly one thing without failing: real ``mamba-ssm`` inference on
a host that does not have the package. Everything else that does not run has to be
*accounted for*, or the pass count is theatre. These tests are the accounting. They
read source text (never live pytest state) on purpose:

  * a live-state check would depend on which files the current selection imported, so
    ``pytest tests/test_honesty_bookkeeping.py`` alone would pass by having nothing to
    check — the failure mode this whole audit is about;
  * ``xfail_strict`` + ``@known_bug`` already make a *fixed* bug loud inside a run;
    what nothing else catches is the paperwork drifting: a marker with no write-up,
    and a write-up that still claims to be locked after the marker was deleted.

So: every ``@known_bug("ID")`` in ``tests/`` must appear as an OPEN entry in
``TODO.md`` 18.1, every 18.1 entry that claims "Locked by N tests" must have such a
marker, and no owned test file may contain an unaccounted escape hatch (a non-strict
xfail, an unexplained skip, ``pytest.skip()`` with no reason, or ``importorskip``).

If one of these rules is wrong for the project, change the rule here in one place —
do not delete the tests that enforce it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODO = ROOT / "TODO.md"
TESTS = Path(__file__).resolve().parent

OWNED = (
    "test_midi_semantics",
    "test_midi_target_roundtrip",
    "test_pipeline_integrity",
    "test_audio_edge_cases",
    "test_backend_contracts",
    "test_api_zip",
    "test_memory_ceiling",
    "test_quality_gate_adversarial",
    "test_checkpoint_load",
    "test_dataset_loaders",
)

_ID = r"[A-Z]{2,6}-\d{1,3}"


def _todo_181() -> list[str]:
    text = TODO.read_text(encoding="utf-8")
    if "## 18.1" not in text:
        raise AssertionError("TODO.md lost its 18.1 section — the audit's defect index")
    return text.split("## 18.1", 1)[1].split("## 18.2", 1)[0].split("\n")


def _ids_on(line: str) -> set[str]:
    return {i for tok in re.findall(rf"\*\*({_ID}(?:/{_ID})*)\*\*", line) for i in tok.split("/")}


def _marked_ids() -> dict[str, list[str]]:
    """Every @known_bug id used anywhere under tests/, mapped to its file:line."""
    found: dict[str, list[str]] = {}
    for path in sorted(TESTS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id == "known_bug") or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.setdefault(first.value, []).append(f"{path.name}:{node.lineno}")
    return found


def _open_entry_ids() -> set[str]:
    return {i for ln in _todo_181() if ln.startswith("- [ ]") for i in _ids_on(ln)}


def _closed_entry_ids() -> set[str]:
    return {i for ln in _todo_181() if ln.startswith("- [x]") for i in _ids_on(ln)}


def test_every_locked_defect_is_written_up_as_open_in_the_todo() -> None:
    """An xfail reason scrolls off a CI log; TODO.md 18.1 is the durable record."""
    marked = _marked_ids()
    # Floor, not an exact count: the audit opened 65 locks and the remediation
    # has been retiring them, so this number falls every time a defect is fixed.
    # It exists to catch a BROKEN SCAN (a refactor moving the markers out of the
    # files `_marked_ids` walks would otherwise silently pass with zero), so it
    # only needs to be well below the real count.
    assert (
        len(marked) >= 10
    ), f"marker scan found only {sorted(marked)} — the scan broke, not the suite"
    missing = sorted(set(marked) - _open_entry_ids())
    assert not missing, (
        f"{missing} lock tests with @known_bug but are not open entries in TODO.md 18.1. "
        "Write them up (id, severity, file:line, the measured number) so the claim is "
        "reviewable — or, if the defect is fixed, delete the marker (strict xfail already "
        "made that fix loud, so this is the paperwork half)"
    )


def test_no_open_todo_entry_claims_a_guard_that_does_not_exist() -> None:
    """The reverse direction: "Locked by N tests" must be true at this commit.

    When a fix lands, the marker goes and the entry should be checked off with the PR
    number. This catches the other half of the drift — an entry that still advertises
    coverage nobody has any more (which is how a fixed bug quietly comes back).
    """
    marked = _marked_ids()
    stale: list[str] = []
    for line in _todo_181():
        if not line.startswith("- [ ]"):
            continue
        ids = _ids_on(line)
        if re.search(r"[Ll]ocked by \d+ test", line) and not (ids & set(marked)):
            stale.append(f"{'/'.join(sorted(ids))}: {line.strip()[:80]}")
    assert not stale, (
        "TODO.md 18.1 says these are locked by tests, but no @known_bug marker uses the id "
        "(the guard was deleted or never written; if the fix landed, mark the entry [x] and "
        "name the PR):\n" + "\n".join(stale)
    )


def test_fixed_entries_keep_their_id_rather_than_being_deleted() -> None:
    """Resolved entries must stay in 18.1 as ``- [x]``, not vanish.

    The id is the join key between a test docstring ("regression guard for MIDI-1"),
    a commit message and the defect index. Deleting the row erases that join and makes
    the regression guards unsearchable.
    """
    closed = _closed_entry_ids()
    if not closed:
        raise AssertionError(
            "TODO.md 18.1 has no closed entries at all — a suite this old has certainly fixed "
            "something; keep the ids and mark them [x] instead of deleting the rows"
        )
    # every closed id must still be referenced somewhere (guard prose or a marker), so the
    # row is a breadcrumb and not a tombstone nobody can follow
    body = "\n".join(p.read_text(encoding="utf-8") for p in TESTS.glob("test_*.py"))
    orphans = sorted(i for i in closed if i not in body)
    assert not orphans, (
        f"{orphans} are marked fixed in TODO.md but no test mentions the id — a fixed defect "
        "needs a pass-today regression guard that names it, or it will regress silently"
    )


def test_no_escape_hatches_in_the_tests_i_own() -> None:
    """The new suite's own no-loophole rules, enforced on itself.

    All four are legal pytest syntax and all four can hide a real failure:
      * ``@pytest.mark.xfail`` without ``strict=True`` — a test that can never fail;
      * ``@pytest.mark.skip`` without ``reason=`` — nobody knows what it is waiting for;
      * ``pytest.skip()`` with no argument — invisible in a CI log;
      * ``pytest.importorskip("torch")`` — a *declared* dependency going missing turns
        into a permanent green skip, which is the exact bug this audit was asked about.

    Capability skips that do carry a reason (``pytest.skip("unneeded on this host:
    mamba_ssm needs CUDA kernels")``) stay legal: the rule is accountability, not the
    absence of skips — ``--fail-on-skip`` is what surfaces them in CI.
    """
    hits: list[str] = []
    checked = 0
    for name in OWNED:
        path = TESTS / f"{name}.py"
        if not path.exists():
            continue
        checked += 1
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted(node.func)
            if dotted.endswith("importorskip"):
                hits.append(f"{path.name}:{node.lineno}: importorskip on a declared dependency")
            elif dotted.endswith("mark.xfail") and not _has_kw(node, "strict"):
                hits.append(f"{path.name}:{node.lineno}: xfail without strict=True")
            elif dotted.endswith("mark.skip") and not _has_kw(node, "reason"):
                hits.append(f"{path.name}:{node.lineno}: unconditional skip without reason=")
            elif dotted.endswith(".skip") and not node.args and not node.keywords:
                hits.append(f"{path.name}:{node.lineno}: pytest.skip() with no reason")
    assert (
        checked >= len(OWNED) - 2
    ), f"only {checked}/{len(OWNED)} owned files found — list drifted"
    assert not hits, "escape hatch in a file this audit owns (see the docstring):\n" + "\n".join(
        hits
    )


def _dotted(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _has_kw(node: ast.Call, name: str) -> bool:
    for kw in node.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
            return True
    # xfail(strict=True) is also spelled via pytest.mark.xfail(...)(...) chains
    return False
