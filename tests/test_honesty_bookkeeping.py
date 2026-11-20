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
    "test_numerics",
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


def _scan_self_check() -> bool:
    """Prove the marker scan still works, without needing an open defect.

    Writes a temporary test file containing a `known_bug("SENTINEL-0", ...)`
    call into the tests/ directory tree that `_marked_ids` walks, runs the
    scanner, and removes it. If the scan is intact it must see the sentinel.
    """

    target = Path(__file__).resolve().parent / "_scan_selfcheck_tmp.py"
    try:
        target.write_text(
            "# SPDX-License-Identifier: Apache-2.0\n"
            "def test_sentinel():\n"
            "    known_bug = lambda *a, **k: (lambda f: f)  # noqa: E731\n"
            '    known_bug("SENTINEL-0", "self-check")\n',
            encoding="utf-8",
        )
        return "SENTINEL-0" in _marked_ids()
    finally:
        target.unlink(missing_ok=True)


def _open_entry_ids() -> set[str]:
    return {i for ln in _todo_181() if ln.startswith("- [ ]") for i in _ids_on(ln)}


def _closed_entry_ids() -> set[str]:
    return {i for ln in _todo_181() if ln.startswith("- [x]") for i in _ids_on(ln)}


def test_every_locked_defect_is_written_up_as_open_in_the_todo() -> None:
    """An xfail reason scrolls off a CI log; TODO.md 18.1 is the durable record."""
    marked = _marked_ids()
    # No floor on the marker count: an empty backlog is a success, not a broken
    # scan. A floor that has to be lowered every time something is fixed measures
    # the backlog, not the scan. The reverse checks below (TODO -> markers) catch
    # an empty TODO with an unexplained zero.
    #
    # But they cannot catch the scan going blind while the TODO also happens to be
    # empty -- the state this repo is now in -- because there is nothing to
    # disagree with. So the scan proves it still works directly, by planting a
    # sentinel marker in a file it walks and requiring that it be seen. That is
    # the one assertion here that does not depend on any defect being open.
    assert _scan_self_check(), (
        "_marked_ids() no longer finds a known_bug marker placed in a scanned file; "
        "the scan is broken, so 'no open defects' would be a false negative"
    )
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


def test_every_open_entry_is_locked_by_a_test_or_says_why_not() -> None:
    """An open defect needs a test, an explicit "no test lock", or it is a rumour.

    The weaker rule above only fires when an entry *claims* "Locked by N tests". That is
    how PR #15 could delete the MIDI-5 and CKPT-1 markers, fix both defects, and leave
    the entries sitting open in the index with nothing behind them: a reader cannot tell
    "unfixed and untested" from "fixed and forgotten", and the next regression is free.

    So each open entry must either have a `@known_bug` marker under one of its ids, or
    carry the literal text `No test lock:` explaining why a test is not the right
    instrument (analytic findings, e.g. the memory model, qualify).
    """
    marked = set(_marked_ids())
    unaccounted: list[str] = []
    for line in _todo_181():
        if not line.startswith("- [ ]"):
            continue
        ids = _ids_on(line)
        if not ids or (ids & marked) or "No test lock:" in line:
            continue
        unaccounted.append(f"{'/'.join(sorted(ids))}: {line.strip()[:90]}")
    assert not unaccounted, (
        "these open TODO.md 18.1 entries have no @known_bug marker and no 'No test lock:' "
        "note — either lock the defect, land the fix and check the entry off, or say why a "
        "test cannot carry it:\n" + "\n".join(unaccounted)
    )


def test_guards_cited_in_the_todo_still_exist() -> None:
    """``tests/test_x.py::test_y`` in 18.1 must resolve here and now.

    Resolved entries are kept on purpose (the id is the join key), and they name the
    guard that replaced the marker. Guards get renamed by the next refactor, and then the
    breadcrumb points at nothing — which reads like "this was fixed and tested" forever.
    Checked by name only, so a line-number or parametrisation edit cannot break it.
    """
    known: set[str] = set()
    for path in TESTS.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                known.add(node.name)
    cited = set(re.findall(r"::(test_[a-z0-9_]+)", "\n".join(_todo_181())))
    missing = sorted(cited - known)
    assert not missing, (
        f"TODO.md 18.1 cites guards that no longer exist: {missing}. Either the test was "
        "renamed (update the citation) or the regression coverage was deleted (put it back "
        "— a fixed defect without its guard is a defect on loan)"
    )


# `importorskip` on a *declared* dependency is the audit's founding defect: torch, numpy,
# soundfile, mido, pyyaml, fastapi, httpx, librosa are all in requirements*.txt, so a host
# missing one has a broken environment and must be told, not handed a green skip. The ten
# OWNED files above are held to zero. The rest of the suite still carries
# _LEGACY_IMPORTORSKIP sites, because converting 31 files in one PR would bury the fixes
# it ships beside - so the debt is capped instead, and the cap only moves down. The count
# is taken from the syntax tree, not from grep: a docstring that *mentions* importorskip
# (this file does) must not be part of the measurement.
_LEGACY_IMPORTORSKIP = 65  # AST-counted, 2026-11-20, 29 files
_DECLARED = (
    "torch",
    "numpy",
    "soundfile",
    "librosa",
    "mido",
    "yaml",
    "fastapi",
    "httpx",
)


def test_importorskip_on_declared_deps_does_not_grow() -> None:
    """A ratchet over the files 18.1's audit has not converted yet.

    `mamba_ssm` is genuinely optional (it needs CUDA to build) and is not counted; every
    name in `_DECLARED` is in requirements*.txt, so gating on it converts "the host is
    broken" into "these N tests were skipped" (TODO.md FAKE-2). The fix is
    `tests.honesty.require(name,
    purpose, module_level=True)`, which fails under `--strict-deps` and skip-with-reasons
    otherwise. 65 sites across 29 files remain (test_api_validation.py and
    test_review_fixes.py carry 7 each). Lower `_LEGACY_IMPORTORSKIP` in the same commit as
    each conversion; a count below the cap is a leftover, not a licence to stop.
    """
    found: list[str] = []
    for path in sorted(TESTS.glob("*.py")):
        if path.stem in OWNED:
            continue  # zero tolerance there, checked by test_no_escape_hatches_in_the_tests_i_own
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not _dotted(node.func).endswith("importorskip"):
                continue
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and first.value in _DECLARED:
                found.append(f"{path.name}:{node.lineno}: importorskip({first.value!r})")
    assert len(found) <= _LEGACY_IMPORTORSKIP, (
        f"{len(found)} importorskip sites on declared dependencies, cap is "
        f"{_LEGACY_IMPORTORSKIP}. Use tests.honesty.require(...) instead - a declared dep "
        "going missing has to fail, not skip:\n" + "\n".join(found[:12])
    )


def test_181_lists_each_defect_id_exactly_once() -> None:
    """Regression guard for DOC-4: a duplicated row forks the index, and merging branches
    is how it happens.

    Two branches that each append to 18.1 concatenate cleanly: git sees no conflict when
    the two copies of an entry land on different lines, so the file ends up with ``MIDI-5``
    twice - which is what `main` @ `abd261f` carried, both rows `- [x]`, one justifying
    itself with a pointer to the remediation plan and the other with "upstream did it in
    `api.py`". A reader cannot tell which is authoritative, and the id stops being a join
    key: a test docstring saying "regression guard for MIDI-5", a commit message and the
    index are supposed to agree through exactly one row.

    Round 3 removed four such duplicates (`MIDI-4`, `MIDI-7`, `EXPR-1`, `MIDI-5`) by hand
    with no rule behind the repair, so the class came back with the next merge. This is
    that rule. The two idioms to fix a failure here: keep ONE row per id, and when the two
    halves disagree about *why* it is closed, merge the justifications rather than picking
    a side - the measured one wins if only one of them has a number in it.
    """
    seen: dict[str, list[str]] = {}
    for line in _todo_181():
        head = re.match(r"^- \[[ x]\] \*\*([^*]+)\*\*", line.strip())
        if head is None:
            continue
        for tok in head.group(1).split("/"):
            if re.fullmatch(_ID, tok):
                seen.setdefault(tok, []).append(line.strip())
    forked = {i: rows for i, rows in seen.items() if len(rows) > 1}
    assert not forked, (
        f"TODO.md 18.1 lists {sorted(forked)} more than once - the id is the join key "
        "between a guard's docstring, the commit that fixed it and this index, so a "
        "duplicate makes them disagree:\n"
        + "\n".join(f"  {i}: {r[:120]}" for i, rows in sorted(forked.items()) for r in rows)
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
