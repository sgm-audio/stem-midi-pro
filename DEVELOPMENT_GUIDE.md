# Development Guide for Stem+MIDI Pro

Guide for contributors. Target platform: **CPU-only, Python 3.13**. License: Apache-2.0 — every source file carries an `# SPDX-License-Identifier: Apache-2.0` header; keep it when editing files and add it to new ones.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Codebase Overview](#codebase-overview)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Extending the Model](#extending-the-model)
7. [Documentation Standards](#documentation-standards)
8. [Code Style](#code-style)
9. [Contributing](#contributing)
10. [Useful Commands](#useful-commands)

## Getting Started

Prerequisites:
- Python **3.13** (pinned — see TODO Section 4.3)
- Git
- No GPU required (CPU-only target)

```bash
git clone <repository-url>
cd stem-midi-pro        # this repo root; all main source files live here
```

Note: there is **no `stem_midi_pro/` package subdirectory** — `main.py`, `api.py`, `train.py` are at the repo root. (`verify_structure.py` predates this layout; see `AGENTS.md` for known quirks.)

## Codebase Overview

### Core components (repo root)
- `main.py` — `StemMidiModel` (currently a NeMo ModelPT subclass; planned migration to a plain `torch.nn.Module`, TODO RF-3.1.7) and the CLI entry point.
- `api.py` — FastAPI service (`POST /process`, `/health`, `/model-info`, `/render-template`).
- `demo.py` — synthetic-audio smoke test.
- `train.py` — training script (PyTorch Lightning today; custom loop planned, TODO TR-12.3).
- `check_syntax.py` — AST syntax check across the repo.
- `verify_structure.py` — structural sanity check (has known path quirks, see `AGENTS.md`).

### Subpackages
- `models/` — `mamba_separator.py`, `mamba_transcriber.py`, `confidence_injector.py`, `losses.py`
- `data/` — `datasets.py` (`AudioDataset`, `Slakh2100YourMT3Dataset`, `MUSDB18HQDataset`, `get_data_loaders`)
- `utils/` — `quality_gates.py`, `template_engine.py`
- `configs/` — `model_config.yaml`
- `user_content/` — the 7 user-facing text templates rendered by `utils/template_engine.py`
- `tests/` — pytest suite (several tests currently skipped pending TODO Section 8)
- `research/` — Mamba-3 experiments, separate from the production path (as-is, `research/README.md`)
- `docs/` — ADRs and operations docs

### Data flow
1. Audio input → validation → load (`soundfile`, mono, float32)
2. `MambaSeparator` → guitar/bass stems
3. `MambaTranscriber` → onset/pitch/velocity/expression/confidence logits
4. `ConfidenceInjector` → MIDI events with CC#127 confidence
5. `utils/quality_gates.route_by_quality` → studio/draft/complex tier
6. `api.py` → ZIP (stems + MIDI + `processing_report.json`)

Before changing pipeline code, read the open bug list in TODO.md Section 2 (C-2.1 … C-2.12) — several components have known defects you should not build assumptions on top of.

## Development Setup

```bash
python3.13 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest black flake8 isort mypy   # dev tools
```

Environment variables for local runs:

```bash
export MODEL_CONFIG_PATH=configs/model_config.yaml
export MODEL_CHECKPOINT_PATH=path/to/checkpoint   # optional
```

The dataset loaders fall back to synthetic data automatically, so no real dataset is needed to develop. See [SETUP.md](SETUP.md) for the full install guide.

## Making Changes

1. Branch: `git checkout -b feature/your-feature-name`
2. Conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`
3. Keep the SPDX license header on every `.py` file.
4. If you touch anything documented in `AGENTS.md`, update `AGENTS.md` too.
5. Update user-facing docs when behavior changes.

### Models
- Respect existing tensor shapes: audio `(B, 1, T)`, spectrogram `(B, F, T)`, mel `(B, n_mels, T)`.
- Architecture hyperparameters live in `configs/model_config.yaml`; read from config rather than hardcoding.
- Remember the CPU target: no CUDA-only ops, no `cuda-` impl flags, keep memory under ~4 GB.

### API
- Keep `/health` and the OpenAPI docs unauthenticated.
- Update [API_DOCUMENTATION.md](API_DOCUMENTATION.md) when endpoints change.

## Testing

```bash
pytest                          # full suite in tests/
pytest -m "not cuda"            # CPU-only (default CI path)
pytest --cov=. tests/           # with coverage
```

- Tests live in `tests/` and use fixtures from `tests/conftest.py`.
- Some tests are marked `skip` pending the bug fixes in TODO Section 2 — un-skip them when you land the corresponding fix (TODO T-8.2.x).
- Coverage gate target: `--cov-fail-under=70` (TODO T-8.4.1).

## Extending the Model

### Adding another stem (e.g. drums)
1. Add a mask head in `models/mamba_separator.py` and update `forward`.
2. Update stem packaging in `api.py:create_response_zip`.
3. Update config and quality gates.
4. Note that TODO ARCH-11.12.1 already tracks this — coordinate there first.

### Changing architecture dimensions
Edit `configs/model_config.yaml` (`separator.d_model`, `n_layer`, etc.). For CPU deployment, use the smaller CPU preset planned in TODO CFG-9.1.3 (`d_model=256`, `n_layer=6`).

## Documentation Standards

- Google-style docstrings with tensor-shape annotations `(B, C, T)` (TODO DOC-10.3.1 covers backfilling these).
- Markdown: `#`/`##`/`###` hierarchy, fenced code blocks, relative links.
- Mark aspirational features with `[PLANNED]` tags, as in `ARCHITECTURE.md` and `SUMMARY.md`.

## Code Style

- Python: PEP 8, 4-space indent, 100-char lines, type hints, `is None` checks.
- Import order: stdlib → third-party → local.
- Linters: `flake8 .`, `black .`, `isort .`, `mypy .` (also what current CI runs). TODO BLD-9.2.2 plans a switch to `ruff`.

## Contributing

1. Fork, branch from `main`, make changes, ensure `pytest` passes.
2. Update docs and tests alongside code.
3. Open a PR with a clear description linked to the relevant TODO.md item ID.
4. Contributions are licensed under the project's Apache License 2.0.

## Useful Commands

```bash
python demo.py                                   # smoke test
python main.py --audio song.wav                  # CLI inference
uvicorn api:app --reload                         # dev API server
pytest -m "not cuda"                             # tests (CPU)
flake8 . && black --check . && isort --check . && mypy .   # lint (CI parity)
python check_syntax.py                           # AST syntax check
python verify_structure.py                       # structural check (see AGENTS.md quirks)
```

**Make targets:** a `Makefile` with `install`, `install-dev`, `lint`, `format`, `test`, `test-cpu`, `demo`, `serve`, `docker-build`, `docker-run` targets is planned (TODO BLD-9.2.5) but does not exist yet — use the commands above.

**CI:** `.github/workflows/ci.yml` runs a structure/syntax job plus lint, and (per the planned rewrite in TODO BLD-9.2.6) a `pytest -m "not cuda"` job on Python 3.13. A `torch.jit.trace` export example is being reworked for the post-NeMo model layout (TODO RF-3.1.7) and is intentionally not documented here until it works on the CPU target.

---
*Development Guide — updated for the CPU-only, Python 3.13 target. See [TODO.md](TODO.md) for the live work queue.*
