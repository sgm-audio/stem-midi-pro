# Stem+MIDI Pro — Agent instructions

Single-package Python audio AI service (Mamba-SSM separation + MIDI transcription). No build system — plain Python with pip.

## Entrypoints

| File | Role | How to run |
|------|------|-----------|
| `demo.py` | Quick smoke test with synthetic audio | `python demo.py` |
| `main.py` | NeMo ModelPT wrapper + CLI inference | `python main.py --config configs/model_config.yaml --audio <file>` |
| `api.py` | FastAPI server (dev) | `python api.py` or `uvicorn api:app --reload` |
| `train.py` | Training with Canadian datasets | `python train.py --config ... --data-config ... --output-dir ./outputs` |

## Dev commands

```bash
pip install -r requirements.txt
pytest                                    # all tests (only 1 test file)
pytest tests/test_payments.py -k "TestName"  # focused
pytest --cov=stem_midi_pro                # with coverage
flake8 stem_midi_pro                      # lint (also black, isort, mypy)
python check_syntax.py                    # AST-level syntax check on all .py files
```

## Testing quirks

- **Only one test file**: `tests/test_payments.py` (credit service + payment API endpoints).
- **conftest.py** is critical — it sets env vars (`DATABASE_URL=sqlite:///:memory:`, `SQUARE_ACCESS_TOKEN=test_token`, `RESEND_API_KEY=test_key`), creates a **separate** FastAPI test app (not `api.py`'s app), and provides fixtures: `db_engine`, `db_session`, `test_user`, `client`, `square_mock`.
- Tests use SQLite in-memory DB with `StaticPool`. Square client is monkeypatched.
- `sys.path.insert(0, ...)` in conftest — tests must run from repo root.
- No model/audio tests exist. Adding tests for `models/` or `api.py` endpoints would need synthetic audio fixtures.

## Architecture

- `main.py` `StemMidiModel(ModelPT)`: separation → transcription → confidence injection → quality routing. All model components in `models/`.
- `api.py` starts by loading the model (`load_model()` on startup). Uses `X-User-ID` header for user auth (no JWT — dev-mode simple). CORS allows all origins.
- Three quality tiers (hardcoded in `utils/quality_gates.py`): Studio (confidence ≥0.85, SI-SDR ≥20dB), Draft (≥0.70), Complex (<0.70).
- Credit system: 3 packages (`services/credit_service.py`). Job cost = `floor(duration_sec / 600) + 1` beyond 30min.
- Two YAML config files: `configs/model_config.yaml` (model architecture) and `example_data_config.yaml` (dataset paths). Both loaded separately.
- DB: SQLAlchemy with SQLite by default (`stem_midi_pro.db`), PostgreSQL-friendly. `init_db()` creates tables.

## Things agents often miss

- **Format with `black` and check with `flake8`** before committing (stated in DEVELOPMENT_GUIDE.md). No pre-commit hook is installed.
- **`check_syntax.py`** does AST-level syntax validation across all `.py` files — quick sanity check.
- **`demo.py`** uses synthetic data only, never touches real audio I/O. Good for CI smoke tests.
- **`api.py`** `/process` endpoint returns a ZIP with stems + MIDI + JSON report. File validation rejects >10min or non-44.1/48kHz audio.
- **Square webhook** at `POST /api/v1/payments/square/webhook` — requires `x-square-hmacsha256-signature` header for production, not enforced in dev.
- Single commit in history — experimental/early stage. No CI, no release process documented beyond Docker build.
- **No `.gitignore`**, `.dockerignore`, or `Makefile` — worth noting if you're adding them.

## When to edit source code

Do NOT edit source files without explicit user approval. This repo's `AGENTS.md` instructions reference `edit: deny` for source code.

## Key docs

- `ARCHITECTURE.md` — component wiring, data flow, streaming
- `DEVELOPMENT_GUIDE.md` — setup, testing, style guide
- `API_DOCUMENTATION.md` — full endpoint reference
- `TEST_PAYMENTS.md` — curl commands for manual Square API testing
