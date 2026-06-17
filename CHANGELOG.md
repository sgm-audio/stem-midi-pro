# Changelog

## v0.1.0 (2026-06-17)

### Bug Fixes
- `models/confidence_injector.py`: Add missing `import torch` (NameError at runtime)
- `tests/conftest.py`: Move `sessionmaker` import to module scope (NameError in test
  fixture)
- `models/losses.py`: Replace mutable default argument with `None` pattern
- `api.py`: Change bare `except:` to `except Exception:`

### Lint & Format
- `ruff format`: 21 Python files reformatted to project style
- `ruff check --fix`: 34 lint issues removed (unused imports, f-string placeholders,
  unused variables)

### New Modules
- `api/v1/`: Payments and users API routers (credit purchase, transaction history)
- `db/`: SQLAlchemy models (User, CreditTransaction) + database setup
- `services/`: Credit service, Square payment integration, Resend email service
- `tests/`: Pytest conftest + credit service + payment endpoint tests

### Documentation
- `AGENTS.md`: OpenCode agent instructions for the repository
- `TEST_PAYMENTS.md`: Curl commands for manual Square API testing
- `API_DOCUMENTATION.md`: Add payment/user endpoint documentation, credit schema

### Scaffold
- `frontend/`: Vite + React + Tailwind frontend scaffold
- `requirements.txt`: Add sqlalchemy, squareup, resend dependencies
- `.gitignore`: Exclude `__pycache__`, `*.pyc`, `*.db`, `.env`, `venv/`
