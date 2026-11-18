# Stem+MIDI Pro — convenience targets
# Python 3.13 / CPU-only target

.PHONY: install install-dev lint format test test-cpu test-cuda demo serve \
        docker-build docker-run clean dist test-all test-quick

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .
	black --check .
	mypy .

format:
	ruff check --fix .
	black .

test:
	pytest

# Canonical CPU run: the honesty gates are on, so an "optional dependency
# missing" skip or a skipped real_mamba (must-run) test fails the build instead
# of scrolling past as 2 skipped.
test-cpu:
	pytest -m "not cuda" --strict-deps --fail-on-skip

test-cuda:
	pytest -m "cuda"

# Everything, including the 8 s / RAM-ceiling perf probes.
test-all:
	pytest --strict-deps --fail-on-skip -m "not cuda"

# Fast feedback: no model inference, no subprocess probes.
test-quick:
	pytest -m "not cuda and not slow and not memory"

demo:
	$(PY) demo.py

serve:
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t stem-midi-pro:latest .

docker-run:
	docker run --rm -p 8000:8000 stem-midi-pro:latest

clean:
	-$(PY) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+glob.glob('*.egg-info')+['build','dist','.pytest_cache','.mypy_cache','.ruff_cache','htmlcov']]; os.path.exists('.coverage') and os.remove('.coverage')"

dist:
	$(PY) -m pip wheel . -w dist --no-deps
