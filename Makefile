# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
# Stem+MIDI Pro — Makefile for development and deployment

.PHONY: install install-dev lint format test test-cpu test-cuda demo serve docker-build docker-run clean dist release-major release-minor release-patch

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

lint:
	ruff check .
	black --check .
	mypy .

format:
	ruff check --fix .
	black .

test:
	pytest -m "not cuda" --cov=. --cov-report=term-missing

test-cpu:
	pytest -m "not cuda" --cov=. --cov-fail-under=70

test-cuda:
	pytest -m "cuda" --cov=. --cov-report=term-missing

demo:
	python demo.py

serve:
	uvicorn api:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 30

docker-build:
	docker build -t stem-midi-pro .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs stem-midi-pro

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info/

dist:
	python -m build

release-major:
	@echo "Use git-cliff or manual version bump"

release-minor:
	@echo "Use git-cliff or manual version bump"

release-patch:
	@echo "Use git-cliff or manual version bump"
