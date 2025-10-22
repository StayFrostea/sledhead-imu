PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup dev env test lint fmt nb

setup: ## create venv, install deps, pre-commit
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt pre-commit black ruff
	pre-commit install
	.venv/bin/python -m ipykernel install --user --name sledhead-imu --display-name "Python (sledhead-imu)"

dev: env
env:
	@echo "Activate with: source .venv/bin/activate"

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests

fmt:
	.venv/bin/black src tests

nb:
	@echo "Start Jupyter: .venv/bin/jupyter lab"
