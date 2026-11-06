# tokenization_101 — developer tasks
# Run `make` or `make help` to list targets.

# Use bash so `printf`, `set -e`, and `&&`/`||` behave consistently.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Tools and paths (override on the command line, e.g. `make test PORT=9000`).
UV            ?= uv
RUN           := $(UV) run
KERNEL_NAME   := tokenization-101
NB_TIMEOUT    ?= 180
PORT          ?= 8501
LAB_PORT      ?= 8888
PKG_DIRS      := src tests app

.PHONY: help setup sync data lint format check typecheck test test-cov \
        notebooks run lab clean reset ci dev all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# --- Environment -----------------------------------------------------------

setup: ## First-time setup: install uv, sync deps, register the Jupyter kernel
	@command -v $(UV) >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --all-extras
	$(RUN) python -m ipykernel install --user --name $(KERNEL_NAME)
	@printf '\n✅ Ready. Run "make data" then "make test".\n'

sync: ## Sync dependencies (incl. dev extras) from pyproject/uv.lock
	$(UV) sync --all-extras

# --- Data ------------------------------------------------------------------

data: ## Build/cache the training corpus and parallel multilingual sentences
	$(RUN) python -m src.datasets.corpora
	@printf '✅ Corpora cached in data/\n'

# --- Quality ---------------------------------------------------------------

lint: format check typecheck ## Run formatter, linter (autofix), and type checker

format: ## Format code with ruff
	$(RUN) ruff format $(PKG_DIRS)

check: ## Lint with ruff (autofixing what it safely can)
	$(RUN) ruff check --fix $(PKG_DIRS)

typecheck: ## Type-check src/ with ty
	$(RUN) ty check src

# --- Tests -----------------------------------------------------------------

test: ## Run the test suite
	$(RUN) pytest

test-cov: ## Run tests with a coverage report
	$(RUN) pytest --cov=src --cov-report=term-missing

# --- Notebooks & apps ------------------------------------------------------

notebooks: data ## Execute all notebooks in place (fails on any error)
	@for nb in notebooks/0*.ipynb; do \
		printf '▶ Executing %s ...\n' "$$nb"; \
		$(RUN) jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=$(NB_TIMEOUT) \
			--ExecutePreprocessor.kernel_name=python3 "$$nb" || exit 1; \
	done
	@printf '\n✅ All notebooks executed.\n'

run: ## Launch the Streamlit app
	$(RUN) streamlit run app/streamlit_app.py --server.port $(PORT)

lab: ## Launch JupyterLab
	$(RUN) jupyter lab --no-browser --port $(LAB_PORT)

# --- Housekeeping ----------------------------------------------------------

clean: ## Remove caches and coverage artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache htmlcov .coverage .pytest_cache .ruff_cache
	@printf '🧹 Cleaned.\n'

reset: clean ## Full reset: also remove the venv and cached corpora
	rm -rf .venv data/corpora

# --- Aggregates ------------------------------------------------------------

ci: sync lint test ## Full CI pipeline: sync, lint, test
dev: lint test ## Fast inner-loop: lint + test
all: lint test notebooks ## Everything: lint, test, run notebooks
