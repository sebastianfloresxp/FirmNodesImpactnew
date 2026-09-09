export PYTHONPATH := $(CURDIR)

.PHONY: data data-full ch3 ch3-full ensemble ch4 ch4-full train verify all-frozen all help \
        lint fmt typecheck deadcode security audit qa qa-install \
        spell spell-fix mdlint mdlint-fix prose docs-qa docs-qa-full

# ---------------------------------------------------------------------------
# Frozen pipeline (no DB required — uses pre-synced artifacts)
# ---------------------------------------------------------------------------
data:  ## Verify core dataset exists (frozen mode — no regeneration needed)
	@if [ -d "data/processed/core/releases/core_v1" ]; then \
		echo "Core dataset found: data/processed/core/releases/core_v1"; \
	else \
		echo "ERROR: Core dataset not found. Sync from Google Drive first."; \
		echo "  rclone sync <DRIVE_FOLDER>:data/processed/ data/processed/"; \
		exit 1; \
	fi

ch3:  ## Ch3 network pipeline (frozen — groups 7-8 only)
	bash src/ch3/run_ch3_pipeline.sh --frozen

ch4:  ## Ch4 analysis pipeline (skip FactSet enrichment step)
	bash src/analysis/chapter4/run_pipeline.sh --config src/analysis/chapter4/config/ch4_v2_fix01.yaml --skip-m9-2

all-frozen: ch3 ch4  ## Run full frozen pipeline (ch3 → ch4, assumes data synced from Drive)

# ---------------------------------------------------------------------------
# Full pipeline (requires FactSet DB + GPU for training)
# ---------------------------------------------------------------------------
data-full:  ## Core data pipeline (full, needs DB)
	python src/data_processing/core/00_run_core_pipeline.py

ch3-full:  ## Ch3 network pipeline (full — all groups, needs DB)
	bash src/ch3/run_ch3_pipeline.sh --full

ensemble:  ## Run ensemble prediction (needs trained model artifacts + results/ensemble/)
	bash src/ensemble/ensemble_predict.sh \
		--tag meta_ranker_v4 \
		--model artifacts/ensemble/meta_ranker/meta_ranker_v4/meta_model.json \
		--meta-dataset results/ensemble/meta_dataset_v4

ch4-full:  ## Ch4 analysis pipeline (full, including FactSet enrichment)
	bash src/analysis/chapter4/run_pipeline.sh --config src/analysis/chapter4/config/ch4_v2_fix01.yaml

all: data-full ch3-full ensemble ch4-full  ## Run full pipeline end-to-end

# ---------------------------------------------------------------------------
# Training and verification
# ---------------------------------------------------------------------------
train:  ## Print instructions for HPO training (requires GPU server)
	@echo "============================================================"
	@echo "  Model Training (HPO)"
	@echo "============================================================"
	@echo ""
	@echo "Training requires a GPU server with CUDA support."
	@echo "Run these scripts on your GPU machine:"
	@echo ""
	@echo "  GraphSAGE:    python src/graphsage/00_graphsage_hpo_two_phase.py --help"
	@echo "  N2V (2-ph):   python src/n2v_temporal/00_n2v_temporal_hpo_two_phase.py --help"
	@echo "  N2V (3-ph):   python src/n2v_temporal/00_n2v_temporal_hpo_three_phase.py --help"
	@echo "  TGNN (2-ph):  python src/tgnn/00_tgnn_hpo_two_phase.py --help"
	@echo "  TGNN (3-ph):  python src/tgnn/00_tgnn_hpo_three_phase.py --help"
	@echo ""
	@echo "See SETUP.md for environment setup and environment_gpu.yml for dependencies."

verify:  ## Verify frozen artifact checksums
	python scripts/verify_checksums.py

# ---------------------------------------------------------------------------
# Code quality (lint, types, dead code, security)
# ---------------------------------------------------------------------------
qa-install:  ## Install full QA toolchain (ruff, pyright, vulture, bandit, pip-audit, codespell, pymarkdown, language-tool-python)
	pip install --upgrade ruff pyright vulture "bandit[toml]" pip-audit codespell pymarkdownlnt language-tool-python

lint:  ## Lint source with ruff (read-only)
	ruff check src scripts

fmt:  ## Auto-format + auto-fix with ruff
	ruff format src scripts
	ruff check --fix src scripts

typecheck:  ## Static type check with pyright
	pyright

deadcode:  ## Detect unused code with vulture
	vulture

security:  ## Static security scan with bandit
	bandit -c pyproject.toml -r src scripts

audit:  ## Audit installed dependencies with pip-audit
	pip-audit --ignore-vuln CVE-2026-32875

qa: lint typecheck deadcode security audit  ## Run the full code-quality suite

# ---------------------------------------------------------------------------
# Docs quality (spell check, markdown lint)
# ---------------------------------------------------------------------------
spell:  ## Spell-check code and docs with codespell
	codespell

spell-fix:  ## Spell-check and auto-apply unambiguous fixes
	codespell -w -i 3

mdlint:  ## Lint Markdown files with pymarkdown
	pymarkdown --config pyproject.toml scan --respect-gitignore .

mdlint-fix:  ## Auto-fix Markdown issues where possible
	pymarkdown --config pyproject.toml fix --respect-gitignore .

prose:  ## Grammar/style check Markdown prose with LanguageTool (requires Java)
	python scripts/prose_check.py

docs-qa: spell mdlint  ## Run the full docs-quality suite (spell + markdown lint)

docs-qa-full: spell mdlint prose  ## docs-qa plus LanguageTool prose check (slow, needs Java)

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help message
	@echo "============================================================"
	@echo "  Semiconductor Supply Chain — Pipeline Targets"
	@echo "============================================================"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-14s %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start (frozen):  make all-frozen"
	@echo "Full pipeline:         make all"
