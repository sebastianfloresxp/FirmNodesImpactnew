# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end semiconductor supply chain vulnerability analysis for Cornell Brooks School Technology Policy Institute. The pipeline flows across three chapters:

1. **Ch2** — Build `core_v1` dataset from FactSet SCR, train link-prediction models (GraphSAGE, N2V, TGNN, TwoTower, heuristics), calibrate ensemble meta-ranker.
2. **Ch3** — Construct DoD-anchored multiplex supply network (contracts + disclosed + predicted + shipping), prune to strict semiconductor subgraph.
3. **Ch4** — Structural vulnerability analysis (deny/delay severity, corridor intermediaries, chokepoint shortlist, action matrix).

All scripts must be run from the **repository root** so relative paths resolve. The `Makefile` sets `PYTHONPATH=$(CURDIR)` automatically; standalone scripts may need `export PYTHONPATH=$(pwd)`.

## Common Commands

### Pipeline (frozen — no DB or GPU required)

```bash
make all-frozen        # ch3 (groups 7-8) + ch4 (skip FactSet enrichment)
make ch3               # Ch3 frozen build only
make ch4               # Ch4 analysis (offline, skips M9.2)
make verify            # Verify SHA256 checksums of all 1,136 frozen artifacts
```

### Pipeline (full — requires FactSet DB + pre-trained artifacts)

```bash
make all               # data-full → ch3-full → ensemble → ch4-full
make data-full         # Core data pipeline
make ch3-full          # Ch3 all groups
make ensemble          # Run ensemble prediction (needs meta_ranker_v4 artifacts)
make ch4-full          # Ch4 including FactSet enrichment (M9.2)
make train             # Print GPU HPO training instructions
```

### Code Quality

```bash
make qa                # lint + typecheck + deadcode + security + audit
make lint              # ruff check src scripts
make fmt               # ruff format + ruff check --fix
make typecheck         # pyright
make deadcode          # vulture
make security          # bandit
make audit             # pip-audit
make qa-install        # Install ruff, pyright, vulture, bandit, pip-audit, codespell, pymarkdownlnt, language-tool-python
```

### Docs Quality

```bash
make docs-qa           # codespell + pymarkdown lint
make docs-qa-full      # docs-qa + LanguageTool prose check (needs Java)
make spell             # codespell
make mdlint            # pymarkdown
make prose             # LanguageTool grammar/style check
```

### Misc Scripts

```bash
bash scripts/regenerate_architecture.sh  # Regenerate pyan3 call-graph HTML docs → docs/
python scripts/verify_checksums.py        # Same as make verify
```

### Report (LaTeX)

Canonical compilation environment is Overleaf (pdfLaTeX + Biber). Main file: `report/report_main.tex`. The Overleaf root must be the **repo root**, not `report/`. Local: ensure `newunicodechar.sty` and all preamble packages are installed.

## Architecture

```text
src/data_processing/    — core_v1 dataset pipeline (FactSet SCR events → candidates → features)
src/{node2vec,graphsage,n2v_temporal,tgnn,twotower,heuristics}/  — link-prediction models
src/ensemble/           — score export, calibration, meta-ranker (logistic) training + inference
src/ch3/                — canonical Ch3 pipeline (contracts → matching → prediction → network → pruning)
  usaspending/          — DoD award ingestion
  matching/             — prime → FactSet entity matching
  prediction/           — candidate pools, scoring, Top-K selection
  network/              — graph assembly (disclosed, predicted, shipping, semiconductor subgraph)
src/analysis/chapter4/  — M0–M9 vulnerability modules + 8 reporting export scripts + ch4_common.py (shared helpers)
src/analysis/chapter2/  — LaTeX table/figure generators for Ch2
src/db_client/          — FactSet SQL connection + query helpers (reads .env for credentials)
src/utils/              — shared utilities
dod_supply_chain_analysis/  — DEPRECATED: early exploratory scripts; canonical Ch3 pipeline is src/ch3/
```

### Ch4 Module Map (M0–M9)

| Module(s) | Purpose |
|-----------|---------|
| M0 | Graph contract, prime weights, code universe, industry enrichment (M0.6, needs DB), semiconductor lens |
| M1 | Topology metrics |
| M2 | Seam analysis — interdiction candidates, DoD harm, hard chokepoints |
| M3 | Supplier-centric refinement, effective reach |
| M4 | Corridor importance + refinement |
| M5 | Redundancy proxy |
| M6 | Weighted disruption, stratified impact, candidate guardrails |
| M9 | Reporting enrichment (M9.2 requires FactSet DB; skip with `--skip-m9-2`) |

Reporting exports live in `src/analysis/chapter4/reporting/` (8 scripts: tables 4.1–4.2, figures 4.1–4.5, appendix A1).

**Data flow:** raw inputs → `data/processed/core/releases/core_v1/` → model artifacts in `artifacts/` → Ch3 network in `artifacts/ch3/network_upstream/` → Ch4 outputs in `artifacts/ch4/v2_fix01/` → LaTeX fragments in `report/tables/` and `figs/`.

## Key Artifacts

| Artifact | Path |
|---|---|
| Core dataset | `data/processed/core/releases/core_v1/` |
| Ch3 semiconductor subgraph (nodes) | `artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet` |
| Ch3 semiconductor subgraph (edges) | `artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet` |
| Ensemble meta-ranker | `artifacts/ensemble/meta_ranker/meta_ranker_v4/meta_model.json` |
| Artifact checksums | `checksums.json` (1,136 files) |

## Configuration

- **Global pipeline config:** `configs/global_config.yaml`
- **Ch4 production config:** `src/analysis/chapter4/config/ch4_v2_fix01.yaml`
- **FactSet DB credentials:** `.env` (copy from `.env.example`; never commit `.env` — a pre-tool hook blocks direct edits)
- **Semiconductor codebooks:** `configs/ch4_semiconductor_codebook_v*.csv`

## QA Toolchain Notes

- **No test suite exists.** There are no `pytest` tests or `tests/` directory. `make qa` (ruff + pyright + vulture + bandit + pip-audit) is the complete QA gate.
- `pyright` is configured in `pyproject.toml` (`typeCheckingMode = "basic"`). `src/analysis/` and `src/ch3/` are excluded from pyright due to research-code patterns.
- `pyright` produces 23 expected `reportMissingImports` warnings — **these are not errors and require no fix.** They arise because `make qa` runs in the CPU linting environment (`supplychain_env_cpu`) where GPU-only packages are absent: `duckdb` (10 files in `src/data_processing/shipping/` and `src/ensemble/`) and `optuna`/`optuna.pruners`/`optuna.samplers` (5 HPO scripts in `src/graphsage/`, `src/n2v_temporal/`, `src/node2vec/`, `src/tgnn/`, `src/twotower/`). To suppress them, run `make qa` inside `supplychain_env_gpu` where both packages are installed.
- `ruff` target is Python 3.10, line length 100. `N806`/`N803` (ML uppercase vars) and `E501` are silenced. Pipeline scripts (`0*_*.py`, `m*_*.py`) are exempt from `E402` (sys.path manipulation before imports is expected).
- `bandit` skips `B101` (assert), `B404`/`B603` (subprocess in pipeline wrappers), `B608` (hardcoded SQL — all DuckDB queries use internal paths).
- A post-edit hook in `.claude/settings.json` auto-runs `ruff check` on any `.py` file after Edit/Write.

## Branching & Commits

Trunk-based development: `main` ↔ Overleaf sync; short-lived `feature/<topic>` branches.

Commit scopes: `[report]`, `[brand]`, `[code]`, `[config]`, `[docs]`, `[fix]`, `[meta]`

Recent commits use conventional prefixes (`fix(scope):`, `chore(scope):`, `docs(scope):`) — match whichever style is already in use on the current branch.

## Large Data

`data/`, `artifacts/`, `results/`, `predictions/`, `logs/` are gitignored. Sync from Google Drive via rclone (see `SETUP.md`). Full dataset is ~700 GB; frozen pipeline needs only ~37 GB (`data/processed/` + `artifacts/`).

## Environment Setup

```bash
# CPU-only (analysis, reporting)
conda env create -f environment_cpu.yml   # creates: supplychain_env_cpu (Python 3.11)
conda activate supplychain_env_cpu

# GPU (model training)
conda env create -f environment_gpu.yml   # creates: supplychain_env_gpu
conda activate supplychain_env_gpu
```

> Note: ruff/pyright target Python 3.10 for linting even though the runtime env is 3.11.

## Known Issues

- HPO scripts (`src/tgnn/`, `src/graphsage/`, `src/n2v_temporal/`) use `SUPPLYCHAIN_ROOT` env var with auto-detection fallback — no manual config needed.
- GitHub repo: `https://github.com/kbsimms/cornell-semiconductor-project.git`. Google Drive folder: `silicon-backbone:cornell-semiconductor-project`.
