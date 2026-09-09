# Semiconductor Supply Chain (DoD) — Technical Analysis

> Prepared by Asymmetric Network Advisory Group LLC for Cornell Brooks School Technology Policy Institute.

This repository contains the end-to-end code and artifacts used to:

- build a leakage-safe supply-chain link prediction dataset from **FactSet Supply Chain Relationships (SCR)** (`core_v1`) (**Chapter 2**),
- train and evaluate multiple link-prediction models and a calibrated **ensemble meta-ranker** (**Chapter 2**),
- construct a **DoD-anchored multiplex supply network** using *contracts (USAspending) + disclosed SCR + predicted links + observed shipping*, then prune to a strict semiconductor-relevant subgraph (**Chapter 3**),
- run **structural vulnerability analysis** on the locked DoD semiconductor network (**Chapter 4**),
- present **policy recommendations** for decision-grade visibility and a layered assurance framework (**Chapter 5**),
- synthesize findings, scope conditions, and next steps (**Chapter 6**).

If you are new to the repo, start here: **`SETUP.md`**.

## Delivery Scope

This repository supports the report *The Silicon Backbone: Mapping and Securing the DoD Semiconductor Supply Chain*. The current delivery (Milestones 1–5) includes the complete written report and all supporting code:

- **Milestone 1 (Section 1):** Introduction, chip primer, manufacturing, supply-chain structure, and risk taxonomy.
- **Milestone 2 (Section 2 + code):** Link-prediction dataset, modeling pipeline, ensemble meta-ranker, and evaluation.
- **Milestone 3 (Section 3 + code):** DoD-anchored network construction (contracts, disclosed, predicted, shipping, semiconductor subgraph).
- **Milestone 4 (Section 4 + code):** Structural vulnerability analysis — deny/delay severity, corridor intermediaries, upstream enablers, decision-grade action matrix, Top-25 chokepoint shortlist.
- **Milestone 5 (Section 5 + Section 6):** Policy recommendations (acquisition levers, layered assurance framework, data governance, incentives, implementation roadmap) and conclusion (synthesis, scope conditions, next steps).

All six sections and three appendices are active in `report/report_main.tex` and compile to a complete report.

## Repo Map (where to look)

### Pipelines (chapters)

- **Chapter 2 — core dataset + modeling**
  - `src/data_processing/core/` — builds `data/processed/core/releases/core_v1/` (events, train snapshot, candidate pools, features).
  - `src/{heuristics,node2vec,graphsage,n2v_temporal,tgnn,twotower}/` — model training/evaluation code.
  - `src/ensemble/` — score export, calibration, meta-ranker training/inference, selection-policy evaluation.
  - `src/analysis/chapter2/` — LaTeX tables/figures for Chapter 2 (see its `README.md`).
- **Chapter 3 — DoD network construction**
  - `src/ch3/` — canonical Chapter 3 pipeline (contracts → matching → prediction → shipping → multiplex merge → pruning).
  - `src/ch3/README.md` — the reproducible “run order” and the canonical output paths.
  - Outputs land primarily under `artifacts/ch3/…`, with LaTeX fragments under `report/tables/` and figures under `figs/chapter3/`.
- **Chapter 4 — vulnerability analysis**
  - `src/analysis/chapter4/` — M0–M9 analysis modules, 8 reporting export scripts, and `run_pipeline.sh` orchestrator (see `src/analysis/chapter4/README.md`).

### Data / outputs

- `data/raw/` — raw inputs (typically not committed; large).
- `data/processed/` — processed datasets, including `core_v1` release and run logs.
- `artifacts/` — trained model checkpoints and intermediate pipeline artifacts (large; typically not committed).
- `report/tables/` — LaTeX `tabular` fragments used in the report.
- `figs/` — figures used in the report.

### Documentation

- `docs/` — Architecture call graphs (pyan3-generated interactive HTML) for all pipeline modules and technical documentation.

- **Sections 5 and 6 — policy and conclusion**
  - `report/chapters/05_illumination_to_assurance.tex` — Policy recommendations: acquisition levers, layered assurance, data governance, incentives, implementation roadmap.
  - `report/chapters/06_conclusion.tex` — Synthesis, scope conditions, inference boundaries, next steps.
  - `report/tables/tab_5_1_implementation_roadmap.tex` — Three-phase implementation roadmap table.

### Legacy / misc

- `dod_supply_chain_analysis/` — earlier exploratory DoD scripts (deprecated); the canonical Chapter 3 pipeline is under `src/ch3/`.

### FactSet SQL access

- `src/db_client/` — FactSet SQL utilities (connection, query/export helpers).
- Requires `.env` with:
  - `DB_SERVER`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD`

## Getting Started

1) Follow **`SETUP.md`** to:

   - create a conda environment (`environment_gpu.yml` or `environment_cpu.yml`),
   - sync large data/artifacts (if you have access to the shared storage),
   - verify your Python environment.

2) (Optional) If you need to run SQL-backed steps (SCR extraction, shipping extraction, RBICS/SIC pulls), add a `.env` file at repo root with the required DB credentials (not committed).

## Reproducing Key Outputs (high level)

### Build the Chapter 2 core dataset (`core_v1`)

This generates the dataset release used throughout the modeling and Chapter 3 inference.

```bash
python src/data_processing/core/00_run_core_pipeline.py --dry-run
# then (if the plan looks right)
python src/data_processing/core/00_run_core_pipeline.py
```

Outputs:

- `data/processed/core/releases/core_v1/…` (events, train snapshot adjacency, candidates, features)

### Build the Chapter 3 DoD multiplex + strict semiconductor subgraph

Use the canonical Chapter 3 runbook:

- `src/ch3/README.md`

Key final artifacts (strict build; Chapter 4 input):

- `artifacts/ch3/network_upstream/dod_semiconductor_nodes_top5_d99_shipping_strict.parquet`
- `artifacts/ch3/network_upstream/dod_semiconductor_edges_top5_d99_shipping_strict.parquet`

### Run the Chapter 4 analysis

```bash
bash src/analysis/chapter4/run_pipeline.sh \
  --config src/analysis/chapter4/config/ch4_v2_fix01.yaml
```

Use `--skip-m9-2` to bypass FactSet DB enrichment for offline runs.
See `src/analysis/chapter4/README.md` for full options and config guide.

## Notes for New Researchers

- This repo mixes **public** data (USAspending) with **proprietary** data (FactSet SCR + shipping). Large inputs and trained artifacts are generally not stored in git.
- Many scripts assume they are run from the **repository root** so relative paths resolve.
- Chapter 3/4 are designed to be reproducible from saved Parquets/JSON summaries under `artifacts/ch3/` and `data/processed/core/releases/core_v1/`.

## License

See `LICENSE`.
