# Reproducing the Semiconductor Supply Chain Analysis

One-page guide to reproducing all results. See `SETUP.md` for detailed environment setup.

## Prerequisites

- Python 3.11.7 (via conda — see `environment_cpu.yml` / `environment_gpu.yml`)
- conda or miniconda
- ~700 GB disk for full data (or ~40 GB for frozen artifacts only)
- FactSet SQL access (full pipeline and entity-name enrichment step — see Known Limitations)
- CUDA GPU (model training only)

## Option A: Accept Artifacts (Frozen Pipeline)

Reproduces all analysis using pre-computed artifacts. No GPU required. One optional step
(`m9_0_semantic_lens_pipeline.py`) queries FactSet for entity display names — see Known Limitations
for how to run without FactSet credentials.

```bash
git clone <repo-url> && cd cornell-semiconductor-project

conda env create -f environment_cpu.yml
conda activate supplychain_env_cpu

# Sync only what the frozen pipeline needs (~37 GB)
rclone sync "silicon-backbone:cornell-semiconductor-project/data/processed/" data/processed/ --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/artifacts/" artifacts/ --progress

# Verify artifact integrity
make verify

# Run the frozen pipeline (Ch3 network assembly → Ch4 analysis)
make all-frozen
```

## Option B: Remake Artifacts (Full Pipeline)

Regenerates all artifacts from raw data. Requires FactSet database credentials.

```bash
cp .env.example .env  # edit with your FactSet credentials

# Sync raw data + trained model artifacts (~100 GB)
rclone sync "silicon-backbone:cornell-semiconductor-project/data/" data/ --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/artifacts/" artifacts/ --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/results/ensemble/" results/ensemble/ --progress

# Run the full pipeline (data processing → Ch3 → ensemble → Ch4)
make all
```

## Option C: Retrain Models (GPU Required)

Retrains all ML models from scratch, then runs the full pipeline.

```bash
conda env create -f environment_gpu.yml
conda activate supplychain_env_gpu

# Sync all data (~660 GB)
rclone sync "silicon-backbone:cornell-semiconductor-project/data/" data/ --progress
rclone sync "silicon-backbone:cornell-semiconductor-project/results/" results/ --progress

# Train models on GPU server (see make train for per-model commands)
make train

# Then run full pipeline
make all
```

## Verify Outputs

```bash
make verify
```

Checks SHA256 hashes of all 1,136 frozen artifacts against `checksums.json`. Reports pass/fail/missing per file.

**What `make verify` does and does not test:**
`make verify` is a **Drive sync integrity check** — it confirms that the artifacts you synced from
Google Drive are byte-identical to the canonical frozen set and were not corrupted in transit.
It is not a pipeline rerun verification tool. If you run `make all-frozen` and then `make verify`,
some parquet files will report FAIL because DuckDB embeds a build-specific hash in its parquet
output; different machines install different compiled binaries of the same DuckDB version, producing
different bytes even from identical input data. The underlying research data is identical — only
the binary envelope of the parquet file changes. To verify that a pipeline rerun produced correct
**research results** (rather than byte-identical files), compare the data values in the output
CSVs against the frozen artifacts rather than relying on SHA256 hashes.

## Make Targets

| Target | Description |
|---|---|
| `make data` | Verify core dataset exists (frozen check) |
| `make data-full` | Core data pipeline (full, needs DB) |
| `make ch3` | Ch3 network pipeline (frozen, groups 7-8) |
| `make ch3-full` | Ch3 network pipeline (full, all groups) |
| `make ensemble` | Ensemble prediction (needs trained models) |
| `make ch4` | Ch4 analysis (skip FactSet enrichment) |
| `make ch4-full` | Ch4 analysis (full) |
| `make train` | Print HPO training instructions |
| `make verify` | Verify frozen artifact checksums |
| `make all-frozen` | Frozen pipeline: ch3 + ch4 (assumes data synced from Drive) |
| `make all` | Full pipeline: data + ch3 + ensemble + ch4 |
| `make help` | List all targets |

## Known Limitations

- **Frozen mode skips model training and ensemble.** The models and ensemble outputs are pre-trained; to retrain, use `make train` on a GPU server and then `make ensemble`.
- **`make all` requires pre-trained model artifacts.** The ensemble step needs `results/ensemble/meta_dataset_v4/` and `artifacts/ensemble/meta_ranker/meta_ranker_v4/`, which are generated during model training (or synced from Drive).
- **FactSet dependency — entity display names only.** The frozen pipeline has one live FactSet
  query: `fetch_factset_names` in `m9_0_semantic_lens_pipeline.py` (M9.3). This fetches
  `entity_proper_name` and `iso_country` for display enrichment in the `m0_7` outputs. It does
  **not** affect any research finding — semiconductor classifications, rankings, and scores are
  derived entirely from frozen M0.6 artifacts. Without FactSet credentials, set
  `fetch_factset_names: false` in `src/analysis/chapter4/config/ch4_v2_fix01.yaml` before running
  `make ch4`. The decision columns and all report table values will be identical; only display-name
  enrichment columns in the `m0_7/` outputs will be empty.
- **FactSet dependency — full pipeline.** Groups 1-6 of Ch3, phase 6 of the core data pipeline,
  and M9.2 (industry enrichment) require FactSet SQL access, which is not publicly available.
  All of these are bypassed in frozen mode.
- **USAspending data.** Raw FY2022-2025 award files (~60 GB) are included in the full data sync but not required for the frozen pipeline.
- **Shipping layer.** `build_shipping_observed_layer.py` in full mode queries FactSet for shipping/sector data. Frozen mode uses pre-built artifacts.
- **Environment pins.** `environment_cpu.yml` and `environment_gpu.yml` pin all data-affecting
  libraries to the exact versions used to generate the frozen artifacts (Python 3.11.7, pandas
  2.1.4, PyArrow 14.0.2, DuckDB 1.4.0). Using unpinned versions will produce byte-different
  parquet files; data values will be identical but `make verify` checksums will not match.

## Troubleshooting

- **`make ch3` fails with "artifact not found"**: Sync artifacts from Google Drive first. Check all 9 required files listed in the error output.
- **`make ch4` fails with "Ch3 node parquet not found"**: Run `make ch3` first to generate the semiconductor subgraph.
- **`make ch4` fails with `MissingEnvironmentError: Missing environment variable 'DB_SERVER'`**:
  The entity display-name enrichment step (`fetch_factset_names` in M9.3) requires FactSet
  credentials. Set `fetch_factset_names: false` in
  `src/analysis/chapter4/config/ch4_v2_fix01.yaml` to run without FactSet. Research findings
  are unaffected — only display-name columns in `artifacts/ch4/.../m0_7/` outputs will be empty.
- **`make verify` reports FAIL**: Re-sync the failing files from Google Drive. Check for partial downloads.
- **`make verify` reports FAIL after re-running the pipeline**: Parquet files written by DuckDB
  embed a build-specific hash in their metadata. Even with the same DuckDB version pinned,
  different conda-forge builds produce different bytes. Data values are identical. If verifying
  pipeline correctness rather than artifact integrity, compare data content rather than file hashes.
- **Import errors**: Ensure `PYTHONPATH` includes the repo root. The Makefile sets this automatically, but standalone scripts may need `export PYTHONPATH=$(pwd)`.
- **conda environment issues**: See `SETUP.md` for detailed environment setup, including PyG wheel URLs.
