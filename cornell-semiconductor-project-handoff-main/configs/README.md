# Configuration Files

This directory contains model hyperparameter configs and reference data for the
semiconductor supply chain pipeline.

## Model Configs

| Directory | Model | Notes |
|-----------|-------|-------|
| `graphsage/` | GraphSAGE | Production config |
| `graphsage_random/` | GraphSAGE (random splits) | Uses `run_graphsage_eval.py` with `--random-splits` flag |
| `node2vec/` | Node2Vec | Production config |

TGNN and TwoTower hyperparameters are specified via command-line arguments
to their respective training scripts (not stored as YAML configs).

## Global Config

`global_config.yaml` defines shared settings (paths, random seed, training
defaults). Training scripts accept these values as CLI arguments but do not
read the YAML at startup — treat it as reference documentation for the
parameter values used in production runs.

## Semiconductor Codebooks (Chapter 4)

Three codebook versions exist for the Chapter 4 severity analysis:

| File | Status |
|------|--------|
| `ch4_semiconductor_codebook_v1.csv` | Superseded |
| `ch4_semiconductor_codebook_v2.csv` | Superseded |
| `ch4_semiconductor_codebook_v2_1.csv` | **Canonical** — used for final Chapter 4 production runs |
