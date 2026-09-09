# TGNN Temporal + Node2Vec

This fork of the temporal TGNN pipeline augments the graph convolution with
per-snapshot Node2Vec embeddings. Provide a directory of files named
`snapshot_000.npy`, `snapshot_001.npy`, … that align with the snapshot windows
built from `--splits-root`. Each `.npy` file must contain a `[num_nodes, dim]`
matrix of float32 embeddings. Pass the directory via `--n2v-dir` to
`03_n2v_temporal_train_eval.py` and the HPO orchestrators. All artifacts, results, and
logs are written under the `n2v_temporal` subfolders to avoid collisions
with the legacy TGNN pipelines.

## Generating Snapshot Node2Vec Embeddings

Use `src/n2v_temporal/04_build_n2v_embeddings.py` to produce the
`snapshot_XXX.npy` files required by the temporal TGNN runner:

```bash
python src/n2v_temporal/04_build_n2v_embeddings.py \
  --splits-root data/.../splits \
  --out-dir artifacts/n2v_temporal/node2vec_snapshots \
  --granularity quarter --embedding-dim 128 --epochs 30
```

The script reuses the TGNN snapshot windows, trains PyTorch Geometric's
`Node2Vec` model on each snapshot, and writes `snapshot_000.npy`,
`snapshot_001.npy`, ... into the chosen output directory along with a metadata
summary JSON.
