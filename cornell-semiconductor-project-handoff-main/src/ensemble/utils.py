"""Shared utilities for ensemble score I/O, calibration, and file discovery.

Provides Platt-scaling calibration helpers, incremental Parquet writers
(ParquetAppend), score-file discovery and metadata (ScoreFile), and SQL
join builders for assembling multi-model score matrices.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "ParquetAppend",
    "ScoreFile",
    "apply_platt",
    "build_join_sql",
    "discover_score_files",
    "ensure_dir",
    "escape_path",
    "load_calibration",
    "load_json",
    "now_iso",
    "save_json",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_calibration(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        data = load_json(path)
    except Exception:
        return None
    params = cast(dict[str, object], data.get("params", {}) if isinstance(data, dict) else {})
    try:
        A = float(cast(float, params.get("A")))
        B = float(cast(float, params.get("B")))
    except Exception:
        return None
    return {"A": A, "B": B}


def apply_platt(logits: np.ndarray, params: dict[str, float] | None) -> np.ndarray:
    if params is None:
        return 1.0 / (1.0 + np.exp(-logits))
    A = float(params.get("A", 1.0))
    B = float(params.get("B", 0.0))
    z = A * logits + B
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ParquetAppend:
    path: Path
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    compression: str = "zstd"

    def __post_init__(self) -> None:
        ensure_dir(self.path.parent)

    def write(self, frame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.schema is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression=self.compression)
        assert self.writer is not None
        if self.schema != table.schema:
            table = table.cast(self.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> ParquetAppend:
        return self

    def __exit__(self, _exc_type, exc, _tb) -> None:
        self.close()


@dataclass
class ScoreFile:
    model: str
    tag: str
    seed: str
    split: str
    path: Path

    @property
    def prob_feature(self) -> str:
        return f"prob_{self.model}_{self.seed}"

    @property
    def logit_feature(self) -> str:
        return f"logit_{self.model}_{self.seed}"


def discover_score_files(roots: Iterable[Path]) -> list[ScoreFile]:
    files: list[ScoreFile] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        parts = root_path.parts
        if len(parts) < 2:
            continue
        model = parts[-2]
        tag = parts[-1]
        for seed_dir in root_path.iterdir():
            if not seed_dir.is_dir():
                continue
            name = seed_dir.name
            if name.startswith("seed_"):
                seed = name.split("_")[-1]
            else:
                seed = name
            for split in ("val", "test"):
                path = seed_dir / f"scores_{split}.parquet"
                if path.exists():
                    files.append(ScoreFile(model=model, tag=tag, seed=seed, split=split, path=path))
    return files


def escape_path(path: Path) -> str:
    return str(path.as_posix()).replace("'", "''")


def build_join_sql(
    files: list[ScoreFile], split: str, include_logits: bool
) -> tuple[str, list[str]]:
    split_files = [f for f in files if f.split == split]
    if not split_files:
        raise RuntimeError(f"No score files discovered for split '{split}'")
    split_files.sort(key=lambda f: (f.model, f.tag, int(f.seed) if f.seed.isdigit() else f.seed))

    select_cols: list[str] = ["base.src_id", "base.dst_id", "base.label"]
    feature_cols: list[str] = []

    base = split_files[0]
    base_path = escape_path(base.path)
    from_clause = f"read_parquet('{base_path}') base"
    joins: list[str] = []

    select_cols.append(f"base.calibrated AS {base.prob_feature}")
    feature_cols.append(base.prob_feature)
    if include_logits:
        select_cols.append(f"base.logit AS {base.logit_feature}")
        feature_cols.append(base.logit_feature)

    for idx, sf in enumerate(split_files[1:], start=1):
        alias = f"t{idx}"
        path = escape_path(sf.path)
        joins.append(f"JOIN read_parquet('{path}') {alias} USING (src_id, dst_id)")
        select_cols.append(f"{alias}.calibrated AS {sf.prob_feature}")
        feature_cols.append(sf.prob_feature)
        if include_logits:
            select_cols.append(f"{alias}.logit AS {sf.logit_feature}")
            feature_cols.append(sf.logit_feature)

    sql = "SELECT " + ", ".join(select_cols) + " FROM " + from_clause
    if joins:
        sql += " " + " ".join(joins)
    return sql, feature_cols
