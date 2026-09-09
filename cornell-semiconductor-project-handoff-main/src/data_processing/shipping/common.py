"""Shared helpers for the shipping data pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

EPOCH = date(1970, 1, 1)


@dataclass(frozen=True)
class TemporalBoundaries:
    min_ts: int
    T0_end: int
    val_start: int
    val_end: int
    test_start: int
    max_ts: int

    @property
    def train_end_date(self) -> date:
        return epoch_days_to_date(self.T0_end)

    @property
    def val_start_date(self) -> date:
        return epoch_days_to_date(self.val_start)

    @property
    def val_end_date(self) -> date:
        return epoch_days_to_date(self.val_end)

    @property
    def test_start_date(self) -> date:
        return epoch_days_to_date(self.test_start)

    @property
    def max_date(self) -> date:
        return epoch_days_to_date(self.max_ts)


def epoch_days_to_date(epoch_days: int) -> date:
    return EPOCH + timedelta(days=int(epoch_days))


def date_to_epoch_days(value: date) -> int:
    return (value - EPOCH).days


def load_temporal_boundaries(meta_path: Path) -> TemporalBoundaries:
    data = json.loads(Path(meta_path).read_text())
    boundaries = data.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError(f"Unexpected temporal meta format: {meta_path}")
    required = {"min_ts", "T0_end", "val_start", "val_end", "test_start", "max_ts"}
    if not required.issubset(boundaries):
        missing = required - set(boundaries)
        raise ValueError(f"Temporal meta missing keys: {missing}")
    return TemporalBoundaries(
        min_ts=int(boundaries["min_ts"]),
        T0_end=int(boundaries["T0_end"]),
        val_start=int(boundaries["val_start"]),
        val_end=int(boundaries["val_end"]),
        test_start=int(boundaries["test_start"]),
        max_ts=int(boundaries["max_ts"]),
    )


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_output_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    ensure_directory(path)
    return path
