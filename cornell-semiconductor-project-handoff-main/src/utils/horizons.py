"""Shared utilities for horizon-based evaluation slices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Each tuple: (lower_bound_days, upper_bound_days_or_None, slice_name)
HORIZON_BUCKETS: Sequence[tuple[int, int | None, str]] = (
    (0, 183, "horizon_0_6"),
    (183, 365, "horizon_6_12"),
    (365, 730, "horizon_12_24"),
    (730, 1095, "horizon_24_36"),
    (1095, 1460, "horizon_36_48"),
    (1460, 1825, "horizon_48_60"),
    (1825, 2190, "horizon_60_72"),
    (2190, None, "horizon_72_plus"),
)


def horizon_slice_names(
    buckets: Sequence[tuple[int, int | None, str]] = HORIZON_BUCKETS,
) -> list[str]:
    """Return the ordered slice names for the configured horizon buckets."""

    return [name for _, _, name in buckets]


def assign_horizon_buckets(
    deltas_days: np.ndarray,
    buckets: Sequence[tuple[int, int | None, str]] = HORIZON_BUCKETS,
) -> np.ndarray:
    """Map an array of day offsets to 1-indexed horizon bucket ids.

    NaNs remain 0. Returned array has dtype int8 to match existing evaluators.
    """

    out = np.zeros_like(deltas_days, dtype=np.int8)
    if deltas_days.size == 0:
        return out
    deltas = np.asarray(deltas_days, dtype=np.float64)
    valid = ~np.isnan(deltas)
    if not np.any(valid):
        return out
    for idx, (lower, upper, _) in enumerate(buckets, start=1):
        mask = valid & (deltas >= lower)
        if upper is not None:
            mask &= deltas < upper
        out[mask] = np.int8(idx)
    return out


__all__ = ["HORIZON_BUCKETS", "assign_horizon_buckets", "horizon_slice_names"]
