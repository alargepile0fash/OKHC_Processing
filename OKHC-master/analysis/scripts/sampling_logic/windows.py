"""Construct historical time windows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def choose_anchor_year(
    input_csv: Path,
    year_col: str,
    start_year: Optional[int],
    step: int,
    chunksize: int,
) -> int:
    """Choose the first time-window boundary."""
    if start_year is not None:
        return start_year

    minimum_year = None
    for chunk in pd.read_csv(
        input_csv,
        usecols=[year_col],
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        years = pd.to_numeric(chunk[year_col], errors="coerce").dropna()
        if not years.empty:
            value = int(years.min())
            minimum_year = value if minimum_year is None else min(minimum_year, value)

    if minimum_year is None:
        raise ValueError(f"No usable years found in {year_col!r}.")

    return minimum_year - (minimum_year % step)


def windows_for_year(year: int, anchor_year: int, width: int, step: int) -> list[tuple[int, int]]:
    """Return every configured window containing a given year."""
    latest_start = anchor_year + ((year - anchor_year) // step) * step

    starts = []
    start = latest_start
    while start >= anchor_year and start + width - 1 >= year:
        starts.append(start)
        start -= step

    return [(start, start + width - 1) for start in reversed(starts)]
