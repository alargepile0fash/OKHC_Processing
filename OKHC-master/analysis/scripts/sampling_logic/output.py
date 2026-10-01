"""Write sampler results and reproducibility information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .corpus import prepare_chunk


def write_outputs(selected: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path) -> None:
    """Write the selected sample and window-level diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selected.to_csv(
        output_dir / "sampled_time_window_wordforms.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        output_dir / "time_window_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics[~diagnostics["time_window_is_eligible_for_sampling"]].to_csv(
        output_dir / "excluded_time_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_selected_token_rows(
    input_csv: Path,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    anchor_year: int,
    output_dir: Path,
) -> None:
    """Write original token rows represented by the selected word forms."""
    keys = set(
        selected["time_window_id"].astype(str) + "\x1e" + selected["wordform_id"].astype(str)
    )
    output = output_dir / "sampled_time_window_original_token_rows.csv"
    wrote_header = False
    columns = None

    for chunk in pd.read_csv(
        input_csv,
        chunksize=args.chunksize,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        if columns is None:
            columns = list(chunk.columns)

        prepared = prepare_chunk(chunk, args, anchor_year)
        if prepared.empty:
            continue

        prepared["selection_key"] = (
            prepared["time_window_id"].astype(str) + "\x1e" + prepared["wordform_id"].astype(str)
        )
        rows = prepared[prepared["selection_key"].isin(keys)].drop(
            columns=["vowel_seq_list", "selection_key"], errors="ignore"
        )
        if rows.empty:
            continue

        rows.to_csv(
            output,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            index=False,
            encoding="utf-8-sig",
        )
        wrote_header = True

    if not wrote_header:
        pd.DataFrame(columns=columns or []).to_csv(output, index=False, encoding="utf-8-sig")


def write_run_config(output_dir: Path, args: argparse.Namespace, anchor_year: int) -> None:
    """Save the exact settings used for this run."""
    settings = vars(args).copy()
    settings["config"] = str(args.config)
    settings["anchor_year"] = anchor_year
    settings["window_overlap_years"] = max(0, args.window_width - args.window_step)
    with open(output_dir / "sampling_config.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)
