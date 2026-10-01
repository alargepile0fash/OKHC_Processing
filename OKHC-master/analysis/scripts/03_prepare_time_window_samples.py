#!/usr/bin/env python3
"""Prepare frequency-based word-form samples for diachronic analysis.

Run this script after the diachronic vowel extractor. Sampling settings live in
analysis/config/sampling_default.json.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sampling_logic.config import parse_args, validate_settings
from sampling_logic.windows import choose_anchor_year
from sampling_logic.corpus import count_wordforms
from sampling_logic.sampling import make_window_diagnostics, select_wordforms
from sampling_logic.output import write_outputs, write_selected_token_rows, write_run_config


def main() -> None:
    args = parse_args()
    validate_settings(args)

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_csv.is_absolute():
        input_csv = PROJECT_ROOT / input_csv
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    header = pd.read_csv(input_csv, nrows=0, encoding="utf-8-sig")
    required = [args.year_col, args.token_col, args.vowels_col, "token_source"]
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    anchor_year = choose_anchor_year(
        input_csv, args.year_col, args.start_year, args.window_step, args.chunksize
    )

    print(f"Window: {args.window_width} years; step: {args.window_step} years")
    print(f"Anchor year: {anchor_year}")

    wordforms = count_wordforms(input_csv, args, anchor_year)
    diagnostics = make_window_diagnostics(wordforms, args)
    selected, diagnostics = select_wordforms(wordforms, diagnostics, args)

    write_outputs(selected, diagnostics, output_dir)
    if not args.skip_token_row_output:
        write_selected_token_rows(input_csv, selected, args, anchor_year, output_dir)
    write_run_config(output_dir, args, anchor_year)

    print(
        f"Selected {len(selected):,} word forms across "
        f"{selected['time_window_id'].nunique():,} time windows."
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
