#!/usr/bin/env python3
"""Build the frequency-based word-form sample used by later analyses.

This script does preprocessing and sampling only. It does not run TP or D2L.
Most settings live in analysis/config/sampling_default.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "analysis" / "config" / "sampling_default.json"
KNOWN_VOWELS = [
    "ㆍ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ",
    "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    """Read the JSON settings file."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    """Load JSON settings, then allow command-line options to override them."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()

    config_path = config_args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    settings = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Prepare the historical Korean word-form sample for phonological analysis."
    )
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--input", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--year-col", default=argparse.SUPPRESS)
    parser.add_argument("--token-col", default=argparse.SUPPRESS)
    parser.add_argument("--vowels-col", default=argparse.SUPPRESS)
    parser.add_argument("--window-width", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--window-step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--start-year", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-word-vowels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--chunksize", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-window-wordforms", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min-window-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-wordforms-per-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--sampling-mode", choices=["cap-preserve-windows", "strict-balanced"], default=argparse.SUPPRESS)
    parser.add_argument("--target-wordforms-per-window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--exclude-windows", nargs="*", default=argparse.SUPPRESS)
    parser.add_argument("--skip-token-row-output", action="store_true", default=argparse.SUPPRESS)

    parser.set_defaults(**settings)
    args = parser.parse_args()
    args.config = config_path
    return args


def validate_settings(args: argparse.Namespace) -> None:
    """Check settings that would otherwise produce an invalid sample."""
    if args.window_width <= 0 or args.window_step <= 0:
        raise ValueError("window_width and window_step must be positive.")
    if args.min_word_vowels < 1:
        raise ValueError("min_word_vowels must be at least 1.")
    if args.target_wordforms_per_window <= 0:
        raise ValueError("target_wordforms_per_window must be positive.")
    if args.min_window_wordforms < 0 or args.min_window_tokens < 0:
        raise ValueError("minimum sample thresholds cannot be negative.")


# ---------------------------------------------------------------------------
# Time windows and vowel parsing
# ---------------------------------------------------------------------------


def choose_anchor_year(input_csv: Path, year_col: str, start_year: Optional[int], chunksize: int) -> int:
    """Choose the first window boundary."""
    if start_year is not None:
        return start_year

    minimum_year = None
    for chunk in pd.read_csv(input_csv, usecols=[year_col], chunksize=chunksize, encoding="utf-8-sig"):
        years = pd.to_numeric(chunk[year_col], errors="coerce").dropna()
        if not years.empty:
            value = int(years.min())
            minimum_year = value if minimum_year is None else min(minimum_year, value)

    if minimum_year is None:
        raise ValueError(f"No usable years found in {year_col!r}.")

    return math.floor(minimum_year / chunksize) * chunksize if False else minimum_year - (minimum_year % 25)


def windows_for_year(year: int, anchor_year: int, width: int, step: int) -> list[tuple[int, int]]:
    """Return every configured window containing a given year."""
    latest_start = anchor_year + math.floor((year - anchor_year) / step) * step
    earliest_start = latest_start - width + 1
    first_start = anchor_year + math.ceil((earliest_start - anchor_year) / step) * step
    return [
        (start, start + width - 1)
        for start in range(first_start, latest_start + 1, step)
        if start >= anchor_year
    ]


def extract_vowels(value: object) -> list[str]:
    """Convert the vowels column into a list of recognized vowel symbols."""
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    for char in "[](){}'\"":
        text = text.replace(char, " ")

    if any(separator in text for separator in " ,;/|"):
        pieces = [piece for piece in text.replace(",", " ").replace(";", " ").replace("/", " ").replace("|", " ").split() if piece]
        result = []
        for piece in pieces:
            result.extend(extract_vowels(piece))
        return result

    symbols = sorted(KNOWN_VOWELS, key=len, reverse=True)
    result = []
    position = 0
    while position < len(text):
        match = next((symbol for symbol in symbols if text.startswith(symbol, position)), None)
        if match is None:
            position += 1
        else:
            result.append(match)
            position += len(match)
    return result


# ---------------------------------------------------------------------------
# Corpus preprocessing
# ---------------------------------------------------------------------------


def prepare_chunk(chunk: pd.DataFrame, args: argparse.Namespace, anchor_year: int) -> pd.DataFrame:
    """Clean a corpus chunk and assign each row to its time window(s)."""
    chunk = chunk.copy()
    year = pd.to_numeric(chunk[args.year_col], errors="coerce")
    chunk = chunk.assign(**{args.year_col: year}).dropna(
        subset=[args.year_col, args.token_col, args.vowels_col]
    )

    if args.start_year is not None:
        chunk = chunk[chunk[args.year_col] >= args.start_year]
    if chunk.empty:
        return chunk

    chunk[args.year_col] = chunk[args.year_col].astype(int)
    chunk[args.token_col] = chunk[args.token_col].astype(str).str.strip()
    chunk = chunk[chunk[args.token_col] != ""]

    chunk["vowel_seq_list"] = chunk[args.vowels_col].apply(extract_vowels)
    chunk["num_vowels_normalized"] = chunk["vowel_seq_list"].str.len()
    chunk = chunk[chunk["num_vowels_normalized"] >= args.min_word_vowels]
    if chunk.empty:
        return chunk

    chunk["vowel_seq"] = chunk["vowel_seq_list"].apply(" ".join)
    chunk["time_windows"] = chunk[args.year_col].apply(
        lambda year: windows_for_year(year, anchor_year, args.window_width, args.window_step)
    )
    chunk = chunk.explode("time_windows", ignore_index=True)
    if chunk.empty:
        return chunk

    chunk[["time_window_start", "time_window_end"]] = pd.DataFrame(
        chunk["time_windows"].tolist(), index=chunk.index
    )
    chunk["time_window_start"] = chunk["time_window_start"].astype(int)
    chunk["time_window_end"] = chunk["time_window_end"].astype(int)
    chunk["time_window_label"] = (
        chunk["time_window_start"].astype(str) + "-" + chunk["time_window_end"].astype(str)
    )
    chunk["time_window_id"] = chunk["time_window_label"]
    chunk["wordform_id"] = chunk[args.token_col] + " || " + chunk["vowel_seq"]
    return chunk.drop(columns=["time_windows"])


# ---------------------------------------------------------------------------
# Frequency sampling
# ---------------------------------------------------------------------------


def count_wordforms(input_csv: Path, args: argparse.Namespace, anchor_year: int) -> pd.DataFrame:
    """Count each word form within each time window."""
    columns = [args.year_col, args.token_col, args.vowels_col, "token_source"]
    groups = []

    for chunk in pd.read_csv(
        input_csv,
        usecols=columns,
        chunksize=args.chunksize,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        prepared = prepare_chunk(chunk, args, anchor_year)
        if prepared.empty:
            continue

        group_columns = [
            "time_window_start", "time_window_end", "time_window_label",
            "time_window_id", "wordform_id", args.token_col, "vowel_seq"
        ]
        groups.append(
            prepared.groupby(group_columns, dropna=False).agg(
                token_count_in_window=("wordform_id", "size"),
                first_observed_year_in_window=(args.year_col, "min"),
                last_observed_year_in_window=(args.year_col, "max"),
                num_vowels_normalized=("num_vowels_normalized", "first"),
                token_sources_present=("token_source", lambda values: "|".join(sorted(set(values.astype(str))))),
                contains_idu_derived_observation=("token_source", lambda values: any("idu" in str(v).lower() for v in values)),
            ).reset_index()
        )

    if not groups:
        raise ValueError("No usable word forms remain after preprocessing.")

    group_columns = [
        "time_window_start", "time_window_end", "time_window_label",
        "time_window_id", "wordform_id", args.token_col, "vowel_seq"
    ]
    return (
        pd.concat(groups, ignore_index=True)
        .groupby(group_columns, dropna=False)
        .agg(
            token_count_in_window=("token_count_in_window", "sum"),
            first_observed_year_in_window=("first_observed_year_in_window", "min"),
            last_observed_year_in_window=("last_observed_year_in_window", "max"),
            num_vowels_normalized=("num_vowels_normalized", "first"),
            token_sources_present=("token_sources_present", lambda values: "|".join(sorted(set("|".join(values).split("|"))))),
            contains_idu_derived_observation=("contains_idu_derived_observation", "max"),
        )
        .reset_index()
    )


def make_window_diagnostics(wordforms: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Decide which time windows can enter the analysis sample."""
    diagnostics = (
        wordforms.groupby(
            ["time_window_start", "time_window_end", "time_window_label", "time_window_id"],
            as_index=False,
        )
        .agg(
            available_wordforms_before_balancing=("wordform_id", "nunique"),
            total_token_count_represented_before_balancing=("token_count_in_window", "sum"),
        )
    )

    diagnostics["sampling_mode"] = args.sampling_mode
    diagnostics["target_wordforms_per_window"] = args.target_wordforms_per_window
    diagnostics["min_window_wordforms_for_warning"] = args.min_window_wordforms
    diagnostics["min_window_tokens"] = args.min_window_tokens
    diagnostics["max_wordforms_per_window"] = args.max_wordforms_per_window
    diagnostics["window_width_years"] = args.window_width
    diagnostics["window_step_years"] = args.window_step
    diagnostics["window_overlap_years"] = max(0, args.window_width - args.window_step)

    diagnostics["time_window_is_low_n_warning"] = (
        diagnostics["available_wordforms_before_balancing"] < args.min_window_wordforms
    )
    diagnostics["time_window_is_eligible_for_sampling"] = (
        diagnostics["total_token_count_represented_before_balancing"] >= args.min_window_tokens
    )

    if args.exclude_windows:
        diagnostics.loc[
            diagnostics["time_window_label"].isin(args.exclude_windows),
            "time_window_is_eligible_for_sampling",
        ] = False

    diagnostics["time_window_exclusion_reason"] = "INCLUDED"
    diagnostics.loc[
        diagnostics["total_token_count_represented_before_balancing"] < args.min_window_tokens,
        "time_window_exclusion_reason",
    ] = "BELOW_MIN_TOKENS"
    diagnostics.loc[
        diagnostics["time_window_label"].isin(args.exclude_windows),
        "time_window_exclusion_reason",
    ] = "MANUALLY_EXCLUDED"
    return diagnostics


def select_wordforms(wordforms: pd.DataFrame, diagnostics: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the highest-frequency word forms for each eligible time window."""
    eligible = diagnostics[diagnostics["time_window_is_eligible_for_sampling"]].copy()

    if eligible.empty:
        raise ValueError("No time windows are eligible for sampling.")

    if args.sampling_mode == "strict-balanced":
        selected_n = int(eligible["available_wordforms_before_balancing"].min())
        if args.max_wordforms_per_window is not None:
            selected_n = min(selected_n, args.max_wordforms_per_window)
    else:
        selected_n = args.target_wordforms_per_window

    diagnostics["selected_sample_size_for_window"] = diagnostics.apply(
        lambda row: min(int(row["available_wordforms_before_balancing"]), selected_n)
        if row["time_window_is_eligible_for_sampling"] else 0,
        axis=1,
    )

    selected = wordforms.merge(
        diagnostics[[
            "time_window_label", "time_window_is_eligible_for_sampling",
            "time_window_is_low_n_warning", "time_window_exclusion_reason",
            "sampling_mode", "target_wordforms_per_window", "selected_sample_size_for_window",
        ]],
        on="time_window_label",
        how="left",
    )
    selected = selected[selected["time_window_is_eligible_for_sampling"]].copy()
    selected = selected.sort_values(
        ["time_window_start", "token_count_in_window", "first_observed_year_in_window", "token", "vowel_seq"],
        ascending=[True, False, True, True, True],
    )
    selected["frequency_rank_in_window"] = selected.groupby("time_window_id").cumcount() + 1
    selected["sample_size_per_time_window"] = selected["selected_sample_size_for_window"]
    selected = selected[
        selected["frequency_rank_in_window"] <= selected["selected_sample_size_for_window"]
    ].copy()
    return selected, diagnostics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(selected: pd.DataFrame, diagnostics: pd.DataFrame, output_dir: Path) -> None:
    """Write the files needed for auditing and later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    selected.to_csv(output_dir / "sampled_time_window_wordforms.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "all_time_window_diagnostics_before_sampling.csv", index=False, encoding="utf-8-sig")
    diagnostics[~diagnostics["time_window_is_eligible_for_sampling"]].to_csv(
        output_dir / "excluded_time_windows_due_to_low_sample.csv", index=False, encoding="utf-8-sig"
    )

    summary = diagnostics.merge(
        selected.groupby(
            ["time_window_start", "time_window_end", "time_window_label", "time_window_id"],
            as_index=False,
        ).agg(
            selected_wordforms_after_sampling=("wordform_id", "nunique"),
            tokens_in_selected_wordforms=("token_count_in_window", "sum"),
            first_selected_year=("first_observed_year_in_window", "min"),
            last_selected_year=("last_observed_year_in_window", "max"),
        ),
        on=["time_window_start", "time_window_end", "time_window_label", "time_window_id"],
        how="left",
    )
    summary.to_csv(output_dir / "time_window_summary.csv", index=False, encoding="utf-8-sig")


def write_selected_token_rows(
    input_csv: Path,
    selected: pd.DataFrame,
    args: argparse.Namespace,
    anchor_year: int,
    output_dir: Path,
) -> None:
    """Write the original token rows represented by the selected word forms."""
    keys = set(
        selected["time_window_label"].astype(str) + "\x1e" + selected["wordform_id"].astype(str)
    )
    output = output_dir / "sampled_time_window_original_token_rows.csv"
    wrote_header = False

    columns = None
    for chunk in pd.read_csv(input_csv, chunksize=args.chunksize, encoding="utf-8-sig", low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
        prepared = prepare_chunk(chunk, args, anchor_year)
        if prepared.empty:
            continue

        prepared["selection_key"] = (
            prepared["time_window_label"].astype(str) + "\x1e" + prepared["wordform_id"].astype(str)
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

    anchor_year = choose_anchor_year(input_csv, args.year_col, args.start_year, args.chunksize)
    print(f"Window: {args.window_width} years; step: {args.window_step} years")
    print(f"Anchor year: {anchor_year}")

    wordforms = count_wordforms(input_csv, args, anchor_year)
    diagnostics = make_window_diagnostics(wordforms, args)
    selected, diagnostics = select_wordforms(wordforms, diagnostics, args)

    write_outputs(selected, diagnostics, output_dir)
    if not args.skip_token_row_output:
        write_selected_token_rows(input_csv, selected, args, anchor_year, output_dir)
    write_run_config(output_dir, args, anchor_year)

    print(f"Selected {len(selected):,} word forms across {selected['time_window_id'].nunique():,} time windows.")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
