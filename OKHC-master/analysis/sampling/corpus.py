"""Prepare the extracted diachronic corpus for sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .windows import windows_for_year


def parse_vowel_sequence(value: object) -> list[str]:
    """Read the standardized comma-separated vowel sequence from the extractor."""
    if pd.isna(value):
        return []
    return [vowel for vowel in str(value).split(",") if vowel]


def prepare_chunk(chunk: pd.DataFrame, args: argparse.Namespace, anchor_year: int) -> pd.DataFrame:
    """Clean a corpus chunk and assign each row to its time window(s)."""
    chunk = chunk.copy()
    chunk[args.year_col] = pd.to_numeric(chunk[args.year_col], errors="coerce")
    chunk = chunk.dropna(subset=[args.year_col, args.token_col, args.vowels_col])

    if args.start_year is not None:
        chunk = chunk[chunk[args.year_col] >= args.start_year]
    if chunk.empty:
        return chunk

    chunk[args.year_col] = chunk[args.year_col].astype(int)
    chunk[args.token_col] = chunk[args.token_col].astype(str).str.strip()
    chunk = chunk[chunk[args.token_col] != ""]

    chunk["vowel_seq_list"] = chunk[args.vowels_col].apply(parse_vowel_sequence)
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
    chunk["time_window_id"] = (
        chunk["time_window_start"].astype(str) + "-" + chunk["time_window_end"].astype(str)
    )
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
            "time_window_start", "time_window_end", "time_window_id",
            "time_window_id", "wordform_id", args.token_col, "vowel_seq"
        ]
        groups.append(
            prepared.groupby(group_columns, dropna=False).agg(
                token_count_in_window=("wordform_id", "size"),
                first_observed_year_in_window=(args.year_col, "min"),
                last_observed_year_in_window=(args.year_col, "max"),
                num_vowels_normalized=("num_vowels_normalized", "first"),
                token_sources_present=("token_source", lambda values: "|".join(sorted(set(values.astype(str))))),
                contains_idu_derived_observation=(
                    "token_source",
                    lambda values: any("idu" in str(value).lower() for value in values),
                ),
            ).reset_index()
        )

    if not groups:
        raise ValueError("No usable word forms remain after preprocessing.")

    group_columns = [
        "time_window_start", "time_window_end", "time_window_id",
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
            token_sources_present=(
                "token_sources_present",
                lambda values: "|".join(sorted(set("|".join(values).split("|")))),
            ),
            contains_idu_derived_observation=("contains_idu_derived_observation", "max"),
        )
        .reset_index()
    )
