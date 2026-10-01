"""Select word forms for each historical time window."""

from __future__ import annotations

import argparse

import pandas as pd


def make_window_diagnostics(wordforms: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Decide which time windows can enter the analysis sample."""
    diagnostics = (
        wordforms.groupby(
            ["time_window_start", "time_window_end", "time_window_id", "time_window_id"],
            as_index=False,
        )
        .agg(
            available_wordforms_before_sampling=("wordform_id", "nunique"),
            total_token_count_represented_before_sampling=("token_count_in_window", "sum"),
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
        diagnostics["available_wordforms_before_sampling"] < args.min_window_wordforms
    )
    diagnostics["time_window_is_eligible_for_sampling"] = True

    if args.sampling_mode == "strict-balanced":
        diagnostics["time_window_is_eligible_for_sampling"] &= (
            diagnostics["available_wordforms_before_sampling"] >= args.min_window_wordforms
        )

    diagnostics["time_window_is_eligible_for_sampling"] &= (
        diagnostics["total_token_count_represented_before_sampling"] >= args.min_window_tokens
    )

    if args.exclude_windows:
        diagnostics.loc[
            diagnostics["time_window_id"].isin(args.exclude_windows),
            "time_window_is_eligible_for_sampling",
        ] = False

    diagnostics["time_window_exclusion_reason"] = "INCLUDED"
    diagnostics.loc[
        diagnostics["sampling_mode"].eq("strict-balanced")
        & diagnostics["time_window_is_low_n_warning"],
        "time_window_exclusion_reason",
    ] = "BELOW_MIN_WORDFORMS"
    diagnostics.loc[
        diagnostics["total_token_count_represented_before_sampling"] < args.min_window_tokens,
        "time_window_exclusion_reason",
    ] = "BELOW_MIN_TOKENS"
    diagnostics.loc[
        diagnostics["time_window_id"].isin(args.exclude_windows),
        "time_window_exclusion_reason",
    ] = "MANUALLY_EXCLUDED"
    return diagnostics


def select_wordforms(
    wordforms: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the highest-frequency word forms for each eligible time window."""
    eligible = diagnostics[diagnostics["time_window_is_eligible_for_sampling"]].copy()
    if eligible.empty:
        raise ValueError("No time windows are eligible for sampling.")

    if args.sampling_mode == "strict-balanced":
        selected_n = int(eligible["available_wordforms_before_sampling"].min())
        if args.max_wordforms_per_window is not None:
            selected_n = min(selected_n, args.max_wordforms_per_window)
    else:
        selected_n = args.target_wordforms_per_window

    diagnostics["selected_sample_size_for_window"] = diagnostics.apply(
        lambda row: min(int(row["available_wordforms_before_sampling"]), selected_n)
        if row["time_window_is_eligible_for_sampling"] else 0,
        axis=1,
    )

    selected = wordforms.merge(
        diagnostics[[
            "time_window_id", "time_window_is_eligible_for_sampling",
            "time_window_is_low_n_warning", "time_window_exclusion_reason",
            "sampling_mode", "target_wordforms_per_window", "selected_sample_size_for_window",
        ]],
        on="time_window_id",
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
