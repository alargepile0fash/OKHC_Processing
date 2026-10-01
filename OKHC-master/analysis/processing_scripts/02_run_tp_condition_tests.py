#!/usr/bin/env python3
"""
02_run_tp_condition_tests.py

Purpose
-------
This script performs ONLY Tolerance Principle testing and descriptive sequence counting.
It assumes frequency filtering has already been completed by 01_build_balanced_period_sample.py.

The script reads:
    balanced_period_wordforms.csv

It writes TP results under multiple conditions:
1. No excluded vowels.
2. Excluding ㅡ, ㅣ, ㅜ from consideration, separately and in every combination.
3. Discarding word forms that contain ㅡ, ㅣ, ㅜ, separately and in every combination.

It computes results at two levels:
- word level: each word form is one TP candidate; the word is an exception if any evaluated
  vowel window inside it is disharmonic.
- candidate-vowel level: each non-initial vowel in the reduced sequence is one TP candidate;
  the candidate is an exception if it disagrees with the immediately preceding considered vowel.

It also writes sequence-count and heatmap-ready files:
- common reduced vowel-symbol sequences by period and condition;
- common reduced harmony-class sequences, e.g. dark-bright or neutral-bright-dark;
- pair-count matrices in long format for heatmaps similar in spirit to Yoon-style summaries.

No D2L analysis is performed anywhere in this script.
Origin/stratum fields are not read or used.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# =============================================================================
# CONFIG: edit these values before running if your theoretical coding differs.
# =============================================================================

DEFAULT_INPUT_CSV = Path("analysis/analyzed_data/tp_period_balanced/balanced_period_wordforms.csv")
DEFAULT_OUTPUT_DIR = Path("analysis/analyzed_data/tp_period_results")
PROJECT_ROOT_ENV_VAR = "OKHC_ROOT"

# Main columns expected from the balancing script.
PERIOD_COL = "period_label"
PERIOD_START_COL = "period_start"
TOKEN_COL = "token"
VOWEL_SEQ_COL = "vowel_seq"
TOKEN_COUNT_COL = "token_count_in_period"
WORDFORM_ID_COL = "wordform_id"

# Vowels that should be tested as excluded-from-consideration and discard-if-present conditions.
TEST_VOWELS = ["ㅡ", "ㅣ", "ㅜ"]

# Harmony-class coding for sequence summaries and TP comparisons.
# IMPORTANT: This is the main theoretical place to check before running.
# The default below follows a broad Korean-style bright/dark split, while treating ㅣ as neutral.
# If your analysis treats ㅡ and/or ㅜ as neutral/opaque for the relevant rule, change their values
# to "NEUTRAL" here before running.
VOWEL_CLASS_MAP = {
    "ㆍ": "BRIGHT",
    "ㅏ": "BRIGHT",
    "ㅐ": "BRIGHT",
    "ㅑ": "BRIGHT",
    "ㅒ": "BRIGHT",
    "ㅗ": "BRIGHT",
    "ㅘ": "BRIGHT",
    "ㅙ": "BRIGHT",
    "ㅚ": "BRIGHT",
    "ㅛ": "BRIGHT",
    "ㅓ": "DARK",
    "ㅔ": "DARK",
    "ㅕ": "DARK",
    "ㅖ": "DARK",
    "ㅜ": "DARK",
    "ㅝ": "DARK",
    "ㅞ": "DARK",
    "ㅟ": "DARK",
    "ㅠ": "DARK",
    "ㅡ": "DARK",
    "ㅢ": "DARK",
    "ㅣ": "NEUTRAL",
}

# Only these classes are treated as ordinary bright/dark harmony values.
CONTRASTIVE_CLASSES = {"BRIGHT", "DARK"}

# How to handle candidate windows involving NEUTRAL when that vowel has NOT been excluded.
# True  = count neutral-involving windows as exceptions, making the no-exclusion baseline stricter.
# False = skip neutral-involving windows as not evaluable.
# For your current goal, True makes it possible to directly see whether excluding neutral/opaque
# vowels improves TP performance.
COUNT_NEUTRAL_WINDOWS_AS_EXCEPTIONS = True

# How to handle unknown vowel symbols. Unknowns are reported in the detailed output.
# True  = skip windows involving unknown vowel classes.
# False = count unknown-involving windows as exceptions.
SKIP_UNKNOWN_WINDOWS = True

# Minimum reduced sequence length needed for a word form to be testable.
MIN_REDUCED_VOWELS_FOR_TEST = 2

# If True, descriptive counts also include token-frequency-weighted counts in addition to
# unweighted word-form counts. TP calculations remain unweighted by default because TP is
# normally interpreted over lexical candidates, not token occurrences.
INCLUDE_TOKEN_WEIGHTED_DESCRIPTIVES = True


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class Condition:
    """A single TP condition."""

    condition_id: str
    condition_family: str
    excluded_vowels: tuple[str, ...]
    description: str




# =============================================================================
# PyCharm/project-root path helpers
# =============================================================================


def looks_like_project_root(path: Path) -> bool:
    """
    Return True when a directory looks like the OKHC project root.

    This script is usually stored at:
        OKHC-master/analysis/processing_scripts/01_build_balanced_period_sample.py

    PyCharm may run the script with a working directory that is NOT OKHC-master.
    Therefore, defaults should be resolved relative to the detected project root,
    not relative to the current terminal/PyCharm working directory.
    """
    return (
        path.is_dir()
        and (path / "analysis" / "analyzed_data").exists()
        and (path / "analysis" / "processing_scripts").exists()
    )


def iter_candidate_roots() -> Iterable[Path]:
    """Yield plausible project-root directories, most reliable first."""
    env_value = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_value:
        yield Path(env_value).expanduser()

    script_path = Path(__file__).resolve()
    yield from script_path.parents

    cwd = Path.cwd().resolve()
    yield cwd
    yield from cwd.parents


def find_project_root(project_root_arg: Optional[Path] = None) -> Path:
    """
    Find the OKHC project root in a way that works from PyCharm or a terminal.

    Priority:
    1. --project-root, if supplied.
    2. OKHC_ROOT environment variable, if supplied.
    3. A parent of this script that contains analysis/analyzed_data and
       analysis/processing_scripts.
    4. A parent of the current working directory with the same structure.
    5. Fallback: two levels above this script when it is inside
       analysis/processing_scripts.
    6. Final fallback: current working directory.
    """
    if project_root_arg is not None:
        root = project_root_arg.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Project root does not exist: {root}")
        return root

    seen: set[Path] = set()
    for candidate in iter_candidate_roots():
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if looks_like_project_root(candidate):
            return candidate

    # Useful fallback for the normal script location even before analyzed_data exists.
    script_path = Path(__file__).resolve()
    if (
        script_path.parent.name == "processing_scripts"
        and script_path.parent.parent.name == "analysis"
    ):
        return script_path.parent.parent.parent

    return Path.cwd().resolve()


def resolve_project_path(path: Path, project_root: Path) -> Path:
    """
    Resolve a path the way users usually expect inside a PyCharm project.

    Absolute paths are left alone. Relative paths are interpreted relative to
    the detected project root instead of PyCharm's current working directory.
    """
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


# =============================================================================
# Argument parsing and basic utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Read command-line options while keeping sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Run TP condition tests on a pre-balanced configured-period word-form CSV."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Root of the OKHC-master project. Usually not needed: the script tries "
            "to infer this from its own location in analysis/processing_scripts."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Balanced word-form CSV from script 01. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for TP outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--level",
        choices=["both", "word", "candidate"],
        default="both",
        help="Which TP level(s) to compute. Default: both.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    """Fail early if the balanced CSV does not have the columns this script needs."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing) +
            "\nAvailable columns are: " + ", ".join(df.columns)
        )


def parse_vowel_sequence(value: object) -> list[str]:
    """
    Convert the normalized vowel_seq string from script 01 back into a list.

    Expected format is space-separated, e.g. "ㅏ ㅗ ㅣ".
    A fallback regex is included in case the sequence arrives without spaces.
    """
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if re.search(r"[\s,;/|]+", text):
        return [x for x in re.split(r"[\s,;/|]+", text) if x]
    # Fallback: extract any known vowel symbols from the compact string.
    symbols = sorted((re.escape(v) for v in VOWEL_CLASS_MAP), key=len, reverse=True)
    return re.findall("|".join(symbols), text)


def make_conditions() -> list[Condition]:
    """
    Build the full condition list requested by the user.

    Conditions include:
    - one baseline with no excluded vowels;
    - all non-empty subsets of TEST_VOWELS as excluded-from-consideration;
    - all non-empty subsets of TEST_VOWELS as discard-if-present.
    """
    conditions: list[Condition] = [
        Condition(
            condition_id="baseline_no_excluded_vowels",
            condition_family="baseline",
            excluded_vowels=tuple(),
            description="No vowels are excluded or discarded; all vowel windows are evaluated under the configured class map.",
        )
    ]

    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(TEST_VOWELS) + 1):
        subsets.extend(tuple(combo) for combo in itertools.combinations(TEST_VOWELS, r))

    for subset in subsets:
        label = "".join(subset)
        conditions.append(
            Condition(
                condition_id=f"exclude_from_consideration_{label}",
                condition_family="exclude_from_consideration",
                excluded_vowels=subset,
                description=(
                    "Keep word forms, but remove " + ", ".join(subset) +
                    " from the vowel sequence before evaluating harmony."
                ),
            )
        )

    for subset in subsets:
        label = "".join(subset)
        conditions.append(
            Condition(
                condition_id=f"discard_wordforms_containing_{label}",
                condition_family="discard_wordforms_containing",
                excluded_vowels=subset,
                description=(
                    "Discard any word form containing at least one of: " + ", ".join(subset) +
                    ". Remaining word forms are evaluated without removing additional vowels."
                ),
            )
        )

    return conditions


def apply_condition_to_sequence(seq: list[str], condition: Condition) -> Optional[list[str]]:
    """
    Return the reduced sequence for this condition.

    For discard_wordforms_containing conditions, None means the whole word form is removed.
    For exclude_from_consideration conditions, only the listed vowels are removed from the sequence.
    """
    excluded = set(condition.excluded_vowels)

    if condition.condition_family == "discard_wordforms_containing":
        if any(v in excluded for v in seq):
            return None
        return list(seq)

    if condition.condition_family == "exclude_from_consideration":
        return [v for v in seq if v not in excluded]

    # Baseline: no change.
    return list(seq)


def classify_vowel(vowel: str) -> str:
    """Return BRIGHT, DARK, NEUTRAL, or UNKNOWN for a vowel symbol."""
    return VOWEL_CLASS_MAP.get(vowel, "UNKNOWN")


def readable_class_sequence(seq: list[str]) -> str:
    """Return labels such as bright-dark-neutral for descriptive tables."""
    if not seq:
        return "EMPTY"
    return "-".join(classify_vowel(v).lower() for v in seq)


def readable_vowel_sequence(seq: list[str]) -> str:
    """Return symbols such as ㅏ-ㅗ-ㅣ for descriptive tables."""
    if not seq:
        return "EMPTY"
    return "-".join(seq)


def evaluate_pair(prev_vowel: str, target_vowel: str) -> dict[str, object]:
    """
    Evaluate one adjacent vowel window.

    The candidate-vowel interpretation is:
    - prev_vowel is the immediately preceding considered vowel;
    - target_vowel is the non-initial candidate vowel being tested;
    - the window is harmonic if both vowels have the same contrastive class;
    - the window is an exception if they disagree, or if neutral windows are configured
      to count as exceptions.

    The function returns a dictionary rather than a boolean so the detailed output can
    explain why every window was counted, skipped, or treated as an exception.
    """
    prev_class = classify_vowel(prev_vowel)
    target_class = classify_vowel(target_vowel)

    result = {
        "prev_vowel": prev_vowel,
        "target_vowel": target_vowel,
        "prev_class": prev_class,
        "target_class": target_class,
        "candidate_is_evaluable": True,
        "candidate_is_exception": False,
        "candidate_status": "harmonic",
        "candidate_status_description": "Both vowels share the same contrastive harmony class.",
    }

    if "UNKNOWN" in {prev_class, target_class}:
        if SKIP_UNKNOWN_WINDOWS:
            result.update(
                candidate_is_evaluable=False,
                candidate_is_exception=False,
                candidate_status="skipped_unknown_class",
                candidate_status_description="Skipped because at least one vowel has UNKNOWN class in VOWEL_CLASS_MAP.",
            )
        else:
            result.update(
                candidate_is_exception=True,
                candidate_status="exception_unknown_class",
                candidate_status_description="Counted as exception because at least one vowel has UNKNOWN class.",
            )
        return result

    if "NEUTRAL" in {prev_class, target_class}:
        if COUNT_NEUTRAL_WINDOWS_AS_EXCEPTIONS:
            result.update(
                candidate_is_exception=True,
                candidate_status="exception_neutral_window",
                candidate_status_description="Counted as exception because neutral-involving windows are not excluded in this condition.",
            )
        else:
            result.update(
                candidate_is_evaluable=False,
                candidate_is_exception=False,
                candidate_status="skipped_neutral_window",
                candidate_status_description="Skipped because neutral-involving windows are configured as not evaluable.",
            )
        return result

    if prev_class in CONTRASTIVE_CLASSES and target_class in CONTRASTIVE_CLASSES:
        if prev_class == target_class:
            return result
        result.update(
            candidate_is_exception=True,
            candidate_status="exception_bright_dark_mismatch",
            candidate_status_description="Counted as exception because adjacent considered vowels disagree in BRIGHT/DARK class.",
        )
        return result

    # Any remaining class combination is not expected, so make it explicit.
    result.update(
        candidate_is_evaluable=False,
        candidate_is_exception=False,
        candidate_status="skipped_unhandled_class_combination",
        candidate_status_description="Skipped because the class combination is not handled by the configured TP rule.",
    )
    return result


def tolerance_threshold(n: int) -> float:
    """
    Return theta_N = N / ln(N).

    TP is not meaningful for N < 2, because ln(1)=0 and a one-item domain cannot provide
    stable evidence of productivity. The caller handles N < 2 separately.
    """
    return n / math.log(n)


def summarize_tp(n: int, e: int) -> dict[str, object]:
    """Return TP summary statistics for a candidate domain."""
    if n < 2:
        return {
            "N_candidates": n,
            "exceptions_e": e,
            "exception_rate_e_over_N": None,
            "theta_N_float": None,
            "theta_N_integer_floor": None,
            "tolerated_exception_rate_theta_over_N": None,
            "tp_passes": False,
            "tp_result_description": "Not testable: N < 2 provides too little evidence for a TP calculation.",
        }

    theta = tolerance_threshold(n)
    passes = e <= theta
    return {
        "N_candidates": n,
        "exceptions_e": e,
        "exception_rate_e_over_N": e / n,
        "theta_N_float": theta,
        "theta_N_integer_floor": math.floor(theta),
        "tolerated_exception_rate_theta_over_N": theta / n,
        "tp_passes": passes,
        "tp_result_description": (
            "PASS: exceptions_e <= theta_N." if passes else "FAIL: exceptions_e > theta_N."
        ),
    }


# =============================================================================
# Core analysis functions
# =============================================================================


def build_condition_dataframe(df: pd.DataFrame, condition: Condition) -> pd.DataFrame:
    """
    Apply one condition to every word form and return an expanded dataframe.

    This output has one row per original word form, with reduced sequences added.
    Rows discarded by the condition remain present but are marked as discarded so the
    summary files can report how much data each condition removes.
    """
    rows = []
    for row in df.itertuples(index=False):
        original_seq = parse_vowel_sequence(getattr(row, VOWEL_SEQ_COL))
        reduced_seq = apply_condition_to_sequence(original_seq, condition)
        discarded = reduced_seq is None
        if reduced_seq is None:
            reduced_seq = []

        rows.append(
            {
                "period_start": getattr(row, PERIOD_START_COL),
                "period_label": getattr(row, PERIOD_COL),
                "wordform_id": getattr(row, WORDFORM_ID_COL),
                "token": getattr(row, TOKEN_COL),
                "token_count_in_period": getattr(row, TOKEN_COUNT_COL),
                "condition_id": condition.condition_id,
                "condition_family": condition.condition_family,
                "condition_excluded_vowels": "+".join(condition.excluded_vowels) if condition.excluded_vowels else "NONE",
                "condition_description": condition.description,
                "original_vowel_sequence": readable_vowel_sequence(original_seq),
                "original_class_sequence": readable_class_sequence(original_seq),
                "reduced_vowel_sequence": readable_vowel_sequence(reduced_seq),
                "reduced_class_sequence": readable_class_sequence(reduced_seq),
                "original_num_vowels": len(original_seq),
                "reduced_num_vowels": len(reduced_seq),
                "wordform_discarded_by_condition": discarded,
                "wordform_has_enough_reduced_vowels": len(reduced_seq) >= MIN_REDUCED_VOWELS_FOR_TEST,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_windows(condition_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand reduced word-form sequences into candidate-vowel windows.

    Each row is one non-initial vowel target in a reduced sequence.
    For sequence A-B-C, this creates two candidate windows:
    - A -> B
    - B -> C
    """
    windows = []
    for row in condition_df.itertuples(index=False):
        if row.wordform_discarded_by_condition or not row.wordform_has_enough_reduced_vowels:
            continue

        seq = [] if row.reduced_vowel_sequence == "EMPTY" else str(row.reduced_vowel_sequence).split("-")
        for idx in range(1, len(seq)):
            prev_vowel = seq[idx - 1]
            target_vowel = seq[idx]
            eval_result = evaluate_pair(prev_vowel, target_vowel)
            windows.append(
                {
                    "period_start": row.period_start,
                    "period_label": row.period_label,
                    "condition_id": row.condition_id,
                    "condition_family": row.condition_family,
                    "condition_excluded_vowels": row.condition_excluded_vowels,
                    "wordform_id": row.wordform_id,
                    "token": row.token,
                    "token_count_in_period": row.token_count_in_period,
                    "original_vowel_sequence": row.original_vowel_sequence,
                    "reduced_vowel_sequence": row.reduced_vowel_sequence,
                    "reduced_class_sequence": row.reduced_class_sequence,
                    "candidate_window_index": idx,
                    "candidate_window_description": "previous considered vowel -> non-initial candidate vowel",
                    **eval_result,
                }
            )
    return pd.DataFrame(windows)


def summarize_candidate_level(candidate_windows: pd.DataFrame) -> pd.DataFrame:
    """Compute TP summaries where each candidate vowel/window is one TP candidate."""
    rows = []
    group_cols = ["period_start", "period_label", "condition_id", "condition_family", "condition_excluded_vowels"]

    for keys, group in candidate_windows.groupby(group_cols, dropna=False):
        period_start, period, condition_id, family, excluded = keys
        evaluable = group[group["candidate_is_evaluable"]]
        n = int(len(evaluable))
        e = int(evaluable["candidate_is_exception"].sum())
        summary = summarize_tp(n, e)
        rows.append(
            {
                "period_start": period_start,
                "period_label": period,
                "analysis_level": "candidate_vowel_level",
                "analysis_level_description": "Each adjacent considered vowel window is one TP candidate; exceptions are disharmonic windows.",
                "condition_id": condition_id,
                "condition_family": family,
                "condition_excluded_vowels": excluded,
                "candidate_windows_total_before_skipping": int(len(group)),
                "candidate_windows_skipped": int((~group["candidate_is_evaluable"]).sum()),
                **summary,
            }
        )
    return pd.DataFrame(rows).sort_values(["period_start", "condition_id"])


def summarize_word_level(condition_df: pd.DataFrame, candidate_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TP summaries where each word form is one TP candidate.

    A word form counts as an exception if at least one evaluable candidate window inside it
    is an exception. This gives a stricter word-level view than candidate-vowel-level testing.
    """
    rows = []
    group_cols = ["period_start", "period_label", "condition_id", "condition_family", "condition_excluded_vowels"]

    # First create a word-level table of exception status from the candidate-window table.
    if candidate_windows.empty:
        word_status = pd.DataFrame(
            columns=group_cols + ["wordform_id", "wordform_has_evaluable_window", "wordform_is_exception"]
        )
    else:
        evaluable = candidate_windows[candidate_windows["candidate_is_evaluable"]].copy()
        if evaluable.empty:
            word_status = pd.DataFrame(
                columns=group_cols + ["wordform_id", "wordform_has_evaluable_window", "wordform_is_exception"]
            )
        else:
            word_status = (
                evaluable.groupby(group_cols + ["wordform_id"], as_index=False)
                .agg(wordform_is_exception=("candidate_is_exception", "max"))
            )
            word_status["wordform_has_evaluable_window"] = True

    for keys, group in condition_df.groupby(group_cols, dropna=False):
        period_start, period, condition_id, family, excluded = keys

        available_before_condition = int(len(group))
        discarded = int(group["wordform_discarded_by_condition"].sum())
        too_short = int((~group["wordform_discarded_by_condition"] & ~group["wordform_has_enough_reduced_vowels"]).sum())

        status_subset = word_status
        for col, value in zip(group_cols, keys):
            status_subset = status_subset[status_subset[col] == value]

        n = int(len(status_subset))
        e = int(status_subset["wordform_is_exception"].sum()) if n else 0
        summary = summarize_tp(n, e)

        rows.append(
            {
                "period_start": period_start,
                "period_label": period,
                "analysis_level": "word_level",
                "analysis_level_description": "Each selected word form is one TP candidate; a word is an exception if any evaluable window inside it is disharmonic.",
                "condition_id": condition_id,
                "condition_family": family,
                "condition_excluded_vowels": excluded,
                "wordforms_available_before_condition": available_before_condition,
                "wordforms_discarded_by_condition": discarded,
                "wordforms_too_short_after_condition": too_short,
                "wordforms_with_evaluable_windows": n,
                **summary,
            }
        )
    return pd.DataFrame(rows).sort_values(["period_start", "condition_id"])


def build_sequence_counts(all_condition_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Count the most common reduced vowel and class sequences by period and condition.

    This is useful for statements such as:
    - "dark-bright sequences become more common after period X";
    - "neutral-bright-dark sequences are concentrated in these periods";
    - "the apparent TP failure is driven by a small set of recurring sequence types."
    """
    usable = all_condition_rows[
        ~all_condition_rows["wordform_discarded_by_condition"]
        & all_condition_rows["wordform_has_enough_reduced_vowels"]
    ].copy()

    group_cols = [
        "period_start",
        "period_label",
        "condition_id",
        "condition_family",
        "condition_excluded_vowels",
        "reduced_vowel_sequence",
        "reduced_class_sequence",
    ]

    aggregations = {
        "wordform_count": ("wordform_id", "nunique"),
    }
    if INCLUDE_TOKEN_WEIGHTED_DESCRIPTIVES:
        aggregations["token_weighted_count"] = ("token_count_in_period", "sum")

    counts = usable.groupby(group_cols, as_index=False).agg(**aggregations)

    # Add within-period shares for easier graphing.
    denom = counts.groupby(["period_label", "condition_id"], as_index=False).agg(
        total_wordforms_in_period_condition=("wordform_count", "sum")
    )
    counts = counts.merge(denom, on=["period_label", "condition_id"], how="left")
    counts["share_of_wordforms_in_period_condition"] = (
        counts["wordform_count"] / counts["total_wordforms_in_period_condition"]
    )

    counts = counts.sort_values(
        ["period_start", "condition_id", "wordform_count", "reduced_class_sequence"],
        ascending=[True, True, False, True],
    )
    counts["rank_within_period_condition"] = counts.groupby(["period_label", "condition_id"]).cumcount() + 1
    return counts


def build_heatmap_pair_counts(candidate_windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build heatmap-ready pair-count tables.

    The class-pair table can produce heatmaps like BRIGHT->DARK, DARK->BRIGHT, etc.
    The symbol-pair table can produce finer vowel-to-vowel heatmaps like ㅏ->ㅓ.
    """
    evaluable_or_exceptions = candidate_windows.copy()
    if evaluable_or_exceptions.empty:
        empty_cols = [
            "period_start", "period_label", "condition_id", "condition_family", "condition_excluded_vowels",
            "prev_class", "target_class", "prev_vowel", "target_vowel", "candidate_count",
            "exception_count", "token_weighted_candidate_count", "token_weighted_exception_count",
        ]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

    # Class-pair counts.
    class_cols = [
        "period_start", "period_label", "condition_id", "condition_family", "condition_excluded_vowels",
        "prev_class", "target_class",
    ]
    class_counts = evaluable_or_exceptions.groupby(class_cols, as_index=False).agg(
        candidate_count=("candidate_is_evaluable", "sum"),
        exception_count=("candidate_is_exception", "sum"),
        token_weighted_candidate_count=(
            "token_count_in_period",
            lambda s: int(s[evaluable_or_exceptions.loc[s.index, "candidate_is_evaluable"]].sum()),
        ),
        token_weighted_exception_count=(
            "token_count_in_period",
            lambda s: int(s[evaluable_or_exceptions.loc[s.index, "candidate_is_exception"]].sum()),
        ),
    )

    # Symbol-pair counts.
    symbol_cols = [
        "period_start", "period_label", "condition_id", "condition_family", "condition_excluded_vowels",
        "prev_vowel", "target_vowel", "prev_class", "target_class",
    ]
    symbol_counts = evaluable_or_exceptions.groupby(symbol_cols, as_index=False).agg(
        candidate_count=("candidate_is_evaluable", "sum"),
        exception_count=("candidate_is_exception", "sum"),
        token_weighted_candidate_count=(
            "token_count_in_period",
            lambda s: int(s[evaluable_or_exceptions.loc[s.index, "candidate_is_evaluable"]].sum()),
        ),
        token_weighted_exception_count=(
            "token_count_in_period",
            lambda s: int(s[evaluable_or_exceptions.loc[s.index, "candidate_is_exception"]].sum()),
        ),
    )

    return class_counts, symbol_counts


def write_variable_descriptions(output_dir: Path) -> None:
    """Write a data dictionary for all TP-analysis outputs."""
    descriptions = [
        ("condition_id", "Unique name for the condition being tested."),
        ("condition_family", "baseline, exclude_from_consideration, or discard_wordforms_containing."),
        ("condition_excluded_vowels", "Vowels targeted by this condition; NONE for baseline."),
        ("condition_description", "Plain-language explanation of what the condition does."),
        ("analysis_level", "word_level or candidate_vowel_level."),
        ("analysis_level_description", "Plain-language explanation of what N means for that output."),
        ("N_candidates", "TP domain size N after the condition is applied and non-evaluable items are removed."),
        ("exceptions_e", "Number of TP exceptions e in that domain."),
        ("exception_rate_e_over_N", "Raw exception rate, e/N."),
        ("theta_N_float", "Tolerance threshold theta_N = N / ln(N)."),
        ("theta_N_integer_floor", "Largest whole-number exception count safely below theta_N."),
        ("tolerated_exception_rate_theta_over_N", "Threshold as a proportion of N; equals 1/ln(N)."),
        ("tp_passes", "TRUE if exceptions_e <= theta_N_float; FALSE otherwise."),
        ("tp_result_description", "Plain-language pass/fail/not-testable explanation."),
        ("wordforms_available_before_condition", "Number of balanced word forms in the period before applying the condition."),
        ("wordforms_discarded_by_condition", "Number of word forms removed by discard_wordforms_containing conditions."),
        ("wordforms_too_short_after_condition", "Number of non-discarded word forms with fewer than two reduced vowels."),
        ("wordforms_with_evaluable_windows", "Number of word forms contributing at least one evaluable vowel window."),
        ("candidate_windows_total_before_skipping", "Number of adjacent reduced-vowel windows before unknown/neutral skipping rules are applied."),
        ("candidate_windows_skipped", "Number of candidate windows skipped because they were configured as not evaluable."),
        ("original_vowel_sequence", "Vowel sequence before the condition is applied."),
        ("reduced_vowel_sequence", "Vowel sequence after exclusion/discard condition is applied."),
        ("original_class_sequence", "Harmony-class sequence before the condition is applied."),
        ("reduced_class_sequence", "Harmony-class sequence after the condition is applied, e.g. dark-bright."),
        ("candidate_window_index", "Position of the non-initial target vowel in the reduced sequence."),
        ("candidate_window_description", "previous considered vowel -> non-initial candidate vowel."),
        ("prev_vowel", "Immediately preceding considered vowel in the reduced sequence."),
        ("target_vowel", "Non-initial candidate vowel being evaluated."),
        ("prev_class", "Harmony class of prev_vowel under VOWEL_CLASS_MAP."),
        ("target_class", "Harmony class of target_vowel under VOWEL_CLASS_MAP."),
        ("candidate_is_evaluable", "TRUE if this window contributes to TP N."),
        ("candidate_is_exception", "TRUE if this evaluable window is counted as a TP exception."),
        ("candidate_status", "Compact label explaining harmonic/exception/skipped status."),
        ("candidate_status_description", "Plain-language explanation of the candidate-status label."),
        ("wordform_count", "Unweighted count of word forms with a given reduced sequence."),
        ("token_weighted_count", "Sum of token frequencies for word forms with a given reduced sequence."),
        ("share_of_wordforms_in_period_condition", "Within-period share of word forms with that reduced sequence under that condition."),
        ("rank_within_period_condition", "Frequency rank of the sequence within the period and condition."),
        ("candidate_count", "Number of candidate windows for a pair in heatmap-ready output."),
        ("exception_count", "Number of exception windows for a pair in heatmap-ready output."),
        ("token_weighted_candidate_count", "Token-frequency-weighted candidate count for heatmap-ready output."),
        ("token_weighted_exception_count", "Token-frequency-weighted exception count for heatmap-ready output."),
    ]
    pd.DataFrame(descriptions, columns=["variable", "description"]).to_csv(
        output_dir / "02_tp_output_variable_descriptions.csv", index=False, encoding="utf-8-sig"
    )


def write_configuration_report(output_dir: Path, conditions: list[Condition]) -> None:
    """Write a machine-readable summary of theoretical/configuration settings used."""
    config_rows = [
        ("TEST_VOWELS", "+".join(TEST_VOWELS), "Vowels tested in exclusion and discard conditions."),
        ("CONTRASTIVE_CLASSES", "+".join(sorted(CONTRASTIVE_CLASSES)), "Classes treated as ordinary harmony values."),
        ("COUNT_NEUTRAL_WINDOWS_AS_EXCEPTIONS", str(COUNT_NEUTRAL_WINDOWS_AS_EXCEPTIONS), "Whether neutral-involving windows count as exceptions when not excluded."),
        ("SKIP_UNKNOWN_WINDOWS", str(SKIP_UNKNOWN_WINDOWS), "Whether windows with UNKNOWN vowel classes are skipped."),
        ("MIN_REDUCED_VOWELS_FOR_TEST", str(MIN_REDUCED_VOWELS_FOR_TEST), "Minimum reduced vowels needed for a word form to be testable."),
    ]
    for vowel, cls in sorted(VOWEL_CLASS_MAP.items()):
        config_rows.append((f"VOWEL_CLASS_MAP[{vowel}]", cls, "Harmony class assigned to this vowel symbol."))

    pd.DataFrame(config_rows, columns=["setting", "value", "description"]).to_csv(
        output_dir / "02_tp_configuration_report.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        [
            {
                "condition_id": c.condition_id,
                "condition_family": c.condition_family,
                "condition_excluded_vowels": "+".join(c.excluded_vowels) if c.excluded_vowels else "NONE",
                "condition_description": c.description,
            }
            for c in conditions
        ]
    ).to_csv(output_dir / "02_tp_conditions_tested.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# Main execution
# =============================================================================


def main() -> None:
    args = parse_args()

    # Resolve default/relative paths relative to the OKHC project root, not relative
    # to whatever working directory PyCharm happens to choose. This lets the script
    # find the CSV produced by script 01 when run directly from PyCharm.
    project_root = find_project_root(args.project_root)
    args.input = resolve_project_path(args.input, project_root)
    args.output_dir = resolve_project_path(args.output_dir, project_root)

    print(f"Detected project root: {project_root}")
    print(f"Resolved input CSV: {args.input}")
    print(f"Resolved output directory: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        raise FileNotFoundError(
            "Input CSV not found after resolving paths.\n"
            f"  Project root: {project_root}\n"
            f"  Input path:   {args.input}\n"
            "Fix options:\n"
            "  1. Run 01_build_balanced_period_sample.py first, or\n"
            "  2. Pass --project-root C:/Users/ashle/PycharmProjects/OKHC_Processing/OKHC-master, or\n"
            "  3. Pass --input with the full absolute balanced_period_wordforms.csv path."
        )

    print(f"Reading balanced word-form sample: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    require_columns(
        df,
        [PERIOD_COL, PERIOD_START_COL, TOKEN_COL, VOWEL_SEQ_COL, TOKEN_COUNT_COL, WORDFORM_ID_COL],
    )
    print(f"Loaded {len(df):,} balanced word-form rows.")

    conditions = make_conditions()
    print(f"Testing {len(conditions)} conditions: baseline + exclusions + discard conditions.")

    condition_frames = []
    candidate_frames = []

    for condition in conditions:
        condition_df = build_condition_dataframe(df, condition)
        windows_df = build_candidate_windows(condition_df)
        condition_frames.append(condition_df)
        candidate_frames.append(windows_df)

    all_condition_rows = pd.concat(condition_frames, ignore_index=True)
    all_candidate_windows = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()

    # -------------------------------------------------------------------------
    # TP summaries.
    # -------------------------------------------------------------------------
    if args.level in {"both", "word"}:
        word_summary = summarize_word_level(all_condition_rows, all_candidate_windows)
        word_summary.to_csv(args.output_dir / "tp_summary_word_level.csv", index=False, encoding="utf-8-sig")
        print(f"Wrote word-level TP summary: {args.output_dir / 'tp_summary_word_level.csv'}")

    if args.level in {"both", "candidate"}:
        candidate_summary = summarize_candidate_level(all_candidate_windows)
        candidate_summary.to_csv(args.output_dir / "tp_summary_candidate_vowel_level.csv", index=False, encoding="utf-8-sig")
        print(f"Wrote candidate-vowel-level TP summary: {args.output_dir / 'tp_summary_candidate_vowel_level.csv'}")

    # -------------------------------------------------------------------------
    # Detailed and descriptive outputs.
    # -------------------------------------------------------------------------
    all_condition_rows.to_csv(
        args.output_dir / "tp_condition_wordform_rows.csv", index=False, encoding="utf-8-sig"
    )
    all_candidate_windows.to_csv(
        args.output_dir / "tp_candidate_vowel_windows.csv", index=False, encoding="utf-8-sig"
    )

    sequence_counts = build_sequence_counts(all_condition_rows)
    sequence_counts.to_csv(
        args.output_dir / "sequence_counts_by_period_condition.csv", index=False, encoding="utf-8-sig"
    )

    class_pair_counts, symbol_pair_counts = build_heatmap_pair_counts(all_candidate_windows)
    class_pair_counts.to_csv(
        args.output_dir / "heatmap_class_pair_counts_long.csv", index=False, encoding="utf-8-sig"
    )
    symbol_pair_counts.to_csv(
        args.output_dir / "heatmap_vowel_symbol_pair_counts_long.csv", index=False, encoding="utf-8-sig"
    )

    write_variable_descriptions(args.output_dir)
    write_configuration_report(args.output_dir, conditions)

    print("\nWrote descriptive/audit outputs:")
    print(f"  - {args.output_dir / 'tp_condition_wordform_rows.csv'}")
    print(f"  - {args.output_dir / 'tp_candidate_vowel_windows.csv'}")
    print(f"  - {args.output_dir / 'sequence_counts_by_period_condition.csv'}")
    print(f"  - {args.output_dir / 'heatmap_class_pair_counts_long.csv'}")
    print(f"  - {args.output_dir / 'heatmap_vowel_symbol_pair_counts_long.csv'}")
    print(f"  - {args.output_dir / '02_tp_output_variable_descriptions.csv'}")
    print(f"  - {args.output_dir / '02_tp_configuration_report.csv'}")
    print(f"  - {args.output_dir / '02_tp_conditions_tested.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
