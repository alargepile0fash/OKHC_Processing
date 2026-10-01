
"""
03_generate_tp_window_exception_rate_graphs.py

Purpose
-------
Generate graphs that show exception rates at *separate vowel windows*.

This script reads:
    tp_candidate_vowel_windows.csv

and computes TP-style exception rates separately for:

    window position 1 = V1 -> V2
    window position 2 = V2 -> V3
    window position 3 = V3 -> V4
    ...

Optionally, it can also compute rates separately for concrete vowel-pair windows:

    /a/ -> /i/
    /i/ -> /ɨ/
    /ə/ -> /a/
    ...

The key difference from the earlier summary graphs is that this script does NOT collapse
all vowel windows together. It asks:

    In each period and condition, what is the exception rate specifically at V1->V2?
    What is the exception rate specifically at V2->V3?
    What is the exception rate specifically at V3->V4?
    etc.

Inputs
------
Expected default input location:

    OKHC-master/analysis/analyzed_data/tp_period_results/tp_candidate_vowel_windows.csv

Outputs
-------
By default, output is written to:

    OKHC-master/analysis/analyzed_data/tp_graphs/tp_window_exception_rates/

Important outputs:

    window_position_rates_by_period.csv
        Aggregated exception-rate data for positional windows.

    vowel_pair_rates_by_period.csv
        Aggregated exception-rate data for concrete vowel-pair windows.

    position/by_condition/
        One graph per condition. Each graph shows separate positional windows over time.

    position/by_window/
        One graph per positional window. Each graph compares conditions over time.

    position/individual_window_vs_threshold/
        One graph per condition per window, comparing raw exception rate against
        the TP threshold rate for that exact window-specific N.

    pair/by_pair/
        One graph per concrete vowel pair, comparing conditions over time.

    pair/by_condition/
        One graph per condition, showing the most frequent concrete vowel pairs.

Methodological interpretation
-----------------------------
For a given period, condition, and window position, this script computes:

    N = number of evaluable candidate windows of that specific type
    e = number of exceptions among those windows
    raw exception rate = e / N
    TP threshold = theta_N / N = (N / ln N) / N = 1 / ln N

If N is very small, the window-specific rate is unstable. Use --min-n to suppress
very small cells from plots. The full CSV still preserves them.

"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# User-editable label settings
# =============================================================================

# These labels are used only in graph titles/legends.
# They do not affect the analysis.
HANGUL_TO_IPA = {
    "ㆍ": "ʌ",
    "ㅏ": "a",
    "ㅓ": "ə",
    "ㅗ": "o",
    "ㅜ": "u",
    "ㅡ": "ɨ",
    "ㅣ": "i",
    "ㅐ": "ɛ",
    "ㅔ": "e",
    "ㅑ": "ja",
    "ㅕ": "jə",
    "ㅛ": "jo",
    "ㅠ": "ju",
    "ㅘ": "wa",
    "ㅝ": "wə",
    "ㅙ": "wɛ",
    "ㅞ": "we",
    "ㅚ": "ø",
    "ㅟ": "y",
    "ㅢ": "ɨj",
}


CONDITION_ORDER = [
    "baseline_no_excluded_vowels",
    "exclude_from_consideration_ㅡ",
    "exclude_from_consideration_ㅣ",
    "exclude_from_consideration_ㅜ",
    "exclude_from_consideration_ㅡㅣ",
    "exclude_from_consideration_ㅡㅜ",
    "exclude_from_consideration_ㅣㅜ",
    "exclude_from_consideration_ㅡㅣㅜ",
    "discard_wordforms_with_ㅡ",
    "discard_wordforms_with_ㅣ",
    "discard_wordforms_with_ㅜ",
    "discard_wordforms_with_ㅡㅣ",
    "discard_wordforms_with_ㅡㅜ",
    "discard_wordforms_with_ㅣㅜ",
    "discard_wordforms_with_ㅡㅣㅜ",
]


REQUIRED_COLUMNS = [
    "period_start",
    "period_label",
    "condition_id",
    "condition_family",
    "condition_excluded_vowels",
    "candidate_window_index",
    "prev_vowel",
    "target_vowel",
    "candidate_is_evaluable",
    "candidate_is_exception",
]


# =============================================================================
# Path handling
# =============================================================================

def find_project_root(script_path: Path, explicit_project_root: Optional[str] = None) -> Path:
    """
    Find OKHC-master project root robustly.

    The expected root is the directory that contains:

        analysis/analyzed_data

    This lets the script work when placed directly in:

        OKHC-master/analysis/processing_scripts/

    It also still works if PyCharm launches it from a different working directory.
    """
    if explicit_project_root:
        root = Path(explicit_project_root).expanduser().resolve()
        if not (root / "analysis" / "analyzed_data").exists():
            raise FileNotFoundError(
                f"--project-root was given, but it does not contain analysis/analyzed_data: {root}"
            )
        return root

    candidates = [script_path.resolve().parent, Path.cwd().resolve()]
    checked = []

    for start in candidates:
        for parent in [start, *start.parents]:
            checked.append(parent)
            if (parent / "analysis" / "analyzed_data").exists():
                return parent

    checked_text = "\n".join(f"  - {p}" for p in checked[:20])
    raise FileNotFoundError(
        "Could not infer project root. Put this script somewhere under OKHC-master "
        "or pass --project-root explicitly.\n"
        "Checked paths included:\n"
        f"{checked_text}"
    )


def default_input_csv(project_root: Path) -> Path:
    return project_root / "analysis" / "analyzed_data" / "tp_period_results" / "tp_candidate_vowel_windows.csv"


def default_output_dir(project_root: Path) -> Path:
    return project_root / "analysis" / "analyzed_data" / "tp_graphs" / "tp_window_exception_rates"


# =============================================================================
# Label helpers
# =============================================================================

def hangul_string_to_ipa(vowels: str) -> str:
    """
    Convert a compact Hangul-vowel string like 'ㅡㅣㅜ' to an IPA label like '/ɨ, i, u/'.
    """
    if pd.isna(vowels) or str(vowels).upper() == "NONE" or str(vowels).strip() == "":
        return "none"

    chars = list(str(vowels).replace("+", "").replace(",", "").replace(" ", ""))
    labels = [HANGUL_TO_IPA.get(ch, ch) for ch in chars]
    return "/" + ", ".join(labels) + "/"


def vowel_to_ipa(v: str) -> str:
    if pd.isna(v):
        return "NA"
    return HANGUL_TO_IPA.get(str(v), str(v))


def pair_to_ipa(prev_vowel: str, target_vowel: str) -> str:
    return f"/{vowel_to_ipa(prev_vowel)}/→/{vowel_to_ipa(target_vowel)}/"


def clean_filename(text: str) -> str:
    """
    Make a filename safe enough for Windows.
    """
    text = str(text)
    text = text.replace("/", "")
    text = text.replace("\\", "")
    text = text.replace("→", "_to_")
    text = text.replace(":", "_")
    text = text.replace("*", "_")
    text = text.replace("?", "_")
    text = text.replace('"', "_")
    text = text.replace("<", "_")
    text = text.replace(">", "_")
    text = text.replace("|", "_")
    text = re.sub(r"\s+", "_", text)
    return text[:180]


def condition_label(row_or_condition_id, excluded_vowels=None, family=None) -> str:
    """
    Make readable labels for graph legends/titles.
    """
    if isinstance(row_or_condition_id, pd.Series):
        condition_id = row_or_condition_id["condition_id"]
        excluded_vowels = row_or_condition_id.get("condition_excluded_vowels", excluded_vowels)
        family = row_or_condition_id.get("condition_family", family)
    else:
        condition_id = str(row_or_condition_id)

    if condition_id == "baseline_no_excluded_vowels" or family == "baseline":
        return "baseline: no excluded vowels"

    ipa = hangul_string_to_ipa(excluded_vowels)

    if family == "exclude" or condition_id.startswith("exclude_from_consideration"):
        return f"exclude from consideration: {ipa}"

    if family == "discard" or condition_id.startswith("discard_wordforms"):
        return f"discard word forms with: {ipa}"

    return condition_id


def window_position_label(index: int) -> str:
    """
    Convert candidate_window_index to V1→V2 style notation.

    candidate_window_index = 1 means first adjacent considered-vowel window:
        V1 -> V2

    candidate_window_index = 2 means:
        V2 -> V3
    """
    index = int(index)
    return f"V{index}→V{index + 1}"


# =============================================================================
# Data loading and aggregation
# =============================================================================

def read_candidate_windows(input_csv: Path) -> pd.DataFrame:
    """
    Read tp_candidate_vowel_windows.csv and validate that it contains the columns
    needed for window-specific rates.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Reading candidate-window CSV: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    print(f"Loaded {len(df):,} candidate-window rows.")
    return df


def normalize_boolean_column(series: pd.Series) -> pd.Series:
    """
    Convert common CSV boolean encodings into actual booleans.
    """
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def filter_condition_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """
    Keep baseline + requested family by default.

    family='exclude' keeps:
        baseline_no_excluded_vowels
        exclude_from_consideration_...

    family='discard' keeps:
        baseline_no_excluded_vowels
        discard_wordforms_with_...

    family='all' keeps all conditions.
    """
    if family == "all":
        return df.copy()

    keep = (df["condition_family"] == family) | (df["condition_family"] == "baseline")
    out = df.loc[keep].copy()

    print(
        f"Condition-family filter: {family}. "
        f"Kept {out['condition_id'].nunique()} conditions."
    )
    return out


def theta_rate(n: int) -> float:
    """
    TP threshold rate = theta_N / N = 1 / ln(N), for N > 1.
    """
    if n <= 1:
        return float("nan")
    return 1.0 / math.log(n)


def aggregate_by_window_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exception rates separately by candidate_window_index.

    Each row of the output corresponds to:

        one period
        one condition
        one positional vowel window, e.g. V1→V2 or V2→V3
    """
    work = df.copy()
    work["candidate_is_evaluable"] = normalize_boolean_column(work["candidate_is_evaluable"])
    work["candidate_is_exception"] = normalize_boolean_column(work["candidate_is_exception"])

    # Only evaluable windows count toward N.
    work = work.loc[work["candidate_is_evaluable"]].copy()

    group_cols = [
        "period_start",
        "period_label",
        "condition_id",
        "condition_family",
        "condition_excluded_vowels",
        "candidate_window_index",
    ]

    out = (
        work.groupby(group_cols, dropna=False)
        .agg(
            N_windows=("candidate_is_exception", "size"),
            exceptions_e=("candidate_is_exception", "sum"),
        )
        .reset_index()
    )

    out["exception_rate_e_over_N"] = out["exceptions_e"] / out["N_windows"]
    out["tp_threshold_rate_theta_over_N"] = out["N_windows"].apply(theta_rate)
    out["window_label"] = out["candidate_window_index"].apply(window_position_label)
    out["condition_label"] = out.apply(condition_label, axis=1)

    out = sort_for_plotting(out)
    return out


def aggregate_by_vowel_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate exception rates separately by concrete adjacent vowel pair.

    Each row of the output corresponds to:

        one period
        one condition
        one concrete vowel pair, e.g. ㅏ→ㅣ or ㅣ→ㅡ
    """
    work = df.copy()
    work["candidate_is_evaluable"] = normalize_boolean_column(work["candidate_is_evaluable"])
    work["candidate_is_exception"] = normalize_boolean_column(work["candidate_is_exception"])
    work = work.loc[work["candidate_is_evaluable"]].copy()

    group_cols = [
        "period_start",
        "period_label",
        "condition_id",
        "condition_family",
        "condition_excluded_vowels",
        "prev_vowel",
        "target_vowel",
    ]

    out = (
        work.groupby(group_cols, dropna=False)
        .agg(
            N_windows=("candidate_is_exception", "size"),
            exceptions_e=("candidate_is_exception", "sum"),
        )
        .reset_index()
    )

    out["exception_rate_e_over_N"] = out["exceptions_e"] / out["N_windows"]
    out["tp_threshold_rate_theta_over_N"] = out["N_windows"].apply(theta_rate)
    out["pair_label"] = out.apply(lambda r: pair_to_ipa(r["prev_vowel"], r["target_vowel"]), axis=1)
    out["condition_label"] = out.apply(condition_label, axis=1)

    out = sort_for_plotting(out)
    return out


def sort_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort period and condition order for readable graphs.
    """
    out = df.copy()

    condition_rank = {cond: i for i, cond in enumerate(CONDITION_ORDER)}
    out["_condition_rank"] = out["condition_id"].map(condition_rank).fillna(999).astype(int)

    sort_cols = ["period_start", "_condition_rank"]
    if "candidate_window_index" in out.columns:
        sort_cols.append("candidate_window_index")
    if "pair_label" in out.columns:
        sort_cols.append("pair_label")

    out = out.sort_values(sort_cols).drop(columns=["_condition_rank"])
    return out


def choose_top_pairs(pair_rates: pd.DataFrame, top_n: int, min_total_n: int) -> list[str]:
    """
    Pick the most frequent concrete vowel pairs for pair-window graphs.
    """
    totals = (
        pair_rates.groupby("pair_label", dropna=False)["N_windows"]
        .sum()
        .sort_values(ascending=False)
    )
    totals = totals.loc[totals >= min_total_n]
    return list(totals.head(top_n).index)


# =============================================================================
# Plotting helpers
# =============================================================================

def save_plot(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Wrote graph: {path}")


def setup_period_axis(df: pd.DataFrame) -> tuple[list[int], list[str]]:
    """
    Return x positions and period labels in chronological order.
    """
    periods = (
        df[["period_start", "period_label"]]
        .drop_duplicates()
        .sort_values("period_start")
    )
    starts = periods["period_start"].tolist()
    labels = periods["period_label"].tolist()
    return starts, labels


def format_rate_axis() -> None:
    plt.ylim(0, 1)
    plt.ylabel("Exception rate")
    plt.grid(True, axis="y", alpha=0.25)
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")


def format_period_xaxis(period_starts: list[int], period_labels: list[str]) -> None:
    plt.xticks(period_starts, period_labels, rotation=45, ha="right")
    plt.xlabel("Period")


def plot_position_by_condition(
    position_rates: pd.DataFrame,
    output_dir: Path,
    min_n: int,
    max_window_position: int,
    show_points: bool,
    dpi: int,
) -> None:
    """
    For each condition:
        plot one line per positional window.

    This directly answers:
        How does the exception rate differ at V1→V2, V2→V3, V3→V4, etc.
        for this vowel-exclusion condition?
    """
    outdir = output_dir / "position" / "by_condition"
    plot_data = position_rates.loc[
        (position_rates["N_windows"] >= min_n)
        & (position_rates["candidate_window_index"] <= max_window_position)
    ].copy()

    for condition_id, sub in plot_data.groupby("condition_id", sort=False):
        label = sub["condition_label"].iloc[0]
        period_starts, period_labels = setup_period_axis(sub)

        plt.figure(figsize=(11, 6.5))

        for window_index, wsub in sub.groupby("candidate_window_index", sort=True):
            marker = "o" if show_points else None
            plt.plot(
                wsub["period_start"],
                wsub["exception_rate_e_over_N"],
                marker=marker,
                linewidth=2,
                label=window_position_label(window_index),
            )

        plt.title(f"Exception rate by vowel-window position\n{label}")
        format_rate_axis()
        format_period_xaxis(period_starts, period_labels)
        plt.legend(title="Vowel window", bbox_to_anchor=(1.02, 1), loc="upper left")

        fname = clean_filename(f"position_by_condition__{condition_id}.png")
        save_plot(outdir / fname, dpi)


def plot_position_by_window(
    position_rates: pd.DataFrame,
    output_dir: Path,
    min_n: int,
    max_window_position: int,
    show_points: bool,
    dpi: int,
) -> None:
    """
    For each positional window:
        plot one line per condition.

    This answers:
        At V1→V2 specifically, how do the different exclusion conditions compare?
        At V2→V3 specifically, how do they compare?
    """
    outdir = output_dir / "position" / "by_window"
    plot_data = position_rates.loc[
        (position_rates["N_windows"] >= min_n)
        & (position_rates["candidate_window_index"] <= max_window_position)
    ].copy()

    for window_index, sub in plot_data.groupby("candidate_window_index", sort=True):
        period_starts, period_labels = setup_period_axis(sub)

        plt.figure(figsize=(12, 7))

        for condition_id, csub in sub.groupby("condition_id", sort=False):
            marker = "o" if show_points else None
            label = csub["condition_label"].iloc[0]
            plt.plot(
                csub["period_start"],
                csub["exception_rate_e_over_N"],
                marker=marker,
                linewidth=2,
                label=label,
            )

        plt.title(f"Exception rate at positional vowel window {window_position_label(window_index)}")
        format_rate_axis()
        format_period_xaxis(period_starts, period_labels)
        plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")

        fname = clean_filename(f"position_by_window__{window_position_label(window_index)}.png")
        save_plot(outdir / fname, dpi)


def plot_individual_position_vs_threshold(
    position_rates: pd.DataFrame,
    output_dir: Path,
    min_n: int,
    max_window_position: int,
    show_points: bool,
    dpi: int,
) -> None:
    """
    For each condition and positional window:
        plot raw exception rate against the TP threshold rate.

    This is the least cluttered way to compare the raw rate and the threshold,
    because each plot has only two lines.
    """
    outdir = output_dir / "position" / "individual_window_vs_threshold"

    plot_data = position_rates.loc[
        (position_rates["N_windows"] >= min_n)
        & (position_rates["candidate_window_index"] <= max_window_position)
    ].copy()

    for (condition_id, window_index), sub in plot_data.groupby(
        ["condition_id", "candidate_window_index"], sort=False
    ):
        label = sub["condition_label"].iloc[0]
        wlabel = window_position_label(window_index)
        period_starts, period_labels = setup_period_axis(sub)

        marker = "o" if show_points else None
        plt.figure(figsize=(10.5, 6))

        plt.plot(
            sub["period_start"],
            sub["exception_rate_e_over_N"],
            marker=marker,
            linewidth=2.5,
            label="raw exception rate e/N",
        )
        plt.plot(
            sub["period_start"],
            sub["tp_threshold_rate_theta_over_N"],
            marker=marker,
            linewidth=2,
            linestyle="--",
            label="TP threshold θN/N",
        )

        plt.title(f"{wlabel}: raw exception rate vs. TP threshold\n{label}")
        format_rate_axis()
        format_period_xaxis(period_starts, period_labels)
        plt.legend()

        fname = clean_filename(f"vs_threshold__{condition_id}__{wlabel}.png")
        save_plot(outdir / fname, dpi)


def plot_all_positions_one_graph(
    position_rates: pd.DataFrame,
    output_dir: Path,
    min_n: int,
    max_window_position: int,
    show_points: bool,
    dpi: int,
) -> None:
    """
    One intentionally dense graph:
        condition + window position are both represented in the legend.

    This can be visually busy, but it gives a single overview.
    """
    outdir = output_dir / "position"
    plot_data = position_rates.loc[
        (position_rates["N_windows"] >= min_n)
        & (position_rates["candidate_window_index"] <= max_window_position)
    ].copy()

    if plot_data.empty:
        return

    period_starts, period_labels = setup_period_axis(plot_data)

    plt.figure(figsize=(14, 8))

    for (condition_id, window_index), sub in plot_data.groupby(
        ["condition_id", "candidate_window_index"], sort=False
    ):
        marker = "o" if show_points else None
        clabel = sub["condition_label"].iloc[0]
        wlabel = window_position_label(window_index)
        plt.plot(
            sub["period_start"],
            sub["exception_rate_e_over_N"],
            marker=marker,
            linewidth=1.7,
            alpha=0.85,
            label=f"{clabel}; {wlabel}",
        )

    plt.title("All positional vowel-window exception rates by period")
    format_rate_axis()
    format_period_xaxis(period_starts, period_labels)
    plt.legend(title="Condition; window", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    save_plot(outdir / "ALL_position_windows_all_conditions.png", dpi)


def plot_pair_by_pair(
    pair_rates: pd.DataFrame,
    output_dir: Path,
    selected_pairs: list[str],
    min_n: int,
    show_points: bool,
    dpi: int,
) -> None:
    """
    For each concrete vowel pair:
        plot one line per condition.
    """
    outdir = output_dir / "pair" / "by_pair"
    plot_data = pair_rates.loc[
        (pair_rates["N_windows"] >= min_n)
        & (pair_rates["pair_label"].isin(selected_pairs))
    ].copy()

    for pair_label, sub in plot_data.groupby("pair_label", sort=False):
        period_starts, period_labels = setup_period_axis(sub)

        plt.figure(figsize=(12, 7))
        for condition_id, csub in sub.groupby("condition_id", sort=False):
            marker = "o" if show_points else None
            label = csub["condition_label"].iloc[0]
            plt.plot(
                csub["period_start"],
                csub["exception_rate_e_over_N"],
                marker=marker,
                linewidth=2,
                label=label,
            )

        plt.title(f"Exception rate for concrete vowel pair {pair_label}")
        format_rate_axis()
        format_period_xaxis(period_starts, period_labels)
        plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")

        fname = clean_filename(f"pair_by_pair__{pair_label}.png")
        save_plot(outdir / fname, dpi)


def plot_pair_by_condition(
    pair_rates: pd.DataFrame,
    output_dir: Path,
    selected_pairs: list[str],
    min_n: int,
    show_points: bool,
    dpi: int,
) -> None:
    """
    For each condition:
        plot one line per selected concrete vowel pair.
    """
    outdir = output_dir / "pair" / "by_condition"
    plot_data = pair_rates.loc[
        (pair_rates["N_windows"] >= min_n)
        & (pair_rates["pair_label"].isin(selected_pairs))
    ].copy()

    for condition_id, sub in plot_data.groupby("condition_id", sort=False):
        label = sub["condition_label"].iloc[0]
        period_starts, period_labels = setup_period_axis(sub)

        plt.figure(figsize=(12, 7))
        for pair_label, psub in sub.groupby("pair_label", sort=False):
            marker = "o" if show_points else None
            plt.plot(
                psub["period_start"],
                psub["exception_rate_e_over_N"],
                marker=marker,
                linewidth=2,
                label=pair_label,
            )

        plt.title(f"Concrete vowel-pair exception rates\n{label}")
        format_rate_axis()
        format_period_xaxis(period_starts, period_labels)
        plt.legend(title="Vowel pair", bbox_to_anchor=(1.02, 1), loc="upper left")

        fname = clean_filename(f"pair_by_condition__{condition_id}.png")
        save_plot(outdir / fname, dpi)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate graphs showing exception rates at separate vowel windows "
            "rather than collapsing all windows together."
        )
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Path to OKHC-master. Usually not needed if the script is somewhere under OKHC-master."
        ),
    )

    parser.add_argument(
        "--input-csv",
        default=None,
        help=(
            "Path to tp_candidate_vowel_windows.csv. "
            "If omitted, the script uses analysis/analyzed_data/tp_period_results/"
            "tp_candidate_vowel_windows.csv under the detected project root."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for graphs and aggregated CSVs. "
            "If omitted, the script uses analysis/analyzed_data/tp_graphs/"
            "tp_window_exception_rates under the detected project root."
        ),
    )

    parser.add_argument(
        "--family",
        choices=["exclude", "discard", "all"],
        default="exclude",
        help=(
            "Which condition family to graph. Baseline is included with exclude/discard. "
            "Use 'all' to include every condition."
        ),
    )

    parser.add_argument(
        "--window-type",
        choices=["position", "pair", "both"],
        default="position",
        help=(
            "position = V1→V2, V2→V3, etc.; "
            "pair = concrete vowel pairs like /a/→/i/; "
            "both = generate both sets."
        ),
    )

    parser.add_argument(
        "--max-window-position",
        type=int,
        default=6,
        help="Highest positional vowel window to plot, e.g. 6 means up to V6→V7.",
    )

    parser.add_argument(
        "--min-n",
        type=int,
        default=10,
        help=(
            "Minimum number of evaluable windows required for a data point to appear in plots. "
            "The full aggregated CSV still includes smaller cells."
        ),
    )

    parser.add_argument(
        "--top-pairs",
        type=int,
        default=12,
        help="For --window-type pair/both, plot only the top N concrete vowel pairs by total count.",
    )

    parser.add_argument(
        "--min-total-pair-n",
        type=int,
        default=50,
        help="For concrete vowel-pair plots, ignore pairs with fewer than this many total windows.",
    )

    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Put markers on line-graph points.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for saved PNGs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__)

    # Only infer the project root when it is actually needed for a default path.
    # This lets you test the script with fully explicit --input-csv and --output-dir paths.
    project_root = None
    if args.project_root or not args.input_csv or not args.output_dir:
        project_root = find_project_root(script_path, args.project_root)

    input_csv = (
        Path(args.input_csv).expanduser().resolve()
        if args.input_csv
        else default_input_csv(project_root)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(project_root)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if project_root is not None:
        print(f"Project root: {project_root}")
    else:
        print("Project root: not needed because --input-csv and --output-dir were both supplied")
    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Condition family: {args.family}")
    print(f"Window type: {args.window_type}")
    print(f"Minimum N for plotted points: {args.min_n}")

    windows = read_candidate_windows(input_csv)
    windows = filter_condition_family(windows, args.family)

    if args.window_type in ["position", "both"]:
        position_rates = aggregate_by_window_position(windows)
        position_csv = output_dir / "window_position_rates_by_period.csv"
        position_rates.to_csv(position_csv, index=False, encoding="utf-8-sig")
        print(f"Wrote position-window summary CSV: {position_csv}")

        plot_position_by_condition(
            position_rates=position_rates,
            output_dir=output_dir,
            min_n=args.min_n,
            max_window_position=args.max_window_position,
            show_points=args.show_points,
            dpi=args.dpi,
        )

        plot_position_by_window(
            position_rates=position_rates,
            output_dir=output_dir,
            min_n=args.min_n,
            max_window_position=args.max_window_position,
            show_points=args.show_points,
            dpi=args.dpi,
        )

        plot_individual_position_vs_threshold(
            position_rates=position_rates,
            output_dir=output_dir,
            min_n=args.min_n,
            max_window_position=args.max_window_position,
            show_points=args.show_points,
            dpi=args.dpi,
        )

        plot_all_positions_one_graph(
            position_rates=position_rates,
            output_dir=output_dir,
            min_n=args.min_n,
            max_window_position=args.max_window_position,
            show_points=args.show_points,
            dpi=args.dpi,
        )

    if args.window_type in ["pair", "both"]:
        pair_rates = aggregate_by_vowel_pair(windows)
        pair_csv = output_dir / "vowel_pair_rates_by_period.csv"
        pair_rates.to_csv(pair_csv, index=False, encoding="utf-8-sig")
        print(f"Wrote concrete vowel-pair summary CSV: {pair_csv}")

        selected_pairs = choose_top_pairs(
            pair_rates=pair_rates,
            top_n=args.top_pairs,
            min_total_n=args.min_total_pair_n,
        )

        selected_pairs_csv = output_dir / "selected_vowel_pairs_for_graphing.csv"
        pd.DataFrame({"pair_label": selected_pairs}).to_csv(
            selected_pairs_csv, index=False, encoding="utf-8-sig"
        )
        print(f"Wrote selected vowel-pair list: {selected_pairs_csv}")
        print("Selected concrete vowel pairs:")
        for pair in selected_pairs:
            print(f"  - {pair}")

        plot_pair_by_pair(
            pair_rates=pair_rates,
            output_dir=output_dir,
            selected_pairs=selected_pairs,
            min_n=args.min_n,
            show_points=args.show_points,
            dpi=args.dpi,
        )

        plot_pair_by_condition(
            pair_rates=pair_rates,
            output_dir=output_dir,
            selected_pairs=selected_pairs,
            min_n=args.min_n,
            show_points=args.show_points,
            dpi=args.dpi,
        )

    print("Done.")


if __name__ == "__main__":
    main()
