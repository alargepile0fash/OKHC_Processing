
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_generate_tp_prefix_exception_rate_graphs.py

Generate TP exception-rate graphs for cumulative vowel prefixes:

    first 2 considered vowels  -> evaluate only V1→V2
    first 3 considered vowels  -> evaluate V1→V2 and V2→V3
    first 4 considered vowels  -> evaluate V1→V2, V2→V3, and V3→V4
    ...

This is different from graphing each separate adjacent window position.  Here, each
prefix length is cumulative: "first 4 vowels" includes all adjacent windows inside
the first four vowels.

Inputs expected from the TP pipeline:
    tp_candidate_vowel_windows.csv
    tp_condition_wordform_rows.csv

Default input directory:
    OKHC-master/analysis/analyzed_data/tp_period_results/

Default output directory:
    OKHC-master/analysis/analyzed_data/tp_graphs/tp_prefix_exception_rates/

The script is designed to work when placed directly in:
    OKHC-master/analysis/processing_scripts/

It should also work if run from PyCharm with the project root as working directory.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# User-editable label settings
# ---------------------------------------------------------------------------

# Labels used in graph titles and legends.  The data files still use Hangul.
HANGUL_TO_IPA = {
    "ㅡ": "ɨ",
    "ㅣ": "i",
    "ㅜ": "u",
    "ㅏ": "a",
    "ㅐ": "ɛ",
    "ㅑ": "ja",
    "ㅒ": "jɛ",
    "ㅓ": "ə",
    "ㅔ": "e",
    "ㅕ": "jə",
    "ㅖ": "je",
    "ㅗ": "o",
    "ㅘ": "wa",
    "ㅙ": "wɛ",
    "ㅚ": "ø",
    "ㅛ": "jo",
    "ㅠ": "ju",
    "ㅝ": "wə",
    "ㅞ": "we",
    "ㅟ": "wi",
    "ㆍ": "ʌ",
}


WINDOWS_FILENAME = "tp_candidate_vowel_windows.csv"
WORDFORMS_FILENAME = "tp_condition_wordform_rows.csv"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Infer the OKHC project root.

    Preferred project root is the directory that contains:
        analysis/analyzed_data
        analysis/processing_scripts

    This function walks upward from the script location and the current working
    directory.  This avoids the earlier failure mode where PyCharm treated a
    nested helper-script folder as the project root.
    """
    candidates: List[Path] = []

    if start is not None:
        candidates.append(start.resolve())

    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd().resolve())

    for seed in candidates:
        for p in [seed, *seed.parents]:
            if (p / "analysis" / "analyzed_data").exists() and (p / "analysis" / "processing_scripts").exists():
                return p

    # Fallback: assume script is in OKHC-master/analysis/processing_scripts.
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "processing_scripts" and script_dir.parent.name == "analysis":
        return script_dir.parent.parent

    # Last fallback: current directory.
    return Path.cwd().resolve()


def default_input_dir(project_root: Path) -> Path:
    return project_root / "analysis" / "analyzed_data" / "tp_period_results"


def default_output_dir(project_root: Path) -> Path:
    return project_root / "analysis" / "analyzed_data" / "tp_graphs" / "tp_prefix_exception_rates"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def safe_filename(text: str) -> str:
    """
    Make a filesystem-safe filename while preserving useful IPA/Hangul labels
    where possible.
    """
    text = str(text)
    text = text.replace("/", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\.\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AFɨəɛøʌ]+", "_", text)
    return text.strip("_") or "unnamed"


def vowel_string_to_ipa(vowels: object) -> str:
    """
    Convert condition_excluded_vowels values to IPA label strings.

    Examples:
        "NONE" -> "none"
        "ㅡ"   -> "/ɨ/"
        "ㅡㅣㅜ" -> "/ɨ, i, u/"
    """
    if pd.isna(vowels):
        return "none"
    s = str(vowels)
    if s.upper() == "NONE" or s == "":
        return "none"

    ipa = []
    for ch in s:
        if ch in HANGUL_TO_IPA:
            ipa.append(HANGUL_TO_IPA[ch])
        elif ch.strip():
            ipa.append(ch)

    if not ipa:
        return "none"
    return "/" + ", ".join(ipa) + "/"


def condition_label(row_or_id: object, excluded: object = None, family: object = None) -> str:
    """
    Human-readable condition label for legends/titles.
    """
    if isinstance(row_or_id, pd.Series):
        condition_id = row_or_id.get("condition_id", "")
        excluded = row_or_id.get("condition_excluded_vowels", excluded)
        family = row_or_id.get("condition_family", family)
    else:
        condition_id = row_or_id

    cid = str(condition_id)
    fam = str(family) if family is not None else ""

    if cid == "baseline_no_excluded_vowels" or fam == "baseline":
        return "baseline: no excluded vowels"

    ipa = vowel_string_to_ipa(excluded)

    if fam == "exclude_from_consideration" or cid.startswith("exclude_from_consideration"):
        return f"exclude from consideration: {ipa}"

    if fam == "discard_wordforms_containing" or cid.startswith("discard_wordforms_containing"):
        return f"discard word forms containing: {ipa}"

    return f"{cid}: {ipa}"


def prefix_label(k: int) -> str:
    return f"first {k} vowels"


def theta_n(N: int) -> float:
    """
    Yang's TP threshold theta_N = N / ln(N).
    For N <= 1, the threshold is undefined for useful TP purposes, so return NaN.
    """
    if N <= 1:
        return float("nan")
    return N / math.log(N)


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def read_inputs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    windows_path = input_dir / WINDOWS_FILENAME
    wordforms_path = input_dir / WORDFORMS_FILENAME

    if not windows_path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {windows_path}")
    if not wordforms_path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {wordforms_path}")

    windows = pd.read_csv(windows_path, encoding="utf-8-sig", low_memory=False)
    wordforms = pd.read_csv(wordforms_path, encoding="utf-8-sig", low_memory=False)

    required_windows = {
        "period_start",
        "period_label",
        "condition_id",
        "condition_family",
        "condition_excluded_vowels",
        "wordform_id",
        "candidate_window_index",
        "candidate_is_exception",
    }
    missing = required_windows - set(windows.columns)
    if missing:
        raise ValueError(f"{WINDOWS_FILENAME} is missing required columns: {sorted(missing)}")

    required_wordforms = {
        "period_start",
        "period_label",
        "condition_id",
        "condition_family",
        "condition_excluded_vowels",
        "wordform_id",
        "reduced_num_vowels",
        "wordform_discarded_by_condition",
        "wordform_has_enough_reduced_vowels",
    }
    missing = required_wordforms - set(wordforms.columns)
    if missing:
        raise ValueError(f"{WORDFORMS_FILENAME} is missing required columns: {sorted(missing)}")

    # Normalize booleans that may have been read as strings.
    windows["candidate_is_exception"] = windows["candidate_is_exception"].astype(str).str.lower().isin(["true", "1", "yes"])

    wordforms["wordform_discarded_by_condition"] = (
        wordforms["wordform_discarded_by_condition"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    wordforms["wordform_has_enough_reduced_vowels"] = (
        wordforms["wordform_has_enough_reduced_vowels"].astype(str).str.lower().isin(["true", "1", "yes"])
    )

    return windows, wordforms


def filter_condition_family(
    windows: pd.DataFrame,
    wordforms: pd.DataFrame,
    family: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter to the requested condition family.

    family values:
        all      -> baseline + exclude + discard
        exclude  -> baseline + exclude_from_consideration
        discard  -> baseline + discard_wordforms_containing
        baseline -> baseline only
    """
    if family == "all":
        return windows.copy(), wordforms.copy()

    if family == "exclude":
        keep = {"baseline", "exclude_from_consideration"}
    elif family == "discard":
        keep = {"baseline", "discard_wordforms_containing"}
    elif family == "baseline":
        keep = {"baseline"}
    else:
        raise ValueError(f"Unknown family: {family}")

    return (
        windows[windows["condition_family"].isin(keep)].copy(),
        wordforms[wordforms["condition_family"].isin(keep)].copy(),
    )


def attach_wordform_lengths(windows: pd.DataFrame, wordforms: pd.DataFrame) -> pd.DataFrame:
    """
    Add reduced_num_vowels from the wordform-level table to every candidate window.
    This lets us ask whether a word actually has at least k vowels when calculating
    the "first k vowels" rate.
    """
    key_cols = ["period_start", "period_label", "condition_id", "wordform_id"]
    meta_cols = key_cols + ["reduced_num_vowels"]

    lengths = wordforms[meta_cols].drop_duplicates()
    merged = windows.merge(lengths, on=key_cols, how="left", validate="many_to_one")

    if merged["reduced_num_vowels"].isna().any():
        n_missing = int(merged["reduced_num_vowels"].isna().sum())
        raise ValueError(
            f"{n_missing:,} candidate-window rows could not be matched to reduced_num_vowels "
            f"in {WORDFORMS_FILENAME}."
        )

    merged["reduced_num_vowels"] = merged["reduced_num_vowels"].astype(int)
    merged["candidate_window_index"] = merged["candidate_window_index"].astype(int)
    return merged


# ---------------------------------------------------------------------------
# Prefix-rate calculations
# ---------------------------------------------------------------------------

def calculate_prefix_rates(
    windows: pd.DataFrame,
    min_prefix: int,
    max_prefix: int,
    include_shorter_words: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate exception rates for cumulative vowel prefixes.

    For prefix length k:
        - candidate-level rate counts all candidate windows inside the first k vowels.
          This means candidate_window_index <= k - 1.
        - word-level rate counts each word form once.  A word is an exception if
          any candidate window inside the first k vowels is an exception.

    If include_shorter_words is False:
        only words with at least k reduced/considered vowels are included for
        prefix length k.  This is the default because "first 4 vowels" is only
        well-defined for words that actually have 4 considered vowels.

    If include_shorter_words is True:
        shorter words are included wherever they have at least one evaluable
        window.  For example, a two-vowel word contributes its V1→V2 window to
        the "first 4 vowels" calculation.  This makes prefix lengths cumulative
        over the whole corpus but changes the interpretation.
    """
    all_candidate_rows = []
    all_word_rows = []

    group_cols = [
        "period_start",
        "period_label",
        "condition_id",
        "condition_family",
        "condition_excluded_vowels",
    ]

    for k in range(min_prefix, max_prefix + 1):
        # Windows contained within the first k considered vowels.
        # k=2 includes window 1 only; k=3 includes windows 1 and 2; etc.
        df_k = windows[windows["candidate_window_index"] <= (k - 1)].copy()

        if not include_shorter_words:
            df_k = df_k[df_k["reduced_num_vowels"] >= k].copy()

        # Candidate-window-level calculation.
        cand = (
            df_k.groupby(group_cols, dropna=False)
            .agg(
                N_candidates=("candidate_is_exception", "size"),
                exceptions_e=("candidate_is_exception", "sum"),
                wordforms_contributing=("wordform_id", "nunique"),
            )
            .reset_index()
        )

        cand["prefix_vowels_counted"] = k
        cand["prefix_description"] = prefix_label(k)
        cand["analysis_level"] = "candidate_window_prefix_level"
        cand["exception_rate_e_over_N"] = cand["exceptions_e"] / cand["N_candidates"]
        cand["theta_N_float"] = cand["N_candidates"].apply(lambda n: theta_n(int(n)))
        cand["tolerated_exception_rate_theta_over_N"] = cand["theta_N_float"] / cand["N_candidates"]
        cand["tp_passes"] = cand["exceptions_e"] <= cand["theta_N_float"]
        all_candidate_rows.append(cand)

        # Word-level calculation over the same prefix.
        if df_k.empty:
            continue

        per_word = (
            df_k.groupby(group_cols + ["wordform_id"], dropna=False)
            .agg(
                word_is_exception=("candidate_is_exception", "max"),
                windows_contributing=("candidate_is_exception", "size"),
            )
            .reset_index()
        )

        word = (
            per_word.groupby(group_cols, dropna=False)
            .agg(
                N_candidates=("wordform_id", "nunique"),
                exceptions_e=("word_is_exception", "sum"),
                candidate_windows_contributing=("windows_contributing", "sum"),
            )
            .reset_index()
        )

        word["prefix_vowels_counted"] = k
        word["prefix_description"] = prefix_label(k)
        word["analysis_level"] = "word_prefix_level"
        word["exception_rate_e_over_N"] = word["exceptions_e"] / word["N_candidates"]
        word["theta_N_float"] = word["N_candidates"].apply(lambda n: theta_n(int(n)))
        word["tolerated_exception_rate_theta_over_N"] = word["theta_N_float"] / word["N_candidates"]
        word["tp_passes"] = word["exceptions_e"] <= word["theta_N_float"]
        all_word_rows.append(word)

    candidate_rates = pd.concat(all_candidate_rows, ignore_index=True) if all_candidate_rows else pd.DataFrame()
    word_rates = pd.concat(all_word_rows, ignore_index=True) if all_word_rows else pd.DataFrame()

    # Helpful label columns.
    for df in [candidate_rates, word_rates]:
        if not df.empty:
            df["condition_label"] = df.apply(condition_label, axis=1)
            df["excluded_vowels_ipa"] = df["condition_excluded_vowels"].apply(vowel_string_to_ipa)
            df["raw_exception_percent"] = df["exception_rate_e_over_N"] * 100.0
            df["tp_threshold_percent"] = df["tolerated_exception_rate_theta_over_N"] * 100.0
            df["tp_margin_exception_minus_threshold"] = (
                df["exception_rate_e_over_N"] - df["tolerated_exception_rate_theta_over_N"]
            )
            df["tp_margin_percent_points"] = df["tp_margin_exception_minus_threshold"] * 100.0

    return candidate_rates, word_rates


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sort_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["period_start", "prefix_vowels_counted", "condition_id"]).copy()


def plot_condition_prefix_lines(
    df: pd.DataFrame,
    outdir: Path,
    analysis_label: str,
    show_points: bool,
    dpi: int,
) -> None:
    """
    One graph per condition.
    Each graph shows exception rate by period for first 2 vowels, first 3 vowels, etc.
    TP threshold for each prefix is drawn as a dashed line.
    """
    ensure_dir(outdir)

    for condition_id, sub in df.groupby("condition_id", sort=False):
        sub = sort_for_plot(sub)
        label = sub["condition_label"].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 7))

        for k, kdf in sub.groupby("prefix_vowels_counted", sort=True):
            kdf = kdf.sort_values("period_start")
            marker = "o" if show_points else None
            line_label = f"{prefix_label(int(k))}: exception rate"
            ax.plot(
                kdf["period_label"],
                kdf["raw_exception_percent"],
                marker=marker,
                linewidth=2,
                label=line_label,
            )
            ax.plot(
                kdf["period_label"],
                kdf["tp_threshold_percent"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.65,
                label=f"{prefix_label(int(k))}: TP threshold",
            )

        ax.set_title(f"{analysis_label}: exception rate by cumulative vowel prefix\n{label}")
        ax.set_xlabel("Period")
        ax.set_ylabel("Percent")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()

        outfile = outdir / f"{safe_filename(condition_id)}__prefix_rates_vs_threshold.png"
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)


def plot_all_conditions_by_prefix(
    df: pd.DataFrame,
    outdir: Path,
    analysis_label: str,
    show_points: bool,
    dpi: int,
) -> None:
    """
    One graph per prefix length.
    Each graph compares all conditions for a given prefix length.
    """
    ensure_dir(outdir)

    for k, sub in df.groupby("prefix_vowels_counted", sort=True):
        sub = sort_for_plot(sub)

        fig, ax = plt.subplots(figsize=(12, 7))

        for condition_id, cdf in sub.groupby("condition_id", sort=False):
            cdf = cdf.sort_values("period_start")
            marker = "o" if show_points else None
            label = cdf["condition_label"].iloc[0]
            ax.plot(
                cdf["period_label"],
                cdf["raw_exception_percent"],
                marker=marker,
                linewidth=2,
                label=label,
            )

        # Threshold varies slightly by condition because N differs, so show the
        # min/max threshold band for this prefix.
        band = (
            sub.groupby("period_label", sort=False)
            .agg(
                period_start=("period_start", "first"),
                threshold_min=("tp_threshold_percent", "min"),
                threshold_max=("tp_threshold_percent", "max"),
            )
            .reset_index()
            .sort_values("period_start")
        )
        ax.fill_between(
            band["period_label"],
            band["threshold_min"],
            band["threshold_max"],
            alpha=0.15,
            label="TP threshold range across conditions",
        )

        ax.set_title(f"{analysis_label}: exception rate for {prefix_label(int(k))}")
        ax.set_xlabel("Period")
        ax.set_ylabel("Exception rate (%)")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        outfile = outdir / f"all_conditions__{int(k)}_vowel_prefix_exception_rates.png"
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)


def plot_all_prefixes_all_conditions(
    df: pd.DataFrame,
    outdir: Path,
    analysis_label: str,
    show_points: bool,
    dpi: int,
) -> None:
    """
    A single dense overview graph.

    Each line is condition × prefix length.  This can get crowded, but it is useful
    as an all-at-once diagnostic.
    """
    ensure_dir(outdir)
    df = sort_for_plot(df)

    fig, ax = plt.subplots(figsize=(14, 8))

    for (condition_id, k), sub in df.groupby(["condition_id", "prefix_vowels_counted"], sort=False):
        sub = sub.sort_values("period_start")
        marker = "o" if show_points else None
        cond_label = sub["condition_label"].iloc[0]
        label = f"{cond_label}; {prefix_label(int(k))}"
        ax.plot(
            sub["period_label"],
            sub["raw_exception_percent"],
            marker=marker,
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )

    threshold_band = (
        df.groupby("period_label", sort=False)
        .agg(
            period_start=("period_start", "first"),
            threshold_min=("tp_threshold_percent", "min"),
            threshold_max=("tp_threshold_percent", "max"),
        )
        .reset_index()
        .sort_values("period_start")
    )

    ax.fill_between(
        threshold_band["period_label"],
        threshold_band["threshold_min"],
        threshold_band["threshold_max"],
        alpha=0.15,
        label="TP threshold range",
    )

    ax.set_title(f"{analysis_label}: all conditions and cumulative vowel prefixes")
    ax.set_xlabel("Period")
    ax.set_ylabel("Exception rate (%)")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()

    outfile = outdir / "all_conditions_all_prefixes_exception_rates.png"
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_variable_descriptions(output_dir: Path) -> None:
    descriptions = [
        ("period_start", "Numeric start year of the period."),
        ("period_label", "Human-readable period label, e.g. 1443-1492."),
        ("condition_id", "Unique condition identifier from the TP analysis."),
        ("condition_family", "Condition family: baseline, exclude_from_consideration, or discard_wordforms_containing."),
        ("condition_excluded_vowels", "Hangul vowel symbols excluded or used for discarding in the condition."),
        ("excluded_vowels_ipa", "IPA rendering of condition_excluded_vowels used for graph labels."),
        ("condition_label", "Human-readable condition label."),
        ("prefix_vowels_counted", "Cumulative number of considered vowels included: 2 means V1→V2 only; 3 means V1→V2 plus V2→V3; etc."),
        ("prefix_description", "Text label for prefix_vowels_counted."),
        ("analysis_level", "candidate_window_prefix_level or word_prefix_level."),
        ("N_candidates", "Number of TP candidates in this period/condition/prefix. At candidate level, this is windows; at word level, this is word forms."),
        ("exceptions_e", "Number of exceptions in this period/condition/prefix."),
        ("exception_rate_e_over_N", "Raw exception rate, exceptions_e divided by N_candidates."),
        ("theta_N_float", "TP threshold theta_N = N / ln(N)."),
        ("tolerated_exception_rate_theta_over_N", "TP threshold as a rate, theta_N divided by N_candidates."),
        ("tp_passes", "True if exceptions_e <= theta_N_float."),
        ("tp_margin_exception_minus_threshold", "Exception rate minus TP threshold rate. Positive values fail the TP."),
        ("tp_margin_percent_points", "The TP margin in percentage points."),
        ("wordforms_contributing", "Candidate-level only: number of word forms contributing at least one window to the prefix calculation."),
        ("candidate_windows_contributing", "Word-level only: number of candidate windows considered inside contributing word forms."),
    ]
    pd.DataFrame(descriptions, columns=["variable", "description"]).to_csv(
        output_dir / "prefix_rate_variable_descriptions.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    project_root = find_project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Generate graphs showing TP exception rates when evaluating cumulative "
            "vowel prefixes: first 2 vowels, first 3 vowels, first 4 vowels, etc."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir(project_root),
        help="Directory containing tp_candidate_vowel_windows.csv and tp_condition_wordform_rows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(project_root),
        help="Directory where graphs and derived CSVs should be written.",
    )
    parser.add_argument(
        "--family",
        choices=["all", "exclude", "discard", "baseline"],
        default="exclude",
        help=(
            "Which condition family to graph. 'exclude' includes baseline plus "
            "exclude_from_consideration conditions."
        ),
    )
    parser.add_argument(
        "--level",
        choices=["candidate", "word", "both"],
        default="both",
        help="Whether to graph candidate-window prefix rates, word prefix rates, or both.",
    )
    parser.add_argument(
        "--min-prefix",
        type=int,
        default=2,
        help="Smallest cumulative vowel prefix to graph. Default: 2.",
    )
    parser.add_argument(
        "--max-prefix",
        type=int,
        default=6,
        help="Largest cumulative vowel prefix to graph. Default: 6.",
    )
    parser.add_argument(
        "--include-shorter-words",
        action="store_true",
        help=(
            "Include words shorter than the requested prefix wherever they have at least "
            "one evaluable window. By default, first-k-vowel rates require words to have "
            "at least k considered vowels."
        ),
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Put markers on line graphs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG graphs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    print(f"Project root: {project_root}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Condition family: {args.family}")
    print(f"Analysis level: {args.level}")
    print(f"Prefix range: first {args.min_prefix} vowels through first {args.max_prefix} vowels")
    print(f"Include shorter words in longer prefixes: {args.include_shorter_words}")

    if args.min_prefix < 2:
        raise ValueError("--min-prefix must be at least 2 because harmony requires at least two vowels.")
    if args.max_prefix < args.min_prefix:
        raise ValueError("--max-prefix must be greater than or equal to --min-prefix.")

    ensure_dir(output_dir)

    print("Reading TP output CSVs...")
    windows, wordforms = read_inputs(input_dir)
    print(f"Loaded candidate windows: {len(windows):,}")
    print(f"Loaded condition wordform rows: {len(wordforms):,}")

    print("Filtering condition family...")
    windows, wordforms = filter_condition_family(windows, wordforms, args.family)
    print(f"Candidate windows after family filter: {len(windows):,}")
    print(f"Wordform rows after family filter: {len(wordforms):,}")

    print("Attaching reduced vowel counts to candidate windows...")
    windows = attach_wordform_lengths(windows, wordforms)

    print("Calculating cumulative prefix exception rates...")
    candidate_rates, word_rates = calculate_prefix_rates(
        windows=windows,
        min_prefix=args.min_prefix,
        max_prefix=args.max_prefix,
        include_shorter_words=args.include_shorter_words,
    )

    # Save derived CSVs.
    if not candidate_rates.empty:
        candidate_csv = output_dir / "candidate_window_prefix_exception_rates_by_period.csv"
        candidate_rates.to_csv(candidate_csv, index=False, encoding="utf-8-sig")
        print(f"Wrote: {candidate_csv}")

    if not word_rates.empty:
        word_csv = output_dir / "word_prefix_exception_rates_by_period.csv"
        word_rates.to_csv(word_csv, index=False, encoding="utf-8-sig")
        print(f"Wrote: {word_csv}")

    write_variable_descriptions(output_dir)
    print(f"Wrote: {output_dir / 'prefix_rate_variable_descriptions.csv'}")

    # Graphs.
    if args.level in {"candidate", "both"} and not candidate_rates.empty:
        label = "Candidate-window level"
        level_dir = output_dir / "candidate_window_level"

        print("Generating candidate-window-level prefix graphs...")
        plot_condition_prefix_lines(
            candidate_rates,
            level_dir / "by_condition",
            label,
            args.show_points,
            args.dpi,
        )
        plot_all_conditions_by_prefix(
            candidate_rates,
            level_dir / "all_conditions_by_prefix",
            label,
            args.show_points,
            args.dpi,
        )
        plot_all_prefixes_all_conditions(
            candidate_rates,
            level_dir / "all_at_once",
            label,
            args.show_points,
            args.dpi,
        )

    if args.level in {"word", "both"} and not word_rates.empty:
        label = "Word level"
        level_dir = output_dir / "word_level"

        print("Generating word-level prefix graphs...")
        plot_condition_prefix_lines(
            word_rates,
            level_dir / "by_condition",
            label,
            args.show_points,
            args.dpi,
        )
        plot_all_conditions_by_prefix(
            word_rates,
            level_dir / "all_conditions_by_prefix",
            label,
            args.show_points,
            args.dpi,
        )
        plot_all_prefixes_all_conditions(
            word_rates,
            level_dir / "all_at_once",
            label,
            args.show_points,
            args.dpi,
        )

    print("Done.")
    print(f"Graphs and derived CSVs are in: {output_dir}")


if __name__ == "__main__":
    main()
