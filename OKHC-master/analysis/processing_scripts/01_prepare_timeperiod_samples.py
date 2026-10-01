#!/usr/bin/env python3
"""
01_prepare_timeperiod_samples.py

Purpose
-------
This script performs ONLY the frequency-filtering and sample-balancing stage.
It does not run any Tolerance Principle calculations.

The script:
1. Reads the original/master token-level CSV in chunks, so very large corpus files
   do not have to be loaded into memory all at once.
2. If --start-year is supplied, drops rows whose year is earlier than that start year.
3. Assigns every remaining token to a configured period based on its year.
4. Collapses token occurrences into period-specific word-form candidates.
5. Counts how often each word form occurs within each configured period.
6. Writes diagnostics for all retained configured periods, including sparse edge periods.
7. Selects high-frequency word forms according to the requested sampling mode.
   The default mode, cap-preserve-periods, keeps every non-manually-excluded period
   after --start-year and caps only the maximum number of word forms per period.
   This avoids creating major gaps just because early periods have fewer forms.
8. Optionally, strict-balanced mode reproduces the older behavior: exclude periods
   below --min-period-wordforms and balance all remaining periods to the smallest
   eligible period.
9. Writes a filtered word-form CSV that the TP-analysis script can read later.
10. Optionally writes the original token rows represented by those selected word forms.

Why this is split off
---------------------
Frequency filtering is separated from TP testing so that:
- the TP-analysis script always runs on the same controlled sample;
- alternative TP conditions do not accidentally change the frequency sample;
- the sample-balancing step can be inspected before any theoretical claims are made;
- sparse edge periods can be preserved in diagnostics without forcing all other
  periods down to the sparse period's sample size.

Default assumptions
-------------------
- Candidate selection is by WORD FORM within each configured period.
- "Highest-frequency" means the word forms with the largest token counts in that period.
- The script ignores origin/stratum fields entirely.
- The script does not sample isolated fixed-interval snapshots. It bins all tokens into contiguous
  configured periods and uses all retained tokens inside each period before balancing.
- If --start-year is provided, it is both the first period boundary and the lower
  year cutoff. Years before that boundary are ignored, so pre-start edge periods
  cannot determine the balanced sample size.

Expected important input columns
--------------------------------
year    : token date/year. Used to assign configured periods.
token   : orthographic word form. Used as the word-form label.
vowels  : vowel sequence for that token. Used to distinguish forms and later run TP tests.

The script is intentionally configurable near the top. If your CSV uses different column
names, edit the CONFIG section or pass command-line arguments.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# =============================================================================
# CONFIG: edit these defaults if your file layout or column names differ.
# =============================================================================

DEFAULT_INPUT_CSV = Path("analysis/data/hangul_vowel_tokens_diachronic.csv")
DEFAULT_OUTPUT_DIR = Path("analysis/data/timeperiod_samples")
PROJECT_ROOT_ENV_VAR = "OKHC_ROOT"

# Input column names. These match the columns shown in your current processing output.
YEAR_COL = "year"
TOKEN_COL = "token"
VOWELS_COL = "vowels"

# Optional metadata columns. These are kept only if present in the original token-row
# audit output. They are not used for sample balancing or TP analysis.
OPTIONAL_METADATA_COLS = [
    "input_file",
    "doc_id",
    "source",
    "corpus",
    "token_source",
    "token_context",
    "harmony_status",
    "vowel_classes",
    "num_vowels",
]

# Each period covers this many years. The user requested configured periods.
PERIOD_SIZE_YEARS = 25

# If START_YEAR is None, the first period boundary is calculated from the earliest
# observed year. If you want historically cleaner bins such as 1400-1424, 1425-1449,
# set START_YEAR = 1400 or pass --start-year 1400.
START_YEAR: Optional[int] = None

# If MIN_WORD_VOWELS is 2, word forms with fewer than 2 vowels are excluded from the
# frequency pool because they cannot provide a vowel-harmony comparison.
MIN_WORD_VOWELS = 2

# Chunk size for reading the master corpus. Larger values can be faster but require
# more memory. Smaller values show progress more frequently and are safer on laptops.
CHUNKSIZE = 250_000

# Minimum number of unique word forms a configured period must have before it is used
# for the balanced TP sample in strict-balanced mode.
# In the default cap-preserve-periods mode, this is reported as a warning threshold
# rather than used to discard periods.
MIN_PERIOD_WORDFORMS_FOR_BALANCING = 1_000

# Default maximum number of highest-frequency word forms to keep per period in
# cap-preserve-periods mode. Periods with fewer than this many forms are kept in full
# and flagged as low-N rather than discarded.
TARGET_WORDFORMS_PER_PERIOD = 1_000

# Default sampling mode.
#   cap-preserve-periods: preserve the full time frame after --start-year by keeping
#       up to TARGET_WORDFORMS_PER_PERIOD per period; low-N periods stay included.
#   strict-balanced: exclude periods below thresholds and force all included periods
#       to exactly the same size.
SAMPLING_MODE = "cap-preserve-periods"

# Optional minimum represented token count for an eligible period. Leave at 0 unless
# you also want to exclude periods with enough unique word forms but very few tokens.
MIN_PERIOD_TOKENS_FOR_BALANCING = 0

# Optional cap on the balanced sample size. None means: use the smallest eligible
# period's full available word-form count. Set this to e.g. 50_000 if the smallest
# eligible period is still too large for practical TP testing.
MAX_WORDFORMS_PER_PERIOD: Optional[int] = None

# Vowel symbols that the parser recognizes when the vowels column is not already
# separated by spaces or commas. Add symbols here if your corpus uses more.
KNOWN_VOWELS = [
    "ㆍ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ",
    "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]




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
        and (path / "analysis" / "data").exists()
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
    3. A parent of this script that contains analysis/data and
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

    # Useful fallback for the normal script location even before data exists.
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
# Helper functions
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Read command-line options while keeping sensible defaults for your project."""
    parser = argparse.ArgumentParser(
        description="Build a period-based, frequency-balanced word-form sample for TP testing."
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
        help=f"Original/master token CSV. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for filtered CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--year-col",
        default=YEAR_COL,
        help=f"Column containing years. Default: {YEAR_COL}",
    )
    parser.add_argument(
        "--token-col",
        default=TOKEN_COL,
        help=f"Column containing word forms. Default: {TOKEN_COL}",
    )
    parser.add_argument(
        "--vowels-col",
        default=VOWELS_COL,
        help=f"Column containing vowel sequences. Default: {VOWELS_COL}",
    )
    parser.add_argument(
        "--period-size",
        type=int,
        default=PERIOD_SIZE_YEARS,
        help="Number of years per period. Default: 25.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=(
            "First period boundary AND lower analysis-year cutoff. For example, "
            "--start-year 1400 creates periods 1400-1424, 1425-1449, etc., "
            "and drops rows dated before 1400. If omitted, the earliest observed "
            "year is floored to the nearest period-size boundary and no earlier "
            "rows are filtered out."
        ),
    )
    parser.add_argument(
        "--min-word-vowels",
        type=int,
        default=MIN_WORD_VOWELS,
        help="Minimum number of vowels a word form must have to enter the sample. Default: 2.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=CHUNKSIZE,
        help="Rows to read per chunk from the master corpus. Default: 250000.",
    )
    parser.add_argument(
        "--min-period-wordforms",
        type=int,
        default=MIN_PERIOD_WORDFORMS_FOR_BALANCING,
        help=(
            "Minimum number of unique word forms a period must have to be included "
            "in the balanced TP sample. Sparse periods are still reported in diagnostics. "
            "Default: 1000. Use 0 only if you truly want every period included."
        ),
    )
    parser.add_argument(
        "--min-period-tokens",
        type=int,
        default=MIN_PERIOD_TOKENS_FOR_BALANCING,
        help=(
            "Minimum total token count represented by usable word forms for a period "
            "to be included in the balanced TP sample. Default: 0."
        ),
    )
    parser.add_argument(
        "--max-wordforms-per-period",
        type=int,
        default=MAX_WORDFORMS_PER_PERIOD,
        help=(
            "Backward-compatible optional cap on sample size per period. In "
            "cap-preserve-periods mode, --target-wordforms-per-period is clearer; "
            "if both are supplied, --target-wordforms-per-period takes priority."
        ),
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["cap-preserve-periods", "strict-balanced"],
        default=SAMPLING_MODE,
        help=(
            "Sampling strategy. cap-preserve-periods keeps every non-manually-excluded "
            "period after --start-year and caps the maximum per-period sample size. "
            "strict-balanced excludes low-N periods and balances all remaining periods "
            "to the smallest eligible period. Default: cap-preserve-periods."
        ),
    )
    parser.add_argument(
        "--target-wordforms-per-period",
        type=int,
        default=TARGET_WORDFORMS_PER_PERIOD,
        help=(
            "Maximum number of highest-frequency word forms to keep per period in "
            "cap-preserve-periods mode. Periods with fewer forms are kept in full and "
            "flagged as low-N rather than discarded. Default: 1000."
        ),
    )
    parser.add_argument(
        "--exclude-periods",
        nargs="*",
        default=[],
        help=(
            "Optional explicit period labels to exclude from the balanced TP sample, "
            "e.g. --exclude-periods 1375-1399 1900-1924. They remain in diagnostics."
        ),
    )
    parser.add_argument(
        "--skip-token-row-output",
        action="store_true",
        help=(
            "Only write balanced_period_wordforms.csv and summaries. "
            "This is faster because it skips the second pass that preserves original token rows."
        ),
    )
    return parser.parse_args()


def require_columns(columns: Iterable[str], required_cols: Iterable[str]) -> None:
    """Fail early with a readable message if the input CSV lacks required columns."""
    available = list(columns)
    missing = [col for col in required_cols if col not in available]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing) +
            "\nAvailable columns are: " + ", ".join(available)
        )


def normalize_vowel_sequence(value: object) -> list[str]:
    """
    Convert the vowels-column value into a clean list of vowel symbols.

    This tries to handle several likely formats:
    - "ㅏ ㅗ ㅣ"
    - "ㅏ,ㅗ,ㅣ"
    - "['ㅏ', 'ㅗ', 'ㅣ']"
    - "ㅏㅗㅣ"

    The output is always a Python list, e.g. ["ㅏ", "ㅗ", "ㅣ"].
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    # Remove common list-like punctuation while preserving Hangul jamo.
    cleaned = re.sub(r"[\[\]\(\)\{\}'\"]", " ", text)

    # If the string contains separators, split on them first.
    if re.search(r"[\s,;/|]+", cleaned):
        pieces = [p for p in re.split(r"[\s,;/|]+", cleaned) if p]
        vowels = []
        for piece in pieces:
            if piece in KNOWN_VOWELS:
                vowels.append(piece)
            else:
                # Fall back to extracting known vowel symbols inside the piece.
                vowels.extend(extract_known_vowels(piece))
        return vowels

    # If there are no separators, extract known symbols in order.
    return extract_known_vowels(cleaned)


def extract_known_vowels(text: str) -> list[str]:
    """
    Extract known vowel symbols from a compact string.

    This uses a longest-symbol-first regex so composite symbols such as ㅘ or ㅢ are
    not accidentally split incorrectly if they appear in the vowels column.
    """
    escaped = sorted((re.escape(v) for v in KNOWN_VOWELS), key=len, reverse=True)
    pattern = "|".join(escaped)
    return re.findall(pattern, text)


def choose_period_anchor_from_min_year(
    min_year: int,
    period_size: int,
    start_year: Optional[int],
) -> int:
    """
    Decide the first period boundary.

    If start_year is provided, that exact value anchors the bins.
    Otherwise, the earliest observed year is rounded down to the nearest period boundary.
    """
    if start_year is not None:
        return int(start_year)
    return math.floor(int(min_year) / period_size) * period_size


def assign_period_start(year: int, anchor_year: int, period_size: int) -> int:
    """
    Assign a year to the start year of its configured period.

    Example with anchor_year=1400 and period_size=25:
    1400 -> 1400
    1424 -> 1400
    1425 -> 1425
    """
    return anchor_year + math.floor((year - anchor_year) / period_size) * period_size


def clean_and_annotate_chunk(
    chunk: pd.DataFrame,
    *,
    year_col: str,
    token_col: str,
    vowels_col: str,
    period_size: int,
    anchor_year: int,
    min_word_vowels: int,
    min_analysis_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Clean one CSV chunk and add the fields needed for frequency balancing.

    This function deliberately performs the same operations on every chunk so that
    streamed processing gives the same logical result as reading the whole CSV at once.
    """
    chunk = chunk.copy()

    # Coerce years to numeric values. Rows with unusable dates cannot be assigned to
    # a period, so they are dropped.
    chunk[year_col] = pd.to_numeric(chunk[year_col], errors="coerce")
    chunk = chunk.dropna(subset=[year_col, token_col, vowels_col])

    if chunk.empty:
        return chunk

    chunk[year_col] = chunk[year_col].astype(int)

    # Important: when --start-year is supplied, it should not merely anchor the
    # period labels. It should also act as the lower bound of the analysis window.
    # Otherwise, a small pre-start edge period such as 1375-1399 could be retained
    # and accidentally determine the balanced sample size.
    if min_analysis_year is not None:
        chunk = chunk[chunk[year_col] >= int(min_analysis_year)]

    if chunk.empty:
        return chunk

    chunk[token_col] = chunk[token_col].astype(str).str.strip()
    chunk = chunk[chunk[token_col] != ""]

    if chunk.empty:
        return chunk

    # Normalize vowels and remove word forms that cannot contribute a harmony window.
    chunk["vowel_seq_list"] = chunk[vowels_col].apply(normalize_vowel_sequence)
    chunk["num_vowels_normalized"] = chunk["vowel_seq_list"].apply(len)
    chunk = chunk[chunk["num_vowels_normalized"] >= min_word_vowels]

    if chunk.empty:
        return chunk

    chunk["vowel_seq"] = chunk["vowel_seq_list"].apply(lambda seq: " ".join(seq))

    # Assign configured period information.
    chunk["period_start"] = chunk[year_col].apply(
        lambda y: assign_period_start(int(y), anchor_year, period_size)
    )
    chunk["period_end"] = chunk["period_start"] + period_size - 1
    chunk["period_label"] = chunk["period_start"].astype(str) + "-" + chunk["period_end"].astype(str)

    # A word form is token + normalized vowel sequence. This avoids collapsing cases
    # where the same orthographic token appears with different extracted vowels.
    chunk["wordform_id"] = chunk[token_col].astype(str) + " || " + chunk["vowel_seq"]

    return chunk


def find_min_usable_year(
    input_csv: Path,
    year_col: str,
    chunksize: int,
) -> int:
    """
    First pass used only when --start-year is omitted.

    Pandas cannot assign period bins until it knows the anchor year. If the user does
    not provide that anchor, this function streams through the year column only and
    finds the earliest usable year without loading the whole corpus.
    """
    min_year: Optional[int] = None
    rows_seen = 0

    print("Finding earliest usable year because --start-year was not provided...")
    for i, chunk in enumerate(
        pd.read_csv(input_csv, usecols=[year_col], chunksize=chunksize, encoding="utf-8-sig"),
        start=1,
    ):
        rows_seen += len(chunk)
        years = pd.to_numeric(chunk[year_col], errors="coerce").dropna()
        if not years.empty:
            chunk_min = int(years.min())
            min_year = chunk_min if min_year is None else min(min_year, chunk_min)

        print(f"  scanned year chunk {i:,}; rows seen so far: {rows_seen:,}", flush=True)

    if min_year is None:
        raise ValueError(f"No usable years found in column {year_col!r}.")

    return min_year


def write_variable_descriptions(output_dir: Path) -> None:
    """Write a separate data dictionary so the output CSVs are self-documenting."""
    descriptions = [
        ("period_start", "First calendar year included in the configured period. If --start-year was supplied, no rows earlier than that year are retained."),
        ("period_end", "Last calendar year included in the configured period."),
        ("period_label", "Human-readable label for the configured period, e.g. 1425-1449."),
        ("wordform_id", "Unique identifier for a period-specific word form: token plus normalized vowel sequence."),
        ("token", "Orthographic word form from the original corpus."),
        ("vowel_seq", "Normalized vowel sequence, separated by spaces."),
        ("num_vowels_normalized", "Number of vowel symbols in vowel_seq after normalization."),
        ("token_count_in_period", "Number of token occurrences of this word form in this configured period."),
        ("first_observed_year_in_period", "Earliest token year for this word form inside the period."),
        ("last_observed_year_in_period", "Latest token year for this word form inside the period."),
        ("frequency_rank_in_period", "Rank after sorting word forms in the period by token_count_in_period descending."),
        ("sampling_mode", "Sampling strategy used by script 01: cap-preserve-periods or strict-balanced."),
        ("target_wordforms_per_period", "Requested maximum number of high-frequency word forms to keep per period in cap-preserve-periods mode."),
        ("selected_sample_size_for_period", "Actual number of word forms selected for this specific period. In cap-preserve-periods mode, this may be smaller than the target when the period has fewer available forms."),
        ("balanced_sample_size_per_period", "Backward-compatible name for selected_sample_size_for_period in the word-form output. In strict-balanced mode this is the same for every included period; in cap-preserve-periods mode it can vary for low-N periods."),
        ("period_is_low_n_warning", "TRUE if the period has fewer available word forms than --min-period-wordforms. In cap-preserve-periods mode this is a warning only, not an exclusion."),
        ("available_wordforms_before_balancing", "Number of candidate word forms available in that period before frequency balancing."),
        ("period_is_eligible_for_balancing", "TRUE if the period is included in the TP input under the chosen sampling mode."),
        ("period_exclusion_reason", "Reason a period was excluded from the TP sample, or INCLUDED if it was retained."),
        ("min_period_wordforms_for_balancing", "Threshold supplied by --min-period-wordforms. In cap-preserve-periods mode this is used as a low-N warning threshold; in strict-balanced mode it is an exclusion threshold."),
        ("min_period_tokens_for_balancing", "Threshold supplied by --min-period-tokens."),
        ("max_wordforms_per_period", "Optional cap supplied by --max-wordforms-per-period; blank means no cap."),
        ("selected_wordforms_after_balancing", "Number of word forms retained after balancing; should equal balanced_sample_size_per_period."),
        ("total_token_count_represented_before_balancing", "Total corpus token count represented by all usable word forms before balancing."),
        ("tokens_in_selected_wordforms", "Original token occurrences represented by the selected high-frequency word forms."),
        ("first_selected_year", "Earliest observed year among selected word forms in the period."),
        ("last_selected_year", "Latest observed year among selected word forms in the period."),
    ]
    out = pd.DataFrame(descriptions, columns=["variable", "description"])
    out.to_csv(output_dir / "01_balanced_sample_variable_descriptions.csv", index=False, encoding="utf-8-sig")


# =============================================================================
# Main processing
# =============================================================================


def main() -> None:
    args = parse_args()

    # Resolve default/relative paths relative to the OKHC project root, not relative
    # to whatever working directory PyCharm happens to choose. This is what allows
    # the script to work when run directly from analysis/processing_scripts.
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
            "  1. Move this script into OKHC-master/analysis/processing_scripts, or\n"
            "  2. Pass --project-root C:/Users/ashle/PycharmProjects/OKHC_Processing/OKHC-master, or\n"
            "  3. Pass --input with the full absolute CSV path."
        )

    file_size_mb = args.input.stat().st_size / 1_000_000
    print(f"Input CSV: {args.input}")
    print(f"File size: {file_size_mb:,.1f} MB")
    print(f"Chunk size: {args.chunksize:,} rows")

    # Read only the header first. This validates the file and avoids loading the full corpus.
    header = pd.read_csv(args.input, nrows=0, encoding="utf-8-sig")
    require_columns(header.columns, [args.year_col, args.token_col, args.vowels_col, "token_source"])

    if args.start_year is None:
        min_year = find_min_usable_year(args.input, args.year_col, args.chunksize)
        anchor_year = choose_period_anchor_from_min_year(min_year, args.period_size, None)
    else:
        anchor_year = args.start_year

    print(f"Using period anchor year: {anchor_year}")
    analysis_start_year = args.start_year
    if analysis_start_year is not None:
        print(f"Filtering out rows earlier than start year: {analysis_start_year}")
    else:
        print("No explicit start-year cutoff was supplied; all usable years will be retained.")
    print("Counting period-specific word forms in chunks...")

    # For the counting pass, read only the columns needed to build the balanced word-form sample.
    count_usecols = [args.year_col, args.token_col, args.vowels_col, "token_source"]

    grouped_chunks: list[pd.DataFrame] = []
    total_rows_seen = 0
    total_rows_after_cleaning = 0
    total_rows_before_start_year = 0

    for i, chunk in enumerate(
        pd.read_csv(
            args.input,
            usecols=count_usecols,
            chunksize=args.chunksize,
            encoding="utf-8-sig",
            low_memory=False,
        ),
        start=1,
    ):
        total_rows_seen += len(chunk)

        # Count pre-start rows explicitly for transparent diagnostics. This count
        # happens before general cleaning, so it may include rows that would later
        # be dropped for other reasons as well.
        if analysis_start_year is not None:
            chunk_years_for_diagnostic = pd.to_numeric(chunk[args.year_col], errors="coerce")
            total_rows_before_start_year += int((chunk_years_for_diagnostic < analysis_start_year).sum())

        cleaned = clean_and_annotate_chunk(
            chunk,
            year_col=args.year_col,
            token_col=args.token_col,
            vowels_col=args.vowels_col,
            period_size=args.period_size,
            anchor_year=anchor_year,
            min_word_vowels=args.min_word_vowels,
            min_analysis_year=analysis_start_year,
        )
        total_rows_after_cleaning += len(cleaned)

        if not cleaned.empty:
            group_cols = [
                "period_start",
                "period_end",
                "period_label",
                "wordform_id",
                args.token_col,
                "vowel_seq",
            ]
            grouped = (
                cleaned.groupby(group_cols, dropna=False)
                .agg(
                    token_count_in_period=("wordform_id", "size"),
                    first_observed_year_in_period=(args.year_col, "min"),
                    last_observed_year_in_period=(args.year_col, "max"),
                    num_vowels_normalized=("num_vowels_normalized", "first"),
                )
                .reset_index()
            )
            grouped_chunks.append(grouped)

        print(
            f"  chunk {i:,}: rows seen={total_rows_seen:,}; "
            f"usable rows so far={total_rows_after_cleaning:,}; "
            f"grouped chunks kept={len(grouped_chunks):,}",
            flush=True,
        )

    if not grouped_chunks:
        raise ValueError(
            "No rows remain after cleaning. Check the year/token/vowels columns and min-word-vowels setting."
        )

    print("Combining chunk-level word-form counts...")
    wordforms = pd.concat(grouped_chunks, ignore_index=True)

    combined_group_cols = [
        "period_start",
        "period_end",
        "period_label",
        "wordform_id",
        args.token_col,
        "vowel_seq",
    ]
    wordforms = (
        wordforms.groupby(combined_group_cols, dropna=False)
        .agg(
            token_count_in_period=("token_count_in_period", "sum"),
            first_observed_year_in_period=("first_observed_year_in_period", "min"),
            last_observed_year_in_period=("last_observed_year_in_period", "max"),
            num_vowels_normalized=("num_vowels_normalized", "first"),
        )
        .reset_index()
    )

    # Make output column names stable even if command-line names differ.
    if args.token_col != "token":
        wordforms = wordforms.rename(columns={args.token_col: "token"})

    if analysis_start_year is not None:
        print(f"Input rows dated before start year and ignored: {total_rows_before_start_year:,}")
    print(f"Usable token rows after cleaning: {total_rows_after_cleaning:,}")
    print(f"Unique period-specific word forms before balancing: {len(wordforms):,}")

    # -------------------------------------------------------------------------
    # Diagnose all periods, then select high-frequency word forms according to the
    # requested sampling mode.
    # -------------------------------------------------------------------------
    period_counts = (
        wordforms.groupby(["period_start", "period_end", "period_label"], as_index=False)
        .agg(
            available_wordforms_before_balancing=("wordform_id", "nunique"),
            total_token_count_represented_before_balancing=("token_count_in_period", "sum"),
        )
        .sort_values("period_start")
    )

    manually_excluded = set(str(p) for p in args.exclude_periods)
    period_counts["sampling_mode"] = args.sampling_mode
    period_counts["min_period_wordforms_for_balancing"] = args.min_period_wordforms
    period_counts["min_period_tokens_for_balancing"] = args.min_period_tokens
    period_counts["target_wordforms_per_period"] = args.target_wordforms_per_period
    period_counts["max_wordforms_per_period"] = args.max_wordforms_per_period
    period_counts["period_is_low_n_warning"] = (
        period_counts["available_wordforms_before_balancing"] < args.min_period_wordforms
    )

    if args.sampling_mode == "strict-balanced":
        # Older behavior: periods below thresholds are removed from the TP input,
        # then every remaining period is forced to the same sample size.
        period_counts["period_is_eligible_for_balancing"] = (
            (period_counts["available_wordforms_before_balancing"] >= args.min_period_wordforms)
            & (period_counts["total_token_count_represented_before_balancing"] >= args.min_period_tokens)
            & (~period_counts["period_label"].astype(str).isin(manually_excluded))
        )
    else:
        # New default behavior: preserve the time frame. Low-N periods are flagged
        # but not discarded, because discarding all pre-1667 periods would create
        # exactly the kind of historical gap this analysis is trying to avoid.
        period_counts["period_is_eligible_for_balancing"] = (
            (period_counts["available_wordforms_before_balancing"] > 0)
            & (period_counts["total_token_count_represented_before_balancing"] >= args.min_period_tokens)
            & (~period_counts["period_label"].astype(str).isin(manually_excluded))
        )

    def explain_period_exclusion(row: pd.Series) -> str:
        """Give a readable reason for each period's inclusion/exclusion status."""
        if str(row["period_label"]) in manually_excluded:
            return "EXCLUDED: manually listed in --exclude-periods"
        if row["total_token_count_represented_before_balancing"] < args.min_period_tokens:
            return (
                "EXCLUDED: represented token count below --min-period-tokens "
                f"({row['total_token_count_represented_before_balancing']:,} < {args.min_period_tokens:,})"
            )
        if args.sampling_mode == "strict-balanced" and row["available_wordforms_before_balancing"] < args.min_period_wordforms:
            return (
                "EXCLUDED: available word forms below --min-period-wordforms "
                f"({row['available_wordforms_before_balancing']:,} < {args.min_period_wordforms:,})"
            )
        if args.sampling_mode == "cap-preserve-periods" and row["available_wordforms_before_balancing"] < args.min_period_wordforms:
            return (
                "INCLUDED WITH LOW-N WARNING: available word forms below --min-period-wordforms "
                f"({row['available_wordforms_before_balancing']:,} < {args.min_period_wordforms:,})"
            )
        return "INCLUDED"

    period_counts["period_exclusion_reason"] = period_counts.apply(explain_period_exclusion, axis=1)

    eligible_periods = period_counts[period_counts["period_is_eligible_for_balancing"]].copy()
    excluded_periods = period_counts[~period_counts["period_is_eligible_for_balancing"]].copy()

    if eligible_periods.empty:
        raise ValueError(
            "No periods are eligible for TP sampling. Lower --min-period-tokens "
            "or remove --exclude-periods."
        )

    if args.sampling_mode == "strict-balanced":
        smallest_eligible_n = int(eligible_periods["available_wordforms_before_balancing"].min())
        selected_n = smallest_eligible_n
        if args.max_wordforms_per_period is not None:
            if args.max_wordforms_per_period <= 0:
                raise ValueError("--max-wordforms-per-period must be positive if provided.")
            selected_n = min(selected_n, int(args.max_wordforms_per_period))
        if selected_n <= 0:
            raise ValueError("Balanced sample size is zero; check eligibility thresholds.")

        period_counts["selected_sample_size_for_period"] = period_counts.apply(
            lambda row: selected_n if row["period_is_eligible_for_balancing"] else 0,
            axis=1,
        )
        print(f"Sampling mode: strict-balanced")
        print(f"Smallest ELIGIBLE period sample: {smallest_eligible_n:,} word forms.")
        if args.max_wordforms_per_period is not None and args.max_wordforms_per_period < smallest_eligible_n:
            print(f"Applying --max-wordforms-per-period cap: {selected_n:,} word forms per eligible period.")
        else:
            print(f"Every eligible period will be limited to {selected_n:,} highest-frequency word forms.")
    else:
        # In cap-preserve-periods mode, the target is an upper cap, not a lower
        # threshold. A period with 612 forms keeps all 612; a period with 60,000
        # forms keeps only the top target_n forms. This controls maximum sample
        # size without deleting sparse historical periods.
        target_n = args.target_wordforms_per_period
        if target_n is None:
            # Backward-compatible fallback. This should usually not happen because
            # the parser default is 1000.
            target_n = args.max_wordforms_per_period
        if target_n is None:
            raise ValueError(
                "cap-preserve-periods mode requires --target-wordforms-per-period "
                "or --max-wordforms-per-period."
            )
        if target_n <= 0:
            raise ValueError("--target-wordforms-per-period must be positive.")

        period_counts["selected_sample_size_for_period"] = period_counts.apply(
            lambda row: min(int(row["available_wordforms_before_balancing"]), int(target_n))
            if row["period_is_eligible_for_balancing"] else 0,
            axis=1,
        )
        print("Sampling mode: cap-preserve-periods")
        print(f"Target/cap per period: {int(target_n):,} highest-frequency word forms.")
        print(
            "Periods with fewer available word forms than this cap will be kept in full "
            "and flagged as low-N, not discarded."
        )
        low_n_count = int(period_counts["period_is_low_n_warning"].sum())
        print(f"Periods below --min-period-wordforms warning threshold: {low_n_count:,}")

    print(f"Periods found: {len(period_counts):,}")
    print(f"Periods included in TP input: {len(eligible_periods):,}")
    print(f"Periods excluded manually/by token threshold: {len(excluded_periods):,}")
    if not excluded_periods.empty:
        preview_cols = [
            "period_label",
            "available_wordforms_before_balancing",
            "total_token_count_represented_before_balancing",
            "period_exclusion_reason",
        ]
        print("Excluded-period preview:")
        print(excluded_periods[preview_cols].to_string(index=False, max_rows=20))

    # -------------------------------------------------------------------------
    # Select the highest-frequency word forms in each included period.
    # -------------------------------------------------------------------------
    wordforms = wordforms.merge(
        period_counts[[
            "period_label",
            "period_is_eligible_for_balancing",
            "period_is_low_n_warning",
            "period_exclusion_reason",
            "sampling_mode",
            "target_wordforms_per_period",
            "selected_sample_size_for_period",
        ]],
        on="period_label",
        how="left",
    )
    wordforms_for_sampling = wordforms[wordforms["period_is_eligible_for_balancing"]].copy()

    wordforms_for_sampling = wordforms_for_sampling.sort_values(
        ["period_start", "token_count_in_period", "first_observed_year_in_period", "token", "vowel_seq"],
        ascending=[True, False, True, True, True],
    )
    wordforms_for_sampling["frequency_rank_in_period"] = wordforms_for_sampling.groupby("period_label").cumcount() + 1

    # Backward-compatible column name used by earlier outputs. In cap-preserve-periods
    # mode this can vary by period; in strict-balanced mode it is constant.
    wordforms_for_sampling["balanced_sample_size_per_period"] = wordforms_for_sampling["selected_sample_size_for_period"]

    balanced_wordforms = wordforms_for_sampling[
        wordforms_for_sampling["frequency_rank_in_period"] <= wordforms_for_sampling["selected_sample_size_for_period"]
    ].copy()

    # -------------------------------------------------------------------------
    # Write the main outputs.
    # -------------------------------------------------------------------------
    balanced_wordforms_path = args.output_dir / "balanced_period_wordforms.csv"
    balanced_tokens_path = args.output_dir / "balanced_period_original_token_rows.csv"
    period_summary_path = args.output_dir / "balanced_period_summary.csv"
    excluded_periods_path = args.output_dir / "excluded_periods_due_to_low_sample.csv"
    all_period_diagnostics_path = args.output_dir / "all_period_diagnostics_before_balancing.csv"

    balanced_wordforms.to_csv(balanced_wordforms_path, index=False, encoding="utf-8-sig")
    period_counts.to_csv(all_period_diagnostics_path, index=False, encoding="utf-8-sig")
    excluded_periods.to_csv(excluded_periods_path, index=False, encoding="utf-8-sig")

    selected_summary = (
        balanced_wordforms.groupby(["period_start", "period_end", "period_label"], as_index=False)
        .agg(
            selected_wordforms_after_balancing=("wordform_id", "nunique"),
            tokens_in_selected_wordforms=("token_count_in_period", "sum"),
            first_selected_year=("first_observed_year_in_period", "min"),
            last_selected_year=("last_observed_year_in_period", "max"),
        )
    )
    period_summary = period_counts.merge(
        selected_summary,
        on=["period_start", "period_end", "period_label"],
        how="left",
    )
    period_summary.to_csv(period_summary_path, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # Optional second pass: preserve original token rows belonging to selected word forms.
    # -------------------------------------------------------------------------
    # This can take a while on a full corpus because it streams the master file again.
    # It is useful for auditing but not required by 02_run_tp_condition_tests.py.
    if not args.skip_token_row_output:
        print("Writing original token rows for selected word forms in a second chunked pass...")
        selected_key_set = set(
            balanced_wordforms["period_label"].astype(str) + "\x1e" + balanced_wordforms["wordform_id"].astype(str)
        )

        wrote_header = False
        selected_token_rows = 0
        rows_seen_second_pass = 0

        for i, chunk in enumerate(
            pd.read_csv(
                args.input,
                chunksize=args.chunksize,
                encoding="utf-8-sig",
                low_memory=False,
            ),
            start=1,
        ):
            rows_seen_second_pass += len(chunk)
            cleaned = clean_and_annotate_chunk(
                chunk,
                year_col=args.year_col,
                token_col=args.token_col,
                vowels_col=args.vowels_col,
                period_size=args.period_size,
                anchor_year=anchor_year,
                min_word_vowels=args.min_word_vowels,
                min_analysis_year=analysis_start_year,
            )

            if not cleaned.empty:
                cleaned["__selection_key"] = cleaned["period_label"].astype(str) + "\x1e" + cleaned["wordform_id"].astype(str)
                selected_rows = cleaned[cleaned["__selection_key"].isin(selected_key_set)].copy()
                selected_rows = selected_rows.drop(columns=["vowel_seq_list", "__selection_key"], errors="ignore")

                if not selected_rows.empty:
                    selected_rows.to_csv(
                        balanced_tokens_path,
                        index=False,
                        encoding="utf-8-sig",
                        mode="w" if not wrote_header else "a",
                        header=not wrote_header,
                    )
                    wrote_header = True
                    selected_token_rows += len(selected_rows)

            print(
                f"  token-row pass chunk {i:,}: rows seen={rows_seen_second_pass:,}; "
                f"selected token rows written={selected_token_rows:,}",
                flush=True,
            )

        if not wrote_header:
            # Create an empty file with a useful header if no selected token rows were found.
            pd.DataFrame().to_csv(balanced_tokens_path, index=False, encoding="utf-8-sig")

    write_variable_descriptions(args.output_dir)

    print("\nWrote frequency-balanced outputs:")
    print(f"  1. {balanced_wordforms_path}")
    print("     Main input for 02_run_tp_condition_tests.py; one row per selected word form per period.")
    if not args.skip_token_row_output:
        print(f"  2. {balanced_tokens_path}")
        print("     Original token rows represented by the selected word forms; useful for auditing.")
    else:
        print("  2. Skipped original token-row audit output because --skip-token-row-output was used.")
    print(f"  3. {period_summary_path}")
    print("     Period-level sample-size and token-count diagnostics, including eligibility status.")
    print(f"  4. {all_period_diagnostics_path}")
    print("     Diagnostics for every configured period before balancing, including sparse periods.")
    print(f"  5. {excluded_periods_path}")
    print("     Periods excluded from the TP input. In cap-preserve-periods mode, this is usually empty unless periods were manually excluded or failed the token threshold.")
    print(f"  6. {args.output_dir / '01_balanced_sample_variable_descriptions.csv'}")
    print("     Variable descriptions for the balancing outputs.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
