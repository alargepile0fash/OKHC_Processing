import math
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# This script is located at:
# OKHC-master/analysis/processing_scripts/evaluate_diachronic_tiers.py
#
# parents[0] = processing_scripts
# parents[1] = analysis
# parents[2] = OKHC-master
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# SETTINGS
# ============================================================

ANALYSIS_OUTPUT_DIR = REPO_ROOT / "analysis" / "analyzed_data"
ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# This is the CSV made by extract_diachronic_hangul_vowels.py
INPUT_FILE = ANALYSIS_OUTPUT_DIR / "hangul_vowel_tokens_diachronic.csv"

# Output files
RESULTS_OUTPUT = ANALYSIS_OUTPUT_DIR / "tier_results_by_period.csv"

EXCLUDE_UNKNOWN_PERIOD = True

# Optional filtering
ALLOWED_CATEGORIES = None
# Example:
# ALLOWED_CATEGORIES = {"ko", "oko", "hj+ko", "hj+oko"}

EXCLUDE_PURE_HJ = False


# ============================================================
# VOWEL CLASSES
# ============================================================

PLUS_RTR = {"ㆍ", "ㅗ", "ㅏ"}
MINUS_RTR = {"ㅡ", "ㅜ", "ㅓ"}
NEUTRAL = {"ㅣ"}

ALL_VOWELS = PLUS_RTR | MINUS_RTR | NEUTRAL


def rtr_class(vowel: str) -> str | None:
    """
    Return the harmony class of a vowel.
    """
    if vowel in PLUS_RTR:
        return "+"
    if vowel in MINUS_RTR:
        return "-"
    if vowel in NEUTRAL:
        return "neutral"
    return None


# ============================================================
# TOLERANCE PRINCIPLE
# ============================================================

def tolerance_threshold(n: int) -> float:
    """
    Yang-style tolerance threshold: n / ln(n).
    """
    if n <= 1:
        return 0.0

    return n / math.log(n)


def satisfies_tp(n: int, c: int) -> bool:
    """
    sat(g,V) = n(g,V) - c(g,V) <= n(g,V) / ln(n(g,V))
    """
    if n <= 1:
        return c == n

    exceptions = n - c
    return exceptions <= tolerance_threshold(n)


# ============================================================
# TIER EVALUATION
# ============================================================

def project_vowels(vowels: list[str], excluded: set[str]) -> list[str]:
    """
    Remove excluded vowels from the tier.
    """
    return [v for v in vowels if v not in excluded]


def evaluate_token_prediction_level(vowels: list[str], excluded: set[str]) -> tuple[int, int]:
    """
    Prediction-level D2L-style counting.

    Each adjacent pair on the projected tier counts as one prediction.

    Returns:
        n = number of predictions
        c = number of correct predictions
    """
    tier = project_vowels(vowels, excluded)

    n = 0
    c = 0

    for left, right in zip(tier, tier[1:]):
        left_class = rtr_class(left)
        right_class = rtr_class(right)

        # Ignore vowels outside the theoretical inventory for now.
        if left_class is None or right_class is None:
            continue

        # If a neutral vowel is still on the tier, it causes a failed prediction.
        # If the tier excludes neutral vowels, these comparisons disappear.
        if left_class == "neutral" or right_class == "neutral":
            n += 1
            continue

        n += 1

        if left_class == right_class:
            c += 1

    return n, c


def evaluate_token_word_level(vowels: list[str], excluded: set[str]) -> tuple[int, int]:
    """
    Word/token-level TP-style counting.

    Each token counts once.
    A token is counted as correct only if all of its relevant predictions are correct.
    """
    n_pred, c_pred = evaluate_token_prediction_level(vowels, excluded)

    # If the token has no usable predictions, do not count it.
    if n_pred == 0:
        return 0, 0

    if n_pred == c_pred:
        return 1, 1

    return 1, 0


@dataclass
class TierResult:
    period: str
    counting_mode: str
    excluded: tuple[str, ...]
    tier: tuple[str, ...]
    n: int
    c: int
    exceptions: int
    threshold: float
    accuracy: float
    satisfies_tp: bool


def evaluate_tier(
    period: str,
    vowel_sequences: list[list[str]],
    excluded: set[str],
    counting_mode: str,
) -> TierResult:
    total_n = 0
    total_c = 0

    for vowels in vowel_sequences:
        if counting_mode == "prediction":
            n, c = evaluate_token_prediction_level(vowels, excluded)
        elif counting_mode == "word":
            n, c = evaluate_token_word_level(vowels, excluded)
        else:
            raise ValueError("counting_mode must be 'prediction' or 'word'")

        total_n += n
        total_c += c

    exceptions = total_n - total_c
    threshold = tolerance_threshold(total_n)
    accuracy = total_c / total_n if total_n else 0.0

    return TierResult(
        period=period,
        counting_mode=counting_mode,
        excluded=tuple(sorted(excluded)),
        tier=tuple(sorted(ALL_VOWELS - excluded)),
        n=total_n,
        c=total_c,
        exceptions=exceptions,
        threshold=threshold,
        accuracy=accuracy,
        satisfies_tp=satisfies_tp(total_n, total_c),
    )


def find_least_stipulative_tier(
    period: str,
    vowel_sequences: list[list[str]],
    counting_mode: str,
) -> TierResult | None:
    """
    Search candidate tiers from least stipulative to most stipulative.

    Least stipulative = smallest number of excluded vowels.
    """
    vowels = sorted(ALL_VOWELS)

    candidate_exclusions = []

    for r in range(len(vowels) + 1):
        for excluded_tuple in itertools.combinations(vowels, r):
            candidate_exclusions.append(excluded_tuple)

    print(
        f"Searching {len(candidate_exclusions)} candidate tiers "
        f"for period={period}, counting_mode={counting_mode}",
        flush=True,
    )

    for i, excluded_tuple in enumerate(candidate_exclusions, start=1):
        if i == 1 or i % 10 == 0 or i == len(candidate_exclusions):
            print(
                f"  Testing candidate {i}/{len(candidate_exclusions)}: "
                f"excluded={excluded_tuple}",
                flush=True,
            )

        excluded = set(excluded_tuple)

        result = evaluate_tier(
            period=period,
            vowel_sequences=vowel_sequences,
            excluded=excluded,
            counting_mode=counting_mode,
        )

        if result.satisfies_tp:
            print(
                f"  Found satisfying tier at candidate {i}/{len(candidate_exclusions)}",
                flush=True,
            )
            return result

    return None

# ============================================================
# DATA LOADING
# ============================================================

def should_keep_row(row: pd.Series) -> bool:
    """
    Decide whether a token row should be included in the analysis.
    """
    if EXCLUDE_UNKNOWN_PERIOD and row.get("period") == "unknown":
        return False

    if EXCLUDE_PURE_HJ and row.get("category") == "hj":
        return False

    if ALLOWED_CATEGORIES is not None and row.get("category") not in ALLOWED_CATEGORIES:
        return False

    return True


def parse_vowel_sequence(value) -> list[str]:
    """
    Convert a CSV vowel field like 'ㅣ,ㅗ,ㅓ' into ['ㅣ', 'ㅗ', 'ㅓ'].
    """
    if pd.isna(value):
        return []

    return [
        v.strip()
        for v in str(value).split(",")
        if v.strip()
    ]


def load_vowel_sequences_by_period(input_file: str) -> dict[str, list[list[str]]]:
    df = pd.read_csv(input_file)

    period_column = "period_50yr"

    if period_column not in df.columns:
        raise KeyError(
            f"Expected column '{period_column}' in {input_file}, "
            f"but found columns: {df.columns.tolist()}"
        )

    if EXCLUDE_UNKNOWN_PERIOD:
        df = df[df[period_column] != "unknown"].copy()

    if EXCLUDE_PURE_HJ and "category" in df.columns:
        df = df[df["category"] != "hj"].copy()

    if ALLOWED_CATEGORIES is not None and "category" in df.columns:
        df = df[df["category"].isin(ALLOWED_CATEGORIES)].copy()

    period_to_sequences = {}

    for period, group in df.groupby(period_column):
        sequences = []

        for value in group["vowels"].dropna():
            vowels = [v.strip() for v in str(value).split(",") if v.strip()]

            if len(vowels) >= 2:
                sequences.append(vowels)

        period_to_sequences[period] = sequences

    return period_to_sequences


# ============================================================
# OUTPUT HELPERS
# ============================================================

def result_to_dict(result: TierResult | None, period: str, counting_mode: str) -> dict:
    if result is None:
        return {
            "period": period,
            "counting_mode": counting_mode,
            "excluded": None,
            "tier": None,
            "n": 0,
            "c": 0,
            "exceptions": None,
            "threshold": None,
            "accuracy": None,
            "satisfies_tp": False,
        }

    return {
        "period": result.period,
        "counting_mode": result.counting_mode,
        "excluded": " ".join(result.excluded),
        "tier": " ".join(result.tier),
        "n": result.n,
        "c": result.c,
        "exceptions": result.exceptions,
        "threshold": result.threshold,
        "accuracy": result.accuracy,
        "satisfies_tp": result.satisfies_tp,
    }


def print_result(result: TierResult | None, period: str, counting_mode: str) -> None:
    if result is None:
        print(f"{period} / {counting_mode}: No tier satisfied the TP.")
        return

    print(f"{period} / {counting_mode}")
    print(f"  Excluded vowels:   {result.excluded}")
    print(f"  Remaining tier:    {result.tier}")
    print(f"  n:                 {result.n}")
    print(f"  c:                 {result.c}")
    print(f"  exceptions:        {result.exceptions}")
    print(f"  threshold n/ln(n): {result.threshold:.4f}")
    print(f"  accuracy:          {result.accuracy:.4f}")
    print(f"  satisfies TP:      {result.satisfies_tp}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting evaluate_diachronic_tiers.py...", flush=True)
    print(f"Input file: {INPUT_FILE}", flush=True)

    period_to_sequences = load_vowel_sequences_by_period(INPUT_FILE)

    if not period_to_sequences:
        print("No vowel sequences found after filtering.", flush=True)
        return

    print(f"Loaded {len(period_to_sequences)} periods.", flush=True)

    for period, sequences in period_to_sequences.items():
        print(
            f"Period {period}: {len(sequences)} vowel sequences loaded.",
            flush=True,
        )

    all_results = []

    for period in sorted(period_to_sequences.keys()):
        vowel_sequences = period_to_sequences[period]

        print("=" * 80, flush=True)
        print(f"Starting period: {period}", flush=True)
        print(f"Number of sequences: {len(vowel_sequences)}", flush=True)

        for counting_mode in ["prediction", "word"]:
            print(f"Starting counting mode: {counting_mode}", flush=True)

            result = find_least_stipulative_tier(
                period=period,
                vowel_sequences=vowel_sequences,
                counting_mode=counting_mode,
            )

            print()
            print_result(result, period, counting_mode)

            all_results.append(
                result_to_dict(
                    result=result,
                    period=period,
                    counting_mode=counting_mode,
                )
            )

        for counting_mode in ["prediction", "word"]:
            result = find_least_stipulative_tier(
                period=period,
                vowel_sequences=vowel_sequences,
                counting_mode=counting_mode,
            )

            print()
            print_result(result, period, counting_mode)

            all_results.append(
                result_to_dict(
                    result=result,
                    period=period,
                    counting_mode=counting_mode,
                )
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8-sig")

    print()
    print(f"Saved period results to {RESULTS_OUTPUT}")


if __name__ == "__main__":
    main()