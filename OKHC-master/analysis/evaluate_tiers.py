import math
import itertools
from dataclasses import dataclass

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "hangul_vowel_tokens.csv"
OUTPUT_FILE = "tier_results_by_period.csv"

# If True, exclude rows where period == unknown.
EXCLUDE_UNKNOWN_PERIOD = True

# Optional category filtering.
# Example:
# ALLOWED_CATEGORIES = {"ko", "oko", "hj+ko", "hj+oko"}
ALLOWED_CATEGORIES = None

# If True, exclude pure Hanmun/Hanja category rows.
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

    for k in range(len(vowels) + 1):
        for excluded_tuple in itertools.combinations(vowels, k):
            excluded = set(excluded_tuple)

            result = evaluate_tier(
                period=period,
                vowel_sequences=vowel_sequences,
                excluded=excluded,
                counting_mode=counting_mode,
            )

            if result.satisfies_tp:
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

    df = df[df.apply(should_keep_row, axis=1)].copy()

    period_to_sequences = {}

    for period, group in df.groupby("period"):
        sequences = []

        for value in group["vowels"].dropna():
            vowels = parse_vowel_sequence(value)

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
    period_to_sequences = load_vowel_sequences_by_period(INPUT_FILE)

    if not period_to_sequences:
        print("No vowel sequences found after filtering.")
        return

    all_results = []

    for period in sorted(period_to_sequences.keys()):
        vowel_sequences = period_to_sequences[period]

        print()
        print("=" * 70)
        print(f"PERIOD: {period}")
        print(f"Number of vowel sequences: {len(vowel_sequences)}")
        print("=" * 70)

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
    results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print(f"Saved period results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()