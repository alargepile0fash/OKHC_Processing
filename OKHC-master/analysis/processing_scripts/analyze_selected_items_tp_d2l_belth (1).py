"""
analyze_selected_items_tp_d2l_belth.py

Run Tolerance Principle and D2L-style tier analyses directly on the selected-items CSV
produced by the earlier top-wordform selection pass.

This script is designed for files like:
    tolerance_year_snapshot_selected_items.csv

Expected core columns:
    snapshot_year
    lexical_item
    item_frequency_in_snapshot_year
    representative_vowels
    sino_proxy_label

Main goals:
    1. Evaluate harmony with no tier projection, for comparison.
    2. Evaluate fixed neutral-vowel tier projections, so "remove ㅣ", "remove ㅡ ㅣ",
       etc. are directly searchable.
    3. Run a D2L-style iterative tier learner inspired by Belth's algorithm:
         - start with the full tier;
         - test a local adjacent Agree rule;
         - if it fails TP, identify vowels involved in failed tier-adjacent predictions;
         - remove the smallest configured natural class containing those bad vowels;
         - prioritize historically unstable/neutral-candidate vowels first;
         - only allow core harmony vowels to be removed in a fallback phase.
    4. Run two scoring modes:
         - wordform_level: one exception max per wordform type;
         - vowel_prediction_level: each adjacent projected vowel prediction counts.
    5. Add sample-size diagnostics and Wilson confidence intervals for reliability.

Important theoretical adaptation:
    Belth's D2L is formally defined for UR/SR alternations and alternating target segments.
    Your dataset is a corpus-level vowel-sequence dataset, not a UR/SR alternation dataset.
    This script therefore implements a transparent corpus adaptation: non-initial projected
    vowels are treated as prediction sites whose harmony class should be predicted from the
    tier-adjacent vowel to their left. Vowels implicated in failed adjacent predictions are
    candidates for tier demotion, with historical neutral candidates prioritized.

Run from OKHC-master/analysis/processing_scripts or anywhere inside the repo.
"""

from __future__ import annotations

import ast
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "analysis").exists() or parent.name == "OKHC-master":
            return parent
    # Fallback for normal placement: OKHC-master/analysis/processing_scripts/script.py
    return here.parents[2]


REPO_ROOT = find_repo_root()
ANALYZED_DIR = REPO_ROOT / "analysis" / "analyzed_data"
ANALYZED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SELECTED_ITEMS_CSV = ANALYZED_DIR / "tolerance_year_snapshot_selected_items.csv"

OUTPUT_TP_FIXED_TIERS = ANALYZED_DIR / "selected_items_tp_fixed_tier_evaluations.csv"
OUTPUT_D2L_LEARNING_PATH = ANALYZED_DIR / "selected_items_d2l_belth_learning_path.csv"
OUTPUT_D2L_FINAL_TIERS = ANALYZED_DIR / "selected_items_d2l_belth_final_tiers.csv"
OUTPUT_D2L_STEP_CANDIDATES = ANALYZED_DIR / "selected_items_d2l_belth_step_problem_vowels.csv"
OUTPUT_MASTER = ANALYZED_DIR / "selected_items_tp_d2l_master_results.csv"
OUTPUT_EXCEPTION_EXAMPLES = ANALYZED_DIR / "selected_items_tp_d2l_exception_examples.csv"
OUTPUT_SAMPLE_SIZE_DIAGNOSTICS = ANALYZED_DIR / "selected_items_sample_size_diagnostics.csv"


# ============================================================
# SETTINGS
# ============================================================

# If your selected-items CSV has a different name, change this or pass a path by editing main().
POSSIBLE_SELECTED_ITEM_FILES = [
    INPUT_SELECTED_ITEMS_CSV,
    ANALYZED_DIR / "tolerance_year_snapshot_selected_items.csv",
]

# Windows. first_N requires at least N raw vowels; whole_word requires at least 2.
VOWEL_WINDOWS = {
    "first_2_vowels": 2,
    "first_3_vowels": 3,
    "first_4_vowels": 4,
    "whole_word": None,
}

# Both requested calculation modes.
COUNTING_MODES = [
    "wordform_level",
    "vowel_prediction_level",
]

# If True, even if a vowel type is removed from the tier, an instance in word-initial
# position is retained as the trigger. This operationalizes your concern that word-initial
# vowels are the harmony trigger for the whole word.
ALWAYS_KEEP_WORD_INITIAL_VOWEL_ON_TIER = True

# Minimum sample-size diagnostics. These do not change pass/fail; they just flag reliability.
MIN_N_FOR_STABLE_INTERPRETATION = 100
MIN_N_FOR_ANY_INTERPRETATION = 30

# Number of exception examples per result group.
MAX_EXCEPTION_EXAMPLES_PER_RESULT = 20

# D2L fallback behavior.
# Primary phase only demotes historically plausible neutral/unstable candidates.
# Fallback phase permits core harmony vowels too, so you can see when a satisfactory tier
# requires removing phonologically crucial vowels.
ALLOW_CORE_HARMONY_VOWELS_ONLY_IN_FALLBACK = True

# Stop if no new vowel gets removed.
MAX_D2L_ITERATIONS = 20


# ============================================================
# VOWEL CLASSES AND DEMOTION PRIORITIES
# ============================================================

# For prediction, retained vowels must have a harmony value. ㅣ is treated as DARK if
# retained, so the analysis can explicitly compare "ㅣ participates" vs. "ㅣ is removed".
# This is deliberately transparent rather than assuming neutral status in advance.
VOWEL_TO_HARMONY_CLASS = {
    # Bright/light/yang values
    "ㆍ": "BRIGHT",
    "ᆞ": "BRIGHT",
    "ㆎ": "BRIGHT",  # historically complex arae+i; retained as bright for diagnostics
    "ㅏ": "BRIGHT",
    "ㅑ": "BRIGHT",
    "ㅐ": "BRIGHT",
    "ㅒ": "BRIGHT",
    "ㅗ": "BRIGHT",
    "ㅛ": "BRIGHT",
    "ㅚ": "BRIGHT",
    "ㅘ": "BRIGHT",
    "ㅙ": "BRIGHT",

    # Dark/yin values
    "ㅡ": "DARK",
    "ㅓ": "DARK",
    "ㅕ": "DARK",
    "ㅔ": "DARK",
    "ㅖ": "DARK",
    "ㅜ": "DARK",
    "ㅠ": "DARK",
    "ㅟ": "DARK",
    "ㅝ": "DARK",
    "ㅞ": "DARK",

    # High front vowel. If retained, classify as DARK for prediction; if neutral, D2L can remove it.
    "ㅣ": "DARK",
    "ㅢ": "DARK",
}

# Core traditional harmony vowels: treated as fully specified and protected in the primary D2L phase.
CORE_TRADITIONAL_HARMONY_VOWELS = {"ㅏ", "ㅗ", "ㅓ", "ㅜ"}

# Historically unstable or likely neutral/demotable candidates. These are prioritized.
# This is a methodological prior, not a claim that all are neutral in every period.
HISTORICAL_NEUTRAL_PRIORITY = [
    "ㅣ",  # classic neutral candidate
    "ㅡ",  # historically important and often neutral-like later
    "ㆍ", "ᆞ", "ㆎ",  # arae-a and related forms
    "ㅢ",
    "ㅐ", "ㅔ", "ㅚ", "ㅟ",  # historically complex/changed vowels included but diagnosable
]

# Secondary candidates: allowed in primary phase if implicated, but lower priority than the above.
SECONDARY_DEMOTION_CANDIDATES = [
    "ㅑ", "ㅕ", "ㅛ", "ㅠ",
    "ㅘ", "ㅙ", "ㅝ", "ㅞ", "ㅒ", "ㅖ",
]

PRIMARY_DEMOTION_CANDIDATES = set(HISTORICAL_NEUTRAL_PRIORITY + SECONDARY_DEMOTION_CANDIDATES)
ALL_KNOWN_VOWELS = set(VOWEL_TO_HARMONY_CLASS)

# Natural classes for D2L-style generalization from problematic vowels to a deletion class.
# The learner chooses the smallest eligible class containing the bad vowels and no protected vowels.
# Singletons are added dynamically, so exact-vowel demotion is always possible.
NAMED_NATURAL_CLASSES = {
    "historical_neutral_i": {"ㅣ"},
    "historical_neutral_i_eu": {"ㅣ", "ㅡ"},
    "arae_related": {"ㆍ", "ᆞ", "ㆎ"},
    "historically_complex_front": {"ㅐ", "ㅔ", "ㅚ", "ㅟ", "ㅢ"},
    "y_glide_vowels": {"ㅑ", "ㅕ", "ㅛ", "ㅠ"},
    "w_glide_vowels": {"ㅘ", "ㅙ", "ㅝ", "ㅞ"},
    "all_historical_priority": set(HISTORICAL_NEUTRAL_PRIORITY),
    "all_primary_demotion_candidates": set(PRIMARY_DEMOTION_CANDIDATES),
    "all_bright_vowels": {v for v, c in VOWEL_TO_HARMONY_CLASS.items() if c == "BRIGHT"},
    "all_dark_vowels": {v for v, c in VOWEL_TO_HARMONY_CLASS.items() if c == "DARK"},
}

# Fixed tiers you explicitly want searchable, including no-removal comparison.
FIXED_TIER_REMOVAL_SETS = [
    tuple(),
    ("ㅣ",),
    ("ㅡ",),
    ("ㆍ",),
    ("ㅣ", "ㅡ"),
    ("ㅣ", "ㆍ"),
    ("ㅡ", "ㆍ"),
    ("ㅣ", "ㅡ", "ㆍ"),
    ("ㅢ",),
    ("ㅐ",),
    ("ㅔ",),
    ("ㅚ",),
    ("ㅟ",),
    ("ㅣ", "ㅡ", "ㅢ"),
    ("ㅣ", "ㅡ", "ㆍ", "ㅢ"),
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Item:
    snapshot_year: int
    lexical_item: str
    frequency: int
    vowels: tuple[str, ...]
    sino_proxy_label: str
    representative_sequence_share: Optional[float] = None
    vowel_variant_count: Optional[int] = None
    token_source: str = "unknown"


@dataclass
class Evaluation:
    counting_mode: str
    snapshot_year: int
    sino_proxy_label: str
    vowel_window: str
    removed_vowels: frozenset[str]
    wordform_types_before_window_filter: int
    wordform_types_entering_evaluation: int
    N: int
    e: int
    theta: float
    passes_tp: bool
    exception_rate: Optional[float]
    tolerated_exception_rate: Optional[float]
    margin_rate: Optional[float]
    coverage_rate: Optional[float]
    retained_vowels_observed: frozenset[str]
    removed_vowels_observed: frozenset[str]
    remaining_tier_has_bright_vowels: bool
    remaining_tier_has_dark_vowels: bool
    failures: list[dict]
    item_records: list[dict]


# ============================================================
# UTILS
# ============================================================

def parse_vowels(value) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
    if "," in text:
        return [v.strip().strip("'\"") for v in text.split(",") if v.strip()]
    if re.search(r"\s", text):
        return [v.strip().strip("'\"") for v in text.split() if v.strip()]
    return list(text)


def norm_label(value) -> str:
    if pd.isna(value):
        return "unknown_sino_proxy"
    s = str(value).strip()
    return s if s else "unknown_sino_proxy"


def format_vowels(vowels: Iterable[str]) -> str:
    vals = [v for v in vowels if v]
    return "NONE" if not vals else " ".join(vals)


def format_removed(vowels: Iterable[str]) -> str:
    vals = sorted(set(vowels), key=vowel_sort_key)
    return "NONE" if not vals else " ".join(vals)


def vowel_sort_key(v: str) -> tuple[int, str]:
    priority = ["ㅣ", "ㅡ", "ㆍ", "ᆞ", "ㆎ", "ㅢ", "ㅐ", "ㅔ", "ㅚ", "ㅟ", "ㅏ", "ㅗ", "ㅓ", "ㅜ"]
    try:
        return (priority.index(v), v)
    except ValueError:
        return (999, v)


def harmony_class(v: str) -> str:
    return VOWEL_TO_HARMONY_CLASS.get(v, "UNKNOWN")


def tolerance_threshold(N: int) -> float:
    if N < 2:
        return 0.0
    return N / math.log(N)


def wilson_interval(e: int, N: int, z: float = 1.96) -> tuple[Optional[float], Optional[float]]:
    """Approx. 95% Wilson interval for binomial exception rate."""
    if N <= 0:
        return None, None
    p = e / N
    denom = 1 + z * z / N
    center = (p + z * z / (2 * N)) / denom
    half = (z * math.sqrt((p * (1 - p) / N) + (z * z / (4 * N * N)))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def reliability_label(e: int, N: int, tolerated_rate: Optional[float]) -> str:
    if N < 2:
        return "uninterpretable_N_less_than_2"
    if N < MIN_N_FOR_ANY_INTERPRETATION:
        return "very_small_sample"
    lo, hi = wilson_interval(e, N)
    if tolerated_rate is None or lo is None or hi is None:
        return "unknown"
    if hi < tolerated_rate:
        return "robust_pass_by_wilson_ci"
    if lo > tolerated_rate:
        return "robust_fail_by_wilson_ci"
    if N < MIN_N_FOR_STABLE_INTERPRETATION:
        return "uncertain_small_sample"
    return "uncertain_ci_overlaps_threshold"


def sample_size_warning(N: int) -> str:
    if N < 2:
        return "N_less_than_2_no_meaningful_tp_test"
    if N < MIN_N_FOR_ANY_INTERPRETATION:
        return f"N_less_than_{MIN_N_FOR_ANY_INTERPRETATION}_very_unstable"
    if N < MIN_N_FOR_STABLE_INTERPRETATION:
        return f"N_less_than_{MIN_N_FOR_STABLE_INTERPRETATION}_interpret_with_caution"
    return "ok"


def choose_input_file() -> Path:
    for p in POSSIBLE_SELECTED_ITEM_FILES:
        if p.exists():
            return p
    # Also allow running from cwd with selected-items file.
    cwd_candidates = [Path.cwd() / "tolerance_year_snapshot_selected_items.csv"]
    for p in cwd_candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find tolerance_year_snapshot_selected_items.csv. "
        f"Looked in: {[str(p) for p in POSSIBLE_SELECTED_ITEM_FILES + cwd_candidates]}"
    )


def find_col(df: pd.DataFrame, names: list[str], label: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"Could not find {label}. Tried {names}. Available columns: {list(df.columns)}")


# ============================================================
# LOAD SELECTED ITEMS
# ============================================================

def load_selected_items(input_file: Path) -> list[Item]:
    print(f"Reading selected items: {input_file}", flush=True)
    preview = pd.read_csv(input_file, nrows=5, encoding="utf-8-sig")
    year_col = find_col(preview, ["snapshot_year", "year"], "snapshot/year column")
    item_col = find_col(preview, ["lexical_item", "wordform", "token"], "lexical item column")
    freq_col = find_col(preview, ["item_frequency_in_snapshot_year", "token_frequency_in_period", "frequency"], "frequency column")
    vowel_col = find_col(preview, ["representative_vowels", "raw_vowels", "vowels", "vowel_sequence"], "vowel column")
    sino_col = find_col(preview, ["sino_proxy_label", "sino_proxy_group", "sino_proxy_stratum", "sino_proxy"], "sino-proxy column")

    optional_cols = {
        "representative_sequence_share_of_item_tokens": "representative_sequence_share",
        "number_of_vowel_sequence_variants_for_item": "vowel_variant_count",
        "token_source": "token_source",
    }

    usecols = [year_col, item_col, freq_col, vowel_col, sino_col]
    for c in optional_cols:
        if c in preview.columns:
            usecols.append(c)

    df = pd.read_csv(input_file, encoding="utf-8-sig", usecols=usecols)
    items: list[Item] = []
    skipped = 0

    for row in df.itertuples(index=False):
        d = row._asdict()
        try:
            year = int(float(d[year_col]))
        except Exception:
            skipped += 1
            continue
        lexical_item = str(d[item_col]).strip()
        if not lexical_item or lexical_item.lower() in {"nan", "none", "null"}:
            skipped += 1
            continue
        vowels = tuple(v for v in parse_vowels(d[vowel_col]) if v)
        if len(vowels) < 2:
            skipped += 1
            continue
        if any(harmony_class(v) == "UNKNOWN" for v in vowels):
            # Keep this conservative; unknown vowels cannot be TP-evaluated transparently.
            skipped += 1
            continue
        try:
            freq = int(float(d[freq_col]))
        except Exception:
            freq = 1

        share = None
        if "representative_sequence_share_of_item_tokens" in d:
            try:
                share = float(d["representative_sequence_share_of_item_tokens"])
            except Exception:
                share = None

        variant_count = None
        if "number_of_vowel_sequence_variants_for_item" in d:
            try:
                variant_count = int(float(d["number_of_vowel_sequence_variants_for_item"]))
            except Exception:
                variant_count = None

        token_source = str(d.get("token_source", "unknown"))

        items.append(Item(
            snapshot_year=year,
            lexical_item=lexical_item,
            frequency=freq,
            vowels=vowels,
            sino_proxy_label=norm_label(d[sino_col]),
            representative_sequence_share=share,
            vowel_variant_count=variant_count,
            token_source=token_source,
        ))

    print(f"Loaded {len(items):,} selected items; skipped {skipped:,} unusable rows.", flush=True)
    return items


# ============================================================
# EVALUATION
# ============================================================

def get_window_vowels(vowels: tuple[str, ...], window: str) -> Optional[tuple[str, ...]]:
    n = VOWEL_WINDOWS[window]
    if n is None:
        return vowels if len(vowels) >= 2 else None
    if len(vowels) < n:
        return None
    return vowels[:n]


def project_window_vowels(window_vowels: tuple[str, ...], removed_vowels: set[str]) -> tuple[tuple[str, int], ...]:
    """
    Return retained tier vowels as (vowel, original_window_index).
    Word-initial vowel is retained even if its symbol is in removed_vowels.
    """
    retained = []
    for i, v in enumerate(window_vowels):
        if i == 0 and ALWAYS_KEEP_WORD_INITIAL_VOWEL_ON_TIER:
            retained.append((v, i))
        elif v not in removed_vowels:
            retained.append((v, i))
    return tuple(retained)


def evaluate_group(
    items: list[Item],
    snapshot_year: int,
    sino_proxy_label: str,
    vowel_window: str,
    removed_vowels: set[str],
    counting_mode: str,
    collect_failures: bool = True,
    collect_item_records: bool = False,
) -> Evaluation:
    wordforms_before = len(items)
    wordforms_entering = 0
    N = 0
    e = 0
    candidate_wordform_count = 0
    failures: list[dict] = []
    item_records: list[dict] = []
    retained_observed: set[str] = set()
    removed_observed: set[str] = set()

    for item in items:
        wv = get_window_vowels(item.vowels, vowel_window)
        if wv is None:
            continue
        wordforms_entering += 1
        for pos, v in enumerate(wv):
            if pos > 0 and v in removed_vowels:
                removed_observed.add(v)
        projected = project_window_vowels(wv, removed_vowels)
        retained_vowels = [v for v, _pos in projected]
        retained_observed.update(retained_vowels)
        classes = [harmony_class(v) for v in retained_vowels]
        if len(projected) < 2:
            if collect_item_records:
                item_records.append({
                    "lexical_item": item.lexical_item,
                    "frequency": item.frequency,
                    "raw_vowels": format_vowels(item.vowels),
                    "window_vowels": format_vowels(wv),
                    "projected_vowels": format_vowels(retained_vowels),
                    "projected_classes": format_vowels(classes),
                    "candidate": False,
                    "disharmonic": False,
                    "exception_pairs": "",
                    "exception_class_pairs": "",
                })
            continue

        pair_count = 0
        pair_errors = 0
        exception_pairs = []
        exception_class_pairs = []
        for j in range(len(projected) - 1):
            context_v, context_pos = projected[j]
            target_v, target_pos = projected[j + 1]
            context_c = harmony_class(context_v)
            target_c = harmony_class(target_v)
            # Unknowns should have been filtered at load time, but preserve safety.
            if context_c == "UNKNOWN" or target_c == "UNKNOWN":
                continue
            pair_count += 1
            correct = context_c == target_c
            if not correct:
                pair_errors += 1
                exception_pairs.append(f"{context_v}-{target_v}")
                exception_class_pairs.append(f"{context_c}-{target_c}")
                if collect_failures:
                    failures.append({
                        "lexical_item": item.lexical_item,
                        "frequency": item.frequency,
                        "window_vowels": format_vowels(wv),
                        "projected_vowels": format_vowels(retained_vowels),
                        "context_vowel": context_v,
                        "target_vowel": target_v,
                        "context_position_in_window": context_pos,
                        "target_position_in_window": target_pos,
                        "context_class": context_c,
                        "target_class": target_c,
                    })

        if pair_count == 0:
            candidate = False
            disharmonic = False
        else:
            candidate = True
            disharmonic = pair_errors > 0

        if candidate:
            candidate_wordform_count += 1

        if counting_mode == "wordform_level":
            if candidate:
                N += 1
                e += 1 if disharmonic else 0
        elif counting_mode == "vowel_prediction_level":
            N += pair_count
            e += pair_errors
        else:
            raise ValueError(f"Unknown counting mode: {counting_mode}")

        if collect_item_records:
            item_records.append({
                "lexical_item": item.lexical_item,
                "frequency": item.frequency,
                "raw_vowels": format_vowels(item.vowels),
                "window_vowels": format_vowels(wv),
                "projected_vowels": format_vowels(retained_vowels),
                "projected_classes": format_vowels(classes),
                "candidate": candidate,
                "disharmonic": disharmonic,
                "exception_pairs": "; ".join(exception_pairs),
                "exception_class_pairs": "; ".join(exception_class_pairs),
            })

    theta = tolerance_threshold(N)
    passes = (e <= theta) if N >= 2 else False
    exception_rate = (e / N) if N else None
    tolerated_rate = (theta / N) if N else None
    margin = (exception_rate - tolerated_rate) if exception_rate is not None and tolerated_rate is not None else None
    coverage = (N / wordforms_entering) if (wordforms_entering and counting_mode == "wordform_level") else None
    if counting_mode == "vowel_prediction_level":
        # For prediction-level N, this is not type coverage. Use candidate wordform rate instead.
        coverage = (candidate_wordform_count / wordforms_entering) if wordforms_entering else None

    remaining_classes = {harmony_class(v) for v in retained_observed}

    return Evaluation(
        counting_mode=counting_mode,
        snapshot_year=snapshot_year,
        sino_proxy_label=sino_proxy_label,
        vowel_window=vowel_window,
        removed_vowels=frozenset(removed_vowels),
        wordform_types_before_window_filter=wordforms_before,
        wordform_types_entering_evaluation=wordforms_entering,
        N=N,
        e=e,
        theta=theta,
        passes_tp=passes,
        exception_rate=exception_rate,
        tolerated_exception_rate=tolerated_rate,
        margin_rate=margin,
        coverage_rate=coverage,
        retained_vowels_observed=frozenset(retained_observed),
        removed_vowels_observed=frozenset(removed_observed),
        remaining_tier_has_bright_vowels="BRIGHT" in remaining_classes,
        remaining_tier_has_dark_vowels="DARK" in remaining_classes,
        failures=failures,
        item_records=item_records,
    )


def eval_to_row(ev: Evaluation, result_type: str, extra: Optional[dict] = None) -> dict:
    ci_low, ci_high = wilson_interval(ev.e, ev.N)
    tolerated = ev.tolerated_exception_rate
    row = {
        "result_type": result_type,
        "counting_mode": ev.counting_mode,
        "snapshot_year": ev.snapshot_year,
        "sino_proxy_label": ev.sino_proxy_label,
        "vowel_window": ev.vowel_window,
        "removed_vowels_from_tier": format_removed(ev.removed_vowels),
        "removed_vowel_count": len(ev.removed_vowels),
        "removed_historical_priority_vowels": format_removed(set(ev.removed_vowels) & set(HISTORICAL_NEUTRAL_PRIORITY)),
        "removed_core_traditional_harmony_vowels": format_removed(set(ev.removed_vowels) & CORE_TRADITIONAL_HARMONY_VOWELS),
        "removes_core_traditional_harmony_vowel": bool(set(ev.removed_vowels) & CORE_TRADITIONAL_HARMONY_VOWELS),
        "word_initial_vowels_always_retained": ALWAYS_KEEP_WORD_INITIAL_VOWEL_ON_TIER,
        "wordform_types_before_window_filter": ev.wordform_types_before_window_filter,
        "wordform_types_entering_evaluation": ev.wordform_types_entering_evaluation,
        "N_for_tolerance_principle": ev.N,
        "disharmonic_exception_count_e": ev.e,
        "tolerance_principle_max_exceptions_theta_N": ev.theta,
        "passes_tolerance_principle": ev.passes_tp,
        "disharmonic_exception_rate": ev.exception_rate,
        "max_tolerated_exception_rate": ev.tolerated_exception_rate,
        "exception_rate_minus_tolerated_rate": ev.margin_rate,
        "candidate_coverage_rate_after_tier_projection": ev.coverage_rate,
        "wilson_95ci_exception_rate_low": ci_low,
        "wilson_95ci_exception_rate_high": ci_high,
        "sample_size_warning": sample_size_warning(ev.N),
        "tp_reliability_label": reliability_label(ev.e, ev.N, tolerated),
        "retained_vowels_observed_on_tier": format_vowels(sorted(ev.retained_vowels_observed, key=vowel_sort_key)),
        "removed_vowels_observed_in_window": format_vowels(sorted(ev.removed_vowels_observed, key=vowel_sort_key)),
        "remaining_tier_has_bright_vowels": ev.remaining_tier_has_bright_vowels,
        "remaining_tier_has_dark_vowels": ev.remaining_tier_has_dark_vowels,
    }
    if extra:
        row.update(extra)
    return row


# ============================================================
# FIXED TIER EVALUATIONS
# ============================================================

def run_fixed_tier_evaluations(groups: dict[tuple[int, str], list[Item]]) -> tuple[list[dict], list[dict]]:
    rows = []
    exception_examples = []
    for (year, label), items in groups.items():
        observed = set(v for item in items for v in item.vowels)
        for window in VOWEL_WINDOWS:
            for mode in COUNTING_MODES:
                for fixed in FIXED_TIER_REMOVAL_SETS:
                    removed = set(fixed)
                    ev = evaluate_group(items, year, label, window, removed, mode, collect_failures=False, collect_item_records=False)
                    requested = set(fixed)
                    rows.append(eval_to_row(ev, "fixed_tier_evaluation", {
                        "fixed_tier_requested_removed_vowels": format_removed(requested),
                        "fixed_tier_requested_vowels_present_in_group": format_removed(requested & observed),
                        "fixed_tier_requested_vowels_absent_from_group": format_removed(requested - observed),
                    }))
                    # Exception examples are generated for D2L final tiers only to keep runtime/output manageable.
    return rows, exception_examples


# ============================================================
# BELTH-STYLE D2L ADAPTATION
# ============================================================

def natural_class_candidates(observed_vowels: set[str], protected: set[str]) -> list[tuple[str, set[str]]]:
    candidates: list[tuple[str, set[str]]] = []
    for v in observed_vowels:
        if v not in protected:
            candidates.append((f"singleton_{v}", {v}))
    for name, cls in NAMED_NATURAL_CLASSES.items():
        cls2 = set(cls) & observed_vowels
        if cls2 and not (cls2 & protected):
            candidates.append((name, cls2))
    # Sort by size, then priority score, then name.
    def score(pair: tuple[str, set[str]]) -> tuple[int, int, str]:
        name, cls = pair
        priority_hits = sum(1 for v in cls if v in HISTORICAL_NEUTRAL_PRIORITY)
        # More historical-priority content is better, so negative.
        return (len(cls), -priority_hits, name)
    return sorted(candidates, key=score)


def choose_deletion_class(
    bad_vowels: set[str],
    observed_vowels: set[str],
    protected_vowels: set[str],
) -> tuple[str, set[str]]:
    """
    Belth-like update: choose smallest configured natural class containing all bad vowels
    and no protected vowels. If none exists, remove bad vowels verbatim.
    """
    bad = set(bad_vowels) & observed_vowels
    bad = bad - protected_vowels
    if not bad:
        return "no_eligible_bad_vowels", set()

    for name, cls in natural_class_candidates(observed_vowels, protected_vowels):
        if bad <= cls:
            return name, cls
    return "verbatim_bad_vowels_no_matching_natural_class", bad


def implicated_bad_vowels_from_failures(
    ev: Evaluation,
    eligible_vowels: set[str],
) -> tuple[set[str], list[dict]]:
    """
    Identify vowels implicated in failed adjacent predictions.

    Strict Belth removes bad adjacent context segments. Because this corpus adaptation does
    not have independently given alternating targets A, the diagnostic records both the
    left context and the target vowel in failed adjacent pairs, but only demotes vowels in
    eligible_vowels and never demotes word-initial positions.
    """
    bad: set[str] = set()
    rows: list[dict] = []
    for f in ev.failures:
        context_v = f["context_vowel"]
        target_v = f["target_vowel"]
        context_pos = int(f["context_position_in_window"])
        target_pos = int(f["target_position_in_window"])

        context_eligible = context_v in eligible_vowels and context_pos > 0
        target_eligible = target_v in eligible_vowels and target_pos > 0

        # Belth-like context failure: the tier-adjacent context cannot predict the target.
        if context_eligible:
            bad.add(context_v)
            rows.append({**f, "implicated_vowel": context_v, "implication_role": "failed_left_context"})

        # Corpus adaptation for neutral vowels: a target that repeatedly fails to agree may be
        # a nonparticipating neutral vowel rather than a true harmony target.
        if target_eligible:
            bad.add(target_v)
            rows.append({**f, "implicated_vowel": target_v, "implication_role": "failed_target_possible_neutral"})

    return bad, rows


def run_d2l_for_group(
    items: list[Item],
    year: int,
    label: str,
    window: str,
    counting_mode: str,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """
    Return learning_path_rows, final_row, problem_vowel_rows, exception_examples.
    """
    observed = set(v for item in items for v in item.vowels)
    primary_eligible = (observed & PRIMARY_DEMOTION_CANDIDATES) - CORE_TRADITIONAL_HARMONY_VOWELS
    fallback_eligible = observed if ALLOW_CORE_HARMONY_VOWELS_ONLY_IN_FALLBACK else primary_eligible

    removed: set[str] = set()
    accumulated_bad: set[str] = set()
    path_rows: list[dict] = []
    problem_rows: list[dict] = []
    exception_examples: list[dict] = []
    phase = "primary_historical_neutral_candidates_only"
    used_fallback = False
    outcome = "not_started"
    final_ev: Optional[Evaluation] = None

    for step in range(MAX_D2L_ITERATIONS + 1):
        ev = evaluate_group(items, year, label, window, removed, counting_mode, collect_failures=True, collect_item_records=False)
        final_ev = ev
        path_rows.append(eval_to_row(ev, "d2l_learning_step", {
            "d2l_step": step,
            "d2l_phase": phase,
            "vowels_newly_removed_this_step": "NONE" if step == 0 else path_rows[-1].get("vowels_newly_removed_next_step", "UNKNOWN"),
            "deletion_class_selected_previous_step": "NONE" if step == 0 else path_rows[-1].get("deletion_class_selected_next_step", "UNKNOWN"),
        }))

        if ev.passes_tp:
            outcome = "passed_tolerance_principle"
            break

        eligible = primary_eligible if not used_fallback else fallback_eligible
        eligible = set(eligible) - removed
        protected = set()
        if not used_fallback:
            protected |= CORE_TRADITIONAL_HARMONY_VOWELS

        bad_now, problem_details = implicated_bad_vowels_from_failures(ev, eligible)
        for pr in problem_details:
            problem_rows.append({
                "counting_mode": counting_mode,
                "snapshot_year": year,
                "sino_proxy_label": label,
                "vowel_window": window,
                "d2l_step": step,
                "d2l_phase": phase,
                "removed_vowels_before_step": format_removed(removed),
                "implicated_vowel": pr["implicated_vowel"],
                "implication_role": pr["implication_role"],
                "lexical_item": pr["lexical_item"],
                "frequency": pr["frequency"],
                "window_vowels": pr["window_vowels"],
                "projected_vowels": pr["projected_vowels"],
                "context_vowel": pr["context_vowel"],
                "target_vowel": pr["target_vowel"],
                "context_class": pr["context_class"],
                "target_class": pr["target_class"],
            })

        accumulated_bad |= bad_now
        class_name, class_to_remove = choose_deletion_class(accumulated_bad, observed, protected | removed)
        new_to_remove = class_to_remove - removed

        if not new_to_remove:
            if not used_fallback and ALLOW_CORE_HARMONY_VOWELS_ONLY_IN_FALLBACK:
                used_fallback = True
                phase = "fallback_all_vowels_core_harmony_vowels_allowed"
                accumulated_bad = set()
                continue
            outcome = "stopped_no_eligible_problem_vowels_to_remove"
            break

        # Store next-step information on current path row.
        path_rows[-1]["vowels_newly_removed_next_step"] = format_removed(new_to_remove)
        path_rows[-1]["deletion_class_selected_next_step"] = class_name
        path_rows[-1]["removed_vowels_after_next_step"] = format_removed(removed | new_to_remove)

        removed |= new_to_remove

        if not removed:
            outcome = "stopped_no_vowels_removed"
            break
    else:
        outcome = "stopped_max_iterations_reached"

    assert final_ev is not None
    final_extra = {
        "d2l_final_outcome": outcome,
        "d2l_used_core_vowel_fallback_phase": used_fallback,
        "d2l_final_phase": phase,
    }
    final_row = eval_to_row(final_ev, "d2l_final_tier", final_extra)
    if counting_mode == "wordform_level":
        final_ev_for_examples = evaluate_group(items, year, label, window, set(final_ev.removed_vowels), counting_mode, collect_failures=False, collect_item_records=True)
        exception_examples.extend(make_exception_examples(final_ev_for_examples, "d2l_final_tier"))
    return path_rows, final_row, problem_rows, exception_examples


def run_d2l(groups: dict[tuple[int, str], list[Item]]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    path_rows: list[dict] = []
    final_rows: list[dict] = []
    problem_rows: list[dict] = []
    exception_examples: list[dict] = []
    for (year, label), items in groups.items():
        for window in VOWEL_WINDOWS:
            for mode in COUNTING_MODES:
                p, f, probs, ex = run_d2l_for_group(items, year, label, window, mode)
                path_rows.extend(p)
                final_rows.append(f)
                problem_rows.extend(probs)
                exception_examples.extend(ex)
    return path_rows, final_rows, problem_rows, exception_examples


# ============================================================
# EXAMPLES AND DIAGNOSTICS
# ============================================================

def make_exception_examples(ev: Evaluation, result_type: str) -> list[dict]:
    examples = [r for r in ev.item_records if r.get("candidate") and r.get("disharmonic")]
    examples.sort(key=lambda r: (-int(r.get("frequency", 0)), str(r.get("lexical_item", ""))))
    out = []
    for r in examples[:MAX_EXCEPTION_EXAMPLES_PER_RESULT]:
        out.append({
            "result_type": result_type,
            "counting_mode": ev.counting_mode,
            "snapshot_year": ev.snapshot_year,
            "sino_proxy_label": ev.sino_proxy_label,
            "vowel_window": ev.vowel_window,
            "removed_vowels_from_tier": format_removed(ev.removed_vowels),
            "lexical_item": r["lexical_item"],
            "frequency": r["frequency"],
            "raw_vowels": r["raw_vowels"],
            "window_vowels": r["window_vowels"],
            "projected_vowels": r["projected_vowels"],
            "projected_classes": r["projected_classes"],
            "exception_pairs": r["exception_pairs"],
            "exception_class_pairs": r["exception_class_pairs"],
        })
    return out


def make_groups(items: list[Item]) -> dict[tuple[int, str], list[Item]]:
    groups: dict[tuple[int, str], list[Item]] = defaultdict(list)
    for item in items:
        groups[(item.snapshot_year, "all_items")].append(item)
        groups[(item.snapshot_year, item.sino_proxy_label)].append(item)
    # Stable sort by year and label for output reproducibility.
    return dict(sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])))


def make_sample_size_diagnostics(groups: dict[tuple[int, str], list[Item]]) -> list[dict]:
    rows = []
    for (year, label), items in groups.items():
        for window in VOWEL_WINDOWS:
            raw_entering = 0
            for item in items:
                if get_window_vowels(item.vowels, window) is not None:
                    raw_entering += 1
            rows.append({
                "snapshot_year": year,
                "sino_proxy_label": label,
                "vowel_window": window,
                "selected_wordform_type_count": len(items),
                "wordform_types_with_enough_vowels_for_window": raw_entering,
                "sample_size_warning_before_tier_projection": sample_size_warning(raw_entering),
                "one_wordform_changes_rate_by": (1 / raw_entering) if raw_entering else None,
                "recommended_action_if_unstable": recommended_action(raw_entering),
            })
    return rows


def recommended_action(N: int) -> str:
    if N < MIN_N_FOR_ANY_INTERPRETATION:
        return "do_not_interpret_alone_merge_adjacent_snapshots_or_report_descriptively"
    if N < MIN_N_FOR_STABLE_INTERPRETATION:
        return "interpret_cautiously_report_confidence_intervals_and_consider_wider_snapshot_window"
    return "ok_report_with_ci"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        print(f"Wrote empty file: {path}", flush=True)
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df):,} rows to {path}", flush=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Starting TP + Belth-style D2L analysis from selected-items CSV.", flush=True)
    print(f"Repo root: {REPO_ROOT}", flush=True)
    input_file = choose_input_file()
    items = load_selected_items(input_file)
    groups = make_groups(items)
    print(f"Analysis groups: {len(groups):,} snapshot/sino groups", flush=True)

    print("Running fixed tier TP evaluations...", flush=True)
    fixed_rows, fixed_examples = run_fixed_tier_evaluations(groups)

    print("Running Belth-style D2L tier learner...", flush=True)
    path_rows, final_rows, problem_rows, d2l_examples = run_d2l(groups)

    sample_rows = make_sample_size_diagnostics(groups)
    master_rows = fixed_rows + final_rows
    example_rows = fixed_examples + d2l_examples

    write_csv(OUTPUT_TP_FIXED_TIERS, fixed_rows)
    write_csv(OUTPUT_D2L_LEARNING_PATH, path_rows)
    write_csv(OUTPUT_D2L_FINAL_TIERS, final_rows)
    write_csv(OUTPUT_D2L_STEP_CANDIDATES, problem_rows)
    write_csv(OUTPUT_MASTER, master_rows)
    write_csv(OUTPUT_EXCEPTION_EXAMPLES, example_rows)
    write_csv(OUTPUT_SAMPLE_SIZE_DIAGNOSTICS, sample_rows)

    print("Done.", flush=True)
    print("Start with:", flush=True)
    print(f"  {OUTPUT_D2L_FINAL_TIERS}", flush=True)
    print(f"  {OUTPUT_TP_FIXED_TIERS}", flush=True)
    print(f"  {OUTPUT_SAMPLE_SIZE_DIAGNOSTICS}", flush=True)


if __name__ == "__main__":
    main()
