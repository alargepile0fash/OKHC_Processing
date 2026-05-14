import csv
import gzip
import json
import sys
import unicodedata as ud
from pathlib import Path

import regex as re
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

# This script is located at:
# OKHC-master/analysis/processing_scripts/extract_diachronic_hangul_vowels.py
#
# parents[0] = processing_scripts
# parents[1] = analysis
# parents[2] = OKHC-master
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# SETTINGS
# ============================================================

# This reads the PROCESSED corpus files created by run_preprocessing.py
INPUT_DIR = REPO_ROOT / "analysis" / "processing_test"

# This is where analysis-ready output files go
OUTPUT_DIR = REPO_ROOT / "analysis" / "processing_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hangul_vowel_tokens_diachronic.csv"

# Recursively search INPUT_DIR for these file types
INPUT_PATTERNS = ["*.jsonl", "*.jsonl.gz"]

# 50-year bins starting in 1493
PERIOD_START_YEAR = 1493
PERIOD_SIZE = 50

# Only keep Hangul tokens with at least this many vowels
MIN_VOWELS = 2

# Optional filtering
EXCLUDE_PURE_HJ = False
ALLOWED_CATEGORIES = None
# Example:
# ALLOWED_CATEGORIES = {"ko", "oko", "hj+ko", "hj+oko"}

PROGRESS_EVERY = 100_000


# ============================================================
# HISTORICAL EVENT SETTINGS
# Adjust these later if your paper uses different dating.
# ============================================================

ARAE_NONINITIAL_MERGER_START = 1500
ARAE_NONINITIAL_MERGER_END = 1599

ARAE_INITIAL_MERGER_START = 1700
ARAE_INITIAL_MERGER_END = 1750

LMK_EMK_TRANSITION_START = 1590
LMK_EMK_TRANSITION_END = 1650


# ============================================================
# PERIODIZATION
# ============================================================

def parse_year(year_value):
    if year_value is None or year_value == "":
        return None

    try:
        return int(float(year_value))
    except (ValueError, TypeError):
        return None


def assign_50yr_period(year_value) -> str:
    year = parse_year(year_value)

    if year is None:
        return "unknown"

    if year < PERIOD_START_YEAR:
        return f"pre_{PERIOD_START_YEAR}"

    offset = year - PERIOD_START_YEAR
    period_index = offset // PERIOD_SIZE

    start = PERIOD_START_YEAR + period_index * PERIOD_SIZE
    end = start + PERIOD_SIZE - 1

    return f"{start}_{end}"


def event_phase(year_value, start_year: int, end_year: int) -> str:
    year = parse_year(year_value)

    if year is None:
        return "unknown"

    if year < start_year:
        return "before"

    if start_year <= year <= end_year:
        return "during"

    return "after"


# ============================================================
# TOKENIZATION
# ============================================================

HANGUL_TOKEN_RE = re.compile(r"[\p{Hangul}]+")
HAN_RE = re.compile(r"\p{Han}")


def classify_token_context(text: str, match) -> str:
    """
    Classify the context of a Hangul token.

    This helps distinguish plain Hangul text from likely Hanja readings,
    e.g. 一本千枝茁(일본천지줄).
    """
    start, end = match.span()

    left = text[max(0, start - 12):start]
    right = text[end:min(len(text), end + 12)]

    token_is_parenthesized = left.endswith("(") and right.startswith(")")
    han_before_parenthesis = bool(re.search(r"\p{Han}\($", left))
    han_near_left = bool(HAN_RE.search(left))
    han_near_right = bool(HAN_RE.search(right))

    if token_is_parenthesized and han_before_parenthesis:
        return "parenthetical_hanja_reading"

    if token_is_parenthesized:
        return "parenthetical_hangul"

    if han_near_left or han_near_right:
        return "near_hanja"

    return "plain_hangul"


# ============================================================
# VOWEL EXTRACTION
# ============================================================

COMPATIBILITY_VOWELS = {
    "ㅏ", "ㅐ", "ㅑ", "ㅒ",
    "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ",
    "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
    "ㆍ", "ㆎ",
}

VOWEL_NORMALIZATION = {
    "ᅡ": "ㅏ",
    "ᅢ": "ㅐ",
    "ᅣ": "ㅑ",
    "ᅤ": "ㅒ",
    "ᅥ": "ㅓ",
    "ᅦ": "ㅔ",
    "ᅧ": "ㅕ",
    "ᅨ": "ㅖ",
    "ᅩ": "ㅗ",
    "ᅪ": "ㅘ",
    "ᅫ": "ㅙ",
    "ᅬ": "ㅚ",
    "ᅭ": "ㅛ",
    "ᅮ": "ㅜ",
    "ᅯ": "ㅝ",
    "ᅰ": "ㅞ",
    "ᅱ": "ㅟ",
    "ᅲ": "ㅠ",
    "ᅳ": "ㅡ",
    "ᅴ": "ㅢ",
    "ᅵ": "ㅣ",

    # arae-a
    "ᆞ": "ㆍ",
    "ㆍ": "ㆍ",
}


PLUS_RTR = {"ㆍ", "ㅗ", "ㅏ"}
MINUS_RTR = {"ㅡ", "ㅜ", "ㅓ"}
NEUTRAL = {"ㅣ"}

ARAE_A = "ㆍ"
ARAE_INITIAL_REFLEX = "ㅏ"
ARAE_NONINITIAL_REFLEX = "ㅡ"


def is_jungseong(char: str) -> bool:
    name = ud.name(char, "")
    return "HANGUL JUNGSEONG" in name


def normalize_vowel(char: str) -> str:
    return VOWEL_NORMALIZATION.get(char, char)


def extract_vowels(token: str) -> list[str]:
    """
    Extract vowels from a Hangul/Old Hangul token.

    Example:
        일본천지줄 -> ㅣ,ㅗ,ㅓ,ㅣ,ㅜ
    """
    decomposed = ud.normalize("NFD", token)

    vowels = []

    for char in decomposed:
        if is_jungseong(char):
            vowels.append(normalize_vowel(char))
        elif char in COMPATIBILITY_VOWELS:
            vowels.append(normalize_vowel(char))

    return vowels


def vowel_class(vowel: str) -> str:
    if vowel in PLUS_RTR:
        return "+RTR"
    if vowel in MINUS_RTR:
        return "-RTR"
    if vowel in NEUTRAL:
        return "NEUTRAL"
    return "OTHER"


def classify_vowel_sequence(vowels: list[str]) -> str:
    classes = [vowel_class(v) for v in vowels]

    non_neutral_classes = [
        c for c in classes
        if c not in {"NEUTRAL", "OTHER"}
    ]

    if not non_neutral_classes:
        return "neutral_or_other_only"

    if len(set(non_neutral_classes)) == 1:
        if "NEUTRAL" in classes:
            return "harmonic_with_neutral"
        return "harmonic"

    return "disharmonic"


def arae_flags(vowels: list[str]) -> dict:
    first_vowel = vowels[0] if vowels else None
    noninitial_vowels = vowels[1:] if len(vowels) > 1 else []

    has_arae = ARAE_A in vowels
    has_a = ARAE_INITIAL_REFLEX in vowels
    has_eu = ARAE_NONINITIAL_REFLEX in vowels

    has_initial_a = first_vowel == ARAE_INITIAL_REFLEX
    has_noninitial_eu = ARAE_NONINITIAL_REFLEX in noninitial_vowels

    has_any_arae_related = has_arae or has_a or has_eu

    return {
        "has_arae_a": has_arae,
        "has_a": has_a,
        "has_eu": has_eu,
        "has_initial_a": has_initial_a,
        "has_noninitial_eu": has_noninitial_eu,
        "has_any_arae_related": has_any_arae_related,
    }


# ============================================================
# FILE HANDLING
# ============================================================

def discover_input_files(input_dir: Path) -> list[Path]:
    files = []

    for pattern in INPUT_PATTERNS:
        files.extend(input_dir.rglob(pattern))

    return sorted(set(files))


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def should_keep_document(row: dict) -> bool:
    category = row.get("category")

    if EXCLUDE_PURE_HJ and category == "hj":
        return False

    if ALLOWED_CATEGORIES is not None and category not in ALLOWED_CATEGORIES:
        return False

    return True


def bool_int(value: bool) -> int:
    return 1 if value else 0


def is_sino_proxy(row: dict, token_context: str) -> bool:
    """
    Heuristic proxy for Sino-Korean / Hanmun-related material.

    This is not proof that a token is etymologically Sino-Korean.
    It flags likely Hanmun/Hanja-reading contexts.
    """
    category = row.get("category")
    language = row.get("language")
    script_han = row.get("script.han")

    try:
        script_han_value = float(script_han)
    except (ValueError, TypeError):
        script_han_value = 0.0

    if token_context == "parenthetical_hanja_reading":
        return True

    if language == "Hanmun":
        return True

    if category in {"hj", "hj+ko", "hj+oko"}:
        return True

    if script_han_value > 0:
        return True

    return False


# ============================================================
# MAIN EXTRACTION
# ============================================================

def process_corpus(input_dir: Path, output_file: Path) -> None:
    input_files = discover_input_files(input_dir)

    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {input_dir} matching {INPUT_PATTERNS}"
        )

    print(f"Found {len(input_files)} input files.")
    for path in input_files[:10]:
        print(f"  {path}")
    if len(input_files) > 10:
        print("  ...")

    fieldnames = [
        "input_file",
        "doc_id",
        "year",
        "period_50yr",
        "arae_noninitial_phase",
        "arae_initial_phase",
        "lmk_emk_transition_phase",
        "source",
        "corpus",
        "language",
        "language_score",
        "category",
        "script",
        "script_kor",
        "script_oko",
        "script_han",
        "token",
        "token_context",
        "sino_proxy",
        "vowels",
        "vowel_classes",
        "num_vowels",
        "harmony_status",
        "has_arae_a",
        "has_a",
        "has_eu",
        "has_initial_a",
        "has_noninitial_eu",
        "has_any_arae_related",
    ]

    docs_seen = 0
    docs_used = 0
    token_rows_written = 0
    bad_json_lines = 0

    with output_file.open("w", encoding="utf-8-sig", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for input_file in tqdm(input_files, desc="Files"):
            with open_text_file(input_file) as infile:
                for line_number, line in enumerate(infile, start=1):
                    if not line.strip():
                        continue

                    docs_seen += 1

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        bad_json_lines += 1
                        continue

                    if not should_keep_document(row):
                        continue

                    text = row.get("text", "")
                    if not isinstance(text, str):
                        continue

                    matches = list(HANGUL_TOKEN_RE.finditer(text))
                    if not matches:
                        continue

                    doc_had_usable_token = False

                    for match in matches:
                        token = match.group(0)
                        vowels = extract_vowels(token)

                        if len(vowels) < MIN_VOWELS:
                            continue

                        token_context = classify_token_context(text, match)
                        flags = arae_flags(vowels)
                        classes = [vowel_class(v) for v in vowels]

                        writer.writerow({
                            "input_file": str(input_file),
                            "doc_id": row.get("id"),
                            "year": row.get("year"),
                            "period_50yr": assign_50yr_period(row.get("year")),
                            "arae_noninitial_phase": event_phase(
                                row.get("year"),
                                ARAE_NONINITIAL_MERGER_START,
                                ARAE_NONINITIAL_MERGER_END,
                            ),
                            "arae_initial_phase": event_phase(
                                row.get("year"),
                                ARAE_INITIAL_MERGER_START,
                                ARAE_INITIAL_MERGER_END,
                            ),
                            "lmk_emk_transition_phase": event_phase(
                                row.get("year"),
                                LMK_EMK_TRANSITION_START,
                                LMK_EMK_TRANSITION_END,
                            ),
                            "source": row.get("source"),
                            "corpus": row.get("corpus"),
                            "language": row.get("language"),
                            "language_score": row.get("language_score"),
                            "category": row.get("category"),
                            "script": row.get("script"),
                            "script_kor": row.get("script.kor"),
                            "script_oko": row.get("script.oko"),
                            "script_han": row.get("script.han"),
                            "token": token,
                            "token_context": token_context,
                            "sino_proxy": bool_int(is_sino_proxy(row, token_context)),
                            "vowels": ",".join(vowels),
                            "vowel_classes": ",".join(classes),
                            "num_vowels": len(vowels),
                            "harmony_status": classify_vowel_sequence(vowels),
                            "has_arae_a": bool_int(flags["has_arae_a"]),
                            "has_a": bool_int(flags["has_a"]),
                            "has_eu": bool_int(flags["has_eu"]),
                            "has_initial_a": bool_int(flags["has_initial_a"]),
                            "has_noninitial_eu": bool_int(flags["has_noninitial_eu"]),
                            "has_any_arae_related": bool_int(flags["has_any_arae_related"]),
                        })

                        token_rows_written += 1
                        doc_had_usable_token = True

                    if doc_had_usable_token:
                        docs_used += 1

                    if docs_seen % PROGRESS_EVERY == 0:
                        print(
                            f"Docs seen: {docs_seen:,} | "
                            f"Docs used: {docs_used:,} | "
                            f"Token rows: {token_rows_written:,} | "
                            f"Bad JSON: {bad_json_lines:,}"
                        )

    print()
    print("Extraction complete.")
    print(f"Documents seen: {docs_seen:,}")
    print(f"Documents with usable Hangul tokens: {docs_used:,}")
    print(f"Token rows written: {token_rows_written:,}")
    print(f"Bad JSON lines skipped: {bad_json_lines:,}")
    print(f"Saved output to: {output_file}")


if __name__ == "__main__":
    process_corpus(INPUT_DIR, OUTPUT_FILE)