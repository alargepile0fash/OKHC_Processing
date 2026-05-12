import json
import csv
import unicodedata as ud
from pathlib import Path

import regex as re


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "sample_processed.jsonl"
OUTPUT_FILE = "hangul_vowel_tokens.csv"

# Set this to True if you want to exclude pure Hanmun/Hanja documents
# even when they contain Hangul readings in parentheses.
EXCLUDE_PURE_HJ = False

# Set this to a set of categories if you want to restrict extraction.
# Example:
# ALLOWED_CATEGORIES = {"ko", "oko", "hj+ko", "hj+oko"}
ALLOWED_CATEGORIES = None

# Only keep tokens with at least this many vowels.
# Use 2 for internal lexical harmony testing.
MIN_VOWELS = 2


# ============================================================
# PERIOD DEFINITIONS
# ============================================================

def assign_period(year):
    """
    Assign a document year to a rough historical period.

    Adjust these bins to match the periodization you want in your paper.
    """
    if year is None or year == "":
        return "unknown"

    try:
        year = int(float(year))
    except (ValueError, TypeError):
        return "unknown"

    if year < 1400:
        return "pre_1400"
    elif year < 1600:
        return "1400_1599"
    elif year < 1800:
        return "1600_1799"
    elif year < 1900:
        return "1800_1899"
    elif year < 1950:
        return "1900_1949"
    else:
        return "1950_present"


# ============================================================
# HANGUL TOKENIZATION
# ============================================================

# Finds continuous stretches of Hangul-script characters.
# Example:
# "一本千枝茁(일본천지줄)" -> ["일본천지줄"]
HANGUL_TOKEN_RE = re.compile(r"[\p{Hangul}]+")


# ============================================================
# VOWEL EXTRACTION
# ============================================================

# Compatibility jamo vowels.
COMPATIBILITY_VOWELS = {
    "ㅏ", "ㅐ", "ㅑ", "ㅒ",
    "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ",
    "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
    "ㆍ", "ㆎ",
}

# Normalize canonical jamo vowels into easier-to-read compatibility symbols.
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

    # Arae-a
    "ᆞ": "ㆍ",
    "ㆍ": "ㆍ",
}


# Middle Korean-style RTR classes.
# Adjust these if your analysis uses different class definitions.
PLUS_RTR = {"ㆍ", "ㅗ", "ㅏ"}
MINUS_RTR = {"ㅡ", "ㅜ", "ㅓ"}
NEUTRAL = {"ㅣ"}


def is_jungseong(char: str) -> bool:
    """
    Return True if a Unicode character is a Hangul medial vowel jamo.
    """
    name = ud.name(char, "")
    return "HANGUL JUNGSEONG" in name


def normalize_vowel(char: str) -> str:
    """
    Normalize canonical vowel jamo to readable compatibility jamo.
    """
    return VOWEL_NORMALIZATION.get(char, char)


def extract_vowels(token: str) -> list[str]:
    """
    Extract vowels from a Hangul or Old Hangul token.

    Modern Hangul syllables are decomposed first:
        한 -> ᄒ + ᅡ + ᆫ

    Then only medial vowel elements are kept.
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
    """
    Map a vowel to its harmony class.
    """
    if vowel in PLUS_RTR:
        return "+RTR"
    if vowel in MINUS_RTR:
        return "-RTR"
    if vowel in NEUTRAL:
        return "NEUTRAL"
    return "OTHER"


def classify_vowel_sequence(vowels: list[str]) -> str:
    """
    Give a rough diagnostic label for the token's vowel sequence.

    This is not the D2L learner; it is just useful for inspecting the data.
    """
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


# ============================================================
# DOCUMENT FILTERING
# ============================================================

def should_keep_document(row: dict) -> bool:
    category = row.get("category")

    if EXCLUDE_PURE_HJ and category == "hj":
        return False

    if ALLOWED_CATEGORIES is not None and category not in ALLOWED_CATEGORIES:
        return False

    return True


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_jsonl(input_file: str, output_file: str) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)

    rows_written = 0
    docs_seen = 0
    docs_used = 0

    fieldnames = [
        "doc_id",
        "year",
        "period",
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
        "vowels",
        "vowel_classes",
        "num_vowels",
        "harmony_status",
    ]

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for line in infile:
            if not line.strip():
                continue

            docs_seen += 1
            row = json.loads(line)

            if not should_keep_document(row):
                continue

            text = row.get("text", "")
            if not isinstance(text, str):
                continue

            tokens = HANGUL_TOKEN_RE.findall(text)
            if not tokens:
                continue

            doc_had_usable_token = False

            for token in tokens:
                vowels = extract_vowels(token)

                if len(vowels) < MIN_VOWELS:
                    continue

                classes = [vowel_class(v) for v in vowels]

                writer.writerow({
                    "doc_id": row.get("id"),
                    "year": row.get("year"),
                    "period": assign_period(row.get("year")),
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
                    "vowels": ",".join(vowels),
                    "vowel_classes": ",".join(classes),
                    "num_vowels": len(vowels),
                    "harmony_status": classify_vowel_sequence(vowels),
                })

                rows_written += 1
                doc_had_usable_token = True

            if doc_had_usable_token:
                docs_used += 1

    print(f"Documents seen: {docs_seen}")
    print(f"Documents with usable Hangul tokens: {docs_used}")
    print(f"Token rows written: {rows_written}")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    process_jsonl(INPUT_FILE, OUTPUT_FILE)