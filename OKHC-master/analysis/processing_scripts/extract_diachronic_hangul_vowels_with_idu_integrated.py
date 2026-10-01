import csv
import gzip
import json
import sys
import unicodedata as ud
from collections import defaultdict
from pathlib import Path

import regex as re
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

# This script is intended to be located at:
# OKHC-master/analysis/processing_scripts/extract_diachronic_hangul_vowels_with_idu_integrated.py
#
# parents[0] = processing_scripts
# parents[1] = analysis
# parents[2] = OKHC-master
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# SETTINGS
# ============================================================

# This reads the PROCESSED corpus files created by run_preprocessing.py.
INPUT_DIR = REPO_ROOT / "analysis" / "analyzed_data"

# This is where analysis-ready output files go.
OUTPUT_DIR = REPO_ROOT / "analysis" / "analyzed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hangul_vowel_tokens_diachronic.csv"

# Recursively search INPUT_DIR for these file types.
INPUT_PATTERNS = ["*.jsonl", "*.jsonl.gz"]

# Keep this legacy column for backward compatibility with older analysis scripts.
# Newer TP/D2L scripts should rebuild 25-year snapshots from the raw year column.
PERIOD_START_YEAR = 1493
PERIOD_SIZE = 50

# Only keep token/readings with at least this many vowels.
MIN_VOWELS = 2

# Optional filtering.
EXCLUDE_PURE_HJ = False
ALLOWED_CATEGORIES = None
# Example:
# ALLOWED_CATEGORIES = {"ko", "oko", "hj+ko", "hj+oko"}

PROGRESS_EVERY = 100_000


# ============================================================
# IDU / KOREAN-STYLE SINITIC SETTINGS
# ============================================================

# Idu resources may be placed in any of these directories.
IDU_RESOURCE_CANDIDATE_DIRS = [
    Path(__file__).resolve().parent / "resources",
    REPO_ROOT / "resources",
    REPO_ROOT / "preprocessing" / "resources",
    REPO_ROOT / "analysis" / "resources",
    REPO_ROOT / "idu" / "resources"
]

IDU_DICTIONARY_FILENAME = "idu_dictionary.jsonl"
IDU_EXCLUSIONS_FILENAME = "idu_exclusions.json"
IDU_NOUN_EXCLUSIONS_FILENAME = "idu_noun_exclusions.json"
IDU_HEADS_FILENAME = "idu_heads.json"

# Emit rows produced from idu_dictionary.jsonl: idu_text -> hangul_text.
# These rows have token_source = idu_dictionary_hangul_correspondence.
ENABLE_IDU_DICTIONARY_HANGUL_CORRESPONDENCES = True

# General exclusions are forms that the Idu classifier treats as false-positive-ish
# for grammatical Idu. For vowel-harmony extraction, the safest default is to skip
# them as dictionary-derived rows, while still extracting ordinary Hangul tokens
# from the text as usual.
EMIT_IDU_GENERAL_EXCLUSION_MATCHES = False

# Noun exclusions are not skipped. They are emitted but flagged with
# idu_is_noun_exclusion = 1. This preserves potentially relevant Korean-style
# Sinitic lexical material while keeping it filterable later.
EMIT_IDU_NOUN_EXCLUSION_MATCHES = True

# If the same idu_text has multiple dictionary rows/readings, emit all unique
# readings by default. This preserves ambiguity for later filtering.
EMIT_ALL_IDU_READING_VARIANTS = True


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
    """
    Legacy 50-year period label retained for backward compatibility.
    The newer analysis scripts should use the raw year column and construct
    25-year snapshots directly.
    """
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
    "ㆎ": "ㆎ",
}

PLUS_RTR = {"ㆍ", "ㅗ", "ㅏ", "ㅐ", "ㅑ", "ㅚ", "ㅛ"}
MINUS_RTR = {"ㅡ", "ㅜ", "ㅓ", "ㅔ", "ㅕ", "ㅟ", "ㅠ"}
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
# IDU RESOURCE LOADING AND MATCHING
# ============================================================

def _load_json_or_jsonl(path: Path):
    if path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(json.loads(s))
        return out

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_resource_path(filename: str) -> Path | None:
    for directory in IDU_RESOURCE_CANDIDATE_DIRS:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def normalize_letters_only(text: str) -> str:
    text = ud.normalize("NFKC", str(text or ""))
    return "".join(ch for ch in text if ud.category(ch).startswith("L"))


def load_idu_resources() -> dict:
    """
    Load Idu dictionary resources.

    The crucial resource is idu_dictionary.jsonl, which is expected to contain
    idu_text and hangul_text fields.
    """
    paths = {
        "dictionary": find_resource_path(IDU_DICTIONARY_FILENAME),
        "exclusions": find_resource_path(IDU_EXCLUSIONS_FILENAME),
        "noun_exclusions": find_resource_path(IDU_NOUN_EXCLUSIONS_FILENAME),
        "heads": find_resource_path(IDU_HEADS_FILENAME),
    }

    if paths["dictionary"] is None:
        print("Idu dictionary not found; Idu-derived Hangul correspondences disabled.")
        return {
            "enabled": False,
            "dictionary_by_idu_text": {},
            "general_exclusions": set(),
            "noun_exclusions": set(),
            "resource_paths": paths,
        }

    general_exclusions = set()
    if paths["exclusions"] is not None:
        raw_exclusions = _load_json_or_jsonl(paths["exclusions"])
        if isinstance(raw_exclusions, list):
            general_exclusions = {normalize_letters_only(x) for x in raw_exclusions if str(x).strip()}
        elif isinstance(raw_exclusions, dict):
            vals = []
            for v in raw_exclusions.values():
                vals.extend(v if isinstance(v, list) else [v])
            general_exclusions = {normalize_letters_only(x) for x in vals if str(x).strip()}

    noun_exclusions = set()
    if paths["noun_exclusions"] is not None:
        raw_nouns = _load_json_or_jsonl(paths["noun_exclusions"])
        if isinstance(raw_nouns, list):
            for item in raw_nouns:
                if isinstance(item, dict):
                    value = item.get("text", "")
                else:
                    value = item
                norm = normalize_letters_only(value)
                if norm:
                    noun_exclusions.add(norm)

    raw_dictionary = _load_json_or_jsonl(paths["dictionary"])

    dictionary_by_idu_text = defaultdict(list)
    for item in raw_dictionary:
        if not isinstance(item, dict):
            continue

        idu_text = normalize_letters_only(item.get("idu_text", ""))
        hangul_text = str(item.get("hangul_text", "") or "").strip()

        if not idu_text or not hangul_text:
            continue

        if idu_text in general_exclusions and not EMIT_IDU_GENERAL_EXCLUSION_MATCHES:
            continue

        if idu_text in noun_exclusions and not EMIT_IDU_NOUN_EXCLUSION_MATCHES:
            continue

        entry = {
            "sequence": item.get("sequence"),
            "idu_text": idu_text,
            "hanja_text": str(item.get("hanja_text", "") or "").strip(),
            "hangul_text": hangul_text,
            "is_general_exclusion": idu_text in general_exclusions,
            "is_noun_exclusion": idu_text in noun_exclusions,
        }

        dictionary_by_idu_text[idu_text].append(entry)

    # Deduplicate exact duplicate readings under each idu_text.
    for idu_text, entries in list(dictionary_by_idu_text.items()):
        seen = set()
        deduped = []
        for entry in entries:
            key = (entry["hangul_text"], entry["hanja_text"], entry["sequence"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        dictionary_by_idu_text[idu_text] = deduped

    print("Loaded Idu resources:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print(f"  dictionary idu_text keys: {len(dictionary_by_idu_text):,}")
    print(f"  general exclusions loaded: {len(general_exclusions):,}")
    print(f"  noun exclusions loaded: {len(noun_exclusions):,}")

    return {
        "enabled": True,
        "dictionary_by_idu_text": dict(dictionary_by_idu_text),
        "general_exclusions": general_exclusions,
        "noun_exclusions": noun_exclusions,
        "resource_paths": paths,
    }


TRIE_TERMINAL = "_entries"


def build_idu_trie(dictionary_by_idu_text: dict[str, list[dict]]) -> dict:
    trie = {}

    for idu_text, entries in dictionary_by_idu_text.items():
        node = trie
        for char in idu_text:
            node = node.setdefault(char, {})
        node.setdefault(TRIE_TERMINAL, []).extend(entries)

    return trie


def find_nonoverlapping_longest_idu_matches(text: str, trie: dict) -> list[dict]:
    """
    Find longest non-overlapping Idu dictionary matches in NFKC-normalized text.

    This is a dependency-free alternative to Aho-Corasick. It favors longer
    dictionary entries over their shorter substrings.
    """
    if not trie:
        return []

    normalized_text = ud.normalize("NFKC", text or "")
    n = len(normalized_text)
    matches = []

    i = 0
    while i < n:
        node = trie
        j = i
        best = None

        while j < n and normalized_text[j] in node:
            node = node[normalized_text[j]]
            j += 1
            if TRIE_TERMINAL in node:
                best = {
                    "start": i,
                    "end": j,
                    "match_text": normalized_text[i:j],
                    "entries": node[TRIE_TERMINAL],
                }

        if best is not None:
            matches.append(best)
            i = best["end"]
        else:
            i += 1

    return matches


def split_idu_hangul_readings(hangul_text: str) -> list[tuple[str, str]]:
    """
    Return one or more Hangul reading candidates from dictionary hangul_text.

    Many dictionary entries contain a single Hangul string. Some contain obvious
    separators; those are split. If there are no separators, the entire string is
    preserved as one candidate rather than guessing boundaries.
    """
    raw = str(hangul_text or "").strip()
    if not raw:
        return []

    # Split only on explicit separators. Do not guess boundaries in concatenated
    # strings, because that would introduce unvalidated readings.
    pieces = re.split(r"[\s,;/|]+", raw)
    pieces = [p.strip() for p in pieces if p.strip()]

    if len(pieces) > 1:
        return [(p, "dictionary_hangul_text_split_on_explicit_separator") for p in pieces]

    # Flag long separatorless strings because they may contain concatenated variants.
    note = "dictionary_hangul_text_single"
    if len(raw) >= 7:
        note = "dictionary_hangul_text_single_or_possible_concatenated_variants"

    return [(raw, note)]


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


def is_sino_proxy(row: dict, token_context: str, token_source: str = "original_hangul_token") -> bool:
    """
    Heuristic proxy for Sino-Korean / Hanmun-related material.

    This is not proof that a token is etymologically Sino-Korean.
    It flags likely Hanmun/Hanja-reading contexts.

    Idu dictionary-derived Hangul correspondences are always marked as a
    Sino/Korean-style Sinitic proxy because they are recovered from Idu/Hanja
    dictionary matches.
    """
    if token_source == "idu_dictionary_hangul_correspondence":
        return True

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


def base_output_row(row: dict, input_file: Path) -> dict:
    year = row.get("year")
    return {
        "input_file": str(input_file),
        "doc_id": row.get("id"),
        "year": year,
        "period_50yr": assign_50yr_period(year),
        "arae_noninitial_phase": event_phase(year, ARAE_NONINITIAL_MERGER_START, ARAE_NONINITIAL_MERGER_END),
        "arae_initial_phase": event_phase(year, ARAE_INITIAL_MERGER_START, ARAE_INITIAL_MERGER_END),
        "lmk_emk_transition_phase": event_phase(year, LMK_EMK_TRANSITION_START, LMK_EMK_TRANSITION_END),
        "source": row.get("source"),
        "corpus": row.get("corpus"),
        "language": row.get("language"),
        "language_score": row.get("language_score"),
        "category": row.get("category"),
        "script": row.get("script"),
        "script_kor": row.get("script.kor"),
        "script_oko": row.get("script.oko"),
        "script_han": row.get("script.han"),
    }


def build_vowel_row(
    *,
    row: dict,
    input_file: Path,
    token: str,
    token_source: str,
    token_context: str,
    vowels: list[str],
    idu_metadata: dict | None = None,
) -> dict:
    flags = arae_flags(vowels)
    classes = [vowel_class(v) for v in vowels]
    idu_metadata = idu_metadata or {}

    out = {
        **base_output_row(row, input_file),
        "token": token,
        "token_source": token_source,
        "token_context": token_context,
        "sino_proxy": bool_int(is_sino_proxy(row, token_context, token_source)),
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

        # Idu-specific metadata. Empty for ordinary Hangul tokens.
        "idu_match_text": idu_metadata.get("idu_match_text", ""),
        "idu_match_start": idu_metadata.get("idu_match_start", ""),
        "idu_match_end": idu_metadata.get("idu_match_end", ""),
        "idu_dictionary_sequence": idu_metadata.get("idu_dictionary_sequence", ""),
        "idu_hanja_text": idu_metadata.get("idu_hanja_text", ""),
        "idu_hangul_text_raw": idu_metadata.get("idu_hangul_text_raw", ""),
        "idu_hangul_reading_used": idu_metadata.get("idu_hangul_reading_used", ""),
        "idu_reading_source_note": idu_metadata.get("idu_reading_source_note", ""),
        "idu_is_general_exclusion": bool_int(bool(idu_metadata.get("idu_is_general_exclusion", False))),
        "idu_is_noun_exclusion": bool_int(bool(idu_metadata.get("idu_is_noun_exclusion", False))),
    }

    return out


# ============================================================
# MAIN EXTRACTION
# ============================================================

def process_corpus(input_dir: Path, output_file: Path) -> None:
    input_files = discover_input_files(input_dir)

    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {input_dir} matching {INPUT_PATTERNS}"
        )

    idu_resources = load_idu_resources() if ENABLE_IDU_DICTIONARY_HANGUL_CORRESPONDENCES else {"enabled": False}
    idu_trie = build_idu_trie(idu_resources["dictionary_by_idu_text"]) if idu_resources.get("enabled") else {}

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
        "token_source",
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
        "idu_match_text",
        "idu_match_start",
        "idu_match_end",
        "idu_dictionary_sequence",
        "idu_hanja_text",
        "idu_hangul_text_raw",
        "idu_hangul_reading_used",
        "idu_reading_source_note",
        "idu_is_general_exclusion",
        "idu_is_noun_exclusion",
    ]

    docs_seen = 0
    docs_used = 0
    original_hangul_token_rows_written = 0
    idu_correspondence_rows_written = 0
    bad_json_lines = 0
    idu_matches_seen = 0
    idu_matches_with_usable_vowels = 0

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

                    doc_had_usable_token = False

                    # --------------------------------------------------------
                    # 1. Original fully-Hangul token extraction.
                    # --------------------------------------------------------
                    for match in HANGUL_TOKEN_RE.finditer(text):
                        token = match.group(0)
                        vowels = extract_vowels(token)

                        if len(vowels) < MIN_VOWELS:
                            continue

                        token_context = classify_token_context(text, match)

                        writer.writerow(build_vowel_row(
                            row=row,
                            input_file=input_file,
                            token=token,
                            token_source="original_hangul_token",
                            token_context=token_context,
                            vowels=vowels,
                        ))

                        original_hangul_token_rows_written += 1
                        doc_had_usable_token = True

                    # --------------------------------------------------------
                    # 2. Idu/Hanja dictionary-derived Hangul correspondences.
                    # --------------------------------------------------------
                    if idu_trie:
                        idu_matches = find_nonoverlapping_longest_idu_matches(text, idu_trie)
                        idu_matches_seen += len(idu_matches)

                        for idu_match in idu_matches:
                            entries = idu_match["entries"]
                            if not EMIT_ALL_IDU_READING_VARIANTS:
                                entries = entries[:1]

                            for entry in entries:
                                for reading, reading_note in split_idu_hangul_readings(entry["hangul_text"]):
                                    vowels = extract_vowels(reading)

                                    if len(vowels) < MIN_VOWELS:
                                        continue

                                    idu_matches_with_usable_vowels += 1

                                    idu_metadata = {
                                        "idu_match_text": idu_match["match_text"],
                                        "idu_match_start": idu_match["start"],
                                        "idu_match_end": idu_match["end"],
                                        "idu_dictionary_sequence": entry.get("sequence"),
                                        "idu_hanja_text": entry.get("hanja_text"),
                                        "idu_hangul_text_raw": entry.get("hangul_text"),
                                        "idu_hangul_reading_used": reading,
                                        "idu_reading_source_note": reading_note,
                                        "idu_is_general_exclusion": entry.get("is_general_exclusion", False),
                                        "idu_is_noun_exclusion": entry.get("is_noun_exclusion", False),
                                    }

                                    writer.writerow(build_vowel_row(
                                        row=row,
                                        input_file=input_file,
                                        token=reading,
                                        token_source="idu_dictionary_hangul_correspondence",
                                        token_context="idu_dictionary_match",
                                        vowels=vowels,
                                        idu_metadata=idu_metadata,
                                    ))

                                    idu_correspondence_rows_written += 1
                                    doc_had_usable_token = True

                    if doc_had_usable_token:
                        docs_used += 1

                    if docs_seen % PROGRESS_EVERY == 0:
                        print(
                            f"Docs seen: {docs_seen:,} | "
                            f"Docs used: {docs_used:,} | "
                            f"Original Hangul rows: {original_hangul_token_rows_written:,} | "
                            f"Idu correspondence rows: {idu_correspondence_rows_written:,} | "
                            f"Idu matches seen: {idu_matches_seen:,} | "
                            f"Bad JSON: {bad_json_lines:,}"
                        )

    print()
    print("Extraction complete.")
    print(f"Documents seen: {docs_seen:,}")
    print(f"Documents with usable tokens/readings: {docs_used:,}")
    print(f"Original Hangul token rows written: {original_hangul_token_rows_written:,}")
    print(f"Idu correspondence rows written: {idu_correspondence_rows_written:,}")
    print(f"Idu dictionary matches seen: {idu_matches_seen:,}")
    print(f"Idu matches with usable vowel readings: {idu_matches_with_usable_vowels:,}")
    print(f"Bad JSON lines skipped: {bad_json_lines:,}")
    print(f"Saved output to: {output_file}")


if __name__ == "__main__":
    process_corpus(INPUT_DIR, OUTPUT_FILE)
