"""Extract historical Hangul vowel sequences from processed OKHC data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import unicodedata as ud
from pathlib import Path

import regex as re
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from idu.dictionary_correspondence import (
    build_idu_trie,
    find_nonoverlapping_longest_idu_matches,
    load_idu_resources,
    split_idu_hangul_readings,
)

DEFAULT_CONFIG = REPO_ROOT / "analysis" / "config" / "extraction_default.json"


def load_config(path: Path) -> dict:
    """Load extraction settings from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = {
        "input_dir",
        "output_file",
        "input_patterns",
        "min_vowels",
        "exclude_pure_hj",
        "allowed_categories",
        "progress_every",
        "arae_noninitial_merger_start",
        "arae_noninitial_merger_end",
        "arae_initial_merger_start",
        "arae_initial_merger_end",
        "lmk_emk_transition_start",
        "lmk_emk_transition_end",
        "idu_dictionary_hangul_correspondences",
        "emit_idu_general_exclusion_matches",
        "emit_idu_noun_exclusion_matches",
        "emit_all_idu_reading_variants",
        "exclude_idu_derived_wordforms",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing extraction config keys: {sorted(missing)}")

    config["input_dir"] = REPO_ROOT / config["input_dir"]
    config["output_file"] = REPO_ROOT / config["output_file"]

    if config["min_vowels"] < 1:
        raise ValueError("min_vowels must be at least 1.")
    if config["progress_every"] < 1:
        raise ValueError("progress_every must be at least 1.")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON configuration file (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


# ============================================================
# TOKENIZATION
# ============================================================

HANGUL_TOKEN_RE = re.compile(r"[\p{Hangul}]+")
HAN_RE = re.compile(r"\p{Han}")

COMPATIBILITY_VOWELS = {
    "ㅏ", "ㅐ", "ㅑ", "ㅒ",
    "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ",
    "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
    "ㆍ", "ㆎ",
}

VOWEL_NORMALIZATION = {
    "ᅡ": "ㅏ", "ᅢ": "ㅐ", "ᅣ": "ㅑ", "ᅤ": "ㅒ",
    "ᅥ": "ㅓ", "ᅦ": "ㅔ", "ᅧ": "ㅕ", "ᅨ": "ㅖ",
    "ᅩ": "ㅗ", "ᅪ": "ㅘ", "ᅫ": "ㅙ", "ᅬ": "ㅚ", "ᅭ": "ㅛ",
    "ᅮ": "ㅜ", "ᅯ": "ㅝ", "ᅰ": "ㅞ", "ᅱ": "ㅟ", "ᅲ": "ㅠ",
    "ᅳ": "ㅡ", "ᅴ": "ㅢ", "ᅵ": "ㅣ",
    "ᆞ": "ㆍ", "ㆍ": "ㆍ", "ㆎ": "ㆎ",
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
    """Extract normalized vowel nuclei from a Hangul/Old Hangul token."""
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

    return {
        "has_arae_a": has_arae,
        "has_a": has_a,
        "has_eu": has_eu,
        "has_initial_a": first_vowel == ARAE_INITIAL_REFLEX,
        "has_noninitial_eu": ARAE_NONINITIAL_REFLEX in noninitial_vowels,
        "has_any_arae_related": has_arae or has_a or has_eu,
    }


def event_phase(year, start: int, end: int) -> str:
    """Classify a year relative to a configurable historical interval."""
    try:
        year = int(year)
    except (TypeError, ValueError):
        return "unknown"

    if year < start:
        return "before"
    if year <= end:
        return "during"
    return "after"


# ============================================================
# FILE HANDLING
# ============================================================

def discover_input_files(input_dir: Path, input_patterns: list[str]) -> list[Path]:
    files = []
    for pattern in input_patterns:
        files.extend(input_dir.rglob(pattern))
    return sorted(set(files))


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def classify_token_context(text: str, match) -> str:
    """Classify a Hangul token's local context."""
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


def should_keep_document(row: dict, config: dict) -> bool:
    category = row.get("category")

    if config["exclude_pure_hj"] and category == "hj":
        return False

    allowed = config["allowed_categories"]
    if allowed is not None and category not in set(allowed):
        return False

    return True


def bool_int(value: bool) -> int:
    return 1 if value else 0


def is_sino_proxy(row: dict, token_context: str, token_source: str = "original_hangul_token") -> bool:
    """Flag likely Hanmun/Hanja-related contexts; this is not etymological proof."""
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


def base_output_row(row: dict, input_file: Path, config: dict) -> dict:
    year = row.get("year")
    return {
        "input_file": str(input_file),
        "doc_id": row.get("id"),
        "year": year,
        "arae_noninitial_phase": event_phase(
            year,
            config["arae_noninitial_merger_start"],
            config["arae_noninitial_merger_end"],
        ),
        "arae_initial_phase": event_phase(
            year,
            config["arae_initial_merger_start"],
            config["arae_initial_merger_end"],
        ),
        "lmk_emk_transition_phase": event_phase(
            year,
            config["lmk_emk_transition_start"],
            config["lmk_emk_transition_end"],
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
    }


def build_vowel_row(
    *,
    row: dict,
    input_file: Path,
    token: str,
    token_source: str,
    token_context: str,
    vowels: list[str],
    config: dict,
    idu_metadata: dict | None = None,
) -> dict:
    flags = arae_flags(vowels)
    classes = [vowel_class(v) for v in vowels]
    idu_metadata = idu_metadata or {}

    return {
        **base_output_row(row, input_file, config),
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


# ============================================================
# MAIN EXTRACTION
# ============================================================

def process_corpus(config: dict) -> None:
    input_dir = config["input_dir"]
    output_file = config["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(input_dir, config["input_patterns"])
    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {input_dir} matching {config['input_patterns']}"
        )

    if config["idu_dictionary_hangul_correspondences"]:
        idu_resources = load_idu_resources(
            emit_general_exclusion_matches=config["emit_idu_general_exclusion_matches"],
            emit_noun_exclusion_matches=config["emit_idu_noun_exclusion_matches"],
        )
    else:
        idu_resources = {"enabled": False}

    idu_trie = (
        build_idu_trie(idu_resources["dictionary_by_idu_text"])
        if idu_resources.get("enabled")
        else {}
    )

    print(f"Found {len(input_files)} input files.")
    for path in input_files[:10]:
        print(f"  {path}")
    if len(input_files) > 10:
        print("  ...")

    fieldnames = [
        "input_file", "doc_id", "year",
        "arae_noninitial_phase", "arae_initial_phase", "lmk_emk_transition_phase",
        "source", "corpus", "language", "language_score", "category",
        "script", "script_kor", "script_oko", "script_han",
        "token", "token_source", "token_context", "sino_proxy",
        "vowels", "vowel_classes", "num_vowels", "harmony_status",
        "has_arae_a", "has_a", "has_eu", "has_initial_a",
        "has_noninitial_eu", "has_any_arae_related",
        "idu_match_text", "idu_match_start", "idu_match_end",
        "idu_dictionary_sequence", "idu_hanja_text", "idu_hangul_text_raw",
        "idu_hangul_reading_used", "idu_reading_source_note",
        "idu_is_general_exclusion", "idu_is_noun_exclusion",
    ]

    docs_seen = 0
    docs_used = 0
    original_hangul_token_rows_written = 0
    idu_correspondence_rows_written = 0
    bad_json_lines = 0
    idu_matches_seen = 0
    idu_matches_with_usable_vowels = 0
    idu_wordforms_excluded = 0

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

                    if not should_keep_document(row, config):
                        continue

                    text = row.get("text", "")
                    if not isinstance(text, str):
                        continue

                    doc_had_usable_token = False

                    for match in HANGUL_TOKEN_RE.finditer(text):
                        token = match.group(0)
                        vowels = extract_vowels(token)

                        if len(vowels) < config["min_vowels"]:
                            continue

                        token_context = classify_token_context(text, match)
                        writer.writerow(build_vowel_row(
                            row=row,
                            input_file=input_file,
                            token=token,
                            token_source="original_hangul_token",
                            token_context=token_context,
                            vowels=vowels,
                            config=config,
                        ))

                        original_hangul_token_rows_written += 1
                        doc_had_usable_token = True

                    if idu_trie:
                        idu_matches = find_nonoverlapping_longest_idu_matches(text, idu_trie)
                        idu_matches_seen += len(idu_matches)

                        for idu_match in idu_matches:
                            entries = idu_match["entries"]
                            if not config["emit_all_idu_reading_variants"]:
                                entries = entries[:1]

                            for entry in entries:
                                for reading, reading_note in split_idu_hangul_readings(entry["hangul_text"]):
                                    vowels = extract_vowels(reading)

                                    if len(vowels) < config["min_vowels"]:
                                        continue

                                    idu_matches_with_usable_vowels += 1

                                    if config["exclude_idu_derived_wordforms"]:
                                        idu_wordforms_excluded += 1
                                        continue

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
                                        config=config,
                                        idu_metadata=idu_metadata,
                                    ))

                                    idu_correspondence_rows_written += 1
                                    doc_had_usable_token = True

                    if doc_had_usable_token:
                        docs_used += 1

                    if docs_seen % config["progress_every"] == 0:
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
    print(f"Idu-derived wordforms excluded: {idu_wordforms_excluded:,}")
    print(f"Bad JSON lines skipped: {bad_json_lines:,}")
    print(f"Saved output to: {output_file}")


if __name__ == "__main__":
    args = parse_args()
    process_corpus(load_config(args.config))
