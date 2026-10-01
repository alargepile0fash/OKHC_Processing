"""Load and match Idu dictionary-to-Hangul correspondences."""

from __future__ import annotations

import json
import unicodedata as ud
from collections import defaultdict
from pathlib import Path

import regex as re


REPO_ROOT = Path(__file__).resolve().parents[1]
IDU_RESOURCE_CANDIDATE_DIRS = [
    REPO_ROOT / "analysis" / "scripts" / "resources",
    REPO_ROOT / "idu" / "resources",
    REPO_ROOT / "preprocessing" / "resources",
    REPO_ROOT / "analysis" / "resources",
    REPO_ROOT / "resources",
]

IDU_DICTIONARY_FILENAME = "idu_dictionary.jsonl"
IDU_EXCLUSIONS_FILENAME = "idu_exclusions.json"
IDU_NOUN_EXCLUSIONS_FILENAME = "idu_noun_exclusions.json"
IDU_HEADS_FILENAME = "idu_heads.json"

ENABLE_IDU_DICTIONARY_HANGUL_CORRESPONDENCES = True
EMIT_IDU_GENERAL_EXCLUSION_MATCHES = False
EMIT_IDU_NOUN_EXCLUSION_MATCHES = True
EMIT_ALL_IDU_READING_VARIANTS = True


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
