"""Run the full OKHC analysis pipeline on the small raw test corpus."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "analysis" / "config" / "testing" / "test_pipeline.json"


def load_config(path: Path) -> dict:
    """Load test-pipeline settings from JSON."""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = {
        "preprocessing_config",
        "extraction_config",
        "sampling_config",
        "output_dir",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing test pipeline config keys: {sorted(missing)}")

    config["preprocessing_config"] = REPO_ROOT / config["preprocessing_config"]
    config["extraction_config"] = REPO_ROOT / config["extraction_config"]
    config["sampling_config"] = REPO_ROOT / config["sampling_config"]
    config["output_dir"] = REPO_ROOT / config["output_dir"]

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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(script: Path, *extra_args: str) -> None:
    """Run one pipeline stage and stop immediately if it fails."""
    command = [sys.executable, str(script), *extra_args]
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def test_harmony_classification() -> None:
    """Check the core/expanded distinction on representative vowel sequences."""
    module_path = REPO_ROOT / "analysis" / "scripts" / "02_extract_diachronic_vowels.py"
    spec = importlib.util.spec_from_file_location("extract_diachronic_vowels", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the extraction module for classification tests.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = [
        (["ㅏ", "ㅗ"], "harmonic", "harmonic"),
        (["ㅏ", "ㅣ", "ㅗ"], "harmonic_with_neutral", "harmonic_with_neutral"),
        (["ㅏ", "ㅓ"], "disharmonic", "disharmonic"),
        (["ㅏ", "ㅑ"], "unclassifiable_due_to_other", "harmonic"),
        (["ㅏ", "ㅘ"], "unclassifiable_due_to_other", "harmonic"),
        (["ㅘ", "ㅝ"], "unclassifiable_due_to_other", "disharmonic"),
        (["ㅏ", "ㅢ"], "unclassifiable_due_to_other", "disharmonic"),
    ]

    for vowels, expected_core, expected_expanded in cases:
        core = module.classify_vowel_sequence(vowels, "core")
        expanded = module.classify_vowel_sequence(vowels, "expanded")
        if core != expected_core or expanded != expected_expanded:
            raise AssertionError(
                f"Classification failed for {''.join(vowels)}: "
                f"core={core!r} (expected {expected_core!r}), "
                f"expanded={expanded!r} (expected {expected_expanded!r})"
            )

    print("PASS: core and nucleus-based expanded harmony classifications behave as expected.")


def test_idu_match_offsets() -> None:
    """Verify Idu matching preserves offsets in the original Unicode text."""
    from idu.dictionary_correspondence import (
        build_idu_trie,
        find_nonoverlapping_longest_idu_matches,
    )

    trie = build_idu_trie({
        "A": [{"sequence": 1, "hangul_text": "가"}],
        "AB": [{"sequence": 2, "hangul_text": "나다"}],
    })
    matches = find_nonoverlapping_longest_idu_matches("xＡB y", trie)

    if len(matches) != 1:
        raise AssertionError(
            f"Idu offset test expected one longest match, got {len(matches)}."
        )

    match = matches[0]
    if match["start"] != 1 or match["end"] != 3 or match["match_text"] != "AB":
        raise AssertionError(
            "Idu offset test failed: normalized matching did not preserve "
            "the original-text span."
        )

    print("PASS: normalized Idu matching preserves original-text offsets.")


def test_sampling_max_word_vowels() -> None:
    """Exercise the upper word-vowel bound with a deterministic synthetic chunk."""
    from argparse import Namespace
    from analysis.scripts.sampling_logic.corpus import prepare_chunk

    args = Namespace(
        year_col="year",
        token_col="token",
        vowels_col="vowels",
        start_year=None,
        min_word_vowels=2,
        max_word_vowels=2,
        window_width=25,
        window_step=25,
    )
    chunk = pd.DataFrame([
        {"year": 1500, "token": "aa", "vowels": "ㅏ,ㅗ", "token_source": "original_hangul_token"},
        {"year": 1500, "token": "bbb", "vowels": "ㅏ,ㅗ,ㅏ", "token_source": "original_hangul_token"},
    ])
    prepared = prepare_chunk(chunk, args, anchor_year=1500)

    if set(prepared["token"]) != {"aa"}:
        raise AssertionError(
            "max_word_vowels test failed: a three-vowel wordform survived "
            "the configured two-vowel upper bound."
        )

    print("PASS: max_word_vowels upper bound excludes over-limit wordforms.")


def main() -> None:
    test_harmony_classification()
    test_idu_match_offsets()
    test_sampling_max_word_vowels()
    config = load_config(parse_args().config)

    if config["output_dir"].exists():
        shutil.rmtree(config["output_dir"])

    config["output_dir"].mkdir(parents=True)

    run(
        REPO_ROOT / "analysis" / "scripts" / "01_preprocess_corpus.py",
        "--config",
        str(config["preprocessing_config"]),
    )
    run(
        REPO_ROOT / "analysis" / "scripts" / "02_extract_diachronic_vowels.py",
        "--config",
        str(config["extraction_config"]),
    )
    run(
        REPO_ROOT / "analysis" / "scripts" / "03_prepare_time_window_samples.py",
        "--config",
        str(config["sampling_config"]),
    )

    extraction_config = load_json(config["extraction_config"])
    sampling_config = load_json(config["sampling_config"])

    extracted_file = REPO_ROOT / extraction_config["output_file"]
    sampling_output_dir = (
        REPO_ROOT
        / sampling_config["runs_dir"]
        / sampling_config["run_name"]
    )
    sampled_file = sampling_output_dir / "sampled_time_window_wordforms.csv"
    extracted = pd.read_csv(extracted_file, encoding="utf-8-sig")
    sampled = pd.read_csv(sampled_file, encoding="utf-8-sig")
    required_harmony_columns = {
        "vowel_classes_core",
        "harmony_status_core",
        "vowel_classes_expanded",
        "harmony_status_expanded",
    }
    missing_harmony_columns = required_harmony_columns - set(extracted.columns)
    if missing_harmony_columns:
        raise AssertionError(
            "Extraction output is missing harmony classification columns: "
            f"{sorted(missing_harmony_columns)}"
        )
    print("PASS: extracted output contains all core and expanded harmony columns.")

    for column in ("vowel_classes", "harmony_status"):
        if column not in extracted.columns:
            raise AssertionError(
                f"Extraction output is missing backward-compatible column: {column}"
            )
    print("PASS: backward-compatible expanded harmony columns are present.")


    # The sampling stage should preserve the extraction-stage harmony columns.
    missing_sampled_harmony_columns = required_harmony_columns - set(sampled.columns)
    if missing_sampled_harmony_columns:
        raise AssertionError(
            "Sampled output is missing harmony classification columns: "
            f"{sorted(missing_sampled_harmony_columns)}"
        )
    print("PASS: sampled output preserves all core and expanded harmony columns.")

    # Idu-derived dictionary readings are excluded from the analytical extraction
    # dataset when the extraction config enables that exclusion.
    if extraction_config["exclude_idu_derived_wordforms"]:
        if "token_source" not in extracted.columns:
            raise AssertionError(
                "Cannot verify Idu exclusion: extraction output is missing token_source."
            )
        idu_rows = int(
            (extracted["token_source"] == "idu_dictionary_hangul_correspondence").sum()
        )
        if idu_rows != 0:
            raise AssertionError(
                "Idu-derived wordforms were expected to be excluded, but "
                f"{idu_rows:,} rows remain in the extraction output."
            )
        print("PASS: no Idu-derived wordforms remain in the extraction output.")


    expected_max = sampling_config["max_word_vowels"]
    if expected_max is None:
        raise ValueError("The test sampling config must set max_word_vowels.")

    extracted_over_max = int(
        (extracted["num_vowels"] > expected_max).sum()
    )
    sampled_over_max = int(
        (sampled["num_vowels_normalized"] > expected_max).sum()
    )

    print("\n=== TEST RESULT ===")
    print(f"Extracted rows: {len(extracted):,}")
    print(
        f"Extracted rows with >{expected_max} vowels: "
        f"{extracted_over_max:,}"
    )
    print(f"Sampled word forms: {len(sampled):,}")
    print(
        f"Sampled word forms with >{expected_max} vowels: "
        f"{sampled_over_max:,}"
    )

    if sampled_over_max != 0:
        raise AssertionError(
            f"The max_word_vowels={expected_max} filter failed: "
            "sampled output contains a word form above the configured limit."
        )

    if extracted_over_max == 0:
        print(
            f"WARNING: the fixture produced no extracted rows with >{expected_max} "
            "vowels, so the upper-limit filter was not directly exercised."
        )
    else:
        print(
            f"PASS: every extracted >{expected_max}-vowel row was excluded "
            "by sampling."
        )

    print(f"Test outputs: {config['output_dir']}")


if __name__ == "__main__":
    main()