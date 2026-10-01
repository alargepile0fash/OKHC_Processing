"""Run the full OKHC analysis pipeline on the small raw test corpus.

This keeps test outputs under analysis/tests/output/ and never touches the
production raw_data/ or analysis/data/ directories.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "analysis" / "tests"
OUTPUT_DIR = TEST_DIR / "output"

RAW_INPUT = TEST_DIR
PROCESSED_OUTPUT = OUTPUT_DIR / "processed"
EXTRACTED_OUTPUT = OUTPUT_DIR / "vowels"
SAMPLED_OUTPUT = OUTPUT_DIR / "samples"

SAMPLING_CONFIG = REPO_ROOT / "analysis" / "config" / "sampling_test.json"


def run(script: Path, env: dict[str, str], *extra_args: str) -> None:
    """Run one pipeline stage and stop immediately if it fails."""
    command = [sys.executable, str(script), *extra_args]
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    sample_file = TEST_DIR / "sample.jsonl"
    if not sample_file.exists():
        raise FileNotFoundError(f"Test fixture not found: {sample_file}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    PROCESSED_OUTPUT.mkdir(parents=True)
    EXTRACTED_OUTPUT.mkdir(parents=True)
    SAMPLED_OUTPUT.mkdir(parents=True)

    env = os.environ.copy()
    env["OKHC_RAW_INPUT"] = str(RAW_INPUT)
    env["OKHC_PROCESSED_OUTPUT"] = str(PROCESSED_OUTPUT)
    env["OKHC_SKIP_EXISTING_OUTPUT"] = "0"
    env["OKHC_EXTRACT_INPUT"] = str(PROCESSED_OUTPUT)
    env["OKHC_EXTRACT_OUTPUT"] = str(EXTRACTED_OUTPUT)

    run(
        REPO_ROOT / "analysis" / "scripts" / "01_preprocess_corpus.py",
        env,
    )
    run(
        REPO_ROOT / "analysis" / "scripts" / "02_extract_diachronic_vowels.py",
        env,
    )
    run(
        REPO_ROOT / "analysis" / "scripts" / "03_prepare_time_window_samples.py",
        env,
        "--config",
        str(SAMPLING_CONFIG),
    )

    extracted_file = EXTRACTED_OUTPUT / "hangul_vowel_tokens_diachronic.csv"
    sampled_file = SAMPLED_OUTPUT / "sampled_time_window_wordforms.csv"

    extracted = pd.read_csv(extracted_file, encoding="utf-8-sig")
    sampled = pd.read_csv(sampled_file, encoding="utf-8-sig")

    extracted_over_5 = int((extracted["num_vowels"] > 5).sum())
    sampled_over_5 = int((sampled["num_vowels_normalized"] > 5).sum())

    print("\n=== TEST RESULT ===")
    print(f"Extracted rows: {len(extracted):,}")
    print(f"Extracted rows with >5 vowels: {extracted_over_5:,}")
    print(f"Sampled word forms: {len(sampled):,}")
    print(f"Sampled word forms with >5 vowels: {sampled_over_5:,}")

    if sampled_over_5 != 0:
        raise AssertionError(
            "The max_word_vowels=5 filter failed: sampled output contains "
            "a word form with more than 5 vowels."
        )

    if extracted_over_5 == 0:
        print(
            "WARNING: the sample produced no extracted rows with >5 vowels, "
            "so the new upper-limit filter was not directly exercised."
        )
    else:
        print("PASS: every extracted >5-vowel row was excluded by sampling.")

    print(f"Test outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
