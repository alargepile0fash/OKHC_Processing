"""Run the full OKHC analysis pipeline on the small raw test corpus."""

from __future__ import annotations

import argparse
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


def main() -> None:
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
    sampled_file = (
        REPO_ROOT
        / sampling_config["output_dir"]
        / "sampled_time_window_wordforms.csv"
    )

    extracted = pd.read_csv(extracted_file, encoding="utf-8-sig")
    sampled = pd.read_csv(sampled_file, encoding="utf-8-sig")

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