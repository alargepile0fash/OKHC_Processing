"""Preprocess raw OKHC JSONL files into classified JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.text_preprocessing import process_dataframe
from preprocessing.classification_logic import classify_dataframe


DEFAULT_CONFIG = REPO_ROOT / "analysis" / "config" / "preprocessing_default.json"


def load_config(path: Path) -> dict:
    """Load preprocessing settings from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = {"input_dir", "output_dir", "chunk_size", "skip_existing_output", "input_patterns"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing preprocessing config keys: {sorted(missing)}")

    config["input_dir"] = REPO_ROOT / config["input_dir"]
    config["output_dir"] = REPO_ROOT / config["output_dir"]
    config["chunk_size"] = int(config["chunk_size"])
    config["skip_existing_output"] = bool(config["skip_existing_output"])

    if config["chunk_size"] < 1:
        raise ValueError("chunk_size must be at least 1.")
    if not config["input_patterns"]:
        raise ValueError("input_patterns must contain at least one pattern.")

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


def discover_input_files(input_dir: Path, input_patterns: list[str]) -> list[Path]:
    """Find all raw JSONL files."""
    files = []
    for pattern in input_patterns:
        files.extend(input_dir.rglob(pattern))
    return sorted(set(files))


def output_path_for(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Keep the raw directory structure in the processed output."""
    relative_parent = input_file.parent.relative_to(input_dir)
    output_dir = output_dir / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_file.name.endswith(".jsonl.gz"):
        name = input_file.name[:-9]
    else:
        name = input_file.stem
    return output_dir / f"{name}.processed.jsonl"


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional OKHC columns when a raw file does not contain them."""
    defaults = {
        "id": None,
        "text": "",
        "year": None,
        "language": None,
        "script": None,
        "source": None,
        "corpus": None,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Run one chunk through normalization, language ID, and classification."""
    chunk = ensure_required_columns(chunk)
    return classify_dataframe(process_dataframe(chunk))


def process_file(input_file: Path, output_file: Path, chunk_size: int) -> None:
    """Process one raw file, using a temporary output until it succeeds."""
    partial_output = output_file.with_suffix(output_file.suffix + ".partial")
    if partial_output.exists():
        partial_output.unlink()

    reader = pd.read_json(
        input_file,
        lines=True,
        chunksize=chunk_size,
        compression="infer",
    )

    try:
        for chunk in tqdm(reader, desc=input_file.name):
            process_chunk(chunk).to_json(
                partial_output,
                orient="records",
                lines=True,
                force_ascii=False,
                mode="a",
            )
        shutil.move(partial_output, output_file)
    except Exception:
        print(f"Error while processing {input_file}.")
        print(f"Partial output left at: {partial_output}")
        raise


def preprocess_corpus(config: dict) -> None:
    """Process every raw JSONL file that does not already have an output."""
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    input_patterns = config["input_patterns"]

    input_files = discover_input_files(input_dir, input_patterns)
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} input files.")

    processed = 0
    skipped = 0
    for input_file in input_files:
        output_file = output_path_for(input_file, input_dir, output_dir)
        if config["skip_existing_output"] and output_file.exists():
            skipped += 1
            continue
        process_file(input_file, output_file, config["chunk_size"])
        processed += 1

    print(f"Preprocessing complete: {processed} processed, {skipped} skipped.")


if __name__ == "__main__":
    args = parse_args()
    preprocess_corpus(load_config(args.config))
