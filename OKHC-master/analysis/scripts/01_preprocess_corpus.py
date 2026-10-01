"""Preprocess raw OKHC JSONL files into classified JSONL files."""

from pathlib import Path
import os
import shutil

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]

from preprocessing.text_preprocessing import process_dataframe
from preprocessing.classification_logic import classify_dataframe


INPUT_DIR = Path(os.environ.get("OKHC_RAW_INPUT", REPO_ROOT / "raw_data"))
OUTPUT_DIR = Path(os.environ.get("OKHC_PROCESSED_OUTPUT", REPO_ROOT / "analysis" / "data"))
CHUNK_SIZE = int(os.environ.get("OKHC_PREPROCESSING_CHUNK_SIZE", "10000"))
SKIP_EXISTING_OUTPUT = os.environ.get("OKHC_SKIP_EXISTING_OUTPUT", "1").lower() not in {
    "0", "false", "no", "n"
}
INPUT_PATTERNS = ["*.jsonl", "*.jsonl.gz"]


def discover_input_files(input_dir: Path) -> list[Path]:
    """Find all raw JSONL files."""
    files = []
    for pattern in INPUT_PATTERNS:
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


def process_file(input_file: Path, output_file: Path) -> None:
    """Process one raw file, using a temporary output until it succeeds."""
    partial_output = output_file.with_suffix(output_file.suffix + ".partial")
    if partial_output.exists():
        partial_output.unlink()

    reader = pd.read_json(
        input_file,
        lines=True,
        chunksize=CHUNK_SIZE,
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


def preprocess_corpus() -> None:
    """Process every raw JSONL file that does not already have an output."""
    input_files = discover_input_files(INPUT_DIR)
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {INPUT_DIR}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} input files.")

    processed = 0
    skipped = 0
    for input_file in input_files:
        output_file = output_path_for(input_file, INPUT_DIR, OUTPUT_DIR)
        if SKIP_EXISTING_OUTPUT and output_file.exists():
            skipped += 1
            continue
        process_file(input_file, output_file)
        processed += 1

    print(f"Preprocessing complete: {processed} processed, {skipped} skipped.")


if __name__ == "__main__":
    preprocess_corpus()
