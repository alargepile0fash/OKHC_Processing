from pathlib import Path
import os
import sys
import shutil

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.text_preprocessing import process_dataframe
from preprocessing.classification_logic import classify_dataframe


# ============================================================
# SETTINGS
# ============================================================

# Folder containing your raw full-corpus JSONL files.
#
# You can override this without editing the file:
#   Windows PowerShell:
#       $env:OKHC_RAW_INPUT="C:\path\to\raw_data"
#   Colab / bash:
#       import os
#       os.environ["OKHC_RAW_INPUT"] = "/content/drive/MyDrive/.../raw_data"
INPUT_DIR = os.environ.get(
    "OKHC_RAW_INPUT",
    str(REPO_ROOT / "raw_data"),
)

# Folder where processed JSONL files will be written.
#
# You can override this with OKHC_PROCESSED_OUTPUT.
OUTPUT_DIR = os.environ.get(
    "OKHC_PROCESSED_OUTPUT",
    str(REPO_ROOT / "analysis" / "analyzed_data"),
)

# The script will recursively search INPUT_DIR for these file types.
INPUT_PATTERNS = ["*.jsonl", "*.jsonl.gz"]

# Number of JSONL records to process at once.
# Increase if your computer handles it easily; decrease if you run out of RAM.
CHUNK_SIZE = int(os.environ.get("OKHC_PREPROCESSING_CHUNK_SIZE", "10000"))

# If True, skip files whose processed output already exists.
SKIP_EXISTING_OUTPUT = os.environ.get("OKHC_SKIP_EXISTING_OUTPUT", "1").strip().lower() not in {
    "0", "false", "no", "n"
}


# ============================================================
# COMPATIBILITY PATCH
# ============================================================

def patch_parallel_apply():
    """
    The OKHC preprocessing code may call .parallel_apply(), but pandas does not
    provide that method by default. For large corpus processing, we safely redirect
    .parallel_apply() to normal .apply().

    This makes the original repo code run without requiring pandarallel.
    """
    if not hasattr(pd.Series, "parallel_apply"):
        pd.Series.parallel_apply = pd.Series.apply

    if not hasattr(pd.DataFrame, "parallel_apply"):
        pd.DataFrame.parallel_apply = pd.DataFrame.apply


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_input_files(input_dir: str) -> list[Path]:
    """
    Find all JSONL files in INPUT_DIR and its subfolders.
    """
    input_path = Path(input_dir)

    files = []

    for pattern in INPUT_PATTERNS:
        files.extend(input_path.rglob(pattern))

    return sorted(set(files))


def output_name_for(input_file: Path) -> str:
    """
    Create a clean output filename.

    Examples:
        part_000.jsonl    -> part_000.processed.jsonl
        part_000.jsonl.gz -> part_000.processed.jsonl
    """
    name = input_file.name

    if name.endswith(".jsonl.gz"):
        base = name[:-len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        base = name[:-len(".jsonl")]
    else:
        base = input_file.stem

    return f"{base}.processed.jsonl"


def output_path_for(input_file: Path, input_dir: str, output_dir: str) -> Path:
    """
    Preserve subfolder structure from INPUT_DIR inside OUTPUT_DIR.
    """
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    relative_parent = input_file.parent.relative_to(input_root)
    output_subdir = output_root / relative_parent
    output_subdir.mkdir(parents=True, exist_ok=True)

    return output_subdir / output_name_for(input_file)


# ============================================================
# PREPROCESSING
# ============================================================

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the dataframe has the columns the OKHC preprocessing/classification
    functions are likely to expect.

    This is defensive: it prevents some KeyErrors if a raw file is missing optional
    metadata columns.
    """
    required_defaults = {
        "id": None,
        "text": "",
        "content": None,
        "year": None,
        "language": None,
        "script": None,
        "source": None,
        "corpus": None,
        "copyright": None,
        "url": None,
        "format": None,
        "metadata": None,
        "analytics": None,
        "translation": None,
    }

    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    return df


def process_one_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Run one dataframe chunk through the OKHC preprocessing and classification logic.
    """
    chunk = ensure_required_columns(chunk)

    chunk = process_dataframe(chunk)
    chunk = classify_dataframe(chunk)

    return chunk


def process_one_file(input_file: Path, output_file: Path) -> None:
    """
    Process one raw JSONL file into one processed JSONL file.

    The output is first written to a .partial file. Once the whole file succeeds,
    the partial file is renamed to the final output name.
    """
    partial_output = output_file.with_suffix(output_file.suffix + ".partial")

    if partial_output.exists():
        partial_output.unlink()

    print()
    print("=" * 80)
    print(f"Processing input:  {input_file}")
    print(f"Writing output:    {output_file}")
    print("=" * 80)

    chunk_count = 0
    row_count = 0

    try:
        reader = pd.read_json(
            input_file,
            lines=True,
            chunksize=CHUNK_SIZE,
            compression="infer",
        )

        for chunk in tqdm(reader, desc=f"Chunks for {input_file.name}"):
            chunk_count += 1
            row_count += len(chunk)

            processed = process_one_chunk(chunk)

            processed.to_json(
                partial_output,
                orient="records",
                lines=True,
                force_ascii=False,
                mode="a",
            )

        shutil.move(partial_output, output_file)

        print(f"Finished {input_file}")
        print(f"Chunks processed: {chunk_count:,}")
        print(f"Rows processed:   {row_count:,}")

    except Exception:
        print()
        print(f"ERROR while processing {input_file}")
        print(f"Partial output left at: {partial_output}")
        raise


def preprocess_corpus(input_dir: str, output_dir: str) -> None:
    patch_parallel_apply()

    input_files = discover_input_files(input_dir)

    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {input_dir!r} matching {INPUT_PATTERNS}"
        )

    print(f"Input directory:  {Path(input_dir).resolve()}")
    print(f"Output directory: {Path(output_dir).resolve()}")
    print(f"Chunk size:       {CHUNK_SIZE:,}")
    print(f"Skip existing:    {SKIP_EXISTING_OUTPUT}")

    print(f"Found {len(input_files)} input files.")
    for path in input_files[:10]:
        print(f"  {path}")
    if len(input_files) > 10:
        print("  ...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    files_processed = 0
    files_skipped = 0

    for input_file in input_files:
        output_file = output_path_for(input_file, input_dir, output_dir)

        if SKIP_EXISTING_OUTPUT and output_file.exists():
            print(f"Skipping existing output: {output_file}")
            files_skipped += 1
            continue

        process_one_file(input_file, output_file)
        files_processed += 1

    print()
    print("Preprocessing complete.")
    print(f"Files processed: {files_processed}")
    print(f"Files skipped:   {files_skipped}")
    print(f"Processed files saved in: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    preprocess_corpus(INPUT_DIR, OUTPUT_DIR)
