import os
import pandas as pd

from preprocessing.text_preprocessing import process_dataframe
from preprocessing.classification_logic import classify_dataframe


INPUT_FILE = "sample.jsonl"
OUTPUT_FILE = "sample_processed.jsonl"
CHUNK_SIZE = 10000


def main():
    print(f"Loading {INPUT_FILE} in chunks...")

    # Delete old output file so we do not accidentally append to a previous run
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    reader = pd.read_json(INPUT_FILE, lines=True, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader, start=1):
        print(f"\nProcessing chunk {i}...")

        if i == 1:
            print("Original columns:")
            print(chunk.columns.tolist())

        print("Running text preprocessing...")
        chunk = process_dataframe(chunk)

        print("Running classification...")
        chunk = classify_dataframe(chunk)

        if i == 1:
            print("Processed columns:")
            print(chunk.columns.tolist())

        print(f"Appending chunk {i} to {OUTPUT_FILE}...")
        chunk.to_json(
            OUTPUT_FILE,
            orient="records",
            lines=True,
            force_ascii=False,
            mode="a",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()