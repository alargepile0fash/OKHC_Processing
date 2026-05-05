import pandas as pd
import regex as re


INPUT_FILE = "sample_processed.jsonl"
OUTPUT_FILE = "sample_hangul_relevant.jsonl"


# Hangul script includes modern Hangul syllables and jamo, including Old Hangul jamo.
HANGUL_TOKEN_RE = re.compile(r"[\p{Hangul}]+")


def has_hangul_token(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(HANGUL_TOKEN_RE.search(text))


def main():
    df = pd.read_json(INPUT_FILE, lines=True)

    # Metadata/script-based filter.
    # This assumes script.kor and script.oko are numeric proportions or counts.
    script_filter = (df["script.kor"].fillna(0) > 0) | (df["script.oko"].fillna(0) > 0)

    # Actual text-content filter.
    text_filter = df["text"].fillna("").apply(has_hangul_token)

    filtered = df[script_filter & text_filter].copy()

    print("Total rows:", len(df))
    print("Rows with Hangul/Old Hangul script metadata:", int(script_filter.sum()))
    print("Rows with actual Hangul-script tokens:", int(text_filter.sum()))
    print("Rows passing both filters:", len(filtered))

    print()
    print(filtered[["id", "year", "language", "category", "script.kor", "script.oko", "script.han", "text"]].head(30).to_string())

    filtered.to_json(
        OUTPUT_FILE,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print()
    print(f"Saved filtered file to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()