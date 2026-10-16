#!/usr/bin/env python3
"""Convert sampled historical Korean vowel sequences into D2L UR/SR pairs.

This script is designed for the current OKHC_Processing sampling pipeline.

Input
-----
analysis/data/runs/<run>/sampled_time_window_wordforms.csv

That file contains one row per sampled lexical wordform in each time window.
The canonical vowel sequence is the sampler's space-separated ``vowel_seq``.

Core D2L representation
-----------------------
The first vowel stays fully specified.

For every later vowel:
    ㅏ -> A
    ㅓ -> EO
    ㅗ -> O
    ㅜ -> U
    ㆍ -> ARAE
    ㅡ -> EU
    ㅣ -> ㅣ

Thus only vowels with a harmonic counterpart become abstract targets.
ㅣ remains fully specified because it has no harmonic opposite.

Examples:
    ㅏ ㅗ ㅣ ㅏ  -> UR: ㅏ O ㅣ A   SR: ㅏ ㅗ ㅣ ㅏ
    ㅓ ㅜ ㅡ     -> UR: ㅓ U EU     SR: ㅓ ㅜ ㅡ

For the first core-only run, any wordform containing a vowel outside the
classical seven-vowel inventory is excluded from D2L input but retained in
an exclusions CSV.

Important type-count issue
--------------------------
algophon's D2L.train() converts UR/SR pairs to a set. Therefore distinct
lexical wordforms with the same vowel-only UR/SR pair would collapse if fed
directly to stock algophon. This converter DOES NOT collapse them. It writes
one row per sampled lexical wordform and reports the potential collapse in
window diagnostics so the downstream runner can handle that decision
explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "analysis" / "config" / "d2l_input_default.json"

CORE_VOWELS = frozenset({"ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ", "ㅣ"})
HARMONIC_VOWELS = frozenset({"ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ"})

ABSTRACT_TARGETS = {
    "ㅏ": "A",
    "ㅓ": "EO",
    "ㅗ": "O",
    "ㅜ": "U",
    "ㆍ": "ARAE",
    "ㅡ": "EU",
}

RTR_CLASS = {
    "ㆍ": "+RTR",
    "ㅏ": "+RTR",
    "ㅗ": "+RTR",
    "ㅡ": "-RTR",
    "ㅓ": "-RTR",
    "ㅜ": "-RTR",
    "ㅣ": "NEUTRAL",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()

    config_path = pre_args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    settings = load_config(config_path)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--input", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--vowel-seq-col", default=argparse.SUPPRESS)
    parser.add_argument("--window-id-col", default=argparse.SUPPRESS)
    parser.add_argument("--wordform-id-col", default=argparse.SUPPRESS)
    parser.add_argument("--token-col", default=argparse.SUPPRESS)
    parser.add_argument("--min-vowels", type=int, default=argparse.SUPPRESS)

    core_group = parser.add_mutually_exclusive_group()
    core_group.add_argument(
        "--core-only",
        dest="core_only",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    core_group.add_argument(
        "--allow-noncore",
        dest="core_only",
        action="store_false",
        default=argparse.SUPPRESS,
    )

    pair_group = parser.add_mutually_exclusive_group()
    pair_group.add_argument(
        "--write-window-pair-files",
        dest="write_window_pair_files",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    pair_group.add_argument(
        "--no-window-pair-files",
        dest="write_window_pair_files",
        action="store_false",
        default=argparse.SUPPRESS,
    )

    parser.set_defaults(**settings)
    args = parser.parse_args()
    args.config = config_path
    return args


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_settings(args: argparse.Namespace) -> None:
    required = [
        "input",
        "output_dir",
        "vowel_seq_col",
        "window_id_col",
        "wordform_id_col",
        "token_col",
        "core_only",
        "min_vowels",
        "write_window_pair_files",
    ]
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise ValueError(f"Missing D2L input settings: {missing}")

    if args.min_vowels < 1:
        raise ValueError("min_vowels must be at least 1.")

    # The current implementation intentionally supports only the validated
    # classical seven-vowel representation.
    if not args.core_only:
        raise ValueError(
            "Non-core D2L conversion is not implemented yet. "
            "Keep core_only=true for the validated seven-vowel analysis."
        )


def parse_vowel_seq(value: object) -> list[str]:
    """Parse the sampler's canonical space-separated vowel_seq."""
    if pd.isna(value):
        return []
    return [part for part in str(value).strip().split() if part]


def convert_sequence(vowels: list[str]) -> tuple[list[str], list[str]]:
    """Return (UR, SR) segment lists for one core-vowel sequence."""
    if not vowels:
        raise ValueError("Cannot convert an empty vowel sequence.")

    ur = [vowels[0]]

    for vowel in vowels[1:]:
        if vowel == "ㅣ":
            ur.append("ㅣ")
        else:
            ur.append(ABSTRACT_TARGETS[vowel])

    return ur, list(vowels)


def exclusion_reason(vowels: list[str], min_vowels: int) -> str | None:
    if len(vowels) < min_vowels:
        return "BELOW_MIN_VOWELS"

    noncore = sorted(set(vowels).difference(CORE_VOWELS))
    if noncore:
        return "NONCORE_VOWEL:" + ",".join(noncore)

    return None


def prepare_rows(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        args.vowel_seq_col,
        args.window_id_col,
        args.wordform_id_col,
        args.token_col,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Sampled wordform file is missing required columns: "
            + ", ".join(missing)
        )

    work = df.copy()
    work["_vowels"] = work[args.vowel_seq_col].apply(parse_vowel_seq)
    work["d2l_exclusion_reason"] = work["_vowels"].apply(
        lambda vowels: exclusion_reason(vowels, args.min_vowels)
    )

    excluded = work[work["d2l_exclusion_reason"].notna()].copy()
    eligible = work[work["d2l_exclusion_reason"].isna()].copy()

    if eligible.empty:
        raise ValueError("No sampled wordforms are eligible for core D2L conversion.")

    converted = eligible["_vowels"].apply(convert_sequence)
    eligible["d2l_ur"] = converted.apply(lambda pair: " ".join(pair[0]))
    eligible["d2l_sr"] = converted.apply(lambda pair: " ".join(pair[1]))
    eligible["d2l_pair_key"] = eligible["d2l_ur"] + "\t" + eligible["d2l_sr"]

    eligible["d2l_initial_vowel"] = eligible["_vowels"].str[0]
    eligible["d2l_initial_rtr_class"] = eligible["d2l_initial_vowel"].map(RTR_CLASS)
    eligible["d2l_num_segments"] = eligible["_vowels"].str.len()
    eligible["d2l_num_targets"] = eligible["_vowels"].apply(
        lambda vowels: sum(vowel in HARMONIC_VOWELS for vowel in vowels[1:])
    )
    eligible["d2l_num_noninitial_i"] = eligible["_vowels"].apply(
        lambda vowels: sum(vowel == "ㅣ" for vowel in vowels[1:])
    )
    eligible["d2l_contains_noninitial_i"] = eligible["d2l_num_noninitial_i"] > 0
    eligible["d2l_contains_arae"] = eligible["_vowels"].apply(lambda vowels: "ㆍ" in vowels)
    eligible["d2l_has_prediction_target"] = eligible["d2l_num_targets"] > 0

    # Lexical-type weight = number of sampled lexical wordform rows in this
    # window that share exactly the same vowel-only D2L UR/SR pair.
    pair_group_cols = [args.window_id_col, "d2l_ur", "d2l_sr"]
    eligible["lexical_type_weight"] = (
        eligible.groupby(pair_group_cols)["d2l_pair_key"].transform("size").astype(int)
    )
    eligible["d2l_pair_is_shared_by_multiple_wordforms"] = (
        eligible["lexical_type_weight"] > 1
    )

    weighted_pairs = (
        eligible.groupby(pair_group_cols, as_index=False)
        .agg(
            lexical_type_weight=(args.wordform_id_col, "size"),
            represented_wordform_ids=(
                args.wordform_id_col,
                lambda values: " ||| ".join(map(str, values)),
            ),
            represented_tokens=(
                args.token_col,
                lambda values: " ||| ".join(map(str, values)),
            ),
            d2l_num_segments=("d2l_num_segments", "first"),
            d2l_num_targets=("d2l_num_targets", "first"),
            d2l_contains_noninitial_i=("d2l_contains_noninitial_i", "first"),
            d2l_contains_arae=("d2l_contains_arae", "first"),
        )
    )

    eligible = eligible.drop(columns=["_vowels", "d2l_exclusion_reason"])
    excluded = excluded.drop(columns=["_vowels"])

    return eligible, excluded, weighted_pairs


def make_window_diagnostics(
    original: pd.DataFrame,
    eligible: pd.DataFrame,
    excluded: pd.DataFrame,
    weighted_pairs: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    window_col = args.window_id_col

    input_counts = (
        original.groupby(window_col, dropna=False)
        .size()
        .rename("sampled_wordforms_input")
    )

    eligible_counts = (
        eligible.groupby(window_col, dropna=False)
        .size()
        .rename("d2l_eligible_lexical_wordforms")
    )

    unique_pairs = (
        weighted_pairs.groupby(window_col, dropna=False)
        .size()
        .rename("unique_vowel_only_d2l_pairs")
    )

    target_counts = (
        eligible.groupby(window_col, dropna=False)["d2l_has_prediction_target"]
        .sum()
        .astype(int)
        .rename("lexical_wordforms_with_at_least_one_target")
    )

    transparent_i_counts = (
        eligible.groupby(window_col, dropna=False)["d2l_contains_noninitial_i"]
        .sum()
        .astype(int)
        .rename("lexical_wordforms_with_noninitial_i")
    )

    excluded_counts = (
        excluded.groupby(window_col, dropna=False)
        .size()
        .rename("excluded_wordforms")
    )

    diagnostics = pd.concat(
        [
            input_counts,
            eligible_counts,
            unique_pairs,
            target_counts,
            transparent_i_counts,
            excluded_counts,
        ],
        axis=1,
    ).fillna(0).reset_index()

    integer_cols = [
        "sampled_wordforms_input",
        "d2l_eligible_lexical_wordforms",
        "unique_vowel_only_d2l_pairs",
        "lexical_wordforms_with_at_least_one_target",
        "lexical_wordforms_with_noninitial_i",
        "excluded_wordforms",
    ]
    diagnostics[integer_cols] = diagnostics[integer_cols].astype(int)

    diagnostics["lexical_types_represented_by_weights_beyond_unique_pairs"] = (
        diagnostics["d2l_eligible_lexical_wordforms"]
        - diagnostics["unique_vowel_only_d2l_pairs"]
    )

    diagnostics["fraction_lexical_types_collapsed_structurally"] = (
        diagnostics["lexical_types_represented_by_weights_beyond_unique_pairs"]
        / diagnostics["d2l_eligible_lexical_wordforms"].replace(0, pd.NA)
    )

    diagnostics["fraction_sample_excluded_from_core_inventory"] = (
        diagnostics["excluded_wordforms"]
        / diagnostics["sampled_wordforms_input"].replace(0, pd.NA)
    )

    return diagnostics


def safe_filename(value: object) -> str:
    text = str(value)
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in text
    )


def write_window_pair_files(
    weighted_pairs: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Write one weighted UR/SR row per unique structural pair in each window."""
    pairs_dir = output_dir / "window_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for window_id, group in weighted_pairs.groupby(args.window_id_col, sort=True):
        output = pairs_dir / f"{safe_filename(window_id)}.tsv"
        group[["d2l_ur", "d2l_sr", "lexical_type_weight"]].to_csv(
            output,
            sep="\t",
            header=False,
            index=False,
            encoding="utf-8",
        )

def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    settings = vars(args).copy()
    settings["config"] = str(args.config)
    settings["input"] = str(resolve_project_path(args.input))
    settings["output_dir"] = str(output_dir)
    settings["core_vowels"] = sorted(CORE_VOWELS)
    settings["abstract_targets"] = ABSTRACT_TARGETS
    settings["representation_note"] = (
        "First vowel specified; later harmonic vowels RTR-underspecified via "
        "named targets; ㅣ remains fully specified."
    )
    settings["weight_definition"] = (
        "Number of sampled lexical wordform rows in the same time window "
        "sharing an identical vowel-only D2L UR/SR pair. Corpus token "
        "frequency is not used as weight."
    )

    with (output_dir / "d2l_input_config.json").open("w", encoding="utf-8") as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    validate_settings(args)

    input_csv = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Sampled wordform input not found: {input_csv}")

    print(f"Reading sampled wordforms: {input_csv}")
    sampled = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)

    eligible, excluded, weighted_pairs = prepare_rows(sampled, args)
    diagnostics = make_window_diagnostics(
        sampled, eligible, excluded, weighted_pairs, args
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    eligible.to_csv(
        output_dir / "d2l_wordforms.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weighted_pairs.to_csv(
        output_dir / "d2l_weighted_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded.to_csv(
        output_dir / "d2l_excluded_wordforms.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics.to_csv(
        output_dir / "d2l_window_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.write_window_pair_files:
        write_window_pair_files(weighted_pairs, output_dir, args)

    write_run_config(output_dir, args)

    print()
    print("D2L conversion complete.")
    print(f"Input sampled lexical wordforms: {len(sampled):,}")
    print(f"Core-D2L eligible lexical wordforms: {len(eligible):,}")
    print(f"Unique structural D2L pairs: {len(weighted_pairs):,}")
    print(
        "Sum of lexical_type_weight: "
        f"{int(weighted_pairs['lexical_type_weight'].sum()):,}"
    )
    print(f"Excluded lexical wordforms: {len(excluded):,}")
    print(
        "Wordforms with at least one D2L target: "
        f"{int(eligible['d2l_has_prediction_target'].sum()):,}"
    )
    print()
    print(
        "d2l_wordforms.csv preserves each sampled lexical type; "
        "d2l_weighted_pairs.csv compresses identical structural pairs while "
        "preserving their lexical-type counts in lexical_type_weight."
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
