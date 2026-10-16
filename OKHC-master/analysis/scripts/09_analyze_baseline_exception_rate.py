#!/usr/bin/env python3
"""Analyze raw baseline harmony exception rates by historical time window.

This stage deliberately runs BEFORE D2L learning. It uses the core seven-vowel
inventory and the lexical wordform rows produced by 07_prepare_d2l_input.py.

Two baseline views are reported:

1. Prediction-level baseline
   Each later harmonic vowel (ㆍ ㅏ ㅗ ㅡ ㅓ ㅜ) is one prediction opportunity.
   ㅣ is not itself scored as a harmony target. A target is an exception when
   its RTR class differs from the initial harmonic trigger's RTR class.

2. Wordform-level baseline
   Each scorable lexical wordform counts once. A wordform is exceptional if
   at least one scored later harmonic vowel disagrees with the trigger.

The prediction-level measure is directly comparable to D2L's target-prediction
n/m statistics. The wordform-level measure preserves the older "one lexical
item, at most one exception" interpretation.

Initial ㅣ policy
----------------
By default, forms whose first vowel is ㅣ are left UNSCORED for the baseline.
This avoids baking the D2L toy-feature assumption ㅣ=[-RTR] into the raw
descriptive baseline. Their counts are reported separately.

Run from the project root:
    python OKHC-master/analysis/scripts/09_analyze_baseline_exception_rate.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "analysis" / "config" / "baseline_default.json"

RTR_CLASS = {
    "ㆍ": "+RTR",
    "ㅏ": "+RTR",
    "ㅗ": "+RTR",
    "ㅡ": "-RTR",
    "ㅓ": "-RTR",
    "ㅜ": "-RTR",
}

HARMONIC_VOWELS = frozenset(RTR_CLASS)
CORE_VOWELS = frozenset({*HARMONIC_VOWELS, "ㅣ"})


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()

    config_path = pre_args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    settings = load_config(config_path)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--input", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--window-id-col", default=argparse.SUPPRESS)
    parser.add_argument("--vowel-seq-col", default=argparse.SUPPRESS)
    parser.add_argument(
        "--initial-i-policy",
        choices=["unscored", "minus-rtr"],
        default=argparse.SUPPRESS,
    )

    detail = parser.add_mutually_exclusive_group()
    detail.add_argument(
        "--write-wordform-details",
        dest="write_wordform_details",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    detail.add_argument(
        "--no-wordform-details",
        dest="write_wordform_details",
        action="store_false",
        default=argparse.SUPPRESS,
    )

    graph = parser.add_mutually_exclusive_group()
    graph.add_argument(
        "--make-graphs",
        dest="make_graphs",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    graph.add_argument(
        "--no-graphs",
        dest="make_graphs",
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


def parse_sequence(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).strip().split() if part]


def tolerance_threshold(n: int) -> float:
    if n <= 1:
        return float("nan")
    return n / math.log(n)


def tolerance_pass(n: int, errors: int) -> bool:
    if n <= 1:
        return False
    correct = n - errors
    return (
        correct > 1
        and correct >= n / 2
        and errors <= tolerance_threshold(n)
    )


def score_wordform(vowels: list[str], initial_i_policy: str) -> dict:
    result = {
        "baseline_status": "SCORABLE",
        "baseline_trigger": None,
        "baseline_trigger_rtr": None,
        "prediction_N": 0,
        "prediction_exceptions": 0,
        "wordform_exception": False,
    }

    if not vowels:
        result["baseline_status"] = "EMPTY_SEQUENCE"
        return result

    noncore = sorted(set(vowels).difference(CORE_VOWELS))
    if noncore:
        result["baseline_status"] = "NONCORE_VOWEL:" + ",".join(noncore)
        return result

    trigger = vowels[0]

    if trigger == "ㅣ":
        if initial_i_policy == "unscored":
            result["baseline_status"] = "INITIAL_I_UNSCORED"
            return result
        trigger_rtr = "-RTR"
    else:
        trigger_rtr = RTR_CLASS[trigger]

    result["baseline_trigger"] = trigger
    result["baseline_trigger_rtr"] = trigger_rtr

    later_targets = [v for v in vowels[1:] if v in HARMONIC_VOWELS]

    if not later_targets:
        result["baseline_status"] = "NO_LATER_HARMONIC_TARGET"
        return result

    errors = sum(RTR_CLASS[v] != trigger_rtr for v in later_targets)

    result["prediction_N"] = len(later_targets)
    result["prediction_exceptions"] = errors
    result["wordform_exception"] = errors > 0
    return result


def analyze_wordforms(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = [args.window_id_col, args.vowel_seq_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Baseline input is missing required columns: "
            + ", ".join(missing)
        )

    details = df.copy()
    parsed = details[args.vowel_seq_col].apply(parse_sequence)
    scored = parsed.apply(
        lambda vowels: score_wordform(vowels, args.initial_i_policy)
    )
    scored_df = pd.DataFrame(scored.tolist(), index=details.index)

    return pd.concat([details, scored_df], axis=1)


def summarize_by_window(details: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    window = args.window_id_col
    rows = []

    for window_id, group in details.groupby(window, sort=False, dropna=False):
        scorable = group[group["baseline_status"].eq("SCORABLE")]

        prediction_n = int(scorable["prediction_N"].sum())
        prediction_e = int(scorable["prediction_exceptions"].sum())
        prediction_correct = prediction_n - prediction_e

        wordform_n = int(len(scorable))
        wordform_e = int(scorable["wordform_exception"].sum())
        wordform_correct = wordform_n - wordform_e

        pred_theta = tolerance_threshold(prediction_n)
        word_theta = tolerance_threshold(wordform_n)

        row = {
            window: window_id,
            "core_wordforms_input": int(len(group)),
            "scorable_wordforms": wordform_n,
            "initial_i_unscored_wordforms": int(
                group["baseline_status"].eq("INITIAL_I_UNSCORED").sum()
            ),
            "no_later_harmonic_target_wordforms": int(
                group["baseline_status"].eq("NO_LATER_HARMONIC_TARGET").sum()
            ),
            "prediction_N": prediction_n,
            "prediction_correct": prediction_correct,
            "prediction_exceptions": prediction_e,
            "prediction_exception_rate": (
                prediction_e / prediction_n if prediction_n else float("nan")
            ),
            "prediction_theta_N": pred_theta,
            "prediction_theta_rate": (
                pred_theta / prediction_n
                if prediction_n > 1
                else float("nan")
            ),
            "prediction_TP_pass": tolerance_pass(
                prediction_n, prediction_e
            ),
            "wordform_N": wordform_n,
            "wordform_correct": wordform_correct,
            "wordform_exceptions": wordform_e,
            "wordform_exception_rate": (
                wordform_e / wordform_n if wordform_n else float("nan")
            ),
            "wordform_theta_N": word_theta,
            "wordform_theta_rate": (
                word_theta / wordform_n
                if wordform_n > 1
                else float("nan")
            ),
            "wordform_TP_pass": tolerance_pass(
                wordform_n, wordform_e
            ),
        }

        for optional in ["time_window_start", "time_window_end"]:
            if optional in group.columns:
                values = group[optional].dropna().unique()
                if len(values):
                    row[optional] = values[0]

        rows.append(row)

    summary = pd.DataFrame(rows)

    sort_cols = [
        col
        for col in ["time_window_start", window]
        if col in summary.columns
    ]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    return summary


def make_graphs(summary: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Graph output requires matplotlib. Install it with: "
            "pip install matplotlib"
        ) from error

    if "time_window_start" in summary.columns:
        x = summary["time_window_start"]
        x_label = "Time window start year"
    else:
        x = range(len(summary))
        x_label = "Historical time window"

    # Prediction-level baseline.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x,
        summary["prediction_exception_rate"],
        marker="o",
        label="Raw exception rate",
    )
    ax.plot(
        x,
        summary["prediction_theta_rate"],
        marker="o",
        label="Tolerance threshold rate",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rate")
    ax.set_title("Baseline prediction-level exception rate by period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "baseline_prediction_exception_rate.png",
        dpi=200,
    )
    plt.close(fig)

    # Wordform-level baseline.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x,
        summary["wordform_exception_rate"],
        marker="o",
        label="Raw exception rate",
    )
    ax.plot(
        x,
        summary["wordform_theta_rate"],
        marker="o",
        label="Tolerance threshold rate",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rate")
    ax.set_title("Baseline wordform-level exception rate by period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "baseline_wordform_exception_rate.png",
        dpi=200,
    )
    plt.close(fig)

    # Direct comparison of the two raw rates.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x,
        summary["prediction_exception_rate"],
        marker="o",
        label="Prediction-level",
    )
    ax.plot(
        x,
        summary["wordform_exception_rate"],
        marker="o",
        label="Wordform-level",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Raw exception rate")
    ax.set_title("Baseline raw exception rates by period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "baseline_raw_exception_rates_comparison.png",
        dpi=200,
    )
    plt.close(fig)


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    settings = vars(args).copy()
    settings["config"] = str(args.config)
    settings["input"] = str(resolve_project_path(args.input))
    settings["output_dir"] = str(output_dir)
    settings["baseline_definition"] = (
        "Initial harmonic vowel triggers RTR agreement on later core harmonic "
        "vowels; ㅣ is not scored as a target. Initial ㅣ handling is controlled "
        "by initial_i_policy."
    )

    with (output_dir / "baseline_analysis_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_csv = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Baseline input not found: {input_csv}\n"
            "Run 07_prepare_d2l_input.py first."
        )

    print(f"Reading baseline input: {input_csv}")
    wordforms = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        low_memory=False,
    )

    details = analyze_wordforms(wordforms, args)
    summary = summarize_by_window(details, args)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "baseline_exception_rate_by_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.write_wordform_details:
        details.to_csv(
            output_dir / "baseline_wordform_details.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if args.make_graphs:
        make_graphs(summary, output_dir, args)

    write_run_config(output_dir, args)

    print()
    print("Baseline analysis complete.")
    print(f"Windows analyzed: {len(summary):,}")
    print(
        "Total core lexical wordforms: "
        f"{int(summary['core_wordforms_input'].sum()):,}"
    )
    print(
        "Scorable lexical wordforms: "
        f"{int(summary['scorable_wordforms'].sum()):,}"
    )
    print(
        "Initial-ㅣ unscored wordforms: "
        f"{int(summary['initial_i_unscored_wordforms'].sum()):,}"
    )
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
