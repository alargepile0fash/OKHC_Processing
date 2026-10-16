#!/usr/bin/env python3
"""Run lexical-type-weighted D2L separately for each historical time window.

Input
-----
The weighted pair table produced by 07_prepare_d2l_input.py:

    analysis/data/runs/<run>/d2l_core/d2l_weighted_pairs.csv

For each time window, this script:
- builds a fresh WeightedD2L model;
- adds the validated Korean core-vowel abstract targets;
- trains on weighted UR/SR pairs;
- records the learned rule, learned tier, excluded surface vowels,
  weighted prediction counts, TP threshold, and weighted accuracies;
- optionally writes per-pair predictions;
- optionally merges the already-computed raw baseline summary and produces
  comparison graphs.

The learner is reset between windows. No information is carried from one
historical period into another.

Run from project root on Windows:
    python -X utf8 OKHC-master/analysis/scripts/10_run_weighted_d2l_by_window.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from algophon import SegStr
from algophon.symbols import UNDERSPECIFIED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from d2l_logic.weighted_d2l import WeightedD2L


DEFAULT_CONFIG = PROJECT_ROOT / "analysis" / "config" / "d2l_run_default.json"

SURFACE_VOWELS = ["ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ", "ㅣ"]
HARMONIC_SURFACE_VOWELS = ["ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ"]

ABSTRACT_TARGETS = {
    "ㅏ": "A",
    "ㅓ": "EO",
    "ㅗ": "O",
    "ㅜ": "U",
    "ㆍ": "ARAE",
    "ㅡ": "EU",
}


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
    parser.add_argument("--feature-file", default=argparse.SUPPRESS)
    parser.add_argument("--baseline-summary", default=argparse.SUPPRESS)
    parser.add_argument("--window-id-col", default=argparse.SUPPRESS)
    parser.add_argument("--ur-col", default=argparse.SUPPRESS)
    parser.add_argument("--sr-col", default=argparse.SUPPRESS)
    parser.add_argument("--weight-col", default=argparse.SUPPRESS)

    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--make-graphs",
        dest="make_graphs",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    graph_group.add_argument(
        "--no-graphs",
        dest="make_graphs",
        action="store_false",
        default=argparse.SUPPRESS,
    )

    detail_group = parser.add_mutually_exclusive_group()
    detail_group.add_argument(
        "--write-pair-predictions",
        dest="write_pair_predictions",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    detail_group.add_argument(
        "--no-pair-predictions",
        dest="write_pair_predictions",
        action="store_false",
        default=argparse.SUPPRESS,
    )

    parser.set_defaults(**settings)
    args = parser.parse_args()
    args.config = config_path
    return args


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_input_columns(df: pd.DataFrame, args: argparse.Namespace) -> None:
    required = [
        args.window_id_col,
        args.ur_col,
        args.sr_col,
        args.weight_col,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Weighted D2L input is missing required columns: "
            + ", ".join(missing)
        )


def build_model(feature_file: Path) -> WeightedD2L:
    model = WeightedD2L(ipa_file_path=str(feature_file))
    model.seginv.add_segs(set(SURFACE_VOWELS))

    for surface, abstract in ABSTRACT_TARGETS.items():
        features = dict(model.seginv[surface].features)
        features["RTR"] = UNDERSPECIFIED
        model.seginv.add_custom(symbol=abstract, features=features)

    return model


def parse_window_bounds(window_id: object) -> tuple[int | None, int | None]:
    try:
        left, right = str(window_id).split("-", maxsplit=1)
        return int(left), int(right)
    except (TypeError, ValueError):
        return None, None


def theta_n(n: int) -> float:
    if n <= 1:
        return float("nan")
    return n / math.log(n)


def tp_pass(n: int, m: int) -> bool:
    if n <= 1:
        return False
    e = n - m
    return m > 1 and m >= n / 2 and e <= theta_n(n)


def make_weighted_pairs(model: WeightedD2L, group: pd.DataFrame, args: argparse.Namespace):
    triples = []
    for row in group.itertuples(index=False):
        ur_text = str(getattr(row, args.ur_col))
        sr_text = str(getattr(row, args.sr_col))
        weight = int(getattr(row, args.weight_col))

        if weight <= 0:
            raise ValueError(
                f"Non-positive lexical type weight in window "
                f"{getattr(row, args.window_id_col)}: {weight}"
            )

        triples.append(
            (
                SegStr(ur_text, seginv=model.seginv),
                SegStr(sr_text, seginv=model.seginv),
                weight,
            )
        )
    return triples


def tier_excluded_surface_segments(model: WeightedD2L) -> list[str]:
    if model.rule is None or model.rule.tier is None:
        return []

    excluded = []
    for symbol in SURFACE_VOWELS:
        seg = model.seginv[symbol]
        if seg not in model.rule.tier:
            excluded.append(symbol)
    return excluded


def tier_included_surface_segments(model: WeightedD2L) -> list[str]:
    if model.rule is None:
        return []
    if model.rule.tier is None:
        return list(SURFACE_VOWELS)

    included = []
    for symbol in SURFACE_VOWELS:
        seg = model.seginv[symbol]
        if seg in model.rule.tier:
            included.append(symbol)
    return included


def model_direction(model: WeightedD2L) -> str | None:
    if model.rule is None:
        return None
    return "left_to_right" if model.rule.left_to_right else "right_to_left"


def model_harmony_type(model: WeightedD2L) -> str | None:
    if model.rule is None:
        return None
    return "harmony" if model.rule.harmony else "disharmony"


def count_targets_in_ur(ur: SegStr, model: WeightedD2L) -> int:
    if model._discrepancy is None:
        return 0
    target = model._discrepancy.get_alternating_UR_segs()
    return sum(seg in target for seg in ur)


def pair_prediction_rows(
    model: WeightedD2L,
    window_id: object,
) -> list[dict]:
    if model.rule is None:
        return []

    rows = []
    for ur, sr in sorted(
        model.training_pairs,
        key=lambda pair: (str(pair[0]), str(pair[1])),
    ):
        weight = int(model.pair_weights[(ur, sr)])
        predicted = model.produce(ur)

        predictions = model.rule._predictions(ur)
        pair_n = len(predictions)
        pair_correct = sum(
            sr[idx] == predicted_seg
            for idx, predicted_seg in predictions
        )
        pair_errors = pair_n - pair_correct

        rows.append(
            {
                "time_window_id": window_id,
                "d2l_ur": str(ur),
                "d2l_sr": str(sr),
                "predicted_sr": str(predicted),
                "lexical_type_weight": weight,
                "pair_full_form_correct": predicted == sr,
                "prediction_N_per_lexical_type": pair_n,
                "prediction_correct_per_lexical_type": pair_correct,
                "prediction_exceptions_per_lexical_type": pair_errors,
                "weighted_prediction_N": pair_n * weight,
                "weighted_prediction_correct": pair_correct * weight,
                "weighted_prediction_exceptions": pair_errors * weight,
            }
        )
    return rows


def analyze_window(
    window_id: object,
    group: pd.DataFrame,
    args: argparse.Namespace,
    feature_file: Path,
) -> tuple[dict, list[dict]]:
    start, end = parse_window_bounds(window_id)

    row = {
        args.window_id_col: window_id,
        "time_window_start": start,
        "time_window_end": end,
        "status": "UNSET",
        "lexical_wordforms_represented": int(group[args.weight_col].sum()),
        "unique_structural_pairs": int(len(group)),
        "pairs_with_noninitial_i": (
            int(group["d2l_contains_noninitial_i"].sum())
            if "d2l_contains_noninitial_i" in group.columns
            else None
        ),
        "learned_rule": None,
        "learned_harmony_type": None,
        "learned_direction": None,
        "learned_tier": None,
        "tier_was_induced": False,
        "tier_included_surface_segments": "",
        "tier_excluded_surface_segments": "",
        "d2l_prediction_N": 0,
        "d2l_prediction_correct": 0,
        "d2l_prediction_exceptions": 0,
        "d2l_exception_rate": float("nan"),
        "d2l_theta_N": float("nan"),
        "d2l_theta_rate": float("nan"),
        "d2l_TP_pass": False,
        "weighted_prediction_accuracy": float("nan"),
        "weighted_full_form_accuracy": float("nan"),
    }

    model = build_model(feature_file)
    weighted_pairs = make_weighted_pairs(model, group, args)

    total_declared_targets = (
        int((group["d2l_num_targets"] * group[args.weight_col]).sum())
        if "d2l_num_targets" in group.columns
        else None
    )
    row["declared_weighted_target_count_from_converter"] = total_declared_targets

    # A window can contain only forms such as ㅏ ㅣ or ㅣ ㅣ, which have no
    # abstract harmony target. Stock D2L cannot build a discrepancy there.
    if total_declared_targets == 0:
        row["status"] = "NO_D2L_TARGETS"
        return row, []

    try:
        model.train_weighted(weighted_pairs)
    except Exception as error:
        row["status"] = "ERROR"
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
        return row, []

    if model.rule is None:
        row["status"] = "NO_PRODUCTIVE_RULE"
        row["weighted_full_form_accuracy"] = model.weighted_pair_accuracy()
        return row, []

    n, m = model.weighted_prediction_stats()
    e = n - m
    theta = theta_n(n)

    included = tier_included_surface_segments(model)
    excluded = tier_excluded_surface_segments(model)

    row.update(
        {
            "status": "RULE_LEARNED",
            "learned_rule": str(model.rule),
            "learned_harmony_type": model_harmony_type(model),
            "learned_direction": model_direction(model),
            "learned_tier": (
                str(model.rule.tier)
                if model.rule.tier is not None
                else "NONE"
            ),
            "tier_was_induced": model.rule.tier is not None,
            "tier_included_surface_segments": " ".join(included),
            "tier_excluded_surface_segments": " ".join(excluded),
            "d2l_prediction_N": int(n),
            "d2l_prediction_correct": int(m),
            "d2l_prediction_exceptions": int(e),
            "d2l_exception_rate": e / n if n else float("nan"),
            "d2l_theta_N": theta,
            "d2l_theta_rate": theta / n if n > 1 else float("nan"),
            "d2l_TP_pass": tp_pass(n, m),
            "weighted_prediction_accuracy": m / n if n else float("nan"),
            "weighted_full_form_accuracy": model.weighted_pair_accuracy(),
        }
    )

    # Sanity check: the final rule returned by D2L should be productive.
    if not row["d2l_TP_pass"]:
        row["status"] = "RULE_RETURNED_BUT_TP_FAILED"

    details = pair_prediction_rows(model, window_id)
    return row, details


def merge_baseline(
    summary: pd.DataFrame,
    baseline_path: Path | None,
    window_col: str,
) -> pd.DataFrame:
    if baseline_path is None or not baseline_path.exists():
        return summary

    baseline = pd.read_csv(
        baseline_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if window_col not in baseline.columns:
        raise ValueError(
            f"Baseline summary does not contain {window_col!r}: "
            f"{baseline_path}"
        )

    keep = [window_col]
    for column in [
        "prediction_N",
        "prediction_exceptions",
        "prediction_exception_rate",
        "prediction_theta_N",
        "prediction_theta_rate",
        "prediction_TP_pass",
        "wordform_N",
        "wordform_exceptions",
        "wordform_exception_rate",
        "wordform_theta_N",
        "wordform_theta_rate",
        "wordform_TP_pass",
        "scorable_wordforms",
        "initial_i_unscored_wordforms",
        "no_later_harmonic_target_wordforms",
    ]:
        if column in baseline.columns:
            keep.append(column)

    baseline = baseline[keep].copy()

    rename = {
        column: f"baseline_{column}"
        for column in keep
        if column != window_col
    }
    baseline = baseline.rename(columns=rename)

    return summary.merge(
        baseline,
        on=window_col,
        how="left",
        validate="one_to_one",
    )


def make_graphs(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Graph output requires matplotlib. Install it with: "
            "pip install matplotlib"
        ) from error

    plot = summary.copy()
    if "time_window_start" in plot.columns:
        plot = plot.sort_values("time_window_start")
        x = plot["time_window_start"]
        xlabel = "Time window start year"
    else:
        x = range(len(plot))
        xlabel = "Historical time window"

    # D2L post-learning exception rate and TP threshold.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x,
        plot["d2l_exception_rate"],
        marker="o",
        label="D2L exception rate",
    )
    ax.plot(
        x,
        plot["d2l_theta_rate"],
        marker="o",
        label="D2L tolerance threshold rate",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rate")
    ax.set_title("Weighted D2L exception rate by period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "d2l_exception_rate_by_window.png",
        dpi=200,
    )
    plt.close(fig)

    # Baseline vs D2L if the baseline file was available.
    baseline_col = "baseline_prediction_exception_rate"
    if baseline_col in plot.columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(
            x,
            plot[baseline_col],
            marker="o",
            label="Raw baseline",
        )
        ax.plot(
            x,
            plot["d2l_exception_rate"],
            marker="o",
            label="After D2L learning",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Prediction-level exception rate")
        ax.set_title("Raw baseline vs weighted D2L by period")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "baseline_vs_d2l_exception_rate.png",
            dpi=200,
        )
        plt.close(fig)

    # Number of excluded core surface vowels.
    tier_counts = plot["tier_excluded_surface_segments"].fillna("").apply(
        lambda value: len(str(value).split()) if str(value).strip() else 0
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x,
        tier_counts,
        marker="o",
        label="Excluded core vowels",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of excluded core surface vowels")
    ax.set_title("Learned tier restrictiveness by period")
    fig.tight_layout()
    fig.savefig(
        output_dir / "d2l_tier_restrictiveness_by_window.png",
        dpi=200,
    )
    plt.close(fig)


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    input_path: Path,
    feature_file: Path,
    baseline_path: Path | None,
) -> None:
    settings = vars(args).copy()
    settings["config"] = str(args.config)
    settings["input"] = str(input_path)
    settings["output_dir"] = str(output_dir)
    settings["feature_file"] = str(feature_file)
    settings["baseline_summary"] = (
        str(baseline_path) if baseline_path is not None else None
    )
    settings["surface_vowels"] = SURFACE_VOWELS
    settings["abstract_targets"] = ABSTRACT_TARGETS
    settings["analysis_note"] = (
        "One fresh lexical-type-weighted D2L model is trained independently "
        "per historical time window."
    )

    with (output_dir / "d2l_run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(settings, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    feature_file = resolve_project_path(args.feature_file)
    baseline_path = resolve_project_path(args.baseline_summary)

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            f"Weighted D2L input not found: {input_path}\n"
            "Run 07_prepare_d2l_input.py first."
        )
    if feature_file is None or not feature_file.exists():
        raise FileNotFoundError(
            f"D2L feature file not found: {feature_file}"
        )

    print(f"Reading weighted D2L input: {input_path}")
    pairs = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    validate_input_columns(pairs, args)

    windows = list(
        pairs.groupby(
            args.window_id_col,
            sort=False,
            dropna=False,
        )
    )

    def window_sort_key(item):
        start, _ = parse_window_bounds(item[0])
        return start if start is not None else float("inf")

    windows.sort(key=window_sort_key)

    print(f"Historical windows to analyze: {len(windows):,}")
    print()

    result_rows = []
    detail_rows = []

    for index, (window_id, group) in enumerate(windows, start=1):
        lexical_n = int(group[args.weight_col].sum())
        print(
            f"[{index}/{len(windows)}] {window_id}: "
            f"{lexical_n:,} lexical types, "
            f"{len(group):,} structural pairs"
        )

        row, details = analyze_window(
            window_id=window_id,
            group=group,
            args=args,
            feature_file=feature_file,
        )
        result_rows.append(row)
        detail_rows.extend(details)

        if row["status"] == "RULE_LEARNED":
            print(
                "    "
                f"tier={row['learned_tier']} | "
                f"errors={row['d2l_prediction_exceptions']}/"
                f"{row['d2l_prediction_N']} | "
                f"accuracy={row['weighted_prediction_accuracy']:.3f}"
            )
        else:
            print(f"    status={row['status']}")

    summary = pd.DataFrame(result_rows)

    if "time_window_start" in summary.columns:
        summary = summary.sort_values(
            "time_window_start",
            na_position="last",
        ).reset_index(drop=True)

    summary = merge_baseline(
        summary,
        baseline_path=baseline_path,
        window_col=args.window_id_col,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir / "d2l_results_by_window.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.write_pair_predictions:
        pd.DataFrame(detail_rows).to_csv(
            output_dir / "d2l_pair_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if args.make_graphs:
        make_graphs(summary, output_dir)

    write_run_config(
        output_dir=output_dir,
        args=args,
        input_path=input_path,
        feature_file=feature_file,
        baseline_path=baseline_path,
    )

    learned = int(summary["status"].eq("RULE_LEARNED").sum())
    no_rule = int(summary["status"].eq("NO_PRODUCTIVE_RULE").sum())
    errors = int(summary["status"].eq("ERROR").sum())

    print()
    print("Weighted D2L diachronic analysis complete.")
    print(f"Windows analyzed: {len(summary):,}")
    print(f"Rules learned: {learned:,}")
    print(f"No productive rule: {no_rule:,}")
    print(f"Errors: {errors:,}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
