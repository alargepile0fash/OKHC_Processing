#!/usr/bin/env python3
"""Run lexical-type-weighted D2L separately for each historical time window.

This runner preserves the final learned model, but also records the complete
failed D2L search path. Diagnostics are written even when D2L returns no rule.

Important outputs include:
- d2l_results_by_window.csv
- d2l_learning_trace.csv
- d2l_failed_predictions.csv
- d2l_failure_context_summary.csv
- d2l_initial_harmony_failure_context_summary.csv
- d2l_failure_target_summary.csv

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

from d2l_logic.korean_weighted_d2l import KoreanProgressiveWeightedD2L


DEFAULT_CONFIG = PROJECT_ROOT / "analysis" / "config" / "d2l_run_default.json"

SURFACE_VOWELS = ["ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ", "ㅣ"]

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


def build_model(feature_file: Path) -> KoreanProgressiveWeightedD2L:
    model = KoreanProgressiveWeightedD2L(ipa_file_path=str(feature_file))
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


def make_weighted_pairs(
    model: KoreanProgressiveWeightedD2L,
    group: pd.DataFrame,
    args: argparse.Namespace,
):
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


def tier_excluded_surface_segments(model: KoreanProgressiveWeightedD2L) -> list[str]:
    if model.rule is None or model.rule.tier is None:
        return []

    return [
        symbol
        for symbol in SURFACE_VOWELS
        if model.seginv[symbol] not in model.rule.tier
    ]


def tier_included_surface_segments(model: KoreanProgressiveWeightedD2L) -> list[str]:
    if model.rule is None:
        return []
    if model.rule.tier is None:
        return list(SURFACE_VOWELS)

    return [
        symbol
        for symbol in SURFACE_VOWELS
        if model.seginv[symbol] in model.rule.tier
    ]


def model_direction(model: KoreanProgressiveWeightedD2L) -> str | None:
    if model.rule is None:
        return None
    return "left_to_right" if model.rule.left_to_right else "right_to_left"


def model_harmony_type(model: KoreanProgressiveWeightedD2L) -> str | None:
    if model.rule is None:
        return None
    return "harmony" if model.rule.harmony else "disharmony"


def pair_prediction_rows(
    model: KoreanProgressiveWeightedD2L,
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
                "prediction_exceptions_per_lexical_type": pair_n - pair_correct,
                "weighted_prediction_N": pair_n * weight,
                "weighted_prediction_correct": pair_correct * weight,
                "weighted_prediction_exceptions": (
                    pair_n - pair_correct
                ) * weight,
            }
        )
    return rows


def trace_summary_fields(model: KoreanProgressiveWeightedD2L) -> dict:
    fields = {}

    for branch in ("harmony", "disharmony"):
        attempts = [
            row for row in model.analysis_trace
            if row["branch"] == branch
        ]
        attempts.sort(key=lambda row: row["attempt_depth"])

        prefix = f"{branch}_"
        fields[prefix + "num_attempts"] = len(attempts)

        if not attempts:
            continue

        initial = attempts[0]
        terminal = attempts[-1]

        fields.update(
            {
                prefix + "initial_direction": initial["selected_direction"],
                prefix + "initial_prediction_N": initial["selected_prediction_N"],
                prefix + "initial_prediction_correct": initial[
                    "selected_prediction_correct"
                ],
                prefix + "initial_prediction_exceptions": initial[
                    "selected_prediction_exceptions"
                ],
                prefix + "initial_exception_rate": (
                    initial["selected_prediction_exceptions"]
                    / initial["selected_prediction_N"]
                    if initial["selected_prediction_N"]
                    else float("nan")
                ),
                prefix + "initial_theta_N": initial["theta_N"],
                prefix + "initial_TP_pass": initial["TP_pass"],
                prefix + "terminal_status": terminal["terminal_status"],
                prefix + "terminal_tier": terminal["tier_after"],
                prefix + "terminal_delset": terminal["delset_after"],
                prefix + "terminal_prediction_N": terminal[
                    "selected_prediction_N"
                ],
                prefix + "terminal_prediction_exceptions": terminal[
                    "selected_prediction_exceptions"
                ],
                prefix + "terminal_TP_pass": terminal["TP_pass"],
            }
        )

    return fields


def annotate_diagnostics(
    rows: list[dict],
    window_id: object,
) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["time_window_id"] = window_id
        out.append(item)
    return out


def analyze_window(
    window_id: object,
    group: pd.DataFrame,
    args: argparse.Namespace,
    feature_file: Path,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    start, end = parse_window_bounds(window_id)

    row = {
        args.window_id_col: window_id,
        "time_window_start": start,
        "time_window_end": end,
        "status": "UNSET",
        "lexical_wordforms_represented": int(group[args.weight_col].sum()),
        "unique_structural_pairs": int(len(group)),
        "learned_rule": None,
        "learned_harmony_type": None,
        "learned_direction": None,
        "learned_tier": None,
        "tier_was_induced": False,
        "tier_included_surface_segments": "",
        "tier_excluded_surface_segments": "",
        "d2l_prediction_N": float("nan"),
        "d2l_prediction_correct": float("nan"),
        "d2l_prediction_exceptions": float("nan"),
        "d2l_exception_rate": float("nan"),
        "d2l_theta_N": float("nan"),
        "d2l_theta_rate": float("nan"),
        "d2l_TP_pass": False,
        "weighted_prediction_accuracy": float("nan"),
        "weighted_full_form_accuracy": float("nan"),
        "analysis_direction_constraint": "left_to_right",
        "analysis_harmony_constraint": "harmony_only",
        "initial_position_forced_to_project": True,
    }

    model = build_model(feature_file)
    weighted_pairs = make_weighted_pairs(model, group, args)

    total_declared_targets = (
        int((group["d2l_num_targets"] * group[args.weight_col]).sum())
        if "d2l_num_targets" in group.columns
        else None
    )
    row["declared_weighted_target_count_from_converter"] = total_declared_targets

    if total_declared_targets == 0:
        row["status"] = "NO_D2L_TARGETS"
        return row, [], [], []

    try:
        model.train_weighted(weighted_pairs)
    except Exception as error:
        row["status"] = "ERROR"
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
        return row, [], [], []

    row.update(trace_summary_fields(model))

    trace_rows = annotate_diagnostics(model.analysis_trace, window_id)
    failure_rows = annotate_diagnostics(model.prediction_events, window_id)

    if model.rule is None:
        row["status"] = "NO_PRODUCTIVE_RULE"
        pair_rows = []
        return row, pair_rows, trace_rows, failure_rows

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

    if not row["d2l_TP_pass"]:
        row["status"] = "RULE_RETURNED_BUT_TP_FAILED"

    return (
        row,
        pair_prediction_rows(model, window_id),
        trace_rows,
        failure_rows,
    )


def build_failure_context_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    group_cols = [
        "time_window_id",
        "branch",
        "attempt_depth",
        "candidate_direction",
        "selected_candidate",
        "context_segment",
        "context_source",
    ]

    work = events.copy()
    work["weighted_eligible_for_delset"] = (
        work["weighted_count"]
        * work["eligible_for_delset_under_current_rule"].astype(int)
    )
    work["weighted_ineligible_for_delset"] = (
        work["weighted_count"]
        * (~work["eligible_for_delset_under_current_rule"]).astype(int)
    )
    work["weighted_prior_misprediction_context"] = (
        work["weighted_count"]
        * work["context_was_prior_misprediction"].astype(int)
    )
    work["weighted_default_errors"] = (
        work["weighted_count"]
        * work["used_default"].astype(int)
    )
    work["weighted_protected_initial_context_errors"] = (
        work["weighted_count"]
        * work["context_is_protected_initial_position"].astype(int)
    )

    summary = (
        work.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            structural_failed_predictions=("weighted_count", "size"),
            weighted_failed_predictions=("weighted_count", "sum"),
            weighted_eligible_for_delset=(
                "weighted_eligible_for_delset",
                "sum",
            ),
            weighted_ineligible_for_delset=(
                "weighted_ineligible_for_delset",
                "sum",
            ),
            weighted_errors_after_prior_misprediction=(
                "weighted_prior_misprediction_context",
                "sum",
            ),
            weighted_default_errors=("weighted_default_errors", "sum"),
            weighted_protected_initial_context_errors=(
                "weighted_protected_initial_context_errors",
                "sum",
            ),
        )
    )

    denom_cols = [
        "time_window_id",
        "branch",
        "attempt_depth",
        "candidate_direction",
        "selected_candidate",
    ]
    totals = summary.groupby(denom_cols, dropna=False)[
        "weighted_failed_predictions"
    ].transform("sum")
    summary["share_of_weighted_failures"] = (
        summary["weighted_failed_predictions"] / totals.replace(0, pd.NA)
    )

    summary["failure_rank_within_candidate"] = (
        summary.groupby(denom_cols, dropna=False)[
            "weighted_failed_predictions"
        ]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    return summary.sort_values(
        denom_cols + ["failure_rank_within_candidate", "context_segment"]
    )


def build_failure_target_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    group_cols = [
        "time_window_id",
        "branch",
        "attempt_depth",
        "candidate_direction",
        "selected_candidate",
        "target_abstract_segment",
        "actual_surface_segment",
        "predicted_surface_segment",
    ]

    summary = (
        events.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            structural_failed_predictions=("weighted_count", "size"),
            weighted_failed_predictions=("weighted_count", "sum"),
        )
    )

    return summary.sort_values(
        [
            "time_window_id",
            "branch",
            "attempt_depth",
            "selected_candidate",
            "weighted_failed_predictions",
        ],
        ascending=[True, True, True, False, False],
    )


def add_top_failure_contexts(
    summary: pd.DataFrame,
    context_summary: pd.DataFrame,
    window_col: str,
) -> pd.DataFrame:
    if context_summary.empty:
        summary["initial_harmony_top_error_contexts"] = ""
        return summary

    focus = context_summary[
        (context_summary["branch"] == "harmony")
        & (context_summary["attempt_depth"] == 0)
        & (context_summary["selected_candidate"] == True)
    ].copy()

    top_strings = {}
    for window_id, group in focus.groupby("time_window_id"):
        group = group.sort_values(
            "weighted_failed_predictions",
            ascending=False,
        ).head(5)
        top_strings[window_id] = "; ".join(
            f"{row.context_segment}:{int(row.weighted_failed_predictions)}"
            for row in group.itertuples(index=False)
        )

    summary["initial_harmony_top_error_contexts"] = (
        summary[window_col].map(top_strings).fillna("")
    )
    return summary


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
    baseline = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in keep
            if column != window_col
        }
    )

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

    plot = summary.sort_values("time_window_start")
    x = plot["time_window_start"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        x,
        plot["harmony_initial_exception_rate"],
        marker="o",
        label="Initial D2L harmony candidate",
    )
    theta_rate = (
        plot["harmony_initial_theta_N"]
        / plot["harmony_initial_prediction_N"]
    )
    ax.plot(
        x,
        theta_rate,
        marker="o",
        label="Tolerance threshold rate",
    )
    ax.set_xlabel("Time window start year")
    ax.set_ylabel("Prediction-level rate")
    ax.set_title("Initial weighted D2L harmony failure by period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "d2l_initial_harmony_failure_by_window.png",
        dpi=200,
    )
    plt.close(fig)

    if "baseline_prediction_exception_rate" in plot.columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(
            x,
            plot["baseline_prediction_exception_rate"],
            marker="o",
            label="Descriptive baseline",
        )
        ax.plot(
            x,
            plot["harmony_initial_exception_rate"],
            marker="o",
            label="D2L initial harmony candidate",
        )
        ax.set_xlabel("Time window start year")
        ax.set_ylabel("Prediction-level exception rate")
        ax.set_title("Baseline vs initial D2L harmony candidate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / "baseline_vs_initial_d2l_harmony.png",
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
        "Fresh Korean-constrained weighted D2L models are trained independently "
        "by window. Harmony is fixed left-to-right, disharmony is not evaluated, "
        "and the word-initial segment is forced to project to every tier. "
        "Recursive failed rule/tier attempts and all failed prediction contexts "
        "are retained for diagnosis."
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
    windows.sort(
        key=lambda item: (
            parse_window_bounds(item[0])[0]
            if parse_window_bounds(item[0])[0] is not None
            else float("inf")
        )
    )

    print(f"Historical windows to analyze: {len(windows):,}")
    print("Analysis mode: progressive harmony only; initial trigger protected")
    print()

    result_rows = []
    pair_rows = []
    trace_rows = []
    failure_rows = []

    for index, (window_id, group) in enumerate(windows, start=1):
        lexical_n = int(group[args.weight_col].sum())
        print(
            f"[{index}/{len(windows)}] {window_id}: "
            f"{lexical_n:,} lexical types, "
            f"{len(group):,} structural pairs"
        )

        row, pairs_out, trace_out, failures_out = analyze_window(
            window_id=window_id,
            group=group,
            args=args,
            feature_file=feature_file,
        )

        result_rows.append(row)
        pair_rows.extend(pairs_out)
        trace_rows.extend(trace_out)
        failure_rows.extend(failures_out)

        if row["status"] == "RULE_LEARNED":
            print(
                "    "
                f"tier={row['learned_tier']} | "
                f"errors={row['d2l_prediction_exceptions']}/"
                f"{row['d2l_prediction_N']} | "
                f"accuracy={row['weighted_prediction_accuracy']:.3f}"
            )
        else:
            print(
                f"    status={row['status']} | "
                f"initial harmony errors="
                f"{row.get('harmony_initial_prediction_exceptions', 'NA')}/"
                f"{row.get('harmony_initial_prediction_N', 'NA')} | "
                f"terminal={row.get('harmony_terminal_status', 'NA')}"
            )

    summary = pd.DataFrame(result_rows).sort_values(
        "time_window_start",
        na_position="last",
    ).reset_index(drop=True)

    trace_df = pd.DataFrame(trace_rows)
    failure_df = pd.DataFrame(failure_rows)
    context_summary = build_failure_context_summary(failure_df)
    target_summary = build_failure_target_summary(failure_df)

    summary = add_top_failure_contexts(
        summary,
        context_summary,
        args.window_id_col,
    )
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

    trace_df.to_csv(
        output_dir / "d2l_learning_trace.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failure_df.to_csv(
        output_dir / "d2l_failed_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    context_summary.to_csv(
        output_dir / "d2l_failure_context_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    target_summary.to_csv(
        output_dir / "d2l_failure_target_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not context_summary.empty:
        initial_context = context_summary[
            (context_summary["branch"] == "harmony")
            & (context_summary["attempt_depth"] == 0)
            & (context_summary["selected_candidate"] == True)
        ].copy()
        initial_context.to_csv(
            output_dir / "d2l_initial_harmony_failure_context_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if args.write_pair_predictions:
        pd.DataFrame(pair_rows).to_csv(
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

    print()
    print("Weighted D2L diachronic analysis complete.")
    print(f"Windows analyzed: {len(summary):,}")
    print(f"Rules learned: {int(summary['status'].eq('RULE_LEARNED').sum()):,}")
    print(
        "No productive rule: "
        f"{int(summary['status'].eq('NO_PRODUCTIVE_RULE').sum()):,}"
    )
    print(f"Failed prediction events recorded: {len(failure_df):,}")
    print()
    print("Diagnostic outputs:")
    print("  d2l_learning_trace.csv")
    print("  d2l_failed_predictions.csv")
    print("  d2l_failure_context_summary.csv")
    print("  d2l_initial_harmony_failure_context_summary.csv")
    print("  d2l_failure_target_summary.csv")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
