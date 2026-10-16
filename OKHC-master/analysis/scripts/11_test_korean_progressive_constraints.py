#!/usr/bin/env python3
"""Sanity checks for the Korean-constrained D2L analysis.

Run:
    python -X utf8 OKHC-master/analysis/scripts/11_test_korean_progressive_constraints.py
"""

from pathlib import Path
import sys

from algophon import SegStr
from algophon.symbols import UNDERSPECIFIED

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from d2l_logic.korean_weighted_d2l import (
    InitialProtectedTier,
    KoreanProgressiveWeightedD2L,
)


FEATURE_FILE = (
    PROJECT_ROOT
    / "analysis"
    / "config"
    / "korean_d2l_core_inventory_features.tsv"
)

SURFACE_VOWELS = {"ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ", "ㅣ"}
ABSTRACT_TARGETS = {
    "ㅏ": "A",
    "ㅓ": "EO",
    "ㅗ": "O",
    "ㅜ": "U",
    "ㆍ": "ARAE",
    "ㅡ": "EU",
}


def build_model():
    model = KoreanProgressiveWeightedD2L(
        ipa_file_path=str(FEATURE_FILE)
    )
    model.seginv.add_segs(SURFACE_VOWELS)

    for surface, abstract in ABSTRACT_TARGETS.items():
        features = dict(model.seginv[surface].features)
        features["RTR"] = UNDERSPECIFIED
        model.seginv.add_custom(
            symbol=abstract,
            features=features,
        )

    return model


def main():
    model = build_model()

    # Position 0 must survive even when its segment class is excluded.
    tier = InitialProtectedTier(
        seginv=model.seginv,
        segs={
            model.seginv["A"],
            model.seginv["EO"],
            model.seginv["O"],
            model.seginv["U"],
            model.seginv["ARAE"],
            model.seginv["EU"],
        },
    )

    form = SegStr("ㅏ ㅣ A", seginv=model.seginv)
    projected = tier.project(form)
    assert str(projected) == "ㅏ A", projected
    assert projected.idxs == [0, 2], projected.idxs

    initial_i = SegStr("ㅣ ㅏ A", seginv=model.seginv)
    projected_i = tier.project(initial_i)
    assert str(projected_i) == "ㅣ A", projected_i
    assert projected_i.idxs == [0, 2], projected_i.idxs

    # Clean progressive harmony should learn only a left-to-right harmony rule.
    weighted_pairs = [
        ("ㅏ A O", "ㅏ ㅏ ㅗ", 10),
        ("ㅓ EO U", "ㅓ ㅓ ㅜ", 10),
        ("ㅗ O A", "ㅗ ㅗ ㅏ", 10),
        ("ㅜ U EO", "ㅜ ㅜ ㅓ", 10),
    ]
    model.train_weighted(weighted_pairs)

    assert model.rule is not None
    assert model.rule.left_to_right
    assert model.rule.harmony
    assert all(
        row["branch"] == "harmony"
        for row in model.analysis_trace
    )
    assert all(
        row["selected_direction"] == "left_to_right"
        for row in model.analysis_trace
    )

    print("Korean progressive D2L constraint checks passed.")
    print(f"Learned rule: {model.rule}")
    print("Initial-position tier protection verified.")


if __name__ == "__main__":
    main()
