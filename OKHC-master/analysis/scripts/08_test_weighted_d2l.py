"""Controlled validation of lexical-type-weighted D2L.

This test combines the already-validated seven-vowel Korean inventory with
lexical-type weights.

Expected behavior:
- 30 unique structural UR/SR pairs;
- 300 lexical types represented (weight 10 per structural pair);
- the initial no-tier rule has n=300 and 90 weighted errors, so it fails TP;
- ㅣ is the only errant local context;
- WeightedD2L then induces a tier excluding ㅣ;
- the final weighted rule has zero errors.

Run from the project root on Windows:
    python -X utf8 OKHC-master/analysis/scripts/08_test_weighted_d2l.py
"""

from pathlib import Path
import math
import sys

from algophon import SegStr
from algophon.symbols import UNDERSPECIFIED
from algophon.utils import tsp

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from d2l_logic.weighted_d2l import WeightedD2L
from d2l_logic.weighted_rule import WeightedRule


FEATURE_FILE = (
    ANALYSIS_DIR
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

WEIGHT_PER_STRUCTURAL_PAIR = 10


def build_model() -> WeightedD2L:
    model = WeightedD2L(ipa_file_path=str(FEATURE_FILE))
    model.seginv.add_segs(SURFACE_VOWELS)

    for surface, abstract in ABSTRACT_TARGETS.items():
        features = dict(model.seginv[surface].features)
        features["RTR"] = UNDERSPECIFIED
        model.seginv.add_custom(symbol=abstract, features=features)

    return model


def raw_weighted_data() -> list[tuple[str, str, int]]:
    """Balanced structural data from the successful seven-vowel test."""
    pairs = [
        # Clean adjacency: all target names represented.
        ("ㅏ A", "ㅏ ㅏ"),
        ("ㅓ EO", "ㅓ ㅓ"),
        ("ㅗ O", "ㅗ ㅗ"),
        ("ㅜ U", "ㅜ ㅜ"),
        ("ㆍ ARAE", "ㆍ ㆍ"),
        ("ㅡ EU", "ㅡ ㅡ"),

        # Same correspondence class under opposite RTR triggers.
        ("ㅏ EO", "ㅏ ㅏ"),
        ("ㅓ A", "ㅓ ㅓ"),
        ("ㅗ U", "ㅗ ㅗ"),
        ("ㅜ O", "ㅜ ㅜ"),
        ("ㆍ EU", "ㆍ ㆍ"),
        ("ㅡ ARAE", "ㅡ ㅡ"),

        # One intervening ㅣ: +RTR cases fail under the no-tier rule.
        ("ㅏ ㅣ A", "ㅏ ㅣ ㅏ"),
        ("ㅏ ㅣ EO", "ㅏ ㅣ ㅏ"),
        ("ㅗ ㅣ O", "ㅗ ㅣ ㅗ"),
        ("ㅗ ㅣ U", "ㅗ ㅣ ㅗ"),
        ("ㆍ ㅣ ARAE", "ㆍ ㅣ ㆍ"),
        ("ㆍ ㅣ EU", "ㆍ ㅣ ㆍ"),

        # -RTR controls.
        ("ㅓ ㅣ A", "ㅓ ㅣ ㅓ"),
        ("ㅓ ㅣ EO", "ㅓ ㅣ ㅓ"),
        ("ㅜ ㅣ O", "ㅜ ㅣ ㅜ"),
        ("ㅜ ㅣ U", "ㅜ ㅣ ㅜ"),
        ("ㅡ ㅣ ARAE", "ㅡ ㅣ ㅡ"),
        ("ㅡ ㅣ EU", "ㅡ ㅣ ㅡ"),

        # Two consecutive transparent ㅣ vowels.
        ("ㅏ ㅣ ㅣ A", "ㅏ ㅣ ㅣ ㅏ"),
        ("ㅗ ㅣ ㅣ U", "ㅗ ㅣ ㅣ ㅗ"),
        ("ㆍ ㅣ ㅣ EU", "ㆍ ㅣ ㅣ ㆍ"),
        ("ㅓ ㅣ ㅣ EO", "ㅓ ㅣ ㅣ ㅓ"),
        ("ㅜ ㅣ ㅣ O", "ㅜ ㅣ ㅣ ㅜ"),
        ("ㅡ ㅣ ㅣ ARAE", "ㅡ ㅣ ㅣ ㅡ"),
    ]

    return [
        (ur, sr, WEIGHT_PER_STRUCTURAL_PAIR)
        for ur, sr in pairs
    ]


def make_weighted_pairs(model: WeightedD2L):
    return [
        (
            SegStr(ur, seginv=model.seginv),
            SegStr(sr, seginv=model.seginv),
            weight,
        )
        for ur, sr, weight in raw_weighted_data()
    ]


def add_defaults(rule, pairs, pair_weights) -> None:
    underextensions = rule.underextension_SRs(
        pairs=pairs,
        pair_weights=pair_weights,
    )
    if not underextensions:
        return

    default_sr = sorted(
        underextensions.items(),
        reverse=True,
        key=lambda item: item[-1],
    )[0][0]

    rule.set_defaults(
        {
            feat: default_sr.features[feat]
            for feat in rule.features
        }
    )


def build_initial_no_tier_rule(
    model: WeightedD2L,
    weighted_pairs,
):
    """Reproduce D2L's first hypothesis without allowing tier induction."""
    pairs = model._weighted_train_setup(weighted_pairs)

    discrepancy = model._discrepancy
    target = discrepancy.get_alternating_UR_segs()

    left_contexts, right_contexts = model._get_tier_adj_contexts(
        discrepancy=discrepancy,
        tier=None,
    )

    left_rule = WeightedRule(
        seginv=model.seginv,
        target=target,
        features=discrepancy.feature_diff,
        left_ctxts=left_contexts,
        tier=None,
        harmony=True,
    )
    right_rule = WeightedRule(
        seginv=model.seginv,
        target=target,
        features=discrepancy.feature_diff,
        right_ctxts=right_contexts,
        tier=None,
        harmony=True,
    )

    add_defaults(left_rule, pairs, model.pair_weights)
    add_defaults(right_rule, pairs, model.pair_weights)

    left_accuracy = left_rule.accuracy(
        pairs,
        pair_weights=model.pair_weights,
    )
    right_accuracy = right_rule.accuracy(
        pairs,
        pair_weights=model.pair_weights,
    )

    rule = left_rule if left_accuracy >= right_accuracy else right_rule
    return rule, pairs


def inspect_initial_rule() -> None:
    model = build_model()
    weighted_pairs = make_weighted_pairs(model)

    rule, pairs = build_initial_no_tier_rule(
        model,
        weighted_pairs,
    )

    print("WEIGHTED D2L TEST")
    print("=================")
    print("unique structural pairs =", len(model.pair_weights))
    print(
        "lexical types represented =",
        sum(model.pair_weights.values()),
    )
    print(
        "weight per structural pair =",
        WEIGHT_PER_STRUCTURAL_PAIR,
    )

    n, m = rule.tsp_stats(
        pairs,
        pair_weights=model.pair_weights,
    )
    errors = n - m
    theta = n / math.log(n)
    errant = rule.errant_ctxts(pairs)
    errant_symbols = {str(segment) for segment in errant}

    print("\nINITIAL NO-TIER RULE")
    print("====================")
    print("rule =", rule)
    print("tier =", rule.tier)
    print("n =", n)
    print("correct =", m)
    print("errors =", errors)
    print("theta_N =", theta)
    print("TP passes =", tsp(n=n, m=m))
    print("errant contexts =", errant)

    assert len(model.pair_weights) == 30
    assert sum(model.pair_weights.values()) == 300
    assert n == 300, f"Expected n=300; got {n}"
    assert errors == 90, f"Expected 90 weighted errors; got {errors}"
    assert not tsp(n=n, m=m), (
        "Initial no-tier rule should fail TP."
    )
    assert errant_symbols == {"ㅣ"}, (
        f"Expected only ㅣ as errant context; got {errant_symbols}"
    )

    print("\nPASS: weighted no-tier rule fails TP as intended.")
    print("PASS: ㅣ is the only errant context.")


def inspect_final_model() -> None:
    model = build_model()
    weighted_pairs = make_weighted_pairs(model)

    model.train_weighted(weighted_pairs)

    print("\n\nFINAL WEIGHTED MODEL")
    print("====================")
    print(model)

    assert model.rule is not None, "Expected D2L to learn a rule."

    n, m = model.weighted_prediction_stats()
    errors = n - m

    print("tier =", model.rule.tier)
    print("n =", n)
    print("correct =", m)
    print("errors =", errors)
    print(
        "weighted prediction accuracy =",
        m / n if n else 0.0,
    )
    print(
        "weighted full-form accuracy =",
        model.weighted_pair_accuracy(),
    )

    assert model.rule.tier is not None, (
        "Expected tier induction after weighted TP failure."
    )

    projection = model.rule.tier.project(
        SegStr("ㅏ ㅣ A", seginv=model.seginv)
    )
    print("projection of ㅏ ㅣ A =", projection)

    assert "ㅣ" not in str(projection), (
        "Expected learned tier to exclude ㅣ."
    )
    assert n == 300, f"Expected final n=300; got {n}"
    assert errors == 0, f"Expected zero final errors; got {errors}"
    assert model.weighted_pair_accuracy() == 1.0

    print("\nPASS: tier was induced.")
    print("PASS: learned tier excludes ㅣ.")
    print("PASS: final weighted accuracy = 1.0.")


def main() -> None:
    inspect_initial_rule()
    inspect_final_model()
    print("\nALL WEIGHTED D2L VALIDATION CHECKS PASSED.")


if __name__ == "__main__":
    main()
