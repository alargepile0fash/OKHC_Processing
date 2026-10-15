"""
Controlled D2L test: transparent Korean ㅣ in [RTR] harmony.

This test has two stages:

1. INITIAL ADJACENCY HYPOTHESIS
   We manually reproduce D2L's first local rule before tier induction.
   Because ㅣ is fully specified as [-RTR], a target immediately after ㅣ
   should initially be predicted from ㅣ. In +RTR forms this should fail.
   We then inspect which segment D2L identifies as the errant context.

2. FINAL D2L MODEL
   We train a fresh D2L model normally and inspect whether it constructs
   a tier that excludes ㅣ and thereby restores the harmony generalization.

Representation:
    ㅏ = [+RTR]
    ㅓ = [-RTR]
    ㅗ = [+RTR]
    ㅜ = [-RTR]
    ㅣ = [-RTR], fully specified

    A = ㅏ/ㅓ-type target, underspecified only for RTR
    O = ㅗ/ㅜ-type target, underspecified only for RTR
"""

from pathlib import Path

from algophon import SegStr
from algophon.models.D2L import D2L, Rule
from algophon.symbols import UNDERSPECIFIED


FEATURE_FILE = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "korean_d2l_transparent_i_features.tsv"
)


def add_underspecified_target(
    d2l: D2L,
    symbol: str,
    surface_example: str,
) -> None:
    """Copy a surface vowel and underspecify only RTR."""
    features = dict(d2l.seginv[surface_example].features)
    features["RTR"] = UNDERSPECIFIED
    d2l.seginv.add_custom(symbol=symbol, features=features)


def build_model() -> D2L:
    d2l = D2L(ipa_file_path=str(FEATURE_FILE))

    d2l.seginv.add_segs({"ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ"})

    add_underspecified_target(d2l, "A", "ㅏ")
    add_underspecified_target(d2l, "O", "ㅗ")

    return d2l


def make_training_pairs(d2l: D2L) -> list[tuple[SegStr, SegStr]]:
    """
    Include ordinary adjacency plus harmony across one or two instances of ㅣ.

    The crucial +RTR cases are:
        ㅏ ㅣ A -> ㅏ ㅣ ㅏ
        ㅗ ㅣ O -> ㅗ ㅣ ㅗ

    Under a strictly adjacent harmony rule, fully specified [-RTR] ㅣ should
    incorrectly predict a [-RTR] realization of the following target.
    """
    raw_pairs = [
        # Clean adjacent harmony
        ("ㅏ A", "ㅏ ㅏ"),
        ("ㅓ A", "ㅓ ㅓ"),
        ("ㅗ O", "ㅗ ㅗ"),
        ("ㅜ O", "ㅜ ㅜ"),

        # One intervening ㅣ
        ("ㅏ ㅣ A", "ㅏ ㅣ ㅏ"),
        ("ㅓ ㅣ A", "ㅓ ㅣ ㅓ"),
        ("ㅗ ㅣ O", "ㅗ ㅣ ㅗ"),
        ("ㅜ ㅣ O", "ㅜ ㅣ ㅜ"),

        # Cross-shape targets across ㅣ
        ("ㅏ ㅣ O", "ㅏ ㅣ ㅗ"),
        ("ㅓ ㅣ O", "ㅓ ㅣ ㅜ"),
        ("ㅗ ㅣ A", "ㅗ ㅣ ㅏ"),
        ("ㅜ ㅣ A", "ㅜ ㅣ ㅓ"),

        # Two consecutive transparent vowels
        ("ㅏ ㅣ ㅣ A", "ㅏ ㅣ ㅣ ㅏ"),
        ("ㅓ ㅣ ㅣ O", "ㅓ ㅣ ㅣ ㅜ"),
        ("ㅗ ㅣ ㅣ A", "ㅗ ㅣ ㅣ ㅏ"),
        ("ㅜ ㅣ ㅣ O", "ㅜ ㅣ ㅣ ㅜ"),
    ]

    return [
        (
            SegStr(ur, seginv=d2l.seginv),
            SegStr(sr, seginv=d2l.seginv),
        )
        for ur, sr in raw_pairs
    ]


def inspect_representation(d2l: D2L) -> None:
    print("FEATURE REPRESENTATION")
    print("======================")
    print("Feature space:")
    print(d2l.seginv.feature_space)

    print("\nRelevant segments:")
    for symbol in ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅣ", "A", "O"]:
        print(symbol, "=", d2l.seginv[symbol].features)

    print("\nTarget feature differences:")
    for underlying, surface in [
        ("A", "ㅏ"),
        ("A", "ㅓ"),
        ("O", "ㅗ"),
        ("O", "ㅜ"),
    ]:
        diff = d2l.seginv.feature_diff(underlying, surface)
        print(f"{underlying} -> {surface}: {diff}")
        assert diff == {"RTR"}, (
            f"Expected only RTR to differ for {underlying} -> {surface}; "
            f"got {diff}"
        )

    # ㅣ is NOT an abstract target.
    assert d2l.seginv["ㅣ"]["hi"] == "+"
    assert d2l.seginv["ㅣ"]["RTR"] == "-"

    # A/O and the four alternating surface vowels are [-hi].
    assert all(
        d2l.seginv[v]["hi"] == "-"
        for v in ["ㅏ", "ㅓ", "ㅗ", "ㅜ", "A", "O"]
    )

    print("\nRepresentation check passed:")
    print("  A/O discrepancy = {RTR}")
    print("  ㅣ = [+hi, -RTR], fully specified")
    print("  A/O are the only abstract harmony targets")


def add_rule_defaults(rule: Rule, pairs: set) -> None:
    """
    Reproduce the default-setting step in D2L.build_rule().
    """
    underextensions = rule.underextension_SRs(pairs=pairs)
    if len(underextensions) > 0:
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


def build_initial_adjacency_rule(
    d2l: D2L,
    pairs: list[tuple[SegStr, SegStr]],
) -> tuple[Rule, set]:
    """
    Reproduce ONLY the first no-tier step of D2L.build_rule().

    This deliberately does not allow recursive tier construction.
    """
    setup_pairs = d2l._train_setup(pairs)

    discrepancy = d2l._discrepancy
    target = discrepancy.get_alternating_UR_segs()

    left_contexts, right_contexts = d2l._get_tier_adj_contexts(
        discrepancy=discrepancy,
        tier=None,
    )

    left_rule = Rule(
        seginv=d2l.seginv,
        target=target,
        features=discrepancy.feature_diff,
        left_ctxts=left_contexts,
        tier=None,
        harmony=True,
    )
    add_rule_defaults(left_rule, setup_pairs)

    right_rule = Rule(
        seginv=d2l.seginv,
        target=target,
        features=discrepancy.feature_diff,
        right_ctxts=right_contexts,
        tier=None,
        harmony=True,
    )
    add_rule_defaults(right_rule, setup_pairs)

    # This is the same left-vs-right choice made inside D2L.build_rule().
    initial_rule = (
        left_rule
        if left_rule.accuracy(setup_pairs) >= right_rule.accuracy(setup_pairs)
        else right_rule
    )

    return initial_rule, setup_pairs


def print_rule_predictions(
    title: str,
    rule: Rule,
    pairs: set,
) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for ur, sr in sorted(
        pairs,
        key=lambda pair: str(pair[0]),
    ):
        pred = rule(ur)
        status = "OK" if pred == sr else "WRONG"
        print(
            f"{ur} -> {pred}   expected {sr}   {status}"
        )


def inspect_initial_hypothesis() -> None:
    """
    Show what D2L sees BEFORE it is allowed to induce a tier.
    """
    d2l = build_model()
    pairs = make_training_pairs(d2l)

    initial_rule, setup_pairs = build_initial_adjacency_rule(
        d2l,
        pairs,
    )

    print("\n\nINITIAL NO-TIER HYPOTHESIS")
    print("==========================")
    print("rule =", initial_rule)
    print("tier =", initial_rule.tier)
    print("left contexts =", initial_rule.left_ctxts)
    print("right contexts =", initial_rule.right_ctxts)

    print_rule_predictions(
        "Initial adjacency predictions",
        initial_rule,
        setup_pairs,
    )

    n, m = initial_rule.tsp_stats(setup_pairs)
    print("\nInitial TSP statistics:")
    print("n =", n)
    print("correct =", m)
    print("errors =", n - m)
    print("accuracy =", m / n if n else 0.0)

    errant = initial_rule.errant_ctxts(setup_pairs)
    print("\nErrant contexts identified by D2L:")
    print(errant)

    print("\nCritical diagnostic:")
    if "ㅣ" in errant:
        print("PASS: ㅣ is identified as an errant local context.")
    else:
        print("FAIL: ㅣ was NOT identified as an errant local context.")

    return errant


def inspect_final_model() -> None:
    """
    Train a completely fresh D2L instance normally, allowing tier induction.
    """
    d2l = build_model()
    pairs = make_training_pairs(d2l)

    d2l.train(pairs)

    print("\n\nFINAL TRAINED D2L MODEL")
    print("=======================")
    print("model =", d2l)

    print("\nDiscrepancy:")
    print("features =", d2l._discrepancy.feature_diff)
    print("alternations =", d2l._discrepancy.alternations)

    print("\nRule details:")
    if d2l.rule is None:
        print("No productive rule was learned.")
        return

    print("rule =", d2l.rule)
    print("harmony =", d2l.rule.harmony)
    print("features =", d2l.rule.features)
    print("target =", d2l.rule.target)
    print("tier =", d2l.rule.tier)
    print("left contexts =", d2l.rule.left_ctxts)
    print("right contexts =", d2l.rule.right_ctxts)

    print_rule_predictions(
        "Final post-learning predictions",
        d2l.rule,
        set(pairs),
    )

    n, m = d2l.rule.tsp_stats(pairs)
    print("\nFinal TSP statistics:")
    print("n =", n)
    print("correct =", m)
    print("errors =", n - m)
    print("accuracy =", m / n if n else 0.0)
    print("D2L accuracy() =", d2l.accuracy(pairs))

    print("\nFinal diagnostic:")
    if d2l.rule.tier is None:
        print(
            "No tier was induced. If the initial adjacency rule failed, "
            "we need to inspect why D2L still accepted a no-tier analysis."
        )
    else:
        print("A tier was induced:", d2l.rule.tier)
        projected_i = d2l.rule.tier.project(
            SegStr("ㅏ ㅣ A", seginv=d2l.seginv)
        )
        print("Projection of ㅏ ㅣ A =", projected_i)

        if "ㅣ" not in str(projected_i):
            print("PASS: learned tier excludes ㅣ.")
        else:
            print("FAIL: learned tier still contains ㅣ.")


def main() -> None:
    representation_model = build_model()
    inspect_representation(representation_model)

    inspect_initial_hypothesis()
    inspect_final_model()


if __name__ == "__main__":
    main()
