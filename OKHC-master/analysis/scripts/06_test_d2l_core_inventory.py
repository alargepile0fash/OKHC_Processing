"""
Controlled D2L test for the seven-vowel core Middle Korean inventory.

Core vowels:
    ㆍ ㅏ ㅗ ㅡ ㅓ ㅜ ㅣ

Harmonic correspondence classes:
    ㅏ ~ ㅓ
    ㅗ ~ ㅜ
    ㆍ ~ ㅡ

ㅣ is fully specified [-RTR] and has no harmonic counterpart.

Named abstract UR targets:
    A, EO, O, U, ARAE, EU

Each target copies every non-RTR feature from its corresponding surface
vowel and replaces only RTR with UNDERSPECIFIED.

The test has two stages:
1. Initial no-tier local harmony rule.
2. Final trained D2L model with tier induction allowed.
"""

from pathlib import Path

from algophon import SegStr
from algophon.models.D2L import D2L, Rule
from algophon.symbols import UNDERSPECIFIED


FEATURE_FILE = (
    Path(__file__).resolve().parents[1]
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


def add_underspecified_target(d2l: D2L, symbol: str, surface_example: str) -> None:
    features = dict(d2l.seginv[surface_example].features)
    features["RTR"] = UNDERSPECIFIED
    d2l.seginv.add_custom(symbol=symbol, features=features)


def build_model() -> D2L:
    d2l = D2L(ipa_file_path=str(FEATURE_FILE))
    d2l.seginv.add_segs(SURFACE_VOWELS)

    for surface, abstract in ABSTRACT_TARGETS.items():
        add_underspecified_target(d2l, abstract, surface)

    return d2l


def make_training_pairs(d2l: D2L) -> list[tuple[SegStr, SegStr]]:
    raw_pairs = [
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

        # One intervening ㅣ.
        ("ㅏ ㅣ A", "ㅏ ㅣ ㅏ"),
        ("ㅏ ㅣ EO", "ㅏ ㅣ ㅏ"),
        ("ㅗ ㅣ O", "ㅗ ㅣ ㅗ"),
        ("ㅗ ㅣ U", "ㅗ ㅣ ㅗ"),
        ("ㆍ ㅣ ARAE", "ㆍ ㅣ ㆍ"),
        ("ㆍ ㅣ EU", "ㆍ ㅣ ㆍ"),

        ("ㅓ ㅣ A", "ㅓ ㅣ ㅓ"),
        ("ㅓ ㅣ EO", "ㅓ ㅣ ㅓ"),
        ("ㅜ ㅣ O", "ㅜ ㅣ ㅜ"),
        ("ㅜ ㅣ U", "ㅜ ㅣ ㅜ"),
        ("ㅡ ㅣ ARAE", "ㅡ ㅣ ㅡ"),
        ("ㅡ ㅣ EU", "ㅡ ㅣ ㅡ"),

        # Two consecutive ㅣ vowels.
        ("ㅏ ㅣ ㅣ A", "ㅏ ㅣ ㅣ ㅏ"),
        ("ㅗ ㅣ ㅣ U", "ㅗ ㅣ ㅣ ㅗ"),
        ("ㆍ ㅣ ㅣ EU", "ㆍ ㅣ ㅣ ㆍ"),
        ("ㅓ ㅣ ㅣ EO", "ㅓ ㅣ ㅣ ㅓ"),
        ("ㅜ ㅣ ㅣ O", "ㅜ ㅣ ㅣ ㅜ"),
        ("ㅡ ㅣ ㅣ ARAE", "ㅡ ㅣ ㅣ ㅡ"),
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

    print("\nSurface vowels:")
    for symbol in ["ㆍ", "ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ", "ㅣ"]:
        print(symbol, "=", d2l.seginv[symbol].features)

    print("\nAbstract targets:")
    for symbol in ["ARAE", "A", "EO", "O", "U", "EU"]:
        print(symbol, "=", d2l.seginv[symbol].features)

    print("\nOwn-surface feature differences:")
    for surface, abstract in ABSTRACT_TARGETS.items():
        diff = d2l.seginv.feature_diff(abstract, surface)
        print(f"{abstract} -> {surface}: {diff}")
        assert diff == {"RTR"}

    opposite = {
        "A": "ㅓ",
        "EO": "ㅏ",
        "O": "ㅜ",
        "U": "ㅗ",
        "ARAE": "ㅡ",
        "EU": "ㆍ",
    }

    print("\nOpposite-member feature differences:")
    for abstract, surface in opposite.items():
        diff = d2l.seginv.feature_diff(abstract, surface)
        print(f"{abstract} -> {surface}: {diff}")
        assert diff == {"RTR"}

    assert d2l.seginv["ㅣ"]["RTR"] == "-"
    assert d2l.seginv["ㅣ"]["front"] == "+"
    assert all(
        d2l.seginv[v]["front"] == "-"
        for v in ["ㆍ", "ㅏ", "ㅓ", "ㅗ", "ㅜ", "ㅡ",
                  "ARAE", "A", "EO", "O", "U", "EU"]
    )

    print("\nRepresentation check passed:")
    print("  all harmonic targets differ from either pair member only by RTR")
    print("  ㅣ is fully specified [-RTR]")
    print("  ㅣ is [+front]; harmonic vowels/targets are [-front]")


def add_rule_defaults(rule: Rule, pairs: set) -> None:
    underextensions = rule.underextension_SRs(pairs=pairs)
    if len(underextensions) > 0:
        default_sr = sorted(
            underextensions.items(),
            reverse=True,
            key=lambda item: item[-1],
        )[0][0]

        rule.set_defaults(
            {feat: default_sr.features[feat] for feat in rule.features}
        )


def build_initial_adjacency_rule(
    d2l: D2L,
    pairs: list[tuple[SegStr, SegStr]],
) -> tuple[Rule, set]:
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

    initial_rule = (
        left_rule
        if left_rule.accuracy(setup_pairs) >= right_rule.accuracy(setup_pairs)
        else right_rule
    )

    return initial_rule, setup_pairs


def print_rule_predictions(title: str, rule: Rule, pairs: set) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for ur, sr in sorted(pairs, key=lambda pair: str(pair[0])):
        pred = rule(ur)
        status = "OK" if pred == sr else "WRONG"
        print(f"{ur} -> {pred}   expected {sr}   {status}")


def inspect_initial_hypothesis() -> None:
    d2l = build_model()
    pairs = make_training_pairs(d2l)

    initial_rule, setup_pairs = build_initial_adjacency_rule(d2l, pairs)

    print("\n\nINITIAL NO-TIER HYPOTHESIS")
    print("==========================")
    print("rule =", initial_rule)
    print("tier =", initial_rule.tier)
    print("features =", initial_rule.features)
    print("target =", initial_rule.target)
    print("left contexts =", initial_rule.left_ctxts)
    print("right contexts =", initial_rule.right_ctxts)

    print_rule_predictions(
        "Initial adjacency predictions",
        initial_rule,
        setup_pairs,
    )

    n, m = initial_rule.tsp_stats(setup_pairs)
    errant = initial_rule.errant_ctxts(setup_pairs)

    print("\nInitial TSP statistics:")
    print("n =", n)
    print("correct =", m)
    print("errors =", n - m)
    print("accuracy =", m / n if n else 0.0)

    print("\nErrant contexts identified by D2L:")
    print(errant)

    errant_symbols = {str(x) for x in errant}

    print("\nCritical diagnostic:")
    if errant_symbols == {"ㅣ"}:
        print("PASS: ㅣ is the only errant local context.")
    elif "ㅣ" in errant_symbols:
        print("PARTIAL PASS: ㅣ is errant, but other contexts are also implicated.")
    else:
        print("FAIL: ㅣ was not identified as an errant context.")


def inspect_final_model() -> None:
    d2l = build_model()
    pairs = make_training_pairs(d2l)

    d2l.train(pairs)

    print("\n\nFINAL TRAINED D2L MODEL")
    print("=======================")
    print("model =", d2l)

    print("\nDiscrepancy:")
    print("features =", d2l._discrepancy.feature_diff)
    print("alternations =", d2l._discrepancy.alternations)

    if d2l.rule is None:
        print("\nNo productive rule was learned.")
        return

    print("\nRule details:")
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

    print("\nTier projections:")
    diagnostics = [
        "ㅏ ㅣ A",
        "ㅗ ㅣ U",
        "ㆍ ㅣ EU",
        "ㅏ ㅣ ㅣ EO",
    ]

    if d2l.rule.tier is None:
        print("No tier was induced.")
        return

    for form in diagnostics:
        ur = SegStr(form, seginv=d2l.seginv)
        projected = d2l.rule.tier.project(ur)
        print(f"{ur} -> {projected}")

    print("\nFinal diagnostic:")
    projection = d2l.rule.tier.project(
        SegStr("ㅏ ㅣ A", seginv=d2l.seginv)
    )
    if "ㅣ" not in str(projection):
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
