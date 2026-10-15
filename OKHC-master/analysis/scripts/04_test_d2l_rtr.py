"""
Minimal Korean [RTR] harmony test for Caleb Belth's algophon D2L model.

Purpose
-------
Test the representation we want before connecting D2L to the historical corpus:

    - the first vowel is fully specified on the surface;
    - every later harmonic vowel is represented by an abstract underlying segment;
    - the abstract segment is underspecified ONLY for RTR;
    - D2L must learn to recover the correct surface RTR value from context.

This is deliberately a toy phonological feature system, not a full phonetic
description of Korean vowels.  Here, A represents the harmonic pair {ㅏ, ㅓ}
and O represents {ㅗ, ㅜ}.  Within each pair, RTR is the only feature that
distinguishes the two surface realizations.
"""

from pathlib import Path

from algophon import SegStr
from algophon.models.D2L import D2L
from algophon.symbols import UNDERSPECIFIED


HERE = Path(__file__).resolve().parent

# If you place the TSV in analysis/config/ and this script in analysis/scripts/,
# replace this with:
FEATURE_FILE = Path(__file__).resolve().parents[1] / "config" / "korean_d2l_toy_features.tsv"

def add_underspecified_target(d2l: D2L, symbol: str, surface_example: str) -> None:
    """
    Add an abstract UR segment by copying a surface vowel and replacing
    only its RTR value with algophon's UNDERSPECIFIED symbol.
    """
    features = dict(d2l.seginv[surface_example].features)
    features["RTR"] = UNDERSPECIFIED
    d2l.seginv.add_custom(symbol=symbol, features=features)


def build_model() -> D2L:
    """
    Build a D2L model with the toy Korean vowel feature inventory.
    """
    d2l = D2L(ipa_file_path=str(FEATURE_FILE))

    # A custom IPA file defines which symbols exist, but they still need to be
    # added to the active segment inventory before SegStr can use them.
    d2l.seginv.add_segs({"ㅏ", "ㅓ", "ㅗ", "ㅜ"})

    # Underlying harmonic pairs:
    #   A -> ㅏ / ㅓ depending on RTR
    #   O -> ㅗ / ㅜ depending on RTR
    add_underspecified_target(d2l, "A", "ㅏ")
    add_underspecified_target(d2l, "O", "ㅗ")

    return d2l


def make_training_pairs(d2l: D2L) -> list[tuple[SegStr, SegStr]]:
    """
    Fully harmonic toy forms.

    First vowel stays fully specified.
    Every following vowel is abstract in the UR.
    """
    raw_pairs = [
        ("ㅏ A O", "ㅏ ㅏ ㅗ"),
        ("ㅏ O A", "ㅏ ㅗ ㅏ"),
        ("ㅗ A O", "ㅗ ㅏ ㅗ"),
        ("ㅓ A O", "ㅓ ㅓ ㅜ"),
        ("ㅓ O A", "ㅓ ㅜ ㅓ"),
        ("ㅜ A O", "ㅜ ㅓ ㅜ"),
    ]

    return [
        (SegStr(ur, seginv=d2l.seginv), SegStr(sr, seginv=d2l.seginv))
        for ur, sr in raw_pairs
    ]


def inspect_representation(d2l: D2L) -> None:
    """
    Sanity checks proving that RTR is the only discrepancy.
    """
    print("Feature space:")
    print(d2l.seginv.feature_space)
    print()

    print("Abstract segments:")
    print("A =", d2l.seginv["A"].features)
    print("O =", d2l.seginv["O"].features)
    print()

    checks = [
        ("A", "ㅏ"),
        ("A", "ㅓ"),
        ("O", "ㅗ"),
        ("O", "ㅜ"),
    ]

    print("Feature differences:")
    for underlying, surface in checks:
        diff = d2l.seginv.feature_diff(underlying, surface)
        print(f"{underlying} -> {surface}: {diff}")
        assert diff == {"RTR"}, (
            f"Expected only RTR to differ for {underlying} -> {surface}, "
            f"but got {diff}"
        )

    print("\nRepresentation check passed: discrepancy is exactly {RTR}.")


def main() -> None:
    d2l = build_model()
    inspect_representation(d2l)

    pairs = make_training_pairs(d2l)

    d2l.train(pairs)

    print("\nLearned model:")
    print(d2l)

    print("\nDiscrepancy:")
    print("features =", d2l._discrepancy.feature_diff)
    print("alternations =", d2l._discrepancy.alternations)

    print("\nTraining predictions:")
    correct = 0
    for ur, sr in pairs:
        pred = d2l.produce(ur)
        ok = pred == sr
        correct += int(ok)
        print(f"{ur} -> {pred}   expected {sr}   {'OK' if ok else 'WRONG'}")

    print(f"\nAccuracy: {correct}/{len(pairs)} = {correct / len(pairs):.3f}")
    print("D2L accuracy():", d2l.accuracy(pairs))

    if d2l.rule is None:
        print("\nNo productive rule was learned.")
    else:
        print("\nRule details:")
        print("rule =", d2l.rule)
        print("harmony =", d2l.rule.harmony)
        print("features =", d2l.rule.features)
        print("target =", d2l.rule.target)
        print("tier =", d2l.rule.tier)
        print("left contexts =", d2l.rule.left_ctxts)
        print("right contexts =", d2l.rule.right_ctxts)


if __name__ == "__main__":
    main()
