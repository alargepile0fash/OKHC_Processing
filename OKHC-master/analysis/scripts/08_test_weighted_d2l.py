"""Controlled sanity test for lexical-type-weighted D2L.

Run from the project root with UTF-8 mode on Windows:
    python -X utf8 OKHC-master/analysis/scripts/08_test_weighted_d2l.py

Expected behavior:
- only three unique structural UR/SR pairs are stored;
- they represent 100 lexical types via weights 80, 15, and 5;
- weighted TSP counts use those lexical weights rather than counting each
  structural pair as one item.
"""

from pathlib import Path
import sys

from algophon import SegStr
from algophon.symbols import UNDERSPECIFIED

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from d2l_logic.weighted_d2l import WeightedD2L


FEATURE_FILE = ANALYSIS_DIR / "config" / "korean_d2l_core_inventory_features.tsv"
SURFACE_VOWELS = {"ㆍ", "ㅏ", "ㅗ", "ㅡ", "ㅓ", "ㅜ", "ㅣ"}
ABSTRACT_TARGETS = {
    "ㅏ": "A",
    "ㅓ": "EO",
    "ㅗ": "O",
    "ㅜ": "U",
    "ㆍ": "ARAE",
    "ㅡ": "EU",
}


def build_model() -> WeightedD2L:
    model = WeightedD2L(ipa_file_path=str(FEATURE_FILE))
    model.seginv.add_segs(SURFACE_VOWELS)

    for surface, abstract in ABSTRACT_TARGETS.items():
        features = dict(model.seginv[surface].features)
        features["RTR"] = UNDERSPECIFIED
        model.seginv.add_custom(symbol=abstract, features=features)

    return model


def main() -> None:
    model = build_model()

    raw = [
        # 80 sampled lexical types share this structural pair.
        ("ㅏ O A", "ㅏ ㅗ ㅏ", 80),
        # 15 sampled lexical types share this structural pair.
        ("ㅓ U A", "ㅓ ㅜ ㅓ", 15),
        # 5 sampled lexical types instantiate the transparent-i pattern.
        ("ㅏ ㅣ A", "ㅏ ㅣ ㅏ", 5),
    ]

    weighted_pairs = [
        (
            SegStr(ur, seginv=model.seginv),
            SegStr(sr, seginv=model.seginv),
            weight,
        )
        for ur, sr, weight in raw
    ]

    model.train_weighted(weighted_pairs)

    print("WEIGHTED D2L TEST")
    print("=================")
    print("unique structural pairs =", len(model.pair_weights))
    print("lexical types represented =", sum(model.pair_weights.values()))

    print("\nPair weights:")
    for (ur, sr), weight in sorted(
        model.pair_weights.items(),
        key=lambda item: str(item[0][0]),
    ):
        print(f"{ur} -> {sr}   weight={weight}")

    print("\nLearned model:")
    print(model)

    if model.rule is None:
        print("No productive rule learned.")
        return

    n, m = model.weighted_prediction_stats()
    print("\nWeighted prediction statistics:")
    print("n =", n)
    print("correct =", m)
    print("errors =", n - m)
    print("accuracy =", m / n if n else 0.0)
    print("weighted full-form accuracy =", model.weighted_pair_accuracy())
    print("tier =", model.rule.tier)

    assert len(model.pair_weights) == 3
    assert sum(model.pair_weights.values()) == 100
    print("\nPASS: structural pairs remain unique while lexical-type counts are preserved.")


if __name__ == "__main__":
    main()
