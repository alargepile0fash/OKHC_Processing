"""Lexical-type-weighted extension of Belth's algophon D2L model.

The learner still discovers discrepancies, contexts, natural classes, and tiers
from UNIQUE structural UR/SR pairs. Numerical productivity evidence is then
computed using lexical-type weights.

Input to train_weighted():
    (UR, SR, lexical_type_weight)

where weight is the number of sampled lexical wordform rows in the same time
window that share that vowel-only UR/SR pair. Corpus token frequency is not
used as weight.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Union

from algophon import SegStr, NatClass
from algophon.models.D2L import D2L, Tier, Discrepancy
from algophon.symbols import BOUNDARIES
from algophon.utils import tsp

from .weighted_rule import WeightedRule


class WeightedD2L(D2L):
    """D2L with lexical-type-weighted TSP/default/accuracy calculations."""

    def __init__(
        self,
        ipa_file_path: Union[None, str] = None,
        sep: str = "\t",
    ) -> None:
        super().__init__(ipa_file_path=ipa_file_path, sep=sep)
        self.pair_weights = {}
        self.training_pairs = set()

    def train_weighted(self, weighted_pairs: Iterable) -> "WeightedD2L":
        """Train on an iterable of (UR, SR, lexical_type_weight) triples."""
        pairs = self._weighted_train_setup(weighted_pairs)
        self.training_pairs = pairs

        harmony_rule = self.build_rule(pairs=pairs, harmony=True)
        disharmony_rule = self.build_rule(pairs=pairs, harmony=False)

        if harmony_rule and not disharmony_rule:
            self.rule = harmony_rule
        elif disharmony_rule and not harmony_rule:
            self.rule = disharmony_rule
        elif harmony_rule and disharmony_rule:
            harmony_acc = harmony_rule.accuracy(
                pairs, pair_weights=self.pair_weights
            )
            disharmony_acc = disharmony_rule.accuracy(
                pairs, pair_weights=self.pair_weights
            )
            self.rule = (
                harmony_rule
                if harmony_acc >= disharmony_acc
                else disharmony_rule
            )
        else:
            self.rule = None
            self.default = None

        return self

    def train(self, pairs: Iterable) -> "WeightedD2L":
        """Compatibility wrapper for either 2-tuples or weighted 3-tuples."""
        materialized = list(pairs)
        if not materialized:
            raise ValueError("Cannot train D2L on an empty dataset.")

        item_len = len(materialized[0])
        if item_len == 2:
            weighted = [(ur, sr, 1) for ur, sr in materialized]
        elif item_len == 3:
            weighted = materialized
        else:
            raise ValueError(
                "Expected (UR, SR) pairs or (UR, SR, weight) triples."
            )
        return self.train_weighted(weighted)

    def _weighted_train_setup(self, weighted_pairs: Iterable) -> set:
        """Reuse stock D2L setup while preserving a lexical weight per pair."""
        weighted_pairs = list(weighted_pairs)
        if not weighted_pairs:
            raise ValueError("Cannot train D2L on an empty dataset.")

        # Merge duplicate input rows before stock D2L turns structural pairs
        # into a set. String keys are sufficient because UR/SR strings are the
        # representation whose lexical multiplicity we intend to preserve.
        raw_weights = defaultdict(int)
        representatives = {}

        for item in weighted_pairs:
            if len(item) != 3:
                raise ValueError(
                    "Weighted training requires (UR, SR, weight) triples."
                )
            ur, sr, weight = item
            weight = int(weight)
            if weight <= 0:
                raise ValueError(
                    "lexical_type_weight must be a positive integer; "
                    f"got {weight}"
                )

            key = (str(ur), str(sr))
            raw_weights[key] += weight
            representatives[key] = (ur, sr)

        # This preserves Belth's abstract-UR inference and Discrepancy logic.
        pairs = super()._train_setup(list(representatives.values()))

        self.pair_weights = {}
        for ur, sr in pairs:
            key = (str(ur), str(sr))
            if key not in raw_weights:
                raise RuntimeError(
                    f"Internal weighting error: no weight found for {key}"
                )
            self.pair_weights[(ur, sr)] = raw_weights[key]

        return pairs

    def weighted_prediction_stats(self, pairs=None) -> tuple[int, int]:
        """Return weighted target-prediction (n, m) for the learned rule."""
        if self.rule is None:
            return 0, 0
        if pairs is None:
            pairs = self.training_pairs
        return self.rule.tsp_stats(pairs, pair_weights=self.pair_weights)

    def weighted_pair_accuracy(self, pairs=None) -> float:
        """Full-form accuracy weighted by represented lexical wordform types."""
        if pairs is None:
            pairs = self.training_pairs

        total_weight = 0
        correct_weight = 0
        for pair in set(pairs):
            ur, sr = pair
            weight = self.pair_weights[pair]
            total_weight += weight
            if self.produce(ur) == sr:
                correct_weight += weight
        return correct_weight / total_weight if total_weight else 0.0

    def accuracy(self, pairs=None) -> float:
        """Return lexical-type-weighted full-form accuracy."""
        return self.weighted_pair_accuracy(pairs=pairs)

    def build_rule(
        self,
        pairs: set,
        delset: set = None,
        tier=None,
        harmony: bool = True,
        discrepancy: Union[None, Discrepancy] = None,
    ):
        """Belth's recursive rule builder with weighted numerical evidence."""
        delset = set() if delset is None else delset
        if discrepancy is None:
            discrepancy = self._discrepancy

        target = discrepancy.get_alternating_UR_segs()
        lctxts, rctxts = self._get_tier_adj_contexts(
            discrepancy=discrepancy,
            tier=tier,
        )

        left_rule = WeightedRule(
            seginv=self.seginv,
            target=target,
            features=discrepancy.feature_diff,
            left_ctxts=lctxts,
            tier=tier,
            harmony=harmony,
        )
        left_underextensions = left_rule.underextension_SRs(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        if left_underextensions:
            left_default_sr = sorted(
                left_underextensions.items(),
                reverse=True,
                key=lambda item: item[-1],
            )[0][0]
            left_rule.set_defaults(
                {
                    feat: left_default_sr.features[feat]
                    for feat in discrepancy.feature_diff
                }
            )

        right_rule = WeightedRule(
            seginv=self.seginv,
            target=target,
            features=discrepancy.feature_diff,
            right_ctxts=rctxts,
            tier=tier,
            harmony=harmony,
        )
        right_underextensions = right_rule.underextension_SRs(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        if right_underextensions:
            right_default_sr = sorted(
                right_underextensions.items(),
                reverse=True,
                key=lambda item: item[-1],
            )[0][0]
            right_rule.set_defaults(
                {
                    feat: right_default_sr.features[feat]
                    for feat in discrepancy.feature_diff
                }
            )

        left_acc = left_rule.accuracy(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        right_acc = right_rule.accuracy(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        rule = left_rule if left_acc >= right_acc else right_rule

        n, m = rule.tsp_stats(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )

        if tsp(n=n, m=m):
            ctxt = set(
                rule.left_ctxts
                if rule.left_ctxts is not None
                else rule.right_ctxts
            )

            if rule.tier is not None:
                acc_before = rule.accuracy(
                    pairs,
                    pair_weights=self.pair_weights,
                )
                rule.set_ctxts(ctxts=rule.tier._tierset)
                acc_after = rule.accuracy(
                    pairs,
                    pair_weights=self.pair_weights,
                )
                if acc_after < acc_before:
                    rule.set_ctxts(ctxt)
            return rule

        # Structural error identities stay unweighted. Their frequency already
        # mattered above, in the weighted TSP decision that got us here.
        old_delset = set(delset)
        delset = delset.union(
            left_rule.errant_ctxts(pairs).union(
                right_rule.errant_ctxts(pairs)
            )
        ).difference(target)

        opts = [
            NatClass(feats={feat}, seginv=self.seginv)
            for feat in self.seginv.feature_intersection(delset)
        ]
        opts = [
            opt for opt in opts
            if not any(ur in opt for ur in target)
        ]

        if opts:
            best = sorted(
                opts,
                key=lambda opt: (
                    len(self.seginv.extension(opt)),
                    f"{opt}",
                ),
            )[0]
            negate_val = {"+": "-", "-": "+"}
            complement = NatClass(
                feats={
                    f"{negate_val[feat[0]]}{feat[1:]}"
                    for feat in best.feats
                },
                seginv=self.seginv,
            )

            if (
                best.extension_complement().difference(BOUNDARIES)
                == complement.extension()
            ):
                tier = Tier(seginv=self.seginv, feats=complement)
            else:
                tier = Tier(
                    seginv=self.seginv,
                    feats=best,
                    as_delset=True,
                )
        else:
            complement = self.seginv.segs.difference(delset)
            tier = Tier(seginv=self.seginv, segs=complement)

        if old_delset == delset:
            return None

        return self.build_rule(
            pairs=pairs,
            delset=delset,
            tier=tier,
            harmony=harmony,
            discrepancy=discrepancy,
        )
