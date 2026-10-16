"""Lexical-type-weighted extension of Belth's algophon D2L model.

The learner still discovers discrepancies, contexts, natural classes, and tiers
from UNIQUE structural UR/SR pairs. Numerical productivity evidence is then
computed using lexical-type weights.

In addition to matching the learning behavior, this class records a diagnostic
trace of every recursive D2L attempt. The trace is observational only: it does
not change which rule or tier is learned.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Union

from algophon import SegStr, NatClass
from algophon.models.D2L import D2L, Tier, Discrepancy
from algophon.symbols import BOUNDARIES
from algophon.utils import tsp

from .weighted_rule import WeightedRule


def _stable_symbols(items) -> str:
    return " ".join(sorted(str(item) for item in items))


class WeightedD2L(D2L):
    """D2L with lexical-type-weighted evidence and diagnostic tracing."""

    def __init__(
        self,
        ipa_file_path: Union[None, str] = None,
        sep: str = "\t",
    ) -> None:
        super().__init__(ipa_file_path=ipa_file_path, sep=sep)
        self.pair_weights = {}
        self.training_pairs = set()
        self.analysis_trace = []
        self.prediction_events = []

    def train_weighted(self, weighted_pairs: Iterable) -> "WeightedD2L":
        """Train on an iterable of (UR, SR, lexical_type_weight) triples."""
        self.analysis_trace = []
        self.prediction_events = []

        pairs = self._weighted_train_setup(weighted_pairs)
        self.training_pairs = pairs

        harmony_rule = self.build_rule(
            pairs=pairs,
            harmony=True,
            _depth=0,
        )
        disharmony_rule = self.build_rule(
            pairs=pairs,
            harmony=False,
            _depth=0,
        )

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

    def _record_prediction_events(
        self,
        rule: WeightedRule,
        pairs: set,
        branch: str,
        depth: int,
        candidate_direction: str,
        selected_candidate: bool,
    ) -> None:
        for event in rule.prediction_diagnostics(
            pairs=pairs,
            pair_weights=self.pair_weights,
            errors_only=True,
        ):
            event = dict(event)
            event.update(
                {
                    "branch": branch,
                    "attempt_depth": depth,
                    "candidate_direction": candidate_direction,
                    "selected_candidate": selected_candidate,
                }
            )
            self.prediction_events.append(event)

    def build_rule(
        self,
        pairs: set,
        delset: set = None,
        tier=None,
        harmony: bool = True,
        discrepancy: Union[None, Discrepancy] = None,
        _depth: int = 0,
    ):
        """Belth's recursive rule builder with weighted evidence and tracing."""
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

        left_n, left_m = left_rule.tsp_stats(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        right_n, right_m = right_rule.tsp_stats(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        left_acc = left_m / left_n if left_n else 0.0
        right_acc = right_m / right_n if right_n else 0.0

        choose_left = left_acc >= right_acc
        rule = left_rule if choose_left else right_rule
        n, m = (left_n, left_m) if choose_left else (right_n, right_m)
        e = n - m
        branch = "harmony" if harmony else "disharmony"
        selected_direction = "left_to_right" if choose_left else "right_to_left"

        self._record_prediction_events(
            left_rule,
            pairs,
            branch,
            _depth,
            "left_to_right",
            choose_left,
        )
        self._record_prediction_events(
            right_rule,
            pairs,
            branch,
            _depth,
            "right_to_left",
            not choose_left,
        )

        trace_row = {
            "branch": branch,
            "attempt_depth": _depth,
            "tier_before": str(tier) if tier is not None else "NONE",
            "delset_before": _stable_symbols(delset),
            "target_segments": _stable_symbols(target),
            "left_contexts": _stable_symbols(lctxts),
            "right_contexts": _stable_symbols(rctxts),
            "left_prediction_N": int(left_n),
            "left_prediction_correct": int(left_m),
            "left_prediction_exceptions": int(left_n - left_m),
            "left_accuracy": left_acc,
            "right_prediction_N": int(right_n),
            "right_prediction_correct": int(right_m),
            "right_prediction_exceptions": int(right_n - right_m),
            "right_accuracy": right_acc,
            "selected_direction": selected_direction,
            "selected_prediction_N": int(n),
            "selected_prediction_correct": int(m),
            "selected_prediction_exceptions": int(e),
            "selected_accuracy": m / n if n else 0.0,
            "theta_N": (n / __import__("math").log(n)) if n > 1 else None,
            "TP_pass": bool(tsp(n=n, m=m)),
            "left_default": (
                _stable_symbols(left_rule.defaults.items())
                if left_rule.defaults is not None
                else ""
            ),
            "right_default": (
                _stable_symbols(right_rule.defaults.items())
                if right_rule.defaults is not None
                else ""
            ),
            "terminal_status": "",
            "left_errant_contexts": "",
            "right_errant_contexts": "",
            "all_errant_contexts": "",
            "protected_target_contexts_removed": "",
            "new_delset_segments": "",
            "delset_after": "",
            "tier_candidate_type": "",
            "tier_after": "",
            "natural_class_options": "",
            "chosen_natural_class": "",
        }

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

            trace_row["terminal_status"] = "TP_PASS"
            trace_row["tier_after"] = (
                str(rule.tier) if rule.tier is not None else "NONE"
            )
            self.analysis_trace.append(trace_row)
            return rule

        left_errant = left_rule.errant_ctxts(pairs)
        right_errant = right_rule.errant_ctxts(pairs)
        all_errant = left_errant.union(right_errant)
        protected = all_errant.intersection(target)

        old_delset = set(delset)
        delset = delset.union(all_errant).difference(target)
        new_delset = delset.difference(old_delset)

        trace_row.update(
            {
                "left_errant_contexts": _stable_symbols(left_errant),
                "right_errant_contexts": _stable_symbols(right_errant),
                "all_errant_contexts": _stable_symbols(all_errant),
                "protected_target_contexts_removed": _stable_symbols(protected),
                "new_delset_segments": _stable_symbols(new_delset),
                "delset_after": _stable_symbols(delset),
            }
        )

        opts = [
            NatClass(feats={feat}, seginv=self.seginv)
            for feat in self.seginv.feature_intersection(delset)
        ]
        opts = [
            opt for opt in opts
            if not any(ur in opt for ur in target)
        ]

        trace_row["natural_class_options"] = " || ".join(
            sorted(str(opt) for opt in opts)
        )

        if opts:
            best = sorted(
                opts,
                key=lambda opt: (
                    len(self.seginv.extension(opt)),
                    f"{opt}",
                ),
            )[0]
            trace_row["chosen_natural_class"] = str(best)

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
                trace_row["tier_candidate_type"] = "NATURAL_CLASS_COMPLEMENT"
            else:
                tier = Tier(
                    seginv=self.seginv,
                    feats=best,
                    as_delset=True,
                )
                trace_row["tier_candidate_type"] = "NATURAL_CLASS_DELETION"
        else:
            complement = self.seginv.segs.difference(delset)
            tier = Tier(seginv=self.seginv, segs=complement)
            trace_row["tier_candidate_type"] = "EXPLICIT_SEGMENT_COMPLEMENT"

        trace_row["tier_after"] = str(tier)

        if old_delset == delset:
            trace_row["terminal_status"] = "NO_NEW_REMOVABLE_CONTEXTS"
            self.analysis_trace.append(trace_row)
            return None

        trace_row["terminal_status"] = "RECURSE"
        self.analysis_trace.append(trace_row)

        return self.build_rule(
            pairs=pairs,
            delset=delset,
            tier=tier,
            harmony=harmony,
            discrepancy=discrepancy,
            _depth=_depth + 1,
        )
