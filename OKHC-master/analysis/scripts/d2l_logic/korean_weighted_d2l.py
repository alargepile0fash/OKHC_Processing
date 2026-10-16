"""Korean-constrained weighted D2L.

This variant keeps the weighted Tolerance Principle and Belth-style recursive
tier induction, but enforces the linguistic assumptions of the Korean
experiment:

1. Harmony is progressive only (left-to-right).
2. Only harmony, not disharmony, is evaluated.
3. The word-initial segment obligatorily projects to every learned tier.
4. Initial-trigger errors are still counted, but an initial trigger occurrence
   cannot itself motivate tier deletion.
5. Failed contexts are retained diagnostically even when they are not legally
   usable for the deletion set.

The tier remains segmental for noninitial positions. The only positional
exception is index 0, which is always preserved as the trigger.
"""

from __future__ import annotations

import math
from typing import Union

from algophon import NatClass, SegStr
from algophon.models.D2L import Tier, Discrepancy
from algophon.symbols import BOUNDARIES
from algophon.utils import tsp

from .weighted_d2l import WeightedD2L, _stable_symbols
from .weighted_rule import WeightedRule


class InitialProtectedTier(Tier):
    """Tier whose first segment always projects, regardless of segment class."""

    def project(self, segstr: SegStr) -> Tier.Projection:
        if not isinstance(segstr, SegStr):
            raise ValueError(
                f"Tier projection not implemented for {type(segstr)}"
            )

        projected = []
        idxs = []
        for idx, seg in enumerate(segstr):
            if idx == 0 or seg in self:
                projected.append(seg)
                idxs.append(idx)

        return Tier.Projection(
            segs=projected,
            idxs=idxs,
            seginv=self.seginv,
        )


class KoreanProgressiveWeightedD2L(WeightedD2L):
    """Weighted D2L constrained to progressive initial-trigger harmony."""

    def train_weighted(self, weighted_pairs):
        self.analysis_trace = []
        self.prediction_events = []

        pairs = self._weighted_train_setup(weighted_pairs)
        self.training_pairs = pairs

        self.rule = self.build_progressive_rule(
            pairs=pairs,
            delset=None,
            tier=None,
            discrepancy=None,
            depth=0,
        )

        if self.rule is None:
            self.default = None

        return self

    def train(self, pairs):
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
                "Expected (UR, SR) pairs or weighted triples."
            )

        return self.train_weighted(weighted)

    def _make_protected_tier(self, **kwargs):
        return InitialProtectedTier(seginv=self.seginv, **kwargs)

    def _eligible_errant_contexts(
        self,
        rule: WeightedRule,
        pairs: set,
    ) -> tuple[set, set, list[dict]]:
        """Return (all failed contexts, eligible deletion contexts, events)."""
        events = rule.prediction_diagnostics(
            pairs=pairs,
            pair_weights=self.pair_weights,
            errors_only=True,
        )

        all_contexts = set()
        eligible = set()

        for event in events:
            segment = event["context_segment"]
            if not event["context_is_boundary"]:
                all_contexts.add(self.seginv[segment])

            if event["eligible_for_delset_under_current_rule"]:
                eligible.add(self.seginv[segment])

        return all_contexts, eligible, events

    def build_progressive_rule(
        self,
        pairs: set,
        delset: set | None = None,
        tier=None,
        discrepancy: Union[None, Discrepancy] = None,
        depth: int = 0,
    ):
        """Recursively build the progressive harmony rule."""
        delset = set() if delset is None else set(delset)

        if discrepancy is None:
            discrepancy = self._discrepancy

        target = discrepancy.get_alternating_UR_segs()
        left_contexts, _ = self._get_tier_adj_contexts(
            discrepancy=discrepancy,
            tier=tier,
        )

        rule = WeightedRule(
            seginv=self.seginv,
            target=target,
            features=discrepancy.feature_diff,
            left_ctxts=left_contexts,
            tier=tier,
            harmony=True,
        )

        underextensions = rule.underextension_SRs(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        if underextensions:
            default_sr = sorted(
                underextensions.items(),
                reverse=True,
                key=lambda item: item[-1],
            )[0][0]
            rule.set_defaults(
                {
                    feat: default_sr.features[feat]
                    for feat in discrepancy.feature_diff
                }
            )

        n, m = rule.tsp_stats(
            pairs=pairs,
            pair_weights=self.pair_weights,
        )
        errors = n - m
        passes = bool(tsp(n=n, m=m))

        all_failed_contexts, eligible_contexts, events = (
            self._eligible_errant_contexts(rule, pairs)
        )

        for event in events:
            item = dict(event)
            item.update(
                {
                    "branch": "harmony",
                    "attempt_depth": depth,
                    "candidate_direction": "left_to_right",
                    "selected_candidate": True,
                }
            )
            self.prediction_events.append(item)

        trace_row = {
            "branch": "harmony",
            "attempt_depth": depth,
            "fixed_direction": "left_to_right",
            "protect_initial_position": True,
            "tier_before": str(tier) if tier is not None else "NONE",
            "delset_before": _stable_symbols(delset),
            "target_segments": _stable_symbols(target),
            "left_contexts": _stable_symbols(left_contexts),
            "right_contexts": "",
            "left_prediction_N": int(n),
            "left_prediction_correct": int(m),
            "left_prediction_exceptions": int(errors),
            "left_accuracy": m / n if n else 0.0,
            "right_prediction_N": "",
            "right_prediction_correct": "",
            "right_prediction_exceptions": "",
            "right_accuracy": "",
            "selected_direction": "left_to_right",
            "selected_prediction_N": int(n),
            "selected_prediction_correct": int(m),
            "selected_prediction_exceptions": int(errors),
            "selected_accuracy": m / n if n else 0.0,
            "theta_N": n / math.log(n) if n > 1 else None,
            "TP_pass": passes,
            "left_default": (
                _stable_symbols(rule.defaults.items())
                if rule.defaults is not None
                else ""
            ),
            "right_default": "",
            "terminal_status": "",
            "left_errant_contexts": _stable_symbols(all_failed_contexts),
            "right_errant_contexts": "",
            "all_errant_contexts": _stable_symbols(all_failed_contexts),
            "eligible_errant_contexts": _stable_symbols(eligible_contexts),
            "protected_target_contexts_removed": "",
            "new_delset_segments": "",
            "delset_after": "",
            "tier_candidate_type": "",
            "tier_after": "",
            "natural_class_options": "",
            "chosen_natural_class": "",
        }

        if passes:
            trace_row["terminal_status"] = "TP_PASS"
            trace_row["tier_after"] = (
                str(rule.tier) if rule.tier is not None else "NONE"
            )
            self.analysis_trace.append(trace_row)
            return rule

        old_delset = set(delset)

        # Abstract targets remain protected. Initial surface trigger occurrences
        # are protected positionally by InitialProtectedTier and are excluded
        # from eligible_contexts by WeightedRule diagnostics.
        delset = delset.union(eligible_contexts).difference(target)
        new_delset = delset.difference(old_delset)

        trace_row["new_delset_segments"] = _stable_symbols(new_delset)
        trace_row["delset_after"] = _stable_symbols(delset)

        if old_delset == delset:
            trace_row["terminal_status"] = "NO_NEW_REMOVABLE_CONTEXTS"
            trace_row["tier_after"] = (
                str(tier) if tier is not None else "NONE"
            )
            self.analysis_trace.append(trace_row)
            return None

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
                tier = self._make_protected_tier(feats=complement)
                trace_row["tier_candidate_type"] = (
                    "NATURAL_CLASS_COMPLEMENT_INITIAL_PROTECTED"
                )
            else:
                tier = self._make_protected_tier(
                    feats=best,
                    as_delset=True,
                )
                trace_row["tier_candidate_type"] = (
                    "NATURAL_CLASS_DELETION_INITIAL_PROTECTED"
                )
        else:
            complement = self.seginv.segs.difference(delset)
            tier = self._make_protected_tier(segs=complement)
            trace_row["tier_candidate_type"] = (
                "EXPLICIT_SEGMENT_COMPLEMENT_INITIAL_PROTECTED"
            )

        trace_row["tier_after"] = str(tier)
        trace_row["terminal_status"] = "RECURSE"
        self.analysis_trace.append(trace_row)

        return self.build_progressive_rule(
            pairs=pairs,
            delset=delset,
            tier=tier,
            discrepancy=discrepancy,
            depth=depth + 1,
        )
