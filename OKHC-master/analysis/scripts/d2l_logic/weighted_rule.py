"""Weighted extension of algophon's D2L Rule.

Stock algophon learns over unique (UR, SR) pairs. In the OKHC experiment,
multiple sampled lexical wordform types can share the same vowel-only pair.
WeightedRule lets one structural pair represent multiple lexical types without
changing the phonological representation itself.

Weights affect numerical evidence (TSP counts, accuracy, default counts), but
not structural operations such as tier projection or errant-context identity.

This module also exposes diagnostic prediction traces. Those traces mirror
algophon's iterative Rule._predictions() exactly, but retain the conditioning
segment, whether that segment is a resolved earlier target, whether the
conditioning relation used the rule proper or the default, and whether the
failed prediction is eligible to contribute to D2L's deletion set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from algophon import SegStr
from algophon.models.D2L import Rule
from algophon.symbols import LWB, RWB


class WeightedRule(Rule):
    """D2L Rule whose counts can be weighted by represented lexical types."""

    @staticmethod
    def _weight_for(pair, pair_weights: Mapping | None) -> int:
        if pair_weights is None:
            return 1
        weight = int(pair_weights.get(pair, 1))
        if weight <= 0:
            raise ValueError(
                f"Pair weights must be positive integers; got {weight} for {pair}"
            )
        return weight

    def tsp_stats(self, pairs, pair_weights: Mapping | None = None) -> tuple[int, int]:
        """Return weighted (n, m) target-prediction counts."""
        n, m = 0, 0
        for ur, sr in set(pairs):
            if isinstance(ur, str):
                ur = SegStr(ur, seginv=self.seginv)
            if isinstance(sr, str):
                sr = SegStr(sr, seginv=self.seginv)

            pair = (ur, sr)
            weight = self._weight_for(pair, pair_weights)

            for idx, pred_sr_seg in self._predictions(ur):
                n += weight
                if sr[idx] == pred_sr_seg:
                    m += weight
        return n, m

    def accuracy(self, pairs, pair_weights: Mapping | None = None) -> float:
        """Weighted target-prediction accuracy."""
        n, m = self.tsp_stats(pairs=pairs, pair_weights=pair_weights)
        return m / n if n > 0 else 0.0

    def underextension_SRs(self, pairs, pair_weights: Mapping | None = None):
        """Count default-candidate SR realizations using lexical-type weights."""
        underex = defaultdict(int)
        for ur, sr in set(pairs):
            if isinstance(ur, str):
                ur = SegStr(ur, seginv=self.seginv)
            if isinstance(sr, str):
                sr = SegStr(sr, seginv=self.seginv)

            pair = (ur, sr)
            weight = self._weight_for(pair, pair_weights)
            pred = self.produce(ur=ur)

            for seg, sr_seg in zip(pred, sr):
                if seg in self.target:
                    underex[sr_seg] += weight
        return underex

    def prediction_diagnostics(
        self,
        pairs,
        pair_weights: Mapping | None = None,
        errors_only: bool = False,
    ) -> list[dict]:
        """Return detailed iterative prediction events.

        This mirrors algophon's Rule._predictions(), but records enough state to
        diagnose why predictions fail. In particular, a context can be an
        original specified segment, a previously resolved target, an unresolved
        target, or a boundary.

        The field eligible_for_delset_under_current_rule matches the logic
        relevant to D2L's errant-context deletion set: the prediction must be
        wrong, must have used an active rule context rather than the default,
        must not be a boundary, and the context itself must not be one of the
        protected abstract target symbols.
        """
        rows = []

        for ur, sr in set(pairs):
            if isinstance(ur, str):
                ur = SegStr(ur, seginv=self.seginv)
            if isinstance(sr, str):
                sr = SegStr(sr, seginv=self.seginv)

            pair = (ur, sr)
            weight = self._weight_for(pair, pair_weights)

            if self.tier is not None:
                projection = self.tier.project(segstr=ur)
                projection_idxs = list(projection.idxs)
            else:
                projection = SegStr(list(ur._segs), seginv=self.seginv)
                projection_idxs = list(range(len(projection)))

            original_projection = list(projection._segs)
            resolved_targets = {}

            tier_ptr = 0 if self.left_to_right else len(projection) - 1

            while 0 <= tier_ptr <= len(projection) - 1:
                seg = projection[tier_ptr]

                if seg in self.target:
                    if self.left_to_right:
                        ctxt_ptr = tier_ptr - 1
                        if ctxt_ptr >= 0:
                            ctxt = projection[ctxt_ptr]
                            ctxt_original_idx = projection_idxs[ctxt_ptr]
                        else:
                            ctxt = self.seginv[LWB]
                            ctxt_original_idx = None
                        active_contexts = self.left_ctxts
                    else:
                        ctxt_ptr = tier_ptr + 1
                        if ctxt_ptr < len(projection):
                            ctxt = projection[ctxt_ptr]
                            ctxt_original_idx = projection_idxs[ctxt_ptr]
                        else:
                            ctxt = self.seginv[RWB]
                            ctxt_original_idx = None
                        active_contexts = self.right_ctxts

                    uses_rule_context = ctxt in active_contexts
                    new_seg = (
                        self._apply(seg=seg, ctxt=ctxt)
                        if uses_rule_context
                        else self._apply_default(seg=seg)
                    )

                    target_original_idx = projection_idxs[tier_ptr]
                    actual_seg = sr[target_original_idx]
                    correct = new_seg == actual_seg

                    if ctxt_original_idx is None:
                        context_source = "BOUNDARY"
                        context_was_resolved_target = False
                        previous_target_correct = None
                        original_context_symbol = None
                    elif ctxt_ptr in resolved_targets:
                        context_source = "RESOLVED_TARGET_CONTEXT"
                        context_was_resolved_target = True
                        previous_target_correct = resolved_targets[ctxt_ptr]["correct"]
                        original_context_symbol = resolved_targets[ctxt_ptr]["abstract_symbol"]
                    else:
                        original_ctxt = original_projection[ctxt_ptr]
                        original_context_symbol = str(original_ctxt)
                        context_was_resolved_target = False
                        previous_target_correct = None
                        context_source = (
                            "UNRESOLVED_TARGET_CONTEXT"
                            if original_ctxt in self.target
                            else "ORIGINAL_SPECIFIED_CONTEXT"
                        )

                    context_symbol = str(ctxt)
                    protected_target_context = ctxt in self.target
                    boundary_context = ctxt_original_idx is None
                    protected_initial_context = ctxt_original_idx == 0
                    eligible_for_delset = (
                        (not correct)
                        and uses_rule_context
                        and (not boundary_context)
                        and (not protected_target_context)
                        and (not protected_initial_context)
                    )

                    event = {
                        "d2l_ur": str(ur),
                        "d2l_sr": str(sr),
                        "lexical_type_weight": weight,
                        "direction": (
                            "left_to_right"
                            if self.left_to_right
                            else "right_to_left"
                        ),
                        "harmony_type": (
                            "harmony"
                            if self.harmony
                            else "disharmony"
                        ),
                        "tier": str(self.tier) if self.tier is not None else "NONE",
                        "target_index": target_original_idx,
                        "target_abstract_segment": str(seg),
                        "actual_surface_segment": str(actual_seg),
                        "predicted_surface_segment": str(new_seg),
                        "correct": bool(correct),
                        "weighted_count": weight,
                        "context_segment": context_symbol,
                        "context_index": ctxt_original_idx,
                        "context_source": context_source,
                        "original_context_symbol": original_context_symbol,
                        "context_was_resolved_target": context_was_resolved_target,
                        "previous_target_prediction_correct": previous_target_correct,
                        "context_was_prior_misprediction": (
                            context_was_resolved_target
                            and previous_target_correct is False
                        ),
                        "used_rule_context": bool(uses_rule_context),
                        "used_default": not bool(uses_rule_context),
                        "context_is_protected_abstract_target": bool(
                            protected_target_context
                        ),
                        "context_is_protected_initial_position": bool(
                            protected_initial_context
                        ),
                        "context_is_boundary": bool(boundary_context),
                        "eligible_for_delset_under_current_rule": bool(
                            eligible_for_delset
                        ),
                    }

                    if not errors_only or not correct:
                        rows.append(event)

                    resolved_targets[tier_ptr] = {
                        "abstract_symbol": str(seg),
                        "predicted_symbol": str(new_seg),
                        "actual_symbol": str(actual_seg),
                        "correct": bool(correct),
                    }

                    projection._segs[tier_ptr] = new_seg

                tier_ptr += 1 if self.left_to_right else -1

        return rows
