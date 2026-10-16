"""Weighted extension of algophon's D2L Rule.

Stock algophon learns over unique (UR, SR) pairs. In the OKHC experiment,
multiple sampled lexical wordform types can share the same vowel-only pair.
WeightedRule lets one structural pair represent multiple lexical types without
changing the phonological representation itself.

Weights affect numerical evidence (TSP counts, accuracy, default counts), but
not structural operations such as tier projection or errant-context identity.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from algophon import SegStr
from algophon.models.D2L import Rule


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
        """Return weighted (n, m) target-prediction counts.

        A structural pair with weight w contributes w for every prediction
        opportunity in that pair. Thus n and m are measured over sampled
        lexical types rather than unique vowel-pattern shapes.
        """
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
