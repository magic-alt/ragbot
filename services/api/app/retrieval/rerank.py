from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def rrf_fuse(
    primary: Iterable[Tuple[str, float]],
    secondary: Iterable[Tuple[str, float]],
    k: int = 60,
    weight_primary: float = 0.5,
    weight_secondary: float = 0.5,
) -> List[Tuple[str, float]]:
    """Fuse two rankings without letting one disjoint modality crowd out the other.

    Equal weights are the safe default for hybrid retrieval.  With the previous
    0.6/0.4 defaults and ``k=60``, a 10-item primary list completely outranked a
    disjoint secondary list: primary rank 10 scored 0.6/70, which is greater
    than secondary rank 1 at 0.4/61.  Callers that have calibrated modality
    weights can still pass explicit values.
    """
    ranks: Dict[str, float] = {}
    for rank, (item_id, _) in enumerate(primary, start=1):
        ranks[item_id] = ranks.get(item_id, 0.0) + weight_primary * (1.0 / (k + rank))
    for rank, (item_id, _) in enumerate(secondary, start=1):
        ranks[item_id] = ranks.get(item_id, 0.0) + weight_secondary * (1.0 / (k + rank))
    return sorted(ranks.items(), key=lambda item: item[1], reverse=True)
