from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def rrf_fuse(
    primary: Iterable[Tuple[str, float]],
    secondary: Iterable[Tuple[str, float]],
    k: int = 60,
    weight_primary: float = 0.6,
    weight_secondary: float = 0.4,
) -> List[Tuple[str, float]]:
    ranks: Dict[str, float] = {}
    for rank, (item_id, _) in enumerate(primary, start=1):
        ranks[item_id] = ranks.get(item_id, 0.0) + weight_primary * (1.0 / (k + rank))
    for rank, (item_id, _) in enumerate(secondary, start=1):
        ranks[item_id] = ranks.get(item_id, 0.0) + weight_secondary * (1.0 / (k + rank))
    return sorted(ranks.items(), key=lambda item: item[1], reverse=True)

