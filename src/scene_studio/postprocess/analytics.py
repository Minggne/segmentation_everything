from __future__ import annotations

import numpy as np

from ..types import Instance

TABLE_HEADERS = ["Đối tượng", "Điểm", "% diện tích", "Độ gần", "Thứ hạng gần"]


def compute_stats(instances: list[Instance], depth: np.ndarray) -> None:
    """Điền area_pct, closeness, depth_rank (1 = gần nhất) trực tiếp vào instances."""
    for ins in instances:
        ins.area_pct = float(ins.mask.mean() * 100)
        ins.closeness = float(depth[ins.mask].mean()) if ins.mask.any() else 0.0
    for rank, ins in enumerate(sorted(instances, key=lambda i: -i.closeness), 1):
        ins.depth_rank = rank


def stats_rows(instances: list[Instance]) -> list[list]:
    return [[i.label, round(i.score, 2), round(i.area_pct, 2),
             round(i.closeness, 2), i.depth_rank] for i in instances]
