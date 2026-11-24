from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class Instance:
    label: str
    score: float
    box: np.ndarray            # [x1, y1, x2, y2]
    mask: np.ndarray           # bool (H, W)
    area_pct: float = 0.0      # % diện tích ảnh
    closeness: float = 0.0     # 0 = xa, 1 = gần
    depth_rank: int = 0        # 1 = gần nhất


@dataclass
class SceneResult:
    image: Image.Image
    instances: list[Instance] = field(default_factory=list)
    depth: np.ndarray | None = None   # float32 (H, W), 1 = gần
    timings: dict[str, float] = field(default_factory=dict)  # giây / stage

    def union_mask(self) -> np.ndarray:
        h, w = self.image.height, self.image.width
        if not self.instances:
            return np.zeros((h, w), dtype=bool)
        return np.any([i.mask for i in self.instances], axis=0)
