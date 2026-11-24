from __future__ import annotations

import numpy as np


class DepthAnythingEstimator:
    """Stage 3: ảnh -> depth map chuẩn hóa [0, 1], 1 = gần camera."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import pipeline

        dev = 0 if device == "cuda" else ("mps" if device == "mps" else -1)
        self.pipe = pipeline("depth-estimation", model=model_name, device=dev)

    def __call__(self, img) -> np.ndarray:
        d = self.pipe(img)["depth"].resize(img.size)
        return np.array(d).astype(np.float32) / 255.0
