from __future__ import annotations

import cv2
import numpy as np


class SAM2Segmenter:
    """Stage 2: boxes -> masks (SAM 2 qua ultralytics)."""

    def __init__(self, weights: str = "sam2.1_b.pt", device: str = "cpu"):
        from ultralytics import SAM

        self.device = device
        self.model = SAM(weights)

    def __call__(self, img, boxes: np.ndarray) -> np.ndarray:
        bgr = np.array(img)[..., ::-1]
        r = self.model(bgr, bboxes=np.asarray(boxes).tolist(),
                       device=self.device, verbose=False)[0]
        masks = r.masks.data.cpu().numpy() > 0.5
        w, h = img.size
        if masks.shape[1:] != (h, w):
            masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h)) > 0 for m in masks])
        return masks
