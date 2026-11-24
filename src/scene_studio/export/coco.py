"""Xuất kết quả sang COCO instance segmentation -> dùng làm công cụ auto-labeling."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..types import SceneResult


def _polygons(mask: np.ndarray) -> list[list[float]]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in cnts if len(c) >= 3]


def to_coco(result: SceneResult, file_name: str = "image.png") -> dict:
    names = sorted({i.label for i in result.instances})
    cat_id = {n: k + 1 for k, n in enumerate(names)}
    anns = []
    for k, ins in enumerate(result.instances, 1):
        x1, y1, x2, y2 = [float(v) for v in ins.box]
        anns.append({
            "id": k, "image_id": 1, "category_id": cat_id[ins.label],
            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": float(ins.mask.sum()),
            "segmentation": _polygons(ins.mask), "iscrowd": 0, "score": ins.score,
        })
    return {
        "images": [{"id": 1, "file_name": file_name,
                    "width": result.image.width, "height": result.image.height}],
        "categories": [{"id": v, "name": n} for n, v in cat_id.items()],
        "annotations": anns,
    }


def write_coco(result: SceneResult, path: str | Path, file_name: str = "image.png") -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_coco(result, file_name), ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return path
