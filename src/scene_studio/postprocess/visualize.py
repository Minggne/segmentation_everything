from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..types import SceneResult

PALETTE = [(255, 80, 80), (80, 200, 120), (80, 140, 255), (255, 200, 60),
           (200, 100, 255), (60, 220, 220), (255, 130, 200), (160, 160, 160)]


def overlay(result: SceneResult) -> Image.Image:
    arr = np.array(result.image).astype(np.float32)
    for i, ins in enumerate(result.instances):
        c = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        arr[ins.mask] = arr[ins.mask] * 0.45 + c * 0.55
    out = Image.fromarray(arr.astype(np.uint8))
    dr = ImageDraw.Draw(out)
    for i, ins in enumerate(result.instances):
        dr.rectangle([float(v) for v in ins.box], outline=PALETTE[i % len(PALETTE)], width=3)
        dr.text((float(ins.box[0]) + 4, float(ins.box[1]) + 4),
                f"{ins.label} {ins.score:.2f}", fill=(255, 255, 255))
    return out


def depth_colormap(depth: np.ndarray) -> Image.Image:
    cm = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return Image.fromarray(cm[..., ::-1])


def crop_cutouts(result: SceneResult) -> list[tuple[Image.Image, str]]:
    base = np.array(result.image)
    out = []
    for ins in result.instances:
        x1, y1, x2, y2 = [int(v) for v in ins.box]
        rgba = np.dstack([base, (ins.mask * 255).astype(np.uint8)])[y1:y2, x1:x2]
        out.append((Image.fromarray(rgba, "RGBA"), ins.label))
    return out
