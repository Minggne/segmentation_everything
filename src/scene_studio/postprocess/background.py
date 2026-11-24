"""Tách nền: alpha mềm viền, thay nền (trong suốt / màu / ảnh), cắt sát vật thể."""
from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image, ImageOps

BG_MODES = ("transparent", "color", "image")


def parse_color(value) -> tuple[int, int, int]:
    """Nhận '#rrggbb', '#rgb', 'rgb(a)(r, g, b, ...)' hoặc tuple; lỗi thì trả về trắng."""
    white = (255, 255, 255)
    try:
        if isinstance(value, (tuple, list)):
            vals = [int(v) for v in value[:3]]
        else:
            s = (value or "#ffffff").strip()
            if s.startswith("#"):
                h = s[1:]
                if len(h) == 3:
                    h = "".join(c * 2 for c in h)
                vals = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
            else:
                vals = [int(float(v)) for v in re.findall(r"[\d.]+", s)[:3]]
    except (ValueError, TypeError):
        return white
    if len(vals) != 3 or any(not 0 <= v <= 255 for v in vals):
        return white
    return (vals[0], vals[1], vals[2])


def refine_alpha(mask: np.ndarray, size: tuple[int, int], feather: float = 2.0) -> np.ndarray:
    """mask (bool/float, kích thước bất kỳ) -> alpha float32 [0,1] theo size=(W,H), viền mềm."""
    m = mask.astype(np.float32)
    w, h = size
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
        # kéo giãn nhẹ để viền không bị mờ nhạt sau khi blur
        m = np.clip((m - 0.5) * 1.6 + 0.5, 0.0, 1.0)
    return m


def compose(img: Image.Image, mask: np.ndarray, mode: str = "transparent",
            color=(255, 255, 255), bg_image: Image.Image | None = None,
            feather: float = 2.0) -> Image.Image:
    if mode not in BG_MODES:
        raise ValueError(f"mode phải là một trong {BG_MODES}")
    img = img.convert("RGB")
    alpha = refine_alpha(mask, img.size, feather)
    arr = np.array(img).astype(np.float32)

    if mode == "transparent":
        rgba = np.dstack([arr, alpha * 255.0]).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    if mode == "image":
        if bg_image is None:
            raise ValueError("Chưa chọn ảnh nền")
        bg = np.array(ImageOps.fit(bg_image.convert("RGB"), img.size)).astype(np.float32)
    else:
        bg = np.empty_like(arr)
        bg[:] = np.array(parse_color(color), dtype=np.float32)

    a = alpha[..., None]
    return Image.fromarray((arr * a + bg * (1 - a)).clip(0, 255).astype(np.uint8))


def crop_to_mask(img: Image.Image, mask: np.ndarray, pad: int = 8) -> Image.Image:
    """Cắt sát vùng mask (mask có thể khác kích thước ảnh)."""
    m = mask.astype(np.float32)
    if m.shape != (img.height, img.width):
        m = cv2.resize(m, img.size, interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(m > 0.5)
    if len(xs) == 0:
        return img
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.width)
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.height)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))
