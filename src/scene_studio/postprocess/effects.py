from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _cutout(img, union, depth, sigma):
    arr = np.array(img)
    return Image.fromarray(np.dstack([arr, (union * 255).astype(np.uint8)]), "RGBA")


def _depth_blur(img, union, depth, sigma):
    arr = np.array(img)
    blur = cv2.GaussianBlur(arr, (0, 0), sigma)
    w = (1.0 - depth)[..., None]  # càng xa càng mờ
    bg = (arr * (1 - w) + blur * w).astype(np.uint8)
    return Image.fromarray(np.where(union[..., None], arr, bg))


def _white_bg(img, union, depth, sigma):
    arr = np.array(img)
    return Image.fromarray(np.where(union[..., None], arr, 255).astype(np.uint8))


def _gray_bg(img, union, depth, sigma):
    arr = np.array(img)
    gray = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return Image.fromarray(np.where(union[..., None], arr, gray))


# key -> (nhãn hiển thị, hàm)
EFFECTS = {
    "cutout": ("Xóa nền (PNG trong suốt)", _cutout),
    "depth_blur": ("Làm mờ nền theo độ sâu", _depth_blur),
    "white_bg": ("Nền trắng", _white_bg),
    "gray_bg": ("Tô xám nền", _gray_bg),
}


def apply_effect(name, img, union, depth, sigma=25.0) -> Image.Image:
    if name not in EFFECTS:
        raise ValueError(f"Effect không tồn tại: {name}. Có: {list(EFFECTS)}")
    return EFFECTS[name][1](img, union, depth, sigma)
