from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


def to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
