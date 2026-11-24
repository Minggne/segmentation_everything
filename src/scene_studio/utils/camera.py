"""Chụp 1 khung hình từ webcam của máy đang chạy app (không phụ thuộc quyền camera của trình duyệt)."""
from __future__ import annotations

import sys

from PIL import Image


def grab_frame(index: int = 0, warmup: int = 8) -> Image.Image:
    import cv2

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(index), backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Không mở được camera index {index}. Thử index 1 hoặc 2, đóng các app đang dùng "
            "camera (Zoom, Teams, Camera...), và bật quyền camera cho ứng dụng desktop trong "
            "Windows: Settings > Privacy & security > Camera.")
    try:
        for _ in range(warmup):   # bỏ vài khung đầu cho camera kịp lấy sáng
            cap.read()
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError("Mở được camera nhưng không đọc được khung hình nào.")
    return Image.fromarray(frame[..., ::-1])
