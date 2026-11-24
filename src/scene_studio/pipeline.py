from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from PIL import Image

from .config import Config
from .postprocess.analytics import compute_stats
from .postprocess.effects import apply_effect
from .types import Instance, SceneResult
from .utils.image import resize_max_side


class ScenePipeline:
    """Detector -> Segmenter -> Depth -> Analytics. Các stage là callable nên dễ thay/mock."""

    def __init__(
        self,
        detector: Callable,      # (img, prompt, thr) -> boxes, labels, scores
        segmenter: Callable,     # (img, boxes) -> bool masks (N, H, W)
        depth_estimator: Callable,  # (img) -> float32 (H, W), 1 = gần
        config: Config | None = None,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.cfg = config or Config()

    @classmethod
    def from_config(cls, config: Config | None = None) -> ScenePipeline:
        from .models import DepthAnythingEstimator, GroundingDINODetector, SAM2Segmenter
        from .utils.device import resolve_device

        cfg = config or Config.load()
        dev = resolve_device(cfg.models.device)
        name = ""
        if dev == "cuda":
            import torch
            name = f" ({torch.cuda.get_device_name(0)})"
        print(f"[scene-studio] device = {dev}{name}")
        return cls(
            GroundingDINODetector(cfg.models.detector, dev),
            SAM2Segmenter(cfg.models.segmenter, dev),
            DepthAnythingEstimator(cfg.models.depth, dev),
            cfg,
        )

    def run(self, image: Image.Image, prompt: str | None = None,
            threshold: float | None = None, use_depth: bool = True,
            max_side: int | None = None) -> SceneResult:
        """use_depth=False bỏ qua Depth Anything (nhanh hơn: dùng cho tách nền và camera live)."""
        rt = self.cfg.runtime
        img = resize_max_side(image.convert("RGB"), max_side or rt.max_side)
        prompt = prompt or rt.default_prompt
        thr = rt.box_threshold if threshold is None else threshold

        t: dict[str, float] = {}
        if use_depth:
            t0 = time.perf_counter()
            depth = self.depth_estimator(img)
            t["depth"] = time.perf_counter() - t0
        else:
            depth = np.zeros((img.height, img.width), dtype=np.float32)

        t0 = time.perf_counter()
        boxes, labels, scores = self.detector(img, prompt, thr)
        t["detect"] = time.perf_counter() - t0

        instances: list[Instance] = []
        if len(boxes):
            t0 = time.perf_counter()
            masks = self.segmenter(img, boxes)
            t["segment"] = time.perf_counter() - t0
            instances = [
                Instance(label=str(l), score=float(s), box=np.asarray(b), mask=m)
                for b, l, s, m in zip(boxes, labels, scores, masks, strict=False)
            ]
            compute_stats(instances, depth)
        t["total"] = sum(t.values())
        return SceneResult(img, instances, depth, t)

    def render(self, result: SceneResult, effect: str | None = None) -> Image.Image:
        rt = self.cfg.runtime
        return apply_effect(effect or rt.default_effect, result.image,
                            result.union_mask(), result.depth, rt.blur_sigma)
