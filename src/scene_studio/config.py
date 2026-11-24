from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

@dataclass
class ModelConfig:
    detector: str = "IDEA-Research/grounding-dino-tiny"
    segmenter: str = "sam2.1_b.pt"
    depth: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"


@dataclass
class RuntimeConfig:
    max_side: int = 1024
    box_threshold: float = 0.3
    default_prompt: str = "person, dog, car"
    default_effect: str = "cutout"
    blur_sigma: float = 25.0


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        cfg = cls()
        path = path or os.environ.get("SCENE_STUDIO_CONFIG") or DEFAULT_CONFIG
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            for section, obj in (("models", cfg.models), ("runtime", cfg.runtime)):
                for k, v in (data.get(section) or {}).items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
        return cfg
