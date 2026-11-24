import numpy as np
import pytest
from PIL import Image

from scene_studio import Config, ScenePipeline

W, H = 100, 80


@pytest.fixture
def image():
    return Image.fromarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))


def fake_detector(img, prompt, thr):
    return np.array([[10, 10, 50, 50], [60, 20, 90, 70]]), ["cat", "dog"], np.array([0.9, 0.6])


def fake_segmenter(img, boxes):
    masks = np.zeros((len(boxes), img.height, img.width), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        masks[i, y1:y2, x1:x2] = True
    return masks


def fake_depth(img):
    d = np.zeros((img.height, img.width), dtype=np.float32)
    d[:, : img.width // 2] = 0.9   # nửa trái gần
    d[:, img.width // 2:] = 0.2    # nửa phải xa
    return d


@pytest.fixture
def pipe():
    return ScenePipeline(fake_detector, fake_segmenter, fake_depth, Config())


@pytest.fixture
def empty_pipe():
    return ScenePipeline(lambda *a: (np.zeros((0, 4)), [], np.zeros(0)),
                         fake_segmenter, fake_depth, Config())
