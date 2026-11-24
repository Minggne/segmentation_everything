import numpy as np
import pytest
from PIL import Image

from scene_studio.postprocess import compose, crop_to_mask, parse_color, refine_alpha


@pytest.fixture
def square():
    img = Image.new("RGB", (100, 80), (10, 200, 30))
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 30:70] = True
    return img, mask


def test_transparent_alpha(square):
    img, mask = square
    out = compose(img, mask, "transparent", feather=0)
    a = np.array(out)[..., 3]
    assert out.mode == "RGBA" and a[40, 50] == 255 and a[5, 5] == 0


def test_color_background(square):
    img, mask = square
    out = np.array(compose(img, mask, "color", color="#ff0000", feather=0))
    assert tuple(out[5, 5]) == (255, 0, 0)       # nền đổi thành đỏ
    assert tuple(out[40, 50]) == (10, 200, 30)   # vật thể giữ nguyên


def test_image_background_and_missing(square):
    img, mask = square
    bg = Image.new("RGB", (300, 300), (0, 0, 255))
    out = np.array(compose(img, mask, "image", bg_image=bg, feather=0))
    assert tuple(out[5, 5]) == (0, 0, 255)
    with pytest.raises(ValueError):
        compose(img, mask, "image", bg_image=None)


def test_invalid_mode(square):
    with pytest.raises(ValueError):
        compose(*square, mode="nope")


def test_feather_softens_edge(square):
    _, mask = square
    hard = refine_alpha(mask, (100, 80), 0)
    soft = refine_alpha(mask, (100, 80), 3)
    assert set(np.unique(hard)) <= {0.0, 1.0}
    assert ((soft > 0) & (soft < 1)).any()


def test_mask_resized_to_image(square):
    img, mask = square
    big = img.resize((200, 160))
    assert compose(big, mask, "transparent").size == (200, 160)   # mask nhỏ hơn ảnh vẫn dùng được


def test_crop_to_mask(square):
    img, mask = square
    out = crop_to_mask(img, mask, pad=0)
    assert out.size == (40, 40)
    assert crop_to_mask(img, np.zeros_like(mask)).size == img.size   # mask rỗng: giữ nguyên ảnh


@pytest.mark.parametrize("value,expected", [
    ("#ff0000", (255, 0, 0)), ("#0f0", (0, 255, 0)),
    ("rgba(10, 20, 30, 1)", (10, 20, 30)), ((1, 2, 3), (1, 2, 3)),
    ("không-phải-màu", (255, 255, 255)), (None, (255, 255, 255)),
])
def test_parse_color(value, expected):
    assert parse_color(value) == expected
