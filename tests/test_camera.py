import pytest

from scene_studio.utils.camera import grab_frame


def test_missing_camera_gives_helpful_error():
    with pytest.raises(RuntimeError, match="camera"):
        grab_frame(index=99, warmup=0)   # index không tồn tại
