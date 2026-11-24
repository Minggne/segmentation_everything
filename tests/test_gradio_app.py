import pytest

gr = pytest.importorskip("gradio")

from scene_studio.apps import gradio_app as app


@pytest.fixture(autouse=True)
def _inject(pipe, monkeypatch):
    monkeypatch.setattr(app, "_pipe", pipe)


def test_build_demo():
    assert app.build_demo() is not None


def test_run_segment(image):
    _ov, _fx, _depth, rows, cuts, coco, msg = app.run_segment(image, "cat, dog", 0.3, "white_bg")
    assert len(rows) == 2 and len(cuts) == 2 and coco.endswith(".json") and "2 đối tượng" in msg


def test_run_matte_modes(image):
    out, _ = app.run_matte(image, "cat", 0.3, "transparent", "#fff", None, 1.0, False)
    assert out.mode == "RGBA"
    out, _ = app.run_matte(image, "cat", 0.3, "color", "#00ff00", None, 1.0, True)
    assert out.size[0] < image.width      # đã cắt sát


def test_run_matte_errors(image, empty_pipe, monkeypatch):
    with pytest.raises(gr.Error):
        app.run_matte(None, "cat", 0.3, "transparent", "#fff", None, 1, False)
    with pytest.raises(gr.Error):
        app.run_matte(image, "cat", 0.3, "image", "#fff", None, 1, False)    # thiếu ảnh nền
    monkeypatch.setattr(app, "_pipe", empty_pipe)
    with pytest.raises(gr.Error, match="Không tìm thấy"):
        app.run_matte(image, "unicorn", 0.3, "transparent", "#fff", None, 1, False)


def test_run_camera_and_live(image):
    out, _msg = app.run_camera(image, "cat", 0.3, "overlay")
    assert out.size == image.size
    assert app.run_camera(None, "cat", 0.3, "overlay")[0] is None
    assert app.run_live(image, "cat", 0.3) is not None
    assert app.run_live(None, "cat", 0.3) is None


def test_grab_server_camera_error_is_gradio_error():
    with pytest.raises(gr.Error):
        app.grab_server_camera(99)
