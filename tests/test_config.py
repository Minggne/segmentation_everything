from scene_studio import Config


def test_defaults():
    assert Config().runtime.max_side == 1024


def test_load_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("runtime:\n  max_side: 320\nmodels:\n  device: cpu\n")
    cfg = Config.load(p)
    assert cfg.runtime.max_side == 320 and cfg.models.device == "cpu"


def test_load_yaml_utf8_vietnamese(tmp_path):
    p = tmp_path / "vi.yaml"
    p.write_text("# chú thích tiếng Việt: ép dùng GPU\nmodels:\n  device: cuda\n", encoding="utf-8")
    assert Config.load(p).models.device == "cuda"
