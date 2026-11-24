"""Tải trước toàn bộ weights để demo không bị chậm lần đầu."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from scene_studio.config import Config
from scene_studio.pipeline import ScenePipeline

if __name__ == "__main__":
    ScenePipeline.from_config(Config.load())
    print("Đã tải xong tất cả model.")
