import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scene_studio.apps.gradio_app import build_demo

demo = build_demo()
if __name__ == "__main__":
    demo.launch()
