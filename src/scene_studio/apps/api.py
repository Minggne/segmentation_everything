from __future__ import annotations

import io
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from ..config import Config
from ..export import to_coco
from ..pipeline import ScenePipeline
from ..postprocess import EFFECTS
from ..utils.image import to_base64_png

app = FastAPI(title="Scene Studio API", version="0.1.0")
_pipe: ScenePipeline | None = None


def _get_pipe() -> ScenePipeline:
    global _pipe
    if _pipe is None:
        _pipe = ScenePipeline.from_config(Config.load())
    return _pipe


@app.get("/health")
def health():
    return {"status": "ok", "effects": list(EFFECTS)}


@app.post("/segment")
async def segment(file: Annotated[UploadFile, File()], prompt: Annotated[str, Form()],
                  threshold: Annotated[float, Form()] = 0.3,
                  effect: Annotated[str, Form()] = "cutout"):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pipe = _get_pipe()
    res = pipe.run(img, prompt, threshold)
    return {
        "instances": [{"label": i.label, "score": i.score, "box": [float(v) for v in i.box],
                       "area_pct": i.area_pct, "closeness": i.closeness,
                       "depth_rank": i.depth_rank} for i in res.instances],
        "coco": to_coco(res, file.filename or "image.png"),
        "result_png_base64": to_base64_png(pipe.render(res, effect)),
    }
