from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _run(args):
    from ..config import Config
    from ..export import write_coco
    from ..pipeline import ScenePipeline
    from ..postprocess import overlay, stats_rows
    from ..utils.image import load_image

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = ScenePipeline.from_config(Config.load(args.config))

    for f in files:
        res = pipe.run(load_image(f), args.prompt, args.threshold)
        pipe.render(res, args.effect).save(out / f"{f.stem}_{args.effect}.png")
        overlay(res).save(out / f"{f.stem}_overlay.png")
        write_coco(res, out / f"{f.stem}_coco.json", f.name)
        print(f"{f.name}: {len(res.instances)} đối tượng | "
              + ", ".join(f"{k} {v:.2f}s" for k, v in res.timings.items()))
        for row in stats_rows(res.instances):
            print("   ", row)


def main(argv=None):
    p = argparse.ArgumentParser(prog="scene-studio")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Chạy trên 1 ảnh hoặc cả thư mục")
    r.add_argument("input")
    r.add_argument("--prompt", required=True)
    r.add_argument("--threshold", type=float, default=0.3)
    r.add_argument("--effect", default="cutout")
    r.add_argument("--out", default="outputs")
    r.add_argument("--config", default=None)
    r.set_defaults(fn=_run)

    d = sub.add_parser("demo", help="Mở Gradio demo")
    d.add_argument("--share", action="store_true")
    d.set_defaults(fn=lambda a: __import__("scene_studio.apps.gradio_app", fromlist=["x"])
                   .build_demo().launch(share=a.share))

    s = sub.add_parser("serve", help="Chạy FastAPI server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=lambda a: __import__("uvicorn").run(
        "scene_studio.apps.api:app", host=a.host, port=a.port))

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
