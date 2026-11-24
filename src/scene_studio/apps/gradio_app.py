from __future__ import annotations

import inspect
import os
import tempfile

from PIL import Image

from ..config import Config
from ..export import write_coco
from ..pipeline import ScenePipeline
from ..postprocess import (
    EFFECTS,
    TABLE_HEADERS,
    compose,
    crop_cutouts,
    crop_to_mask,
    depth_colormap,
    overlay,
    parse_color,
    stats_rows,
)
from ..utils.camera import grab_frame

# ---- kích thước: mọi thứ tính theo chiều cao khung nhìn (vh) nên không bao giờ scroll trang ----
H_IN = "max(200px, calc(100vh - 281px))"      # ảnh vào (có khối điều khiển bên dưới)
H_OUT = "max(200px, calc(100vh - 142px))"     # kết quả trong tab con
H_WIDE = "max(200px, calc(100vh - 228px))"   # ảnh trái/phải, điều khiển nằm hàng dưới
H_CAM = "max(200px, calc(100vh - 332px))"     # camera + 3 hàng điều khiển
H_LIVE = "max(200px, calc(100vh - 212px))"        # live: 2 ảnh + 1 hàng điều khiển + 1 dòng chú thích
H_CAM_OUT = "max(200px, calc(100vh - 136px))"  # kết quả camera + dòng thông báo

CSS = """
html, body, gradio-app { height: 100vh !important; overflow: hidden !important; }
.gradio-container { max-width: 100% !important; height: 100vh !important;
  overflow: hidden !important; padding: 4px 10px !important;
  --layout-gap: 6px; --block-padding: 4px; --form-gap-width: 6px; }
.gradio-container .column, .gradio-container .row { gap: 6px !important; }
.html-container { padding: 0 !important; }
#hdr_w { padding: 0 !important; min-height: 0 !important; }
.gradio-container .main, .gradio-container .wrap, .gradio-container .contain {
  padding-top: 0 !important; padding-bottom: 0 !important; }
footer, .built-with { display: none !important; }
#hdr { padding: 2px 4px !important; min-height: 0 !important; font-size: 15px; }
#hdr small { opacity: .65; margin-left: 8px; }
button[role="tab"] { padding: 4px 14px !important; font-size: 14px !important; }
.tabitem, [role="tabpanel"] { padding: 4px 0 0 0 !important; }
.gradio-container .gap, .gradio-container .form { gap: 6px !important; }
.gradio-container textarea, .gradio-container input[type="text"],
.gradio-container input[type="number"] { padding: 6px 8px !important; min-height: 0 !important; }
.gradio-container .block { padding: 4px !important; }
.gradio-container .table-wrap { max-height: calc(100vh - 230px) !important; overflow: auto; }
.nowrap { flex-wrap: nowrap !important; }
.nowrap > *, .nowrap .form > * { min-width: 0 !important; }
.nowrap .form { flex-wrap: nowrap !important; }
.tight-md { min-height: 22px !important; }
.tight-md p { margin: 0 !important; font-size: 13px; }
"""

EFFECT_CHOICES = [(label, key) for key, (label, _) in EFFECTS.items()]
BG_CHOICES = [("Trong suốt", "transparent"), ("Màu đơn", "color"), ("Ảnh nền", "image")]
CAM_MODES = [("Mask + nhãn", "overlay"), ("Tách nền (trong suốt)", "cutout"),
             ("Nền trắng", "white_bg"), ("Mờ nền theo độ sâu", "depth_blur")]

_pipe: ScenePipeline | None = None


def get_pipeline() -> ScenePipeline:
    global _pipe
    if _pipe is None:
        _pipe = ScenePipeline.from_config(Config.load())
    return _pipe


def _mk(cls, **kw):
    """Tạo component Gradio, bỏ qua tham số mà phiên bản đang cài không hỗ trợ (5.x vs 6.x)."""
    ok = set(inspect.signature(cls.__init__).parameters)
    return cls(**{k: v for k, v in kw.items() if k in ok})


def _need(img, prompt):
    import gradio as gr

    if img is None:
        raise gr.Error("Chưa có ảnh. Hãy tải ảnh lên hoặc chụp từ camera.")
    if not (prompt or "").strip():
        raise gr.Error("Hãy nhập đối tượng cần tìm, ví dụ: person, dog, car.")


def _path(f) -> str:
    """UploadButton trả về str / list / object có .name tùy phiên bản Gradio."""
    if isinstance(f, (list, tuple)):
        f = f[0]
    return str(getattr(f, "name", f))


def _timings(res) -> str:
    return " · ".join(f"**{k}** {v:.2f}s" for k, v in res.timings.items())


# ------------------------------------------------------------------ handlers
def run_segment(img, prompt, thr, effect):
    _need(img, prompt)
    pipe = get_pipeline()
    res = pipe.run(img, prompt, thr)
    fd, coco = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    write_coco(res, coco)
    msg = f"{len(res.instances)} đối tượng · {_timings(res)}"
    return (overlay(res), pipe.render(res, effect), depth_colormap(res.depth),
            stats_rows(res.instances), crop_cutouts(res), coco, msg)


def run_matte(img, prompt, thr, mode, color, bg_img, feather, crop):
    import gradio as gr

    _need(img, prompt)
    if mode == "image" and bg_img is None:
        raise gr.Error("Bạn chọn 'Ảnh nền' nhưng chưa tải ảnh nền lên.")
    if mode == "image":
        bg_img = Image.open(_path(bg_img)).convert("RGB")
    res = get_pipeline().run(img, prompt, thr, use_depth=False)
    if not res.instances:
        raise gr.Error(f"Không tìm thấy '{prompt}'. Thử hạ ngưỡng tin cậy hoặc đổi từ khóa.")
    full = img.convert("RGB")        # ghép trên ảnh gốc để giữ độ phân giải
    mask = res.union_mask()
    out = compose(full, mask, mode, parse_color(color), bg_img, feather)
    if crop:
        out = crop_to_mask(out, mask)
    return out, f"{len(res.instances)} đối tượng · {_timings(res)}"


def run_camera(img, prompt, thr, mode):
    import gradio as gr

    if img is None:
        return None, "Chưa có ảnh từ camera."
    _need(img, prompt)
    pipe = get_pipeline()
    res = pipe.run(img, prompt, thr, use_depth=(mode == "depth_blur"))
    if not res.instances:
        raise gr.Error(f"Không tìm thấy '{prompt}' trong ảnh.")
    out = overlay(res) if mode == "overlay" else pipe.render(res, mode)
    return out, f"{len(res.instances)} đối tượng · {_timings(res)}"


def run_live(frame, prompt, thr):
    if frame is None or not (prompt or "").strip():
        return None
    res = get_pipeline().run(frame, prompt, thr, use_depth=False, max_side=480)
    return overlay(res)


def grab_server_camera(index):
    import gradio as gr

    try:
        return grab_frame(int(index or 0))
    except RuntimeError as e:
        raise gr.Error(str(e)) from e


# ------------------------------------------------------------------ UI
def build_demo(preload: bool = True):
    import gradio as gr
    if preload:
        get_pipeline()
    cfg = Config.load()
    rt = cfg.runtime
    img_kw = {"type": "pil", "show_label": False, "format": "png"}

    with gr.Blocks(title="Scene Studio") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML("<div id='hdr'><b>Scene Studio</b></div>", elem_id="hdr_w")

        with gr.Tabs():
            # ============ 1. Phân đoạn ============
            with gr.Tab("Phân đoạn"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=280):
                        s_in = _mk(gr.Image, sources=["upload", "clipboard"], height=H_IN, **img_kw)
                        s_prompt = gr.Textbox(rt.default_prompt, show_label=False, max_lines=1,
                                              placeholder="Đối tượng cần tìm (tiếng Anh, cách nhau dấu phẩy)")
                        with gr.Row(elem_classes="nowrap"):
                            s_thr = gr.Slider(0.1, 0.8, rt.box_threshold, step=0.05, label="Ngưỡng", scale=3)
                            s_eff = gr.Dropdown(EFFECT_CHOICES, value=rt.default_effect,
                                                label="Hiệu ứng", scale=3)
                        s_btn = gr.Button("Chạy phân đoạn", variant="primary")
                    with gr.Column(scale=6, min_width=320), gr.Tabs():
                        with gr.Tab("Mask"):
                            s_mask = _mk(gr.Image, height=H_OUT, interactive=False, **img_kw)
                        with gr.Tab("Hiệu ứng"):
                            s_fx = _mk(gr.Image, height=H_OUT, interactive=False, **img_kw)
                        with gr.Tab("Depth"):
                            s_depth = _mk(gr.Image, height=H_OUT, interactive=False, **img_kw)
                        with gr.Tab("Thống kê"):
                            s_table = _mk(gr.Dataframe, headers=TABLE_HEADERS, interactive=False,
                                          max_height=300, show_label=False)
                            s_msg = gr.Markdown(elem_classes="tight-md")
                        with gr.Tab("Từng vật thể"):
                            s_gal = _mk(gr.Gallery, height=H_OUT, show_label=False, columns=4)
                        with gr.Tab("COCO"):
                            s_coco = _mk(gr.File, label="COCO JSON (auto-label)")
                s_btn.click(run_segment, [s_in, s_prompt, s_thr, s_eff],
                            [s_mask, s_fx, s_depth, s_table, s_gal, s_coco, s_msg])

            # ============ 2. Tách nền ============
            with gr.Tab("Tách nền"):
                with gr.Row(equal_height=False):
                    m_in = _mk(gr.Image, sources=["upload", "clipboard"], height=H_WIDE, **img_kw)
                    m_out = _mk(gr.Image, height=H_WIDE, interactive=False, **img_kw)
                with gr.Row(elem_classes="nowrap"):
                    m_prompt = gr.Textbox("person", label="Đối tượng giữ lại", max_lines=1, scale=3)
                    m_thr = gr.Slider(0.1, 0.8, rt.box_threshold, step=0.05, label="Ngưỡng", scale=2)
                    m_feather = gr.Slider(0, 8, 2, step=0.5, label="Viền mềm", scale=2)
                    m_mode = gr.Dropdown(BG_CHOICES, value="transparent", label="Nền mới", scale=3)
                    m_color = gr.ColorPicker("#ffffff", label="Màu", visible=False, scale=1)
                    m_bg = _mk(gr.UploadButton, label="Chọn ảnh nền", file_types=["image"],
                               visible=False, scale=2)
                    m_crop = gr.Checkbox(False, label="Cắt sát", scale=1)
                    m_btn = gr.Button("Tách nền", variant="primary", scale=1)
                m_msg = gr.Markdown(elem_classes="tight-md")
                m_bg.upload(lambda f: gr.update(label="Ảnh: " + os.path.basename(_path(f))), m_bg, m_bg)
                m_mode.change(lambda m: (gr.update(visible=m == "color"),
                                         gr.update(visible=m == "image")),
                              m_mode, [m_color, m_bg])
                m_btn.click(run_matte, [m_in, m_prompt, m_thr, m_mode, m_color, m_bg, m_feather, m_crop],
                            [m_out, m_msg])

            # ============ 3. Camera (chụp 1 ảnh) ============
            with gr.Tab("Camera"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=280):
                        c_in = _mk(gr.Image, sources=["webcam"], height=H_CAM, mirror_webcam=False, **img_kw)
                        c_prompt = gr.Textbox("person", show_label=False, max_lines=1,
                                              placeholder="Đối tượng cần tìm")
                        with gr.Row(elem_classes="nowrap"):
                            c_thr = gr.Slider(0.1, 0.8, rt.box_threshold, step=0.05, label="Ngưỡng", scale=3)
                            c_mode = gr.Dropdown(CAM_MODES, value="overlay", label="Kết quả", scale=3)
                        with gr.Row(elem_classes="nowrap"):
                            c_idx = gr.Number(0, precision=0, label="Cam #", scale=1, minimum=0)
                            c_cv = gr.Button("Camera máy", scale=2)
                            c_btn = gr.Button("Xử lý", variant="primary", scale=2)
                    with gr.Column(scale=6, min_width=320):
                        c_out = _mk(gr.Image, height=H_CAM_OUT, interactive=False, **img_kw)
                        c_msg = gr.Markdown(elem_classes="tight-md")
                c_args = [c_in, c_prompt, c_thr, c_mode]
                c_btn.click(run_camera, c_args, [c_out, c_msg])
                c_in.change(run_camera, c_args, [c_out, c_msg])   # tự xử lý khi vừa chụp
                c_cv.click(grab_server_camera, c_idx, c_in)

            # ============ 4. Live ============
            with gr.Tab("Live"):
                with gr.Row(equal_height=False):
                    l_in = _mk(gr.Image, sources=["webcam"], streaming=True, height=H_LIVE,
                               mirror_webcam=False, **img_kw)
                    l_out = _mk(gr.Image, height=H_LIVE, interactive=False, **img_kw)
                with gr.Row(elem_classes="nowrap"):
                    l_prompt = gr.Textbox("person", label="Đối tượng", max_lines=1, scale=3)
                    l_thr = gr.Slider(0.1, 0.8, rt.box_threshold, step=0.05, label="Ngưỡng", scale=2)
                gr.Markdown("Live chỉ chạy Grounding DINO + SAM 2 ở độ phân giải thấp (bỏ depth) để mượt hơn.",
                            elem_classes="tight-md")
                stream_kw = {"stream_every": 0.5, "time_limit": 120, "concurrency_limit": 1}
                ok = set(inspect.signature(l_in.stream).parameters)
                l_in.stream(run_live, [l_in, l_prompt, l_thr], l_out,
                            **{k: v for k, v in stream_kw.items() if k in ok})
    return demo
