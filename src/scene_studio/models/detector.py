from __future__ import annotations


class GroundingDINODetector:
    """Stage 1: text prompt -> boxes (open-vocabulary)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.to(device).eval()

    @staticmethod
    def build_query(prompt: str) -> str:
        items = [p.strip().lower() for p in prompt.split(",") if p.strip()]
        return ". ".join(items) + "."

    def __call__(self, img, prompt: str, threshold: float):
        inputs = self.processor(images=img, text=self.build_query(prompt),
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=threshold, text_threshold=threshold,
            target_sizes=[img.size[::-1]])[0]
        labels = res.get("text_labels", res["labels"])
        return res["boxes"].cpu().numpy(), list(labels), res["scores"].cpu().numpy()
