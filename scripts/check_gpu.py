"""Kiểm tra torch có thấy GPU không và đo tốc độ thử."""
import time

import torch

print("torch        :", torch.__version__)
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if (major, minor) < (2, 6):
    print("CẢNH BÁO     : cần torch >= 2.6 (transformers mới yêu cầu). "
          "Cài bản cu124: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
print("CUDA build   :", torch.version.cuda)
print("CUDA sẵn sàng:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        x @ x
    torch.cuda.synchronize()
    print(f"matmul 4096² x20: {time.perf_counter() - t:.2f}s  -> GPU OK")
else:
    print("\nChưa dùng được GPU. Nếu máy có NVIDIA, bạn đang cài torch bản CPU.")
    print("Xem hướng dẫn: https://pytorch.org/get-started/locally/")
