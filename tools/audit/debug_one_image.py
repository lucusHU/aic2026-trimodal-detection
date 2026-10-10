import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

DFINE_ROOT = "/home/newbie/bigdatan/jingsai/D-FINE"
CONFIG = "/home/newbie/bigdatan/jingsai/bisai/configs/baseline_800.yml"
WEIGHTS = "/home/newbie/bigdatan/jingsai/bisai/output/baseline_800/best_stg1.pth"

# 改成你刚才 find 出来的那一张 val 图片
IMAGE = "/home/newbie/bigdatan/jingsai/bisai/dataset/images/val/000005_004_00000286.png"

DEVICE = "cuda:0"
IMGSZ = 800

if DFINE_ROOT not in sys.path:
    sys.path.insert(0, DFINE_ROOT)

from src.core import YAMLConfig

device = torch.device(DEVICE)

cfg = YAMLConfig(
    CONFIG,
    resume=WEIGHTS,
)

if "HGNetv2" in cfg.yaml_cfg:
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

checkpoint = torch.load(
    WEIGHTS,
    map_location="cpu",
    weights_only=False,
)

print("checkpoint keys:", checkpoint.keys())

if "ema" in checkpoint and checkpoint["ema"] is not None:
    state = checkpoint["ema"]["module"]
    print("Using EMA weights")
elif "model" in checkpoint:
    state = checkpoint["model"]
    print("Using model weights")
else:
    raise RuntimeError("checkpoint里没有ema/model")

cfg.model.load_state_dict(state)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        labels, boxes, scores = self.postprocessor(
            outputs,
            orig_target_sizes,
        )
        return labels, boxes, scores

model = Model().to(device)
model.eval()

image = Image.open(IMAGE).convert("RGB")
orig_w, orig_h = image.size

transform = T.Compose([
    T.Resize((IMGSZ, IMGSZ)),
    T.ToTensor(),
])

x = transform(image).unsqueeze(0).to(device)

orig_sizes = torch.tensor(
    [[orig_w, orig_h]],
    dtype=torch.float32,
    device=device,
)

with torch.no_grad():
    labels, boxes, scores = model(x, orig_sizes)

labels = labels[0].detach().cpu()
boxes = boxes[0].detach().cpu()
scores = scores[0].detach().cpu()

order = torch.argsort(scores, descending=True)

print()
print("image:", IMAGE)
print("original size:", orig_w, orig_h)
print("num predictions:", len(scores))
print()
print("Top 20:")
print("=" * 80)

for rank, idx in enumerate(order[:20], start=1):
    cls = int(labels[idx].item())
    score = float(scores[idx].item())
    x1, y1, x2, y2 = boxes[idx].tolist()

    print(
        f"{rank:02d} | "
        f"class={cls:2d} | "
        f"score={score:.6f} | "
        f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
    )
