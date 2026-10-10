import sys
import torch
import torchvision.transforms as T
from PIL import Image

DFINE_ROOT = "/home/newbie/bigdatan/jingsai/D-FINE"
CONFIG = "/home/newbie/bigdatan/jingsai/bisai/configs/baseline_800.yml"
WEIGHTS = "/home/newbie/bigdatan/jingsai/bisai/output/baseline_800/best_stg1.pth"
IMAGE = "/home/newbie/bigdatan/jingsai/bisai/dataset/images/val/000005_004_00000286.png"

DEVICE = "cuda:0"
IMGSZ = 800

if DFINE_ROOT not in sys.path:
    sys.path.insert(0, DFINE_ROOT)

from src.core import YAMLConfig


def load_cfg():
    cfg = YAMLConfig(CONFIG, resume=WEIGHTS)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(
        WEIGHTS,
        map_location="cpu",
        weights_only=False,
    )

    if "ema" in checkpoint and checkpoint["ema"] is not None:
        state = checkpoint["ema"]["module"]
        print("Using EMA")
    elif "model" in checkpoint:
        state = checkpoint["model"]
        print("Using model")
    else:
        raise RuntimeError("checkpoint 里既没有 'ema' 也没有 'model'")

    cfg.model.load_state_dict(state)
    return cfg


device = torch.device(DEVICE)

image = Image.open(IMAGE).convert("RGB")
orig_w, orig_h = image.size

transform = T.Compose([
    T.Resize((IMGSZ, IMGSZ)),
    T.ToTensor(),
])

x = transform(image).unsqueeze(0).to(device)

orig_size = torch.tensor(
    [[orig_w, orig_h]],
    dtype=torch.float32,
    device=device,
)

print("=" * 90)
print("Image:", IMAGE)
print("Original size:", orig_w, orig_h)
print("=" * 90)

# A: deploy path used by infer_bisai.py
cfg1 = load_cfg()
model_deploy = cfg1.model.deploy().to(device)
model_deploy.eval()
post_deploy = cfg1.postprocessor.deploy()

with torch.no_grad():
    outputs1 = model_deploy(x)
    labels1, boxes1, scores1 = post_deploy(outputs1, orig_size)

labels1 = labels1[0].detach().cpu()
boxes1 = boxes1[0].detach().cpu()
scores1 = scores1[0].detach().cpu()

# B: non-deploy validation-style path
cfg2 = load_cfg()
model_normal = cfg2.model.to(device)
model_normal.eval()
post_normal = cfg2.postprocessor

with torch.no_grad():
    outputs2 = model_normal(x)
    results2 = post_normal(outputs2, orig_target_sizes=orig_size)

result2 = results2[0]
labels2 = result2["labels"].detach().cpu()
boxes2 = result2["boxes"].detach().cpu()
scores2 = result2["scores"].detach().cpu()


def show(name, labels, boxes, scores, topk=20):
    print()
    print("=" * 90)
    print(name)
    print("=" * 90)
    print("num predictions:", len(scores))

    order = torch.argsort(scores, descending=True)

    for rank, idx in enumerate(order[:topk], start=1):
        cls = int(labels[idx].item())
        score = float(scores[idx].item())
        x1, y1, x2, y2 = boxes[idx].tolist()
        print(
            f"{rank:02d} | class={cls:2d} | score={score:.6f} | "
            f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )


show("A = DEPLOY（当前 infer_bisai.py 路径）", labels1, boxes1, scores1)
show("B = NORMAL（官方 validation 风格路径）", labels2, boxes2, scores2)

print()
print("=" * 90)
print("直接比较")
print("=" * 90)
print("deploy 数量:", len(scores1))
print("normal 数量:", len(scores2))

n = min(len(scores1), len(scores2))
if n == 0:
    print("没有可比较的预测。")
else:
    print("score最大差值:", torch.max(torch.abs(scores1[:n] - scores2[:n])).item())
    print("box最大差值:", torch.max(torch.abs(boxes1[:n] - boxes2[:n])).item())
    print("label不同数量:", int(torch.sum(labels1[:n] != labels2[:n]).item()))
print("=" * 90)
