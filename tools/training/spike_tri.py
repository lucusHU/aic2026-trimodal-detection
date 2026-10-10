# -*- coding: utf-8 -*-
"""
5 通道（RGB + Infrared + Depth）早融合技术验证。
验证三件事：
  1. 数据集能否产出 5 通道 tensor 并走通 transforms（去掉 photometric）
  2. HGNetv2 stem 改 5 通道后能否前向
  3. 从已训练的 RGB 权重热启动时，新增通道如何初始化才能保持初始等价
"""
import os
import sys

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

REPO = "/home/adlab7/ssd/bigdatan/jingsai/D-FINE"
sys.path.insert(0, REPO)

print("=" * 70)
print("[1] 检查 5 通道能否走通 torchvision v2 transforms")
print("=" * 70)

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torchvision import tv_tensors

# 构造 5 通道 image
img5 = torch.rand(5, 360, 640).float()
boxes = tv_tensors.BoundingBoxes(
    torch.tensor([[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 260.0]]),
    format="XYXY", canvas_size=(360, 640),
)
labels = torch.tensor([1, 2])

for name, op in [
    ("RandomHorizontalFlip", T.RandomHorizontalFlip(p=1.0)),
    ("Resize", T.Resize((640, 640))),
    ("RandomZoomOut", T.RandomZoomOut(fill=0)),
    ("RandomIoUCrop", T.RandomIoUCrop()),
    ("SanitizeBoundingBoxes", T.SanitizeBoundingBoxes(min_size=1)),
]:
    try:
        if name == "RandomZoomOut":
            out_i, out_b, out_l = op(img5, boxes, labels)
        elif name == "RandomIoUCrop":
            out_i, out_b, out_l = op(img5, boxes, labels)
        elif name == "SanitizeBoundingBoxes":
            out_i, out_b, out_l = op(img5, boxes, labels)
        else:
            out_i, out_b, out_l = op(img5, boxes, labels)
        shape = out_i.shape if hasattr(out_i, "shape") else type(out_i)
        print(f"  [OK]   {name:22s} -> image {tuple(shape)}")
    except Exception as e:
        print(f"  [FAIL] {name:22s} -> {type(e).__name__}: {str(e)[:120]}")

# photometric（预期失败，我们会在配置里去掉）
try:
    T.RandomPhotometricDistort(p=1.0)(img5, boxes, labels)
    print("  [OK]   RandomPhotometricDistort -> 5ch 可用")
except Exception as e:
    print(f"  [SKIP] RandomPhotometricDistort -> {type(e).__name__}: {str(e)[:100]}")

print()
print("=" * 70)
print("[2] HGNetv2 stem 改 5 通道 + 权重热启动等价性")
print("=" * 70)

from src.nn.backbone.hgnetv2 import HGNetv2

# 原 3 通道 stem
bb3 = HGNetv2(name="B4", pretrained=False, use_lab=True)
stem_conv = None
# 找到 stem 里第一个卷积
for m in bb3.stem.modules():
    if isinstance(m, nn.Conv2d):
        stem_conv = m
        break
print(f"  原 stem 第一层卷积: in={stem_conv.in_channels} out={stem_conv.out_channels} "
      f"k={stem_conv.kernel_size} weight={tuple(stem_conv.weight.shape)}")

# 方案：在 stem 前插 1x1 适配层 5->3，identity 初始化
adapt = nn.Conv2d(5, 3, kernel_size=1, bias=False)
with torch.no_grad():
    adapt.weight.zero_()
    # 输出通道 i 只连输入通道 i（identity for RGB）
    for i in range(3):
        adapt.weight[i, i, 0, 0] = 1.0
    # 红外/深度通道初始化为小随机，避免一开始完全无梯度
    nn.init.normal_(adapt.weight[:, 3:, :, :], std=0.01)

x5 = torch.rand(2, 5, 320, 320)
rgb = x5[:, :3]
with torch.no_grad():
    out_adapt = adapt(x5)
    diff = (out_adapt - rgb).abs().max().item()
print(f"  适配层输出 vs 原RGB 最大差: {diff:.6f}  (应≈0，仅红外/深度通道的小扰动)")

# 完整前向：stem(adapt(x))
class StemWrapper(nn.Module):
    def __init__(self, stem, adapt):
        super().__init__()
        self.adapt = adapt
        self.stem = stem

    def forward(self, x):
        return self.stem(self.adapt(x))

bb3.eval()
wrapped = StemWrapper(bb3.stem, adapt).eval()
with torch.inference_mode():
    try:
        out = wrapped(x5)
        print(f"  [OK] stem(adapt(5ch)) 前向成功 -> {tuple(out.shape)}")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")

print()
print("=" * 70)
print("[3] 检查 dataset 与 collate 对 5 通道的假设")
print("=" * 70)
from src.data._misc import convert_to_tv_tensor
try:
    t = convert_to_tv_tensor(torch.rand(5, 100, 100), key="image")
    print(f"  convert_to_tv_tensor(5ch) -> {type(t).__name__} {tuple(t.shape)}")
except Exception as e:
    print(f"  convert_to_tv_tensor(5ch) FAIL: {type(e).__name__}: {str(e)[:150]}")

# collate
from src.data.collate_fn import BatchImageCollateFunction
cf = BatchImageCollateFunction(base_size=640, base_size_repeat=4)
print(f"  collate_fn base_size={cf.base_size} (生成多尺度，按 batch 内最大尺寸 pad)")

print()
print("=" * 70)
print("[4] 数据集三模态对齐抽查")
print("=" * 70)
base = "/home/adlab7/ssd/bigdatan/jingsai/bisai/dataset"
import glob
for split in ["train", "val"]:
    vis_dir = f"{base}/images/{split}"
    ir_dir = f"{base}/tri/{split}/infrared"
    dp_dir = f"{base}/tri/{split}/depth"
    vis = {os.path.splitext(f)[0] for f in os.listdir(vis_dir)}
    ir = {os.path.splitext(f)[0] for f in os.listdir(ir_dir)} if os.path.isdir(ir_dir) else set()
    dp = {os.path.splitext(f)[0] for f in os.listdir(dp_dir)} if os.path.isdir(dp_dir) else set()
    print(f"  {split}: visible={len(vis)} infrared={len(ir)} depth={len(dp)} "
          f"| 缺IR={len(vis - ir)} 缺DP={len(vis - dp)}")

    # 抽查尺寸
    s = sorted(vis)[:3]
    for stem in s:
        v = Image.open(f"{vis_dir}/{stem}.jpg" if os.path.exists(f"{vis_dir}/{stem}.jpg")
                       else f"{vis_dir}/{stem}.png")
        i = Image.open(f"{ir_dir}/{stem}.png")
        d = Image.open(f"{dp_dir}/{stem}.png")
        print(f"    {stem}: vis{v.size}{v.mode} ir{i.size}{i.mode} dp{d.size}{d.mode}")
        break

print()
print("DONE")
