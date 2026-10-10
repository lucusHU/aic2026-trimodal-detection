#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spike: 验证 D-FINE 能否改成 5 通道输入（RGB3 + IR1 + Depth1）并跑通前向。"""
import sys, os, torch
sys.path.insert(0, '/home/adlab7/ssd/bigdatan/jingsai/D-FINE')
os.chdir('/home/adlab7/ssd/bigdatan/jingsai/bisai')

from src.core import YAMLConfig

cfg = YAMLConfig('configs/baseline_800.yml', resume='')
model = cfg.model
print('=== stem 结构 ===')
for n, m in model.backbone.stem.named_modules():
    if hasattr(m, 'weight') and m.weight is not None:
        print('  ', n, tuple(m.weight.shape))

w = model.backbone.stem.stem1.conv.weight
print()
print('stem1.conv.weight:', tuple(w.shape), 'in_channels =', w.shape[1])
print('stem1 requires_grad:', model.backbone.stem.stem1.conv.weight.requires_grad)

# 加载 seed1 训练好的权重，确认能加载
ck = torch.load('output/seed1_800/best_stg2.pth', map_location='cpu')
print()
print('checkpoint keys:', list(ck.keys()))
state = ck['ema']['module'] if 'ema' in ck else ck['model']
print('has backbone.stem.stem1.conv.weight:',
      'backbone.stem.stem1.conv.weight' in state)
print('  shape:', tuple(state['backbone.stem.stem1.conv.weight'].shape))

# === 改成 5 通道 ===
import torch.nn as nn
old_conv = model.backbone.stem.stem1.conv
new_conv = nn.Conv2d(5, old_conv.out_channels,
                     kernel_size=old_conv.kernel_size,
                     stride=old_conv.stride,
                     padding=old_conv.padding,
                     bias=(old_conv.bias is not None))
with torch.no_grad():
    ow = old_conv.weight  # [O, 3, K, K]
    # 幅值保持：新增两通道取 RGB 三通道均值，整体按 3/5 缩放
    mean_w = ow.mean(dim=1, keepdim=True)      # [O, 1, K, K]
    nw = torch.cat([ow, mean_w.expand(-1, 2, -1, -1)], dim=1)  # [O,5,K,K]
    new_conv.weight.copy_(nw)
    if old_conv.bias is not None:
        new_conv.bias.copy_(old_conv.bias)
model.backbone.stem.stem1.conv = new_conv
print()
print('新 stem1 in_channels =', model.backbone.stem.stem1.conv.weight.shape[1])

# === 前向测试 ===
model.eval()
x = torch.randn(2, 5, 640, 640)
orig = torch.tensor([[640, 360], [640, 360]], dtype=torch.float32)
with torch.no_grad():
    out = model(x, orig)
print()
print('=== 前向成功 ===')
if isinstance(out, (tuple, list)):
    for o in out:
        print('  ', [tuple(t.shape) for t in o] if isinstance(o, (tuple, list)) else tuple(o.shape))

# === 检查有没有其他依赖 3 通道的地方 ===
print()
print('=== 检查 mean/std 归一化 ===')
print('cfg.yaml_cfg HGNetv2:', cfg.yaml_cfg.get('HGNetv2'))
