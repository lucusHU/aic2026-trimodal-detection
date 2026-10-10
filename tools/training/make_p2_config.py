#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成并校验 P2 检测头配置（豆包方案）。

改动只用配置完成（源码无需修改），依据：
  hgnetv2.py:577      if idx in self.return_idx            → 接受任意索引
  hybrid_encoder.py:343 out_channels = [hidden]*len(in_channels) → 层数由 in_channels 长度决定
  dfine_decoder.py:67 assert len(num_points) == num_levels

HGNetv2-B4 各 stage 输出（hgnetv2.py:402-410）：
  stage1 128 (stride4) / stage2 512 (8) / stage3 1024 (16) / stage4 2048 (32)
"""
import os
import sys

os.chdir('/home/adlab7/ssd/bigdatan/jingsai/bisai')
sys.path.insert(0, '/home/adlab7/ssd/bigdatan/jingsai/D-FINE')

SRC = 'configs/seed1_800.yml'
DST = 'configs/seed1_p2.yml'

s = open(SRC).read()
assert 'p2' not in s

# 去掉原来的 output_dir 行，换成新的
import re
s = re.sub(r'(?m)^output_dir:.*$', 'output_dir: ./output/seed1_p2', s)

# 追加覆盖块（YAML 顶层键会覆盖 __include__ 里的同名键）
s += '''

# ==========================================================
# P2 小目标检测头（豆包方案 / Codex 用配置实现，不改源码）
#   1) backbone 多返回一级 stage1（stride 4, 128ch）
#   2) encoder in_channels / feat_strides 加一项
#   3) decoder num_levels 3->4, num_points 补一项
# ==========================================================

HGNetv2:
  name: 'B4'
  return_idx: [0, 1, 2, 3]          # 原 [1,2,3]
  freeze_stem_only: True
  freeze_at: 0
  freeze_norm: True
  pretrained: False

HybridEncoder:
  in_channels: [128, 512, 1024, 2048]   # ★ 首项必须是 128，不是 256
  feat_strides: [4, 8, 16, 32]
  use_encoder_idx: [3]                  # 原 [2]

DFINETransformer:
  num_levels: 4                         # 原 3
  feat_channels: [256, 256, 256, 256]   # ★ 必须显式给 4 项
  feat_strides: [4, 8, 16, 32]          # 原 [8,16,32]
  num_points: [3, 6, 3, 3]              # 原 [3,6,3]
'''

open(DST, 'w').write(s)
print('[1] wrote', DST)

# ---------------------------------------------------------------- 构模型校验
import torch  # noqa: E402

from src.core import YAMLConfig  # noqa: E402

cfg = YAMLConfig(DST)
m = cfg.model
n_all = sum(p.numel() for p in m.parameters())
print('[2] model built OK, params = %.2fM' % (n_all / 1e6))

# 结构自检
bb = m.backbone
fb = torch.zeros(1, 3, 800, 800)      # 必须用 eval_spatial_size，否则 encoder 的 pos_embed 尺寸不匹配
with torch.no_grad():
    feats = bb(fb)
print('[3] backbone out channels =', [f.shape[1] for f in feats],
      ' spatial strides =', [800 // f.shape[-1] for f in feats])
assert [f.shape[1] for f in feats] == [128, 512, 1024, 2048], 'channel mismatch!'

m.eval()
with torch.no_grad():
    outs = m(fb)
print('[4] full forward OK; outputs =', type(outs).__name__)

# 载入 seed1 权重，看哪些新参数没被覆盖（预期只有 P2 相关的新参数）
raw = torch.load('output/seed1_800/best_stg2.pth', map_location='cpu')
sd = raw['ema']['module']
r = m.load_state_dict(sd, strict=False)
print('[5] missing = %d  unexpected = %d' % (len(r.missing_keys), len(r.unexpected_keys)))
print('    前 12 个 missing:', r.missing_keys[:12])
assert not r.unexpected_keys, r.unexpected_keys[:5]

print()
print('[OK] P2 配置可用；新增未覆盖的参数只有 P2 相关的那几个。')
