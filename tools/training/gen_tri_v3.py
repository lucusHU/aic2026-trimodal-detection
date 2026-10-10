#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 v3 实验的配置与初始化权重（由 Codex 接管执行）。

只用到了一个关键事实：
  D-FINE 的 -t 走 load_tuning_state() -> state['ema']['module']，
  fuse.*.gate.1.bias 是普通参数，会被 checkpoint 覆盖。
  => 想让门初始是关的（2*sigmoid(-4)=0.036），必须重建 init 权重，改 yml 没用。
"""
import os
import re
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
BISAI = "/home/adlab7/ssd/bigdatan/jingsai/bisai"
os.chdir(BISAI)
sys.path.insert(0, "/home/adlab7/ssd/bigdatan/jingsai/D-FINE")

# ---------------------------------------------------------------- 1. 生成配置
src = open("configs/tri_p3.yml").read()
s = src


def rep(a, b):
    global s
    assert a in s, "NOT FOUND: " + a[:70]
    s = s.replace(a, b)


rep("output_dir: ./output/tri_p3", "output_dir: ./output/tri_v3")
rep("gate_bias: 1.0", "gate_bias: -4.0")
rep("total_batch_size: 16", "total_batch_size: 4")
rep("      lr: 0.0000125", "      lr: 0.000001")
rep("  lr: 0.00006", "  lr: 0.000001")

# epochs 在服务器上被改过好几次，用正则兜底（只改顶层那一行）
s, n_sub = re.subn(r"(?m)^epochs:\s*\d+\s*$", "epochs: 12", s)
assert n_sub == 1, "epochs line not found exactly once (found %d)" % n_sub

OLD_OPT = (
    "    -\n"
    "      params: '^(?=.*(?:aux_stem|aux_s2|aux_s3|aux_s4|aux_s5|fuse)).*$'\n"
    "      lr: 0.0005\n"
    "      weight_decay: 0."
)
NEW_OPT = (
    "    -\n"
    "      params: '^(?=.*fuse\\.\\d+\\.gate).*$'\n"
    "      lr: 0.002\n"
    "      weight_decay: 0.\n"
    "\n"
    "    -\n"
    "      params: '^(?=.*(?:aux_stem|aux_s2|aux_s3|aux_s4|aux_s5|fuse\\.\\d+\\.proj)).*$'\n"
    "      lr: 0.0002\n"
    "      weight_decay: 0."
)
assert OLD_OPT in s, "optimizer block not found"
s = s.replace(OLD_OPT, NEW_OPT)

open("configs/tri_v3.yml", "w").write(s)
print("[1] wrote configs/tri_v3.yml  (%d bytes)" % len(s))

# ------------------------------------------------------- 2. 重建初始化权重
import torch  # noqa: E402

from src.core import YAMLConfig  # noqa: E402

cfg = YAMLConfig("configs/tri_v3.yml")
m = cfg.model

raw = torch.load("output/seed1_800/best_stg2.pth", map_location="cpu")
sd = raw["ema"]["module"]  # ★ 必须是 ema，不是 model
r = m.load_state_dict(sd, strict=False)
assert not r.unexpected_keys, r.unexpected_keys[:5]
bad = [k for k in r.missing_keys if not ("aux_" in k or k.startswith("fuse"))]
assert not bad, bad[:5]

gate = float(m.fuse[0].gate[1].bias.abs().max())
proj = float(m.fuse[0].proj.weight.abs().max())
print("[2] built model: gate.bias absmax=%.4f  proj.weight absmax=%.6f" % (gate, proj))
assert abs(gate - 4.0) < 1e-6, "gate bias is NOT -4.0 -- config not applied"
assert proj == 0.0, "proj is not exactly zero"

st = m.state_dict()
new = dict(raw)
new["model"] = st
new["ema"] = {"module": st, "updates": 0}
new["last_epoch"] = 0
torch.save(new, "output/tri_init_v3.pth")
print("[3] saved output/tri_init_v3.pth")

# ------------------------------------------------- 3. 复核：优化器分组互斥性
groups = [
    ("backbone", r"^(?=.*backbone)(?!.*norm|bn).*$"),
    ("encdec-norm", r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$"),
    ("gate", r"^(?=.*fuse\.\d+\.gate).*$"),
    ("aux+proj", r"^(?=.*(?:aux_stem|aux_s2|aux_s3|aux_s4|aux_s5|fuse\.\d+\.proj)).*$"),
]
names = [k for k, v in m.named_parameters() if v.requires_grad]
seen = {}
for gname, pat in groups:
    hit = [k for k in names if len(re.findall(pat, k)) > 0]
    print("[4] group %-11s -> %3d params" % (gname, len(hit)))
    for k in hit:
        seen.setdefault(k, []).append(gname)

dup = {k: v for k, v in seen.items() if len(v) > 1}
print("[4] overlap count =", len(dup))
assert not dup, ("PARAMS IN MULTIPLE GROUPS: %s" % list(dup.items())[:3])
print("[OK] config + init generated, groups are mutually exclusive")
