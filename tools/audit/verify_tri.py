"""
验证三模态模型在加载 RGB 权重后与原模型**完全等价**。
这是"不可能降分"的数学依据：若最大输出差为 0，则第 0 步的 mAP 与
原 RGB 基线完全相同，训练只能在此基础上改善。
"""
import sys

import torch

sys.path.insert(0, "../D-FINE")
from src.core import YAMLConfig  # noqa: E402

CKPT = "output/seed1_800/best_stg2.pth"

print("=" * 60)
print("[1] 加载权重")
raw = torch.load(CKPT, map_location="cpu")
if "model" in raw and isinstance(raw["model"], dict) and len(raw["model"]) > 10:
    sd = raw["model"]
elif "ema" in raw and isinstance(raw["ema"], dict) and "module" in raw["ema"]:
    sd = raw["ema"]["module"]
else:
    sd = raw
sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
print(f"    checkpoint keys: {len(sd)}")

print("=" * 60)
print("[2] 原版 DFINE")
cfg0 = YAMLConfig("configs/seed1_800.yml")
m0 = cfg0.model
r0 = m0.load_state_dict(sd, strict=False)
print(f"    missing={len(r0.missing_keys)} unexpected={len(r0.unexpected_keys)}")
m0.eval()

print("=" * 60)
print("[3] TriModalDFINE")
cfg1 = YAMLConfig("configs/tri_p3.yml")
m1 = cfg1.model
r1 = m1.load_state_dict(sd, strict=False)
print(f"    missing={len(r1.missing_keys)} unexpected={len(r1.unexpected_keys)}")
print(f"    missing keys: {r1.missing_keys[:6]}")
assert all(
    ("aux_" in k or k.startswith("fuse")) for k in r1.missing_keys
), "有非新增模块的权重没加载上！"
n_new = sum(p.numel() for n, p in m1.named_parameters() if "aux_" in n or n.startswith("fuse"))
n_all = sum(p.numel() for p in m1.parameters())
print(f"    新增参数 {n_new/1e6:.2f}M / 总参数 {n_all/1e6:.2f}M ({n_new/n_all*100:.2f}%)")
m1.eval()

print("=" * 60)
print("[4] 等价性检验（三模态输入 5 通道 vs 原模型 3 通道）")
torch.manual_seed(1234)
x = torch.rand(2, 5, 800, 800)
with torch.no_grad():
    o0 = m0(x[:, :3])
    o1 = m1(x)


def flat(o):
    if isinstance(o, dict):
        return torch.cat([v.reshape(-1) for v in o.values() if torch.is_tensor(v)])
    if isinstance(o, (list, tuple)):
        return torch.cat([v.reshape(-1) for v in o if torch.is_tensor(v)])
    return o.reshape(-1)


a, b = flat(o0), flat(o1)
print(f"    元素数 {a.numel()} vs {b.numel()}")
diff = (a - b).abs().max().item()
print(f"    最大绝对差 = {diff:.3e}")
print(f"    原模型输出范数 = {a.abs().mean().item():.6f}")

print("=" * 60)
print("[5] 纯 RGB 回退（3 通道输入走原路径）")
with torch.no_grad():
    o2 = m1(x[:, :3])
d2 = (flat(o0) - flat(o2)).abs().max().item()
print(f"    3ch 输入最大绝对差 = {d2:.3e}")

print("=" * 60)
print("[6] 数据集 5 通道")
ds = cfg1.train_dataloader.dataset
img, tgt = ds[0]
print(f"    image type={type(img).__name__} shape={tuple(img.shape)} dtype={img.dtype}")
print(f"    target keys={sorted(tgt.keys())}")

print("=" * 60)
ok = (diff < 1e-6) and (d2 < 1e-6) and (tuple(img.shape)[0] == 5)
print("结论:", "✅ 等价性成立 —— 三模态起点 == RGB 基线，不会降分" if ok else "❌ 不等价，需要排查")
