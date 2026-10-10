"""决定性诊断：把未训练的 tri 模型（= seed1 权重 + 零初始化辅助支路）在 tri 管线里评估。
   若结果 != 0.5660，说明 tri 管线本身让 RGB 输入发生了变化，与训练无关。"""
import sys

sys.path.insert(0, '../D-FINE')
import torch

from src.core import YAMLConfig
from src.solver.det_engine import evaluate

CFG = sys.argv[1] if len(sys.argv) > 1 else 'configs/tri_p3f.yml'
CKPT = sys.argv[2] if len(sys.argv) > 2 else 'output/tri_init.pth'

print('=' * 70)
print('config =', CFG)
print('ckpt   =', CKPT)
print('=' * 70)

cfg = YAMLConfig(CFG)
model = cfg.model
device = torch.device('cuda:0')

raw = torch.load(CKPT, map_location='cpu')
sd = raw.get('model', raw)
sd = {k.replace('module.', ''): v for k, v in sd.items()}
res = model.load_state_dict(sd, strict=False)
print('missing  =', len(res.missing_keys), res.missing_keys[:4])
print('unexpected =', len(res.unexpected_keys), list(res.unexpected_keys)[:4])

model.to(device)

stats = evaluate(
    model,
    cfg.criterion,
    cfg.postprocessor,
    cfg.val_dataloader,
    cfg.evaluator,
    device,
    0,
    False,
    output_dir=None,
)
print('=' * 70)
print('结果 stats =', stats)
print('=' * 70)
