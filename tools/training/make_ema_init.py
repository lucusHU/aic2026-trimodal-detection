"""从 seed1 的 EMA 权重构建供 -t 使用的初始化权重（RGB 模型）。
   注意 1：D-FINE 的 -t 只读 raw['model']，而真正的 0.5660 分数在 raw['ema']['module'] 里。
   注意 2：换分辨率会改变 decoder.anchors / decoder.valid_mask 的形状（按分辨率生成的缓冲量，
          不是学习权重），必须允许它们缺失。"""
import sys

sys.path.insert(0, '../D-FINE')
import torch

from src.core import YAMLConfig

CFG = sys.argv[1] if len(sys.argv) > 1 else 'configs/seed1_1024.yml'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'output/seed1_ema_init.pth'

cfg = YAMLConfig(CFG)
m = cfg.model

raw = torch.load('output/seed1_800/best_stg2.pth', map_location='cpu')
ema = {k.replace('module.', ''): v for k, v in raw['ema']['module'].items()}

# 过滤掉形状不匹配的缓冲量（分辨率相关）
tgt = m.state_dict()
drop = [k for k, v in ema.items() if k in tgt and tgt[k].shape != v.shape]
print('dropped (shape mismatch, resolution buffers):', drop)
ema = {k: v for k, v in ema.items() if k not in drop}

r = m.load_state_dict(ema, strict=False)
print('missing=%d unexpected=%d' % (len(r.missing_keys), len(r.unexpected_keys)))
print('missing keys:', list(r.missing_keys)[:10])
assert not r.unexpected_keys, r.unexpected_keys[:5]
allowed = {'decoder.anchors', 'decoder.valid_mask'}
bad = [k for k in r.missing_keys if k not in allowed]
assert not bad, ('unexpected missing', bad[:5])

new = dict(raw)
new['model'] = m.state_dict()
new['ema'] = {'module': m.state_dict(), 'updates': 0}
new['last_epoch'] = 0
torch.save(new, OUT)
print('saved', OUT)
