import sys, torch
sys.path.insert(0, '../D-FINE')
from src.core import YAMLConfig
cfg = YAMLConfig('configs/tri_p3.yml')
m = cfg.model
raw = torch.load('output/seed1_800/best_stg2.pth', map_location='cpu')
sd = raw['ema']['module']          # <-- 用 EMA 权重热启动 (val 0.5641)
r = m.load_state_dict(sd, strict=False)
assert not r.unexpected_keys, r.unexpected_keys[:5]
bad = [k for k in r.missing_keys if not ('aux_' in k or k.startswith('fuse'))]
assert not bad, bad[:5]
print('missing (new modules only):', len(r.missing_keys))
new = dict(raw)
new['model'] = m.state_dict()
new['ema'] = {'module': m.state_dict(), 'updates': 0}
new['last_epoch'] = 0
torch.save(new, 'output/tri_init.pth')
print('saved ema-based output/tri_init.pth')
