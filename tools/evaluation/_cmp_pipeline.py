import sys
sys.path.insert(0, '../D-FINE')
import torch
from src.core import YAMLConfig

print('=' * 70)
print('[1] 两条管线的 val transforms 对象')
print('=' * 70)
c1 = YAMLConfig('configs/seed1_800.yml')      # 基线（纯 RGB）
c2 = YAMLConfig('configs/tri_p3f.yml')        # 三模态
print('baseline:', repr(c1.val_dataloader.dataset._transforms))
print()
print('tri     :', repr(c2.val_dataloader.dataset._transforms))
print()

print('=' * 70)
print('[2] 同一张图的 RGB 张量是否一致（这是关键）')
print('=' * 70)
a, ta = c1.val_dataloader.dataset[0]
b, tb = c2.val_dataloader.dataset[0]
print('baseline image:', type(a).__name__, tuple(a.shape), a.dtype)
print('tri      image:', type(b).__name__, tuple(b.shape), b.dtype)
print('同一张图?', ta['image_path'].split('/')[-1], tb['image_path'].split('/')[-1])
if a.shape[0] == 3 and b.shape[0] == 5:
    d = (a.float() - b[:3].float()).abs()
    print('RGB 最大绝对差 = %.6e   平均绝对差 = %.6e' % (d.max().item(), d.mean().item()))
    print('baseline RGB 范围 [%.4f, %.4f]  mean=%.4f' % (a.min(), a.max(), a.mean()))
    print('tri      RGB 范围 [%.4f, %.4f]  mean=%.4f' % (b[:3].min(), b[:3].max(), b[:3].mean()))
else:
    print('形状不符，无法对比')
print()

print('=' * 70)
print('[3] 辅助通道统计（IR / Depth）')
print('=' * 70)
ir, dp = b[3], b[4]
print('infrared: shape=%s range[%.3f, %.3f] mean=%.4f std=%.4f zero%%=%.1f' %
      (tuple(ir.shape), ir.min(), ir.max(), ir.mean(), ir.std(), 100.0 * (ir == 0).float().mean()))
print('depth   : shape=%s range[%.3f, %.3f] mean=%.4f std=%.4f zero%%=%.1f' %
      (tuple(dp.shape), dp.min(), dp.max(), dp.mean(), dp.std(), 100.0 * (dp == 0).float().mean()))
print()

print('=' * 70)
print('[4] 基线 train transforms vs tri train transforms')
print('=' * 70)
print('baseline train:', repr(c1.train_dataloader.dataset._transforms))
print()
print('tri      train:', repr(c2.train_dataloader.dataset._transforms))
