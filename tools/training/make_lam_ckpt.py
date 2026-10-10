#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write a checkpoint whose tri-modal fusion residual is scaled by --lam.

Only fuse.*.proj.{weight,bias} are touched -> forward becomes
  p + lam * proj(a) * 2*sigmoid(gate(a))   (proj is linear, exact).
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--lam', type=float, required=True)
a = ap.parse_args()

ck = torch.load(a.src, map_location='cpu')
n = 0


def scale(sd):
    global n
    for k in list(sd.keys()):
        if k.startswith('fuse.') and '.proj.' in k:
            sd[k] = sd[k] * a.lam
            n += 1


if isinstance(ck.get('ema'), dict) and 'module' in ck['ema']:
    scale(ck['ema']['module'])
if 'model' in ck:
    scale(ck['model'])
torch.save(ck, a.dst)
print('scaled %d tensors by %g -> %s' % (n, a.lam, a.dst))
