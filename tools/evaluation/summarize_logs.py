#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总各 output/*/log.txt 中每轮的 test_coco_eval_bbox，打印最优 mAP"""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
out = root / 'output'
rows = []
for d in sorted(out.iterdir()):
    if not d.is_dir():
        continue
    log = d / 'log.txt'
    if not log.exists():
        continue
    best = (-1, -1, None)
    last = None
    n = 0
    for line in log.open('r', encoding='utf-8', errors='ignore'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        v = j.get('test_coco_eval_bbox')
        if not v or len(v) < 12:
            continue
        ep = j.get('epoch')
        n += 1
        last = (ep, v[0])
        if v[0] > best[0]:
            best = (v[0], ep, v)
    if best[2] is None:
        continue
    b = best[2]
    rows.append((best[0], d.name, best[1], b, last, n))

rows.sort(reverse=True)
print('%-24s %8s %6s %8s %8s %8s %8s %8s %8s' %
      ('run', 'best_mAP', 'ep', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'last'))
for m, name, ep, b, last, n in rows:
    print('%-24s %8.4f %6s %8.4f %8.4f %8.4f %8.4f %8.4f  ep%s=%.4f (%d evals)' %
          (name, m, ep, b[1], b[2], b[3], b[4], b[5], last[0], last[1], n))
