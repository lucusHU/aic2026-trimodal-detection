#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用版提交包 clamp + 合规校验。

用法: python fix_submit_any.py <src.zip> [dst.zip]
如果 dst 省略，则输出 <src>_fixed.zip
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

SRC = sys.argv[1]
DST = sys.argv[2] if len(sys.argv) > 2 else SRC.replace('.zip', '_fixed.zip')
IMG_DIR = '/home/adlab7/ssd/bigdatan/jingsai/bisai/dataset/data/AIC2026_PHASE_2_1000/visible'

TMP = Path('/tmp/fixzip_any')
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)

z = zipfile.ZipFile(SRC)
fixed_boxes = 0
dropped = 0
total = 0
names = []

for n in z.namelist():
    if not n.lower().endswith('.txt'):
        continue
    names.append(n)
    raw = z.read(n).decode('utf-8').strip()
    out = []
    if raw:
        for ln in raw.split('\n'):
            p = ln.split()
            if len(p) != 6:
                continue
            c = int(p[0])
            cx, cy, w, h = [float(x) for x in p[1:5]]
            s = float(p[5])
            total += 1
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            o = (x1, y1, x2, y2)
            x1, x2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
            y1, y2 = min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)
            if (x1, y1, x2, y2) != o:
                fixed_boxes += 1
            nw, nh = x2 - x1, y2 - y1
            if nw <= 0 or nh <= 0:
                dropped += 1
                continue
            out.append('%d %.6f %.6f %.6f %.6f %.6f'
                       % (c, (x1 + x2) / 2, (y1 + y2) / 2, nw, nh, s))
    (TMP / Path(n).name).write_text('\n'.join(out) + ('\n' if out else ''), encoding='utf-8')

with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as zo:
    for f in sorted(TMP.iterdir()):
        zo.write(f, f.name)

print('原框总数:', total)
print('修正过的框:', fixed_boxes)
print('因 clamp 后无效丢弃:', dropped)
print('输出:', DST)

# ------------------------------------------------------- 合规复核
z2 = zipfile.ZipFile(DST)
entries = [n for n in z2.namelist() if n.lower().endswith('.txt')]
stems = {Path(n).stem for n in entries}
imgs = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR)}

bad = 0
nb = 0
mx = 0
cls = set()
for n in entries:
    txt = z2.read(n).decode('utf-8').strip()
    lines = [l for l in txt.split('\n') if l.strip()]
    nb += len(lines)
    mx = max(mx, len(lines))
    for l in lines:
        p = l.split()
        if len(p) != 6:
            bad += 1
            continue
        try:
            c = int(p[0])
            v = [float(x) for x in p[1:]]
        except ValueError:
            bad += 1
            continue
        cls.add(c)
        if not (0 <= c <= 11 and all(0.0 <= x <= 1.0 for x in v)):
            bad += 1

print()
print('=== 合规复核 ===')
print('txt 文件数        :', len(entries))
print('嵌套目录          :', sum(1 for n in entries if '/' in n.strip('/')))
print('与测试图同名缺失  :', len(imgs - stems))
print('多余文件          :', len(stems - imgs))
print('总框数            :', nb)
print('单图最大框数      :', mx, '(上限 100)')
print('非法行/越界       :', bad)
print('类别集合          :', sorted(cls))

ok = (bad == 0 and mx <= 100 and len(imgs - stems) == 0 and len(stems - imgs) == 0
      and len(entries) == len(imgs))
print()
print('VERDICT:', 'OK —— 可以上传' if ok else 'PROBLEM —— 不要上传')
