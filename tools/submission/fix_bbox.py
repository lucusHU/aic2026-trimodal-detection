#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把提交 zip 里超出图像边界的预测框裁剪回图内。

只处理 cx-w/2 < 0 或 cx+w/2 > 1（以及 y 方向）的框：
按边界裁剪，保持框与图像的真实交集，中心随之调整。
其余行保持原样（不改动数值精度以外的东西）。
"""
import argparse, shutil, sys, zipfile
from pathlib import Path


def fix_line(p):
    c = p[0]
    cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
    x1, x2 = cx - w / 2.0, cx + w / 2.0
    y1, y2 = cy - h / 2.0, cy + h / 2.0
    changed = False
    if x1 < 0.0 or x2 > 1.0 or y1 < 0.0 or y2 > 1.0:
        changed = True
        x1 = min(max(x1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        y2 = min(max(y2, 0.0), 1.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    return c, cx, cy, w, h, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    src = Path(args.zip)
    out = Path(args.out) if args.out else src.with_name(src.stem + '_clamped.zip')
    tmp = Path(str(src) + '.fixdir')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    zin = zipfile.ZipFile(src)
    n_line = n_fix = 0
    for name in zin.namelist():
        if not name.endswith('.txt'):
            continue
        data = zin.read(name).decode('utf-8')
        lines = []
        for line in data.splitlines():
            if not line.strip():
                continue
            p = line.split()
            n_line += 1
            c, cx, cy, w, h, ch = fix_line(p)
            if ch:
                n_fix += 1
            lines.append('%s %.6f %.6f %.6f %.6f %s' % (c, cx, cy, w, h, p[5]))
        (tmp / name).write_text('\n'.join(lines), encoding='utf-8')
    shutil.make_archive(str(out.with_suffix('')), 'zip', tmp)
    shutil.rmtree(tmp)
    print('lines=%d  fixed=%d' % (n_line, n_fix))
    print('out: %s' % out)


if __name__ == '__main__':
    main()
