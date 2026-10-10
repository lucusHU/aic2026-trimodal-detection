#!/bin/bash
# usage: _finalize_any.sh <name>     # expects submit/<name>/<name>/*.txt and zip/<name>.zip
B=/home/adlab7/ssd/bigdatan/jingsai/bisai
cd $B || exit 1
NAME=$1
date "+%H:%M:%S"
echo "name=$NAME"

/home/newbie/miniconda3/envs/jingsai/bin/python - "$NAME" <<'PY'
import sys, shutil, zipfile
from pathlib import Path

NAME = sys.argv[1]
RAW = Path('zip') / (NAME + '.zip')
SRCROOT = Path('submit') / NAME
txts = list(SRCROOT.rglob('*.txt'))
if not txts:
    raise SystemExit('NO TXT under %s' % SRCROOT)
SRC = txts[0].parent
print('source dir :', SRC, '(%d txt)' % len(txts))

DST = Path('submit') / (NAME + '_fixed')
FIXED = Path('zip') / (NAME + '_fixed.zip')
if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

total = clamped = dropped = 0
for p in sorted(SRC.glob('*.txt')):
    out = []
    for line in p.read_text().strip().splitlines():
        if not line.strip():
            continue
        total += 1
        c, cx, cy, w, h, s = line.split()
        c = int(float(c)); cx, cy, w, h, s = map(float, (cx, cy, w, h, s))
        x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        nx1, nx2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
        ny1, ny2 = min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)
        nw, nh = nx2-nx1, ny2-ny1
        if nw <= 0 or nh <= 0:
            dropped += 1; continue
        ncx, ncy = (nx1+nx2)/2, (ny1+ny2)/2
        if abs(ncx-cx) > 1e-9 or abs(ncy-cy) > 1e-9 or abs(nw-w) > 1e-9 or abs(nh-h) > 1e-9:
            clamped += 1
        out.append('%d %.6f %.6f %.6f %.6f %.6f' % (c, ncx, ncy, nw, nh, min(max(s,0.0),1.0)))
    (DST / p.name).write_text('\n'.join(out) + ('\n' if out else ''))

print('boxes=%d clamped=%d dropped=%d' % (total, clamped, dropped))
if FIXED.exists():
    FIXED.unlink()
with zipfile.ZipFile(FIXED, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DST.glob('*.txt')):
        z.write(p, arcname=p.name)
print('fixed zip :', FIXED, FIXED.stat().st_size, 'bytes')

TEST = Path('dataset/data/AIC2026_PHASE_2_1000/visible')
stems = {p.stem for p in TEST.iterdir()}
TOL = 1e-5

def check(zpath):
    z = zipfile.ZipFile(zpath)
    names = [n for n in z.namelist() if not n.endswith('/')]
    txts = [n for n in names if n.endswith('.txt')]
    nested = [n for n in txts if '/' in n.strip('/')]
    miss = stems - {Path(n).stem for n in txts}
    extra = {Path(n).stem for n in txts} - stems
    nbox = badf = badc = badconf = over = wh1 = out = empty = 0
    for n in txts:
        rows = [l.split() for l in z.read(n).decode().strip().splitlines() if l.strip()]
        if not rows: empty += 1
        if len(rows) > 100: over += 1
        for r in rows:
            nbox += 1
            if len(r) != 6:
                badf += 1; continue
            try:
                c = int(float(r[0])); cx, cy, w, h, s = map(float, r[1:6])
            except Exception:
                badf += 1; continue
            if not (0 <= c <= 11): badc += 1
            if not (0.0 <= s <= 1.0): badconf += 1
            if w > 1+TOL or h > 1+TOL: wh1 += 1
            if (cx-w/2 < -TOL or cx+w/2 > 1+TOL or cy-h/2 < -TOL or cy+h/2 > 1+TOL
                or w <= 0 or h <= 0): out += 1
    ok = (badf==badc==badconf==over==wh1==0 and len(txts)==1000 and not nested
          and not miss and not extra)
    print('%-42s txt=%4d boxes=%6d empty=%d field=%d class=%d conf=%d over100=%d '
          'w/h>1=%d extent_out=%d nested=%d miss=%d extra=%d -> %s'
          % (Path(zpath).name, len(txts), nbox, empty, badf, badc, badconf, over,
             wh1, out, len(nested), len(miss), len(extra), 'PASS' if ok else 'FAIL'))
    return ok

a = check(FIXED)
b = check(RAW) if RAW.exists() else None
print('OVERALL:', 'PASS' if a else 'FAIL')
PY
date "+%H:%M:%S"
echo ===DONE===
