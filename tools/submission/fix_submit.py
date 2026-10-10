"""把提交 txt 中所有框 clamp 回 [0,1]，重打包为合规 zip。"""
import zipfile, shutil
from pathlib import Path

SRC = "/home/adlab7/ssd/bigdatan/jingsai/bisai/zip/seed1_ms800_928_nms07.zip"
DST = "/home/adlab7/ssd/bigdatan/jingsai/bisai/zip/seed1_ms800_928_nms07_fixed.zip"
TMP = Path("/tmp/fixzip")
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)

z = zipfile.ZipFile(SRC)
fixed_boxes = 0
dropped = 0
total = 0

for n in z.namelist():
    if not n.lower().endswith('.txt'):
        continue
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
            # 转 x1y1x2y2 后 clamp
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            ox1, oy1, ox2, oy2 = x1, y1, x2, y2
            x1, x2 = min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)
            y1, y2 = min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)
            if (x1, y1, x2, y2) != (ox1, oy1, ox2, oy2):
                fixed_boxes += 1
            nw, nh = x2 - x1, y2 - y1
            if nw <= 0 or nh <= 0:
                dropped += 1
                continue
            ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
            out.append(f"{c} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f} {s:.6f}")
    (TMP / Path(n).name).write_text("\n".join(out) + ("\n" if out else ""), encoding='utf-8')

with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as zo:
    for f in sorted(TMP.iterdir()):
        zo.write(f, f.name)

print("原框总数:", total)
print("修正过的框:", fixed_boxes)
print("因 clamp 后无效丢弃:", dropped)
print("输出:", DST)

# 复核
z2 = zipfile.ZipFile(DST)
bad = 0
cnt = 0
for n in z2.namelist():
    for ln in z2.read(n).decode().strip().split('\n'):
        if not ln:
            continue
        p = ln.split()
        cnt += 1
        v = [float(x) for x in p[1:5]]
        if any(x < 0 or x > 1 for x in v):
            bad += 1
print("复核: txt 数", len(z2.namelist()), "框数", cnt, "越界", bad)
