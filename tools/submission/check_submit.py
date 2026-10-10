import os, sys, zipfile, io
from pathlib import Path

TEST_DIR = Path("/home/adlab7/ssd/bigdatan/jingsai/bisai/dataset/data/AIC2026_PHASE_2_1000/visible")
ZIP = "/home/adlab7/ssd/bigdatan/jingsai/bisai/zip/seed1_ms800_928_nms07.zip"
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

imgs = sorted(p for p in TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
img_stems = set(p.stem for p in imgs)
print("测试图片数:", len(imgs))

z = zipfile.ZipFile(ZIP)
names = [n for n in z.namelist() if n.lower().endswith('.txt')]
print("zip 内 txt 数:", len(names))
print("zip 内非 txt 条目:", [n for n in z.namelist() if not n.lower().endswith('.txt')][:10])

txt_stems = set(Path(n).name[:-4] for n in names)
# 处理可能带目录前缀
missing = img_stems - txt_stems
extra = txt_stems - img_stems
print("缺图(测试集有但 zip 无):", len(missing), list(missing)[:5])
print("多余(zip 有但测试集无):", len(extra), list(extra)[:5])

# 是否有嵌套目录
dirs = set(os.path.dirname(n) for n in names)
print("zip 内目录层级:", dirs)

err_cls, err_coor, err_score, empty = 0, 0, 0, 0
cls_set = set()
nfields_bad = 0
total = 0
per_img_max = 0
over100 = []
for n in names:
    data = z.read(n).decode('utf-8').strip()
    if not data:
        empty += 1
        continue
    lines = data.split('\n')
    per_img_max = max(per_img_max, len(lines))
    if len(lines) > 100:
        over100.append((n, len(lines)))
    for ln in lines:
        parts = ln.split()
        total += 1
        if len(parts) != 6:
            nfields_bad += 1
            continue
        c = int(parts[0]); vals = [float(x) for x in parts[1:5]]; s = float(parts[5])
        cls_set.add(c)
        if not (0 <= c <= 11):
            err_cls += 1
        for v in vals:
            if v < 0.0 or v > 1.0:
                err_coor += 1
                break
        if s < 0.0 or s > 1.0:
            err_score += 1

print("总预测框数:", total)
print("空 txt 数:", empty)
print("单图最大框数:", per_img_max, "| 超 100 框的图:", over100[:3], len(over100))
print("字段数非 6 的行:", nfields_bad)
print("类别 id 集合:", sorted(cls_set))
print("类别越界行数:", err_cls)
print("坐标越界行数:", err_coor)
print("分数越界行数:", err_score)
