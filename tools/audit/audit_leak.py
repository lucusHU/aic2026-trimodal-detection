"""合规审计：训练/验证集 与 测试集 是否存在任何交叉污染。"""
import hashlib, json, os
from pathlib import Path

ROOT = Path("/home/adlab7/ssd/bigdatan/jingsai/bisai")
TEST = ROOT / "dataset/data/AIC2026_PHASE_2_1000/visible"
TRAIN = ROOT / "dataset/images/train"
VAL = ROOT / "dataset/images/val"
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def lis(d):
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXT)

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

test_f, train_f, val_f = lis(TEST), lis(TRAIN), lis(VAL)
print(f"test={len(test_f)} train={len(train_f)} val={len(val_f)}")

# 1) 文件名交叉
tn = {p.name for p in test_f}
print("文件名重叠 train∩test:", len({p.name for p in train_f} & tn))
print("文件名重叠 val∩test  :", len({p.name for p in val_f} & tn))

# 2) 内容 md5 交叉（最严格）
print("计算 md5 ...")
th = {md5(p) for p in test_f}
trh = {md5(p) for p in train_f}
vah = {md5(p) for p in val_f}
print("内容重叠 train∩test:", len(trh & th))
print("内容重叠 val∩test  :", len(vah & th))

# 3) 标注 json 里是否出现测试集图片
for jf in ["instances_train.json", "instances_val.json"]:
    d = json.load(open(ROOT / "dataset/annotations" / jf))
    names = {im["file_name"] for im in d["images"]}
    hit = names & tn
    print(f"{jf}: 图片条目 {len(names)}, 与测试集重名 {len(hit)}",
          sorted(hit)[:3] if hit else "")
    # 标注框是否越界 / 是否含测试图
    print(f"   类别数={len(d['categories'])} 标注框数={len(d['annotations'])}")

# 4) 磁盘上图片文件是否真的落在各自目录
print("train 目录文件是否都在 json 里:",
      len({p.name for p in train_f} - {im["file_name"] for im in
          json.load(open(ROOT/'dataset/annotations/instances_train.json'))["images"]}))
