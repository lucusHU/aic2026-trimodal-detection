#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO(darknet) -> COCO 转换脚本（在服务器 bisai 目录下运行）
读取已解压的 data/images + data/labels，生成 D-Fine/mmdet 需要的
    dataset/images/train, dataset/images/val, dataset/annotations/instances_{train,val}.json
会自动把训练集切出 VAL_RATIO 作为验证集（比赛 val/ 无标注，不能用来训练评估）。
不改动原 data/ 目录。
"""
import os, json, random, glob
from PIL import Image
from shutil import copyfile

# =============== 按实际情况改这里 ===============
IMG_DIR   = "dataset/data/images"   # YOLO 图片目录
LBL_DIR   = "dataset/data/labels"   # YOLO 标注目录
VAL_RATIO = 0.15                         # 从训练集切出验证集比例
OUT_DIR   = "dataset"                    # COCO 输出根目录
NUM_CLASSES = 12                         # 类别数（index 0~11）
SEED      = 42
# ==============================================

random.seed(SEED)

def get_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h)

def main():
    os.makedirs(os.path.join(OUT_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "images", "val"),   exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "annotations"),     exist_ok=True)

    # 1. 收集 (img_path, label_path, stem)
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    samples = []
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(LBL_DIR, stem + ".txt")
        if not os.path.exists(lbl):
            print("跳过无标注图片:", os.path.basename(img))
            continue
        samples.append((img, lbl, stem))

    if not samples:
        print("未找到带标注的图片，请检查 IMG_DIR/LBL_DIR 路径！")
        return

    random.shuffle(samples)
    n_val = max(1, int(len(samples) * VAL_RATIO))
    val_keys = set(s for _, _, s in samples[:n_val])
    print(f"共 {len(samples)} 张带标注图片，划分 train={len(samples)-n_val} val={n_val}")

    # 2. 类别表（与 YOLO 原始 index 一致，方便后续映射回类别）
    categories = [{"id": i, "name": f"class_{i}", "supercategory": "none"} for i in range(NUM_CLASSES)]

    def build(sel):
        images, annotations = [], []
        image_id, ann_id = 1, 1
        for img, lbl, stem in sel:
            w, h = get_size(img)
            ext = os.path.splitext(img)[1].lower()
            images.append({
                "id": image_id,
                "file_name": stem + ext,
                "width": w, "height": h,
            })
            with open(lbl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, cx, cy, bw, bh = line.split()
                    cls = int(cls); cx, cy, bw, bh = (float(x) for x in (cx, cy, bw, bh))
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    ww = bw * w
                    hh = bh * h
                    annotations.append({
                        "id": ann_id, "image_id": image_id,
                        "category_id": cls,
                        "bbox": [x, y, ww, hh],
                        "area": ww * hh,
                        "iscrowd": 0,
                    })
                    ann_id += 1
            image_id += 1
        return {"images": images, "annotations": annotations, "categories": categories}

    train_sel = [s for s in samples if s[2] not in val_keys]
    val_sel   = [s for s in samples if s[2] in val_keys]

    for split, sel in [("train", train_sel), ("val", val_sel)]:
        data = build(sel)
        with open(os.path.join(OUT_DIR, "annotations", f"instances_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f)
        # 硬链接图片到对应目录（同盘不占额外空间；失败则复制）
        for img, lbl, stem in sel:
            dst = os.path.join(OUT_DIR, "images", split, os.path.basename(img))
            if not os.path.exists(dst):
                try:
                    os.link(img, dst)
                except OSError:
                    copyfile(img, dst)
        print(f"已生成 instances_{split}.json（{len(sel)} 张）")

    print("\n完成！输出目录：", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()

