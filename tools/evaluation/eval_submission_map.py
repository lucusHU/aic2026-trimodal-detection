import json
from pathlib import Path

from faster_coco_eval import COCO, COCOeval_faster

GT_JSON = "/home/newbie/bigdatan/jingsai/bisai/dataset/annotations/instances_val.json"
PRED_DIR = Path("/home/newbie/bigdatan/jingsai/bisai/submit/val_top100")
OUT_JSON = "/home/newbie/bigdatan/jingsai/bisai/val_top100_predictions_coco.json"

# ---------------------------------------------------------
# 1. 读取 GT
# ---------------------------------------------------------
with open(GT_JSON, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

# stem -> image info
stem_to_img = {}
for img in gt_data["images"]:
    stem = Path(img["file_name"]).stem
    stem_to_img[stem] = img

# ---------------------------------------------------------
# 2. 把提交 TXT 转成 COCO detection result JSON
#    TXT:
#    class cx cy w h confidence
#    坐标为 0~1 归一化
#
#    COCO result:
#    image_id, category_id, bbox[x,y,w,h], score
# ---------------------------------------------------------
predictions = []
missing = []

for txt_path in sorted(PRED_DIR.glob("*.txt")):
    stem = txt_path.stem

    if stem not in stem_to_img:
        missing.append(txt_path.name)
        continue

    img = stem_to_img[stem]
    image_id = img["id"]
    W = img["width"]
    H = img["height"]

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 6:
                raise ValueError(
                    f"{txt_path.name} 存在非6列数据: {line}"
                )

            cls = int(float(parts[0]))
            cx = float(parts[1]) * W
            cy = float(parts[2]) * H
            bw = float(parts[3]) * W
            bh = float(parts[4]) * H
            score = float(parts[5])

            x = cx - bw / 2.0
            y = cy - bh / 2.0

            predictions.append({
                "image_id": int(image_id),
                "category_id": int(cls),
                "bbox": [
                    float(x),
                    float(y),
                    float(bw),
                    float(bh),
                ],
                "score": float(score),
            })

print("=" * 70)
print("TXT -> COCO 转换完成")
print("预测目录 :", PRED_DIR)
print("GT JSON  :", GT_JSON)
print("TXT数量  :", len(list(PRED_DIR.glob('*.txt'))))
print("预测框数 :", len(predictions))
print("GT找不到的TXT:", len(missing))

if missing:
    print("前10个未匹配文件:", missing[:10])

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(predictions, f)

print("COCO预测JSON:", OUT_JSON)
print("=" * 70)

# ---------------------------------------------------------
# 3. 官方风格 COCO bbox 评估
# ---------------------------------------------------------
coco_gt = COCO(GT_JSON)
coco_dt = coco_gt.loadRes(predictions)

coco_eval = COCOeval_faster(
    coco_gt,
    coco_dt,
    iouType="bbox",
)

# 对齐官方 COCO evaluator
coco_eval.params.maxDets = [1, 10, 100]

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print()
print("=" * 70)
print("关键结果")
print("=" * 70)

stats = coco_eval.stats

print(f"mAP50-95 : {stats[0]:.4f}")
print(f"mAP50    : {stats[1]:.4f}")
print(f"mAP75    : {stats[2]:.4f}")
print(f"AR@1     : {stats[6]:.4f}")
print(f"AR@10    : {stats[7]:.4f}")
print(f"AR@100   : {stats[8]:.4f}")

print()
print("对照：你刚才官方 test-only 的 mAP50-95 ≈ 0.553")
print("=" * 70)
