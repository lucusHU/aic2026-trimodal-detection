import json
from pathlib import Path
from collections import defaultdict

JSON_PATH = "/home/newbie/bigdatan/jingsai/bisai/dataset/annotations/instances_val.json"

# 这里改成你要检查的预测目录
PRED_DIR = Path("/home/newbie/bigdatan/jingsai/bisai/submit/val_top100")

IOU_THRESH = 0.5

CONF_THRESH = 0.0


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


# =========================
# 读取验证集 GT
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = {}
stem_to_id = {}

for img in data["images"]:
    images[img["id"]] = img
    stem_to_id[Path(img["file_name"]).stem] = img["id"]

gt_dict = defaultdict(list)

for ann in data["annotations"]:
    image_id = ann["image_id"]
    cls = int(ann["category_id"])

    x, y, w, h = ann["bbox"]

    gt_dict[image_id].append(
        {
            "class": cls,
            "box": [x, y, x + w, y + h],
        }
    )


# =========================
# 开始统计
# =========================
TP = 0
FP = 0
FN = 0

num_images = 0
num_gt = 0
num_preds = 0

missing_files = 0


for txt_path in PRED_DIR.glob("*.txt"):

    stem = txt_path.stem

    if stem not in stem_to_id:
        print("JSON里找不到:", txt_path.name)
        missing_files += 1
        continue

    image_id = stem_to_id[stem]

    img_info = images[image_id]

    img_w = img_info["width"]
    img_h = img_info["height"]

    gt_list = gt_dict[image_id]

    pred_list = []

    # =========================
    # 读取预测 TXT
    # 格式：
    # class cx cy w h confidence
    # =========================
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 6:
                continue

            cls = int(float(parts[0]))

            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            conf = float(parts[5])
            if conf < CONF_THRESH:
                continue

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            pred_list.append(
                {
                    "class": cls,
                    "box": [x1, y1, x2, y2],
                    "conf": conf,
                }
            )

    # 高置信度优先
    pred_list.sort(key=lambda x: x["conf"], reverse=True)

    used_gt = set()

    # =========================
    # 匹配预测框和 GT
    # 类别相同 + IoU >= 0.5
    # 才算 TP
    # =========================
    for pred in pred_list:

        best_iou = 0.0
        best_gt_idx = None

        for i, gt in enumerate(gt_list):

            if i in used_gt:
                continue

            # 类别必须相同
            if pred["class"] != gt["class"]:
                continue

            current_iou = iou(pred["box"], gt["box"])

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i

        if best_gt_idx is not None and best_iou >= IOU_THRESH:

            TP += 1
            used_gt.add(best_gt_idx)

        else:

            FP += 1

    # 没被匹配到的 GT 就是漏检
    FN += len(gt_list) - len(used_gt)

    num_images += 1
    num_gt += len(gt_list)
    num_preds += len(pred_list)


precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


print()
print("=" * 50)
print("检测结果统计")
print("=" * 50)

print("图片数:", num_images)
print("GT总框数:", num_gt)
print("预测总框数:", num_preds)
print("JSON里找不到的预测文件:", missing_files)

print()
print("TP 正确检测:", TP)
print("FP 误检:", FP)
print("FN 漏检:", FN)

print()
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

print("=" * 50)
