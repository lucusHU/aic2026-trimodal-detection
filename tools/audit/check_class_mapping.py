import os
import json
from collections import defaultdict
from pathlib import Path

JSON_PATH = "/home/newbie/bigdatan/jingsai/bisai/dataset/annotations/instances_val.json"
PRED_DIR = Path("/home/newbie/bigdatan/jingsai/bisai/submit/val_check_conf01")

IOU_THRESH = 0.5


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0

    return inter / union


with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# image_id -> 图片信息
images = {}
stem_to_id = {}

for img in data["images"]:
    images[img["id"]] = img

    stem = os.path.splitext(img["file_name"])[0]
    stem_to_id[stem] = img["id"]


# image_id -> GT
gt_dict = defaultdict(list)

for ann in data["annotations"]:
    image_id = ann["image_id"]
    cls = int(ann["category_id"])

    x, y, w, h = ann["bbox"]

    box = [
        x,
        y,
        x + w,
        y + h
    ]

    gt_dict[image_id].append({
        "class": cls,
        "box": box
    })


# mapping[gt_class][pred_class] = 次数
mapping = defaultdict(lambda: defaultdict(int))

matched_count = 0
checked_images = 0
missing_images = 0


for filename in os.listdir(PRED_DIR):

    if not filename.endswith(".txt"):
        continue

    stem = os.path.splitext(filename)[0]

    if stem not in stem_to_id:
        print("JSON里找不到:", filename)
        missing_images += 1
        continue

    image_id = stem_to_id[stem]

    img_info = images[image_id]

    img_w = img_info["width"]
    img_h = img_info["height"]

    gt_list = gt_dict[image_id]

    pred_list = []

    pred_path = os.path.join(PRED_DIR, filename)

    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))

            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h

            conf = float(parts[5]) if len(parts) >= 6 else 1.0

            box = [
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2
            ]

            pred_list.append({
                "class": cls,
                "box": box,
                "conf": conf
            })

    checked_images += 1

    # 高置信度优先
    pred_list.sort(key=lambda x: x["conf"], reverse=True)

    used_gt = set()

    for pred in pred_list:

        best_iou = 0
        best_gt_idx = None

        for i, gt in enumerate(gt_list):

            if i in used_gt:
                continue

            current_iou = iou(pred["box"], gt["box"])

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i

        if best_gt_idx is not None and best_iou >= IOU_THRESH:

            gt = gt_list[best_gt_idx]

            mapping[gt["class"]][pred["class"]] += 1

            used_gt.add(best_gt_idx)

            matched_count += 1


print("\n==============================")
print("检查图片数:", checked_images)
print("成功匹配框数:", matched_count)
print("JSON里找不到的预测文件:", missing_images)
print("==============================\n")

print("GT类别 -> Pred类别统计\n")

for gt_cls in range(12):

    row = mapping[gt_cls]

    total = sum(row.values())

    print(f"GT {gt_cls}  总匹配 {total}")

    if total == 0:
        print("   没有匹配到框")
        continue

    for pred_cls, count in sorted(row.items(), key=lambda x: -x[1]):

        percentage = count / total * 100

        print(
            f"   -> Pred {pred_cls}: "
            f"{count} 次  ({percentage:.1f}%)"
        )

    print()