# bisai/infer_submit.py

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# ============================================================
# 1. 路径和参数
# ============================================================

# D-FINE 仓库和 bisai 同级
DFINE_ROOT = "../D-FINE"

# 一定改成：训练这个权重时所用的 yml
CONFIG = "configs/你的真实配置文件.yml"

# 先推理 best_stg1
WEIGHT = "output/dfine_hgnetv2_l_bisai/best_stg1.pth"

# 改成真实测试图片目录
# 例如 dataset/test 或 dataset/test/visible
TEST_DIR = "dataset/test"

# 原始 txt 输出
SUBMIT_DIR = "submit"

# 程序内部使用 cuda:0
# 运行时用 CUDA_VISIBLE_DEVICES 指定物理 GPU
DEVICE = "cuda:0"

# mAP 比赛不要把阈值设太高
CONF_THRES = 0.01

# 官方要求每张图最多 100 框
MAX_DET = 100

# D-FINE 默认评估尺寸是 640x640
INPUT_SIZE = (640, 640)


# ============================================================
# 2. 导入 D-FINE
# ============================================================

sys.path.insert(0, os.path.abspath(DFINE_ROOT))

from src.core import YAMLConfig


# ============================================================
# 3. 加载模型
# ============================================================

def load_model():
    print("加载配置:", CONFIG)
    print("加载权重:", WEIGHT)

    cfg = YAMLConfig(CONFIG, resume=WEIGHT)

    checkpoint = torch.load(WEIGHT, map_location="cpu")

    # D-FINE checkpoint 通常可能同时保存 model 和 ema。
    # 推理优先使用 EMA。
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        state_dict = checkpoint["ema"]["module"]
        print("使用 EMA 权重")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("使用 model 权重")
    else:
        # 防止某些 checkpoint 本身就是 state_dict
        state_dict = checkpoint
        print("checkpoint 本身作为 state_dict 加载")

    cfg.model.load_state_dict(state_dict)

    class DFineInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = DFineInference().to(DEVICE)
    model.eval()

    return model


# ============================================================
# 4. 图像预处理
# ============================================================

transform = T.Compose([
    T.Resize(INPUT_SIZE),
    T.ToTensor(),
])


# ============================================================
# 5. 单张图片推理
# ============================================================

@torch.no_grad()
def infer_one(model, image_path):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    # D-FINE postprocessor 需要原图尺寸来恢复 bbox
    orig_size = torch.tensor(
        [[width, height]],
        dtype=torch.float32,
        device=DEVICE
    )

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    labels, boxes, scores = model(image_tensor, orig_size)

    labels = labels[0].detach().cpu()
    boxes = boxes[0].detach().cpu()
    scores = scores[0].detach().cpu()

    # 先过滤过低置信度
    keep = scores >= CONF_THRES
    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    # 按 confidence 降序排序
    order = torch.argsort(scores, descending=True)
    labels = labels[order]
    boxes = boxes[order]
    scores = scores[order]

    # 每张图最多 100 个预测
    labels = labels[:MAX_DET]
    boxes = boxes[:MAX_DET]
    scores = scores[:MAX_DET]

    return labels, boxes, scores, width, height


# ============================================================
# 6. 写比赛要求的 TXT
# ============================================================

def write_submission_txt(
    txt_path,
    labels,
    boxes,
    scores,
    width,
    height
):
    # 即便没有框，也会创建一个空 txt
    with open(txt_path, "w", encoding="utf-8") as f:

        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box.tolist()

            # 防止预测框稍微越过图像边界
            x1 = max(0.0, min(x1, float(width)))
            x2 = max(0.0, min(x2, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            y2 = max(0.0, min(y2, float(height)))

            box_w = x2 - x1
            box_h = y2 - y1

            # 无效框不要写
            if box_w <= 0 or box_h <= 0:
                continue

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # 比赛要求归一化坐标
            norm_center_x = center_x / width
            norm_center_y = center_y / height
            norm_w = box_w / width
            norm_h = box_h / height

            # 再保险限制到 [0, 1]
            norm_center_x = min(max(norm_center_x, 0.0), 1.0)
            norm_center_y = min(max(norm_center_y, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            class_id = int(label.item())
            confidence = float(score.item())

            f.write(
                f"{class_id} "
                f"{norm_center_x:.6f} "
                f"{norm_center_y:.6f} "
                f"{norm_w:.6f} "
                f"{norm_h:.6f} "
                f"{confidence:.6f}\n"
            )


# ============================================================
# 7. 批量推理整个测试集
# ============================================================

def main():
    os.makedirs(SUBMIT_DIR, exist_ok=True)

    test_dir = Path(TEST_DIR)

    if not test_dir.exists():
        raise FileNotFoundError(
            f"测试集目录不存在: {TEST_DIR}"
        )

    if not Path(CONFIG).exists():
        raise FileNotFoundError(
            f"配置文件不存在: {CONFIG}"
        )

    if not Path(WEIGHT).exists():
        raise FileNotFoundError(
            f"权重不存在: {WEIGHT}"
        )

    model = load_model()

    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    image_paths = sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )

    print()
    print("=" * 60)
    print("测试图片数量:", len(image_paths))
    print("TEST_DIR:", TEST_DIR)
    print("SUBMIT_DIR:", SUBMIT_DIR)
    print("CONF_THRES:", CONF_THRES)
    print("MAX_DET:", MAX_DET)
    print("=" * 60)

    if len(image_paths) == 0:
        raise RuntimeError(
            f"{TEST_DIR} 里没有找到图片，请检查测试集路径"
        )

    total_dets = 0

    for i, image_path in enumerate(image_paths, start=1):

        labels, boxes, scores, width, height = infer_one(
            model,
            image_path
        )

        txt_path = (
            Path(SUBMIT_DIR)
            / f"{image_path.stem}.txt"
        )

        write_submission_txt(
            txt_path,
            labels,
            boxes,
            scores,
            width,
            height
        )

        total_dets += len(labels)

        if (
            i == 1
            or i % 50 == 0
            or i == len(image_paths)
        ):
            print(
                f"[{i}/{len(image_paths)}] "
                f"{image_path.name} -> "
                f"{len(labels)} detections"
            )

    print()
    print("=" * 60)
    print("推理完成")
    print("生成 TXT 数量:", len(image_paths))
    print("预测框总数:", total_dets)
    print("结果目录:", SUBMIT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()