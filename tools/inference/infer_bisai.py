import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="/home/newbie/bigdatan/jingsai/bisai/configs/x_baseline_800_800.yml",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="/home/newbie/bigdatan/jingsai/bisai/output/x_baseline_800_800/best_stg1.pth",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/home/newbie/bigdatan/jingsai/bisai/dataset/data/test/visible",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="/home/newbie/bigdatan/jingsai/bisai/predict_txt",
    )

    parser.add_argument("--imgsz", type=int, default=800)

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. 加入 D-FINE 项目路径
    # ---------------------------------------------------------
    dfine_root = "/home/newbie/bigdatan/jingsai/D-FINE"

    if dfine_root not in sys.path:
        sys.path.insert(0, dfine_root)

    from src.core import YAMLConfig

    device = torch.device(args.device)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("D-FINE inference")
    print(f"config  : {args.config}")
    print(f"weights : {args.weights}")
    print(f"input   : {input_dir}")
    print(f"output  : {output_dir}")
    print(f"imgsz   : {args.imgsz}")
    print(f"conf    : {args.conf}")
    print(f"device  : {device}")
    print("=" * 70)

    # ---------------------------------------------------------
    # 2. 创建模型
    # ---------------------------------------------------------
    cfg = YAMLConfig(
        args.config,
        resume=args.weights,
    )

    # 防止推理时再次下载 / 加载 backbone 预训练权重
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    print("[1/4] Loading checkpoint...")

    checkpoint = torch.load(
    args.weights,
    map_location="cpu",
    weights_only=False,
    )

    # D-FINE 官方训练保存的 checkpoint
    # 优先使用 EMA
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        state = checkpoint["ema"]["module"]
        print("Using EMA weights.")
    elif "model" in checkpoint:
        state = checkpoint["model"]
        print("Using model weights.")
    else:
        raise RuntimeError(
            "Checkpoint 中既没有 'ema' 也没有 'model'，"
            "请把 checkpoint.keys() 发给我。"
        )

    cfg.model.load_state_dict(state)

    # ---------------------------------------------------------
    # 3. 按官方 deploy 方式组合 model + postprocessor
    # ---------------------------------------------------------
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)

            labels, boxes, scores = self.postprocessor(
                outputs,
                orig_target_sizes,
            )

            return labels, boxes, scores

    model = Model().to(device)
    model.eval()

    print("[2/4] Model loaded.")

    # ---------------------------------------------------------
    # 4. 图像预处理
    #
    # D-FINE 官方 torch inference：
    # Resize -> ToTensor
    #
    # 你的训练 / 验证尺寸是 800
    # ---------------------------------------------------------
    transform = T.Compose(
        [
            T.Resize((args.imgsz, args.imgsz)),
            T.ToTensor(),
        ]
    )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_paths = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"在测试集目录中没有找到图片：{input_dir}"
        )

    print(f"[3/4] Found {len(image_paths)} images.")

    total_boxes = 0

    # ---------------------------------------------------------
    # 5. 逐张推理
    # ---------------------------------------------------------
    with torch.no_grad():

        batch_size = args.batch_size
        total_boxes = 0

        for start_idx in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[start_idx:start_idx + batch_size]

            batch_tensors = []
            batch_orig_sizes = []
            batch_meta = []

        # 1. 读入一个 batch
            for image_path in batch_paths:
                image = Image.open(image_path).convert("RGB")
                orig_w, orig_h = image.size

                image_tensor = transform(image)

                batch_tensors.append(image_tensor)
                batch_orig_sizes.append([orig_w, orig_h])
                batch_meta.append((image_path, orig_w, orig_h))

        # 2. 拼 batch
            images_tensor = torch.stack(batch_tensors, dim=0).to(device)
            orig_sizes = torch.tensor(
                batch_orig_sizes,
                dtype=torch.float32,
                device=device,
            )

        # 3. 推理
            labels_batch, boxes_batch, scores_batch = model(
                images_tensor,
                orig_sizes,
            )

        # 4. 逐张写 txt
            for i in range(len(batch_meta)):
                image_path, orig_w, orig_h = batch_meta[i]

                labels = labels_batch[i].detach().cpu()
                boxes = boxes_batch[i].detach().cpu()
                scores = scores_batch[i].detach().cpu()

                keep = scores >= args.conf
                labels = labels[keep]
                boxes = boxes[keep]
                scores = scores[keep]

                txt_path = output_dir / (image_path.stem + ".txt")
                lines = []

                for label, box, score in zip(labels, boxes, scores):
                    class_id = int(label.item())

                    if class_id < 0 or class_id > 11:
                        continue

                    x1, y1, x2, y2 = box.tolist()

                    x1 = max(0.0, min(x1, orig_w))
                    x2 = max(0.0, min(x2, orig_w))
                    y1 = max(0.0, min(y1, orig_h))
                    y2 = max(0.0, min(y2, orig_h))

                    box_w = max(0.0, x2 - x1)
                    box_h = max(0.0, y2 - y1)

                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    norm_center_x = center_x / orig_w
                    norm_center_y = center_y / orig_h
                    norm_w = box_w / orig_w
                    norm_h = box_h / orig_h
                    confidence = float(score.item())

                    line = (
                        f"{class_id} "
                        f"{norm_center_x:.6f} "
                        f"{norm_center_y:.6f} "
                        f"{norm_w:.6f} "
                        f"{norm_h:.6f} "
                        f"{confidence:.6f}"
                    )
                    lines.append(line)

                with open(txt_path, "w", encoding="utf-8") as f:
                    if lines:
                        f.write("\n".join(lines))
                        f.write("\n")

                total_boxes += len(lines)

    print()
    print("[4/4] Inference finished.")
    print(f"Images     : {len(image_paths)}")
    print(f"Total boxes: {total_boxes}")
    print(f"TXT saved  : {output_dir}")


if __name__ == "__main__":
    main()