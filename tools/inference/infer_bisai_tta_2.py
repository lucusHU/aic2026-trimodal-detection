#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
D-FINE-X 多尺度 TTA 推理（动态 eval_spatial_size 版）

TTA:
  768  original + flip
  1024 original + flip
  1280 original + flip

关键点：
  每个尺度分别构建一个 eval_spatial_size 与输入尺寸一致的 D-FINE-X 模型，
  避免 HybridEncoder 位置编码长度与特征长度不匹配。

最终：
  6组预测 -> WBF -> Top100 -> TXT -> ZIP
"""

from ensemble_boxes import weighted_boxes_fusion
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as T


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TTA_SIZES = [800, 1024, 1280]
TTA_FLIP = True


def parse_args():
    p = argparse.ArgumentParser("D-FINE-X dynamic-size TTA -> TXT + ZIP")
    p.add_argument("-c", "--config", required=True, help="D-FINE-X训练配置")
    p.add_argument("-r", "--resume", required=True, help="D-FINE-X best权重")
    p.add_argument("-i", "--input", required=True, help="测试图片目录或单张图片")
    p.add_argument("--repo", default="../D-FINE", help="D-FINE仓库路径")
    p.add_argument("--device", default="cuda:0", help="例如 cuda:0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--cat-offset", type=int, default=0)
    p.add_argument(
        "--submit-dir",
        default="/home/newbie/bigdatan/jingsai/bisai/submit",
    )
    p.add_argument(
        "--zip-dir",
        default="/home/newbie/bigdatan/jingsai/bisai/zip",
    )
    p.add_argument("--zip-name", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")

    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    # deploy时由当前尺寸重新生成
    state = dict(state)
    state.pop("decoder.anchors", None)
    state.pop("decoder.valid_mask", None)
    return state


def build_model_for_size(args, eval_size, state):
    """
    每个TTA尺度分别构建一个模型。
    最关键的是在 cfg.model 第一次被创建之前，
    把 eval_spatial_size 改成当前尺寸。
    """
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.resume)

    # ----------------------------------------------------------
    # 关键修复：
    # 768输入 -> [768,768]
    # 1024输入 -> [1024,1024]
    # 1280输入 -> [1280,1280]
    # 必须在 cfg.model 被访问/创建之前修改
    # ----------------------------------------------------------
    cfg.yaml_cfg["eval_spatial_size"] = [eval_size, eval_size]

    # 如果具体模块配置中也显式缓存了 eval_spatial_size，
    # 一并覆盖（存在才改，不存在不新增无效参数）
    for key in ("HybridEncoder", "DFINETransformer"):
        if key in cfg.yaml_cfg and "eval_spatial_size" in cfg.yaml_cfg[key]:
            cfg.yaml_cfg[key]["eval_spatial_size"] = [eval_size, eval_size]

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # 此时才真正创建模型
    base_model = cfg.model
    missing = base_model.load_state_dict(state, strict=False)

    class Model(torch.nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    model = Model(base_model, cfg.postprocessor).to(args.device).eval()

    print(
        f"[model] {eval_size}x{eval_size} loaded | "
        f"eval_spatial_size={cfg.yaml_cfg.get('eval_spatial_size')}",
        flush=True,
    )

    # 只做提示，不因此终止
    if getattr(missing, "missing_keys", None):
        print(f"[model] missing keys: {len(missing.missing_keys)}", flush=True)
    if getattr(missing, "unexpected_keys", None):
        print(f"[model] unexpected keys: {len(missing.unexpected_keys)}", flush=True)

    return model


def infer_batch(model, images, size, device, amp=False, flip=False):
    batch_tensors = []
    orig_sizes = []
    widths = []

    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])

    for image in images:
        w, h = image.size
        widths.append(w)

        if flip:
            if hasattr(Image, "Transpose"):
                input_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            else:
                input_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            input_image = image

        batch_tensors.append(transform(input_image))
        orig_sizes.append([w, h])

    image_tensor = torch.stack(batch_tensors, dim=0).to(device)
    orig_size = torch.tensor(orig_sizes, dtype=torch.float32, device=device)

    with torch.inference_mode():
        if amp and str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                labels, boxes, scores = model(image_tensor, orig_size)
        else:
            labels, boxes, scores = model(image_tensor, orig_size)

    out_labels, out_boxes, out_scores = [], [], []

    for i in range(len(images)):
        lb = labels[i].detach().cpu()
        bx = boxes[i].detach().cpu()
        sc = scores[i].detach().cpu()

        if flip and bx.numel() > 0:
            bx = bx.clone()
            w = widths[i]
            old_x1 = bx[:, 0].clone()
            old_x2 = bx[:, 2].clone()
            bx[:, 0] = w - old_x2
            bx[:, 2] = w - old_x1

        out_labels.append(lb)
        out_boxes.append(bx)
        out_scores.append(sc)

    return out_labels, out_boxes, out_scores


def fuse_tta_predictions(
    all_labels,
    all_boxes,
    all_scores,
    width,
    height,
    iou_thr=0.55,
    skip_box_thr=0.001,
):
    boxes_list = []
    scores_list = []
    labels_list = []
    total_valid = 0

    for labels, boxes, scores in zip(all_labels, all_boxes, all_scores):
        b_list, s_list, l_list = [], [], []

        for lb, box, sc in zip(labels, boxes, scores):
            score = float(sc.item())
            if score < skip_box_thr:
                continue

            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            x2 = max(0.0, min(x2, float(width)))
            y2 = max(0.0, min(y2, float(height)))

            if x2 <= x1 or y2 <= y1:
                continue

            b_list.append([
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ])
            s_list.append(score)
            l_list.append(int(lb.item()))
            total_valid += 1

        boxes_list.append(b_list)
        scores_list.append(s_list)
        labels_list.append(l_list)

    if total_valid == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    weights = [1.0] * len(boxes_list)

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height

    return (
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
    )


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入不存在: {input_path}")

    # 图片列表
    if input_path.is_dir():
        img_list = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXT
        )
    else:
        img_list = [input_path]

    if not img_list:
        raise RuntimeError("没有找到测试图片")

    stems = [p.stem for p in img_list]
    if len(stems) != len(set(stems)):
        raise RuntimeError("存在同名图片stem，无法安全生成TXT")

    # 输出目录
    weight_stem = Path(args.resume).stem
    zip_name = args.zip_name or f"{weight_stem}_X_TTA_dynamic"

    submit_root = Path(args.submit_dir)
    submit_root.mkdir(parents=True, exist_ok=True)

    txt_dir = submit_root / zip_name
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)

    zip_dir = Path(args.zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("[TTA] D-FINE-X dynamic eval_spatial_size", flush=True)
    print(f"[TTA] sizes      : {TTA_SIZES}", flush=True)
    print(f"[TTA] flip       : {TTA_FLIP}", flush=True)
    print(f"[TTA] passes/img : {len(TTA_SIZES) * 2}", flush=True)
    print(f"[TTA] batch-size : {args.batch_size}", flush=True)
    print(f"[TTA] images     : {len(img_list)}", flush=True)
    print("=" * 72, flush=True)

    # ----------------------------------------------------------
    # 只读一次权重
    # ----------------------------------------------------------
    state = load_checkpoint_state(args.resume)

    # ----------------------------------------------------------
    # 构建3个不同eval_spatial_size模型
    # ----------------------------------------------------------
    models = {}
    for size in TTA_SIZES:
        models[size] = build_model_for_size(args, size, state)

    print("[info] all TTA-size models ready", flush=True)

    t0 = time.time()
    total_boxes = 0
    empty_count = 0

    batch_size = max(1, args.batch_size)
    done_count = 0

    for batch_start in range(0, len(img_list), batch_size):
        batch_paths = img_list[batch_start:batch_start + batch_size]
        batch_t0 = time.time()

        images = [Image.open(p).convert("RGB") for p in batch_paths]

        batch_all_labels = [[] for _ in batch_paths]
        batch_all_boxes = [[] for _ in batch_paths]
        batch_all_scores = [[] for _ in batch_paths]

        for size in TTA_SIZES:
            model = models[size]

            print(
                f"[batch] {batch_start + 1}-{batch_start + len(batch_paths)}/"
                f"{len(img_list)} | {size} original",
                flush=True,
            )

            labels_b, boxes_b, scores_b = infer_batch(
                model=model,
                images=images,
                size=size,
                device=args.device,
                amp=args.amp,
                flip=False,
            )

            for i in range(len(batch_paths)):
                keep = scores_b[i] >= args.conf
                batch_all_labels[i].append(labels_b[i][keep])
                batch_all_boxes[i].append(boxes_b[i][keep])
                batch_all_scores[i].append(scores_b[i][keep])

            if TTA_FLIP:
                print(
                    f"[batch] {batch_start + 1}-{batch_start + len(batch_paths)}/"
                    f"{len(img_list)} | {size} flip",
                    flush=True,
                )

                labels_b, boxes_b, scores_b = infer_batch(
                    model=model,
                    images=images,
                    size=size,
                    device=args.device,
                    amp=args.amp,
                    flip=True,
                )

                for i in range(len(batch_paths)):
                    keep = scores_b[i] >= args.conf
                    batch_all_labels[i].append(labels_b[i][keep])
                    batch_all_boxes[i].append(boxes_b[i][keep])
                    batch_all_scores[i].append(scores_b[i][keep])

        for i, p in enumerate(batch_paths):
            im = images[i]
            w, h = im.size

            labels, boxes, scores = fuse_tta_predictions(
                batch_all_labels[i],
                batch_all_boxes[i],
                batch_all_scores[i],
                width=w,
                height=h,
                iou_thr=args.wbf_iou,
                skip_box_thr=args.conf,
            )

            if scores.numel() > 0:
                order = torch.argsort(scores, descending=True)
                order = order[:args.max_det]
                labels = labels[order]
                boxes = boxes[order]
                scores = scores[order]

            lines = []

            for lb, box, sc in zip(labels, boxes, scores):
                x1, y1, x2, y2 = [float(x) for x in box.tolist()]

                x1 = max(0.0, min(x1, float(w)))
                y1 = max(0.0, min(y1, float(h)))
                x2 = max(0.0, min(x2, float(w)))
                y2 = max(0.0, min(y2, float(h)))

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                class_id = int(lb.item()) + args.cat_offset
                confidence = float(sc.item())

                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                nw = bw / w
                nh = bh / h

                lines.append(
                    f"{class_id} "
                    f"{clamp01(cx):.6f} {clamp01(cy):.6f} "
                    f"{clamp01(nw):.6f} {clamp01(nh):.6f} "
                    f"{clamp01(confidence):.6f}"
                )

            txt_path = txt_dir / f"{p.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

            total_boxes += len(lines)
            if not lines:
                empty_count += 1

            done_count += 1
            print(
                f"[done] {done_count}/{len(img_list)} | "
                f"{p.name} | boxes={len(lines)} | "
                f"batch={time.time()-batch_t0:.1f}s | "
                f"total={time.time()-t0:.1f}s",
                flush=True,
            )

    # 一张图一个TXT
    txt_files = list(txt_dir.glob("*.txt"))
    if len(txt_files) != len(img_list):
        raise RuntimeError(
            f"TXT数量({len(txt_files)}) != 图片数量({len(img_list)})"
        )

    created_zip = shutil.make_archive(
        str(zip_dir / zip_name),
        "zip",
        root_dir=txt_dir,
    )

    print("=" * 72, flush=True)
    print("[done] finished", flush=True)
    print(f"[done] images    : {len(img_list)}", flush=True)
    print(f"[done] boxes     : {total_boxes}", flush=True)
    print(f"[done] empty txt : {empty_count}", flush=True)
    print(f"[done] zip       : {created_zip}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
