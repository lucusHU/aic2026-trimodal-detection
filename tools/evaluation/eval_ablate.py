#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通道消融评测：同一个三模态权重，在 val300 上做配对比较。

用途：判断 IR / Depth 到底是帮忙还是噪声。配对比较（同模型、同图、只动一个通道）
的方差远小于 300 张图的 ±1.6pp 非配对噪声，0.3pp 量级就有意义。

  --zero-ch 3   把红外通道置 0
  --zero-ch 4   把深度通道置 0
  不加          完整 5 通道

注意：模型没见过"某通道全 0"的输入，属分布外，**必须在 val 上量过才可用于出包**。
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.ops import batched_nms  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
AUX_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_aux(folder, stem):
    for e in AUX_EXT:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return None


def load_state(path, use_ema=True):
    ck = torch.load(path, map_location='cpu')
    if use_ema and isinstance(ck.get('ema'), dict) and 'module' in ck['ema']:
        state = ck['ema']['module']
    elif 'model' in ck:
        state = ck['model']
    else:
        state = ck
    state = dict(state)
    state.pop('decoder.anchors', None)
    state.pop('decoder.valid_mask', None)
    return state


def build(args, state, size):
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg['eval_spatial_size'] = [size, size]
    for key in ('HybridEncoder', 'DFINETransformer'):
        if key in cfg.yaml_cfg and 'eval_spatial_size' in cfg.yaml_cfg[key]:
            cfg.yaml_cfg[key]['eval_spatial_size'] = [size, size]
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    base = cfg.model
    base.load_state_dict(state, strict=False)

    class M(torch.nn.Module):
        def __init__(self, model, post):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = post.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    return M(base, cfg.postprocessor).to(args.device).eval()


def load_5ch(p, aux_folders, size, zero_ch=None):
    img = Image.open(p).convert('RGB')
    w, h = img.size
    rgb = np.asarray(img.resize((size, size), Image.BILINEAR))
    chans = [torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)]
    stem = Path(p).stem
    for folder in aux_folders:
        ap = find_aux(folder, stem)
        if ap is None:
            chans.append(torch.zeros(1, size, size, dtype=chans[0].dtype))
            continue
        im = Image.open(ap)
        if im.mode != 'L':
            im = im.convert('L')
        a = np.asarray(im.resize((size, size), Image.BILINEAR))
        chans.append(torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0))
    x = torch.cat(chans, dim=0).to(torch.float32).div_(255.0).contiguous()
    if zero_ch is not None and 0 <= zero_ch < x.shape[0]:
        x[zero_ch] = 0.0
    return x, (w, h)


@torch.inference_mode()
def run(model, paths, size, device, batch, aux_folders, zero_ch):
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ts, origs = [], []
        for p in chunk:
            x, (w, h) = load_5ch(p, aux_folders, size, zero_ch)
            ts.append(x); origs.append([w, h])
        xs = torch.stack(ts, 0).to(device)
        o = torch.tensor(origs, dtype=torch.float32, device=device)
        labels, boxes, scores = model(xs, o)
        for j in range(len(chunk)):
            out.append((labels[j].cpu(), boxes[j].cpu().float(), scores[j].cpu().float()))
    return out


def to_dt(per_size, paths, img_map, iou, maxdet, conf=0.001, size_filter=None):
    sizes = [s for s in per_size if (size_filter is None or s == size_filter)]
    dets = []
    for i, p in enumerate(paths):
        lbs, bxs, scs = [], [], []
        for s in sizes:
            lb, bx, sc = per_size[s][i]
            m = sc >= conf
            if not bool(m.any()):
                continue
            lbs.append(lb[m]); bxs.append(bx[m]); scs.append(sc[m])
        if not lbs:
            continue
        lbb = torch.cat(lbs); bxb = torch.cat(bxs); scb = torch.cat(scs)
        k = batched_nms(bxb.float(), scb.float(), lbb, iou)[:maxdet]
        iid = img_map.get(p.stem)
        if iid is None:
            continue
        for idx in k:
            x1, y1, x2, y2 = bxb[idx].tolist()
            dets.append({'image_id': iid, 'category_id': int(lbb[idx]),
                         'bbox': [x1, y1, x2 - x1, y2 - y1],
                         'score': float(scb[idx])})
    return dets


def evaluate(coco, dets):
    if not dets:
        return 0.0
    dts = coco.loadRes(dets)
    ev = COCOeval(coco, dts, 'bbox')
    ev.evaluate(); ev.accumulate()
    ev.summarize()          # ← stats 由 summarize() 填充，漏了这句 ev.stats 就是空的
    return float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--aux', nargs='+', default=['dataset/tri/val/infrared', 'dataset/tri/val/depth'])
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--size', type=int, nargs='+', default=[800, 928])
    ap.add_argument('--nms-iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--zero-ch', type=int, default=None, help='3=zero IR, 4=zero Depth')
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {Path(v['file_name']).stem: k for k, v in coco.imgs.items()}
    tag = 'full-5ch' if args.zero_ch is None else ('zero-IR(ch3)' if args.zero_ch == 3 else 'zero-Depth(ch4)')
    print('[info] %d images | mode=%s | sizes=%s | nms=%.2f md=%d'
          % (len(paths), tag, sorted(set(args.size)), args.nms_iou, args.max_det), flush=True)

    state = load_state(args.resume, use_ema=not args.no_ema)
    t0 = time.time()
    per_size = {}
    for s in sorted(set(args.size)):
        model = build(args, state, s)
        per_size[s] = run(model, paths, s, args.device, args.batch_size, args.aux, args.zero_ch)
        print('[fwd] size=%d  %.0fs' % (s, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    m = evaluate(coco, to_dt(per_size, paths, img_map, args.nms_iou, args.max_det))
    print('[RESULT] %-13s  mAP@0.5:0.95 = %.4f   (%.0fs)'
          % (tag, m, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
