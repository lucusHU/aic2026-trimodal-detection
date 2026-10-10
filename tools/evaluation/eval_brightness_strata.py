#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""亮度分层配对 Δ：回答"IR/Depth 到底有没有价值"。

动机（WorkBuddy 提出）：
  IR 的价值可能是**稀疏的**——只在暗光/夜间样本上大。
  在 300 张全量上平均，一个只在 20% 样本上 +5pp 的增益会被稀释成 +1pp，
  再被 ±1.6pp 的噪声吃掉。**平均为 0 ≠ 处处为 0。**

做法：
  1. 算每张 val 图的 RGB 平均亮度，按分位数分成 N 层
  2. 对「完整 / IR置0 / Depth置0」三种输入分别前向（同一权重）
  3. 在**每一层内**单独算 mAP，给出配对 Δ

先在小尺度（800 单尺度）做快速筛，若某层出现显著 Δ 再用完整尺度集复核。
"""

import argparse
import contextlib
import io
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
        a = np.asarray(Image.open(ap))
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.dtype == np.uint16:
            a = np.clip(a.astype(np.float32) / 80.0, 0, 255).astype(np.uint8)
        im2 = Image.fromarray(a)
        if im2.size != (size, size):
            im2 = im2.resize((size, size), Image.BILINEAR)
        chans.append(torch.from_numpy(np.ascontiguousarray(np.asarray(im2))).unsqueeze(0))
    x = torch.cat(chans, 0).to(torch.float32).div_(255.0).contiguous()
    if zero_ch is not None and 0 <= zero_ch < x.shape[0]:
        x[zero_ch] = 0.0
    return x, (w, h)


@torch.inference_mode()
def forward_all(model, paths, size, device, batch, aux_folders, zero_ch):
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


def per_image_dets(cache, sizes, paths, img_map, iou, maxdet, conf):
    """返回 {image_id: dets}，用于按任意子集重算 mAP。"""
    res = {}
    for i, p in enumerate(paths):
        lbs, bxs, scs = [], [], []
        for size in sizes:
            lb, bx, sc = cache[size][i]
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
        res[iid] = [{'image_id': iid, 'category_id': int(lbb[j]),
                     'bbox': [bxb[j][0].item(), bxb[j][1].item(),
                              (bxb[j][2] - bxb[j][0]).item(), (bxb[j][3] - bxb[j][1]).item()],
                     'score': float(scb[j])} for j in k]
    return res


def eval_subset(coco, per_img, ids):
    dets = []
    for i in ids:
        dets.extend(per_img.get(i, []))
    if not dets:
        return 0.0
    dts = coco.loadRes(dets)
    ev = COCOeval(coco, dts, 'bbox')
    ev.params.imgIds = sorted(ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--aux', nargs='+', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:4')
    ap.add_argument('--sizes', type=int, nargs='+', default=[800])
    ap.add_argument('--strata', type=int, default=4)
    ap.add_argument('--nms-iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    stem2id = {Path(v['file_name']).stem: k for k, v in coco.imgs.items()}
    img_map = {p.stem: stem2id[p.stem] for p in paths if p.stem in stem2id}

    # 亮度
    bright = {}
    for p in paths:
        with Image.open(p) as im:
            g = np.asarray(im.convert('L').resize((64, 64), Image.BILINEAR), dtype=np.float32)
        bright[img_map[p.stem]] = float(g.mean())
    ids_sorted = sorted(img_map.values(), key=lambda i: bright[i])
    n = len(ids_sorted)
    strata = [ids_sorted[i * n // args.strata:(i + 1) * n // args.strata]
              for i in range(args.strata)]
    print('[info] val=%d strata=%d 亮度分位=' %
          (n, args.strata) +
          ' | '.join('%.1f~%.1f' % (bright[s[0]], bright[s[-1]]) for s in strata), flush=True)

    state = load_state(args.resume, use_ema=True)
    modes = [('full', None), ('zero-IR(ch3)', 3), ('zero-Depth(ch4)', 4)]
    t0 = time.time()
    tables = {}
    for tag, zc in modes:
        cache = {}
        for s in sorted(set(args.sizes)):
            model = build(args, state, s)
            cache[s] = forward_all(model, paths, s, args.device, args.batch_size, args.aux, zc)
            print('[fwd] %-14s size=%d  %.0fs' % (tag, s, time.time() - t0), flush=True)
            del model
            torch.cuda.empty_cache()
        per_img = per_image_dets(cache, args.sizes, paths, img_map,
                                 args.nms_iou, args.max_det, args.conf)
        tables[tag] = [eval_subset(coco, per_img, s) for s in strata]
        tables[tag + '|ALL'] = eval_subset(coco, per_img, ids_sorted)
        print('[RESULT] %-14s 各层 mAP = %s' %
              (tag, ' '.join('%.4f' % v for v in tables[tag])), flush=True)

    print()
    print('=== 亮度分层配对 Δ（同一批图、同一权重，只改输入通道）===')
    hdr = '%-16s' % '模式'
    for i in range(args.strata):
        hdr += '  层%d(暗→亮)' % (i + 1)
    hdr += '    FULL-ALL'
    print(hdr)
    for tag in ['full', 'zero-IR(ch3)', 'zero-Depth(ch4)']:
        row = '%-16s' % tag
        for v in tables[tag]:
            row += '  %9.4f' % v
        row += '  %9.4f' % tables[tag + '|ALL']
        print(row)
    print()
    for tag in ['zero-IR(ch3)', 'zero-Depth(ch4)']:
        row = '%-16s' % ('Δ ' + tag)
        for a, b in zip(tables[tag], tables['full']):
            row += '  %+9.4f' % (a - b)
        row += '  %+9.4f' % (tables[tag + '|ALL'] - tables['full|ALL'])
        print(row)
    print()
    print('判读：若"最暗层"的 Δ 显著为负（置零后掉分）= 该模态在暗光下有真实增量信息；')
    print('      若所有层的 Δ 都 ≈ 0 = 该模态在本数据集上确实没被用上。')


if __name__ == '__main__':
    main()
