#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多尺度框投票合并（Weighted Box Fusion 思路，但仅限**同一模型**的多尺度 TTA 输出）。

合规：单模型、单权重的多尺度推理增强，不属于"多个不同结构/训练阶段模型的集成"。
原理：对 800/928 两个尺度检出的同一目标簇做 score 加权坐标平均 -> 更紧的框 -> 提 AP75。
复用 sweep_post.py 缓存的原始检测，纯 CPU，秒级扫描。
"""
import argparse, json, os, pickle, sys, copy
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
NCLS = 12


def iou_one(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, x2 - x1); ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    a2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(1e-6, a1 + a2 - inter)


def wbf(boxes, scores, labels, iou_thr, score_rule, maxdet):
    out_b, out_s, out_l = [], [], []
    for c in labels.unique():
        m = labels == c
        b = boxes[m].float().numpy()
        s = scores[m].float().numpy()
        if len(s) == 0:
            continue
        order = np.argsort(-s)
        b, s = b[order], s[order]
        used = np.zeros(len(s), dtype=bool)
        for i in range(len(s)):
            if used[i]:
                continue
            grp = (~used) & (iou_one(b[i], b) > iou_thr)
            used |= grp
            idx = np.nonzero(grp)[0]
            w = s[idx]
            nb = (b[idx] * w[:, None]).sum(0) / max(1e-6, w.sum())
            if score_rule == 'max':
                ns = w.max()
            elif score_rule == 'sum':
                ns = w.sum()
            elif score_rule == 'maxn':      # WBF 原版风格：max * n
                ns = w.max() * len(idx)
            else:
                ns = w.mean()
            out_b.append(nb); out_s.append(float(ns)); out_l.append(int(c))
    if not out_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.stack(out_b); S = np.array(out_s); L = np.array(out_l)
    if len(S) > maxdet:
        top = np.argsort(-S)[:maxdet]
        B, S, L = B[top], S[top], L[top]
    return B, S, L


def make_dets(per_size, paths, img_map, sizes, iou_thr, score_rule, maxdet, conf=0.001):
    dets = []
    for i, p in enumerate(paths):
        lbs, bxs, scs = [], [], []
        for s in sizes:
            lb, bx, sc = per_size[s][i]
            if len(lb) == 0:
                continue
            km = sc >= conf
            if not bool(km.any()):
                continue
            lbs.append(lb[km]); bxs.append(bx[km]); scs.append(sc[km])
        if not lbs:
            continue
        lbb = torch.cat(lbs); bxb = torch.cat(bxs).float(); scb = torch.cat(scs).float()
        B, S, L = wbf(bxb, scb, lbb, iou_thr, score_rule, maxdet)
        iid = img_map.get(p.stem)
        if iid is None:
            continue
        for j in range(len(S)):
            x1, y1, x2, y2 = B[j].tolist()
            dets.append({'image_id': iid, 'category_id': int(L[j]),
                         'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                         'score': float(S[j])})
    return dets


def fast_map(coco, dets):
    ids = set(coco.imgs.keys())
    dets = [d for d in dets if d['image_id'] in ids]
    if not dets:
        return 0.0
    ev = COCOeval(coco, coco.loadRes(dets), 'bbox')
    ev.evaluate(); ev.accumulate()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return float(ev.stats[0])


def subset_coco(coco, stems):
    d = copy.deepcopy(coco.dataset)
    keep = set(iid for iid, info in coco.imgs.items()
               if Path(info['file_name']).stem in stems)
    d['images'] = [im for im in d['images'] if im['id'] in keep]
    d['annotations'] = [a for a in d['annotations'] if a['image_id'] in keep]
    sub = COCO(); sub.dataset = d; sub.createIndex()
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--out', default='vote_results.json')
    args = ap.parse_args()

    per_size = pickle.load(open(args.cache, 'rb'))
    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {Path(info['file_name']).stem: iid for iid, info in coco.imgs.items()}
    sizes = sorted(per_size.keys())
    print('[info] cache sizes %s, imgs %d' % (sizes, len(paths)), flush=True)

    stems = sorted([p.stem for p in paths if p.stem in img_map])
    cocoA = subset_coco(coco, set(stems[0::2]))
    cocoB = subset_coco(coco, set(stems[1::2]))

    # baseline: plain batched NMS (与已提交配置同)
    from torchvision.ops import batched_nms
    def make_nms(iou, maxdet=100, conf=0.001):
        dets = []
        for i, p in enumerate(paths):
            lbs, bxs, scs = [], [], []
            for s in sizes:
                lb, bx, sc = per_size[s][i]
                if len(lb) == 0:
                    continue
                km = sc >= conf
                if not bool(km.any()):
                    continue
                lbs.append(lb[km]); bxs.append(bx[km]); scs.append(sc[km])
            if not lbs:
                continue
            lbb = torch.cat(lbs); bxb = torch.cat(bxs).float(); scb = torch.cat(scs).float()
            k = batched_nms(bxb, scb, lbb, iou)[:maxdet]
            iid = img_map.get(p.stem)
            if iid is None:
                continue
            for idx in k:
                x1, y1, x2, y2 = bxb[idx].tolist()
                dets.append({'image_id': iid, 'category_id': int(lbb[idx]),
                             'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                             'score': float(scb[idx])})
        return dets

    base = fast_map(coco, make_nms(0.7))
    print('[baseline NMS iou=0.7] mAP=%.4f' % base, flush=True)

    res = []
    for thr in [0.5, 0.6, 0.7, 0.8]:
        for rule in ['avg', 'max', 'sum', 'maxn']:
            d = make_dets(per_size, paths, img_map, sizes, thr, rule, 100)
            m = fast_map(coco, d)
            mA = fast_map(cocoA, d); mB = fast_map(cocoB, d)
            bA = fast_map(cocoA, make_nms(0.7)); bB = fast_map(cocoB, make_nms(0.7))
            print('[wbf thr=%.2f rule=%s] mAP=%.4f  (foldA %.4f vs base %.4f | foldB %.4f vs base %.4f)'
                  % (thr, rule, m, mA, bA, mB, bB), flush=True)
            res.append((m, dict(thr=thr, rule=rule, map=round(m, 4),
                                foldA=round(mA, 4), foldB=round(mB, 4),
                                foldA_base=round(bA, 4), foldB_base=round(bB, 4))))
    res.sort(key=lambda x: -x[0])
    print('\n=== RANK ===', flush=True)
    for r in res[:8]:
        print('  mAP=%.4f %s' % (r[0], r[1]), flush=True)
    json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
