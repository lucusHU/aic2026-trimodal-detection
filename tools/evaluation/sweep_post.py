#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""后处理扫描：一次前向缓存原始检测，离线秒级扫描
  - 分类别置信度阈值（city之光实证手段）
  - NMS IoU / max_det
  - Soft-NMS (gaussian)
并做 foldA/foldB 分半校验，防止在 val-300 噪声上过拟合。

合规说明：全部为**单模型**推理后处理，不涉及任何多模型集成/伪标签。
"""
import argparse, copy, json, os, sys, time, pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tri import load_state, build, run, find_aux, IMG_EXT  # noqa

NCLS = 12


# ---------------------------- post-processing ----------------------------
def _nms(boxes, scores, labels, iou, maxdet, soft_sigma=None, soft_thresh=0.0001):
    """batched NMS (or gaussian soft-NMS per class). boxes: Nx4 xyxy."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if soft_sigma is None:
        from torchvision.ops import batched_nms
        return batched_nms(boxes.float(), scores.float(), labels, iou)[:maxdet]

    # gaussian soft-nms, class-wise
    keep_all = []
    for c in labels.unique():
        idxs = torch.nonzero(labels == c).flatten()
        b = boxes[idxs].float().numpy()
        s = scores[idxs].float().numpy()
        order = s.argsort()[::-1]
        b, s = b[order], s[order]
        n = len(s)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        keep = []
        while len(s) > 0 and len(keep) < maxdet:
            i = int(np.argmax(s))
            keep.append(i)
            if len(keep) >= maxdet or len(s) == 1:
                break
            cur = b[i]
            xx1 = np.maximum(cur[0], x1); yy1 = np.maximum(cur[1], y1)
            xx2 = np.minimum(cur[2], x2); yy2 = np.minimum(cur[3], y2)
            iw = np.maximum(0.0, xx2 - xx1); ih = np.maximum(0.0, yy2 - yy1)
            inter = iw * ih
            ov = inter / np.maximum(1e-6, areas[i] + areas - inter)
            decay = np.exp(-(ov * ov) / soft_sigma)
            s = s * decay
            s[i] = -1.0
            alive = s >= soft_thresh
            if not alive.any():
                break
            b, s, areas = b[alive], s[alive], areas[alive]
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        keep_all.append(idxs[torch.as_tensor(np.array(keep, dtype=np.int64))])
    if not keep_all:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(keep_all)[:maxdet]


def make_dets(per_size, paths, img_map, sizes, iou, maxdet, confs, soft_sigma=None):
    conf_t = torch.as_tensor(confs, dtype=torch.float32)
    dets = []
    for i, p in enumerate(paths):
        lbs, bxs, scs = [], [], []
        for s in sizes:
            lb, bx, sc = per_size[s][i]
            if len(lb) == 0:
                continue
            km = sc >= conf_t[lb.long()]
            if not bool(km.any()):
                continue
            lbs.append(lb[km]); bxs.append(bx[km]); scs.append(sc[km])
        if not lbs:
            continue
        lbb = torch.cat(lbs); bxb = torch.cat(bxs).float(); scb = torch.cat(scs).float()
        k = _nms(bxb, scb, lbb, iou, maxdet, soft_sigma)
        iid = img_map.get(p.stem)
        if iid is None:
            continue
        for idx in k:
            x1, y1, x2, y2 = bxb[idx].tolist()
            dets.append({'image_id': iid, 'category_id': int(lbb[idx]),
                         'bbox': [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                         'score': float(scb[idx])})
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
    sub = COCO()
    sub.dataset = d
    sub.createIndex()
    return sub


# ---------------------------- sweep ----------------------------
def coord_descent(make, coco_tune, init_conf, cands, rounds=2, verbose=True):
    confs = list(init_conf)
    best = fast_map(coco_tune, make(confs))
    for r in range(rounds):
        improved = False
        for c in range(NCLS):
            cur = confs[c]
            loc_b, loc_t = best, cur
            for t in cands:
                if t == cur:
                    continue
                confs[c] = t
                m = fast_map(coco_tune, make(confs))
                if m > loc_b + 1e-6:
                    loc_b, loc_t = m, t
            confs[c] = loc_t if loc_t != cur or loc_b > best else cur
            if loc_b > best + 1e-6:
                best = loc_b
                improved = True
        if verbose:
            print('[cd] round %d  tune_mAP=%.4f  confs=%s' % (r, best, [round(x, 3) for x in confs]), flush=True)
        if not improved:
            break
    return confs, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--aux', nargs='*', default=['dataset/tri/val/infrared', 'dataset/tri/val/depth'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--size', type=int, nargs='+', default=[800, 928])
    ap.add_argument('--nms-iou', type=float, nargs='+', default=[0.6, 0.7, 0.8])
    ap.add_argument('--max-det', type=int, nargs='+', default=[100])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--cache', default='cache_dets.pkl')
    ap.add_argument('--out', default='sweep_post_results.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--soft', type=float, nargs='*', default=[])
    ap.add_argument('--rounds', type=int, default=2)
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {Path(info['file_name']).stem: iid for iid, info in coco.imgs.items()}
    print('[info] val imgs %d matched %d' % (len(paths), sum(1 for p in paths if p.stem in img_map)), flush=True)

    # ---- forward ----
    if os.path.exists(args.cache):
        per_size = pickle.load(open(args.cache, 'rb'))
        print('[info] loaded cache %s  sizes=%s' % (args.cache, sorted(per_size)), flush=True)
    else:
        state = load_state(args.resume, use_ema=True)
        per_size = {}
        t0 = time.time()
        for s in sorted(set(args.size)):
            model = build(args, state, s)
            per_size[s] = run(model, paths, s, args.device, args.batch_size, args.aux)
            print('[fwd] size=%d %.0fs' % (s, time.time() - t0), flush=True)
            del model
            torch.cuda.empty_cache()
        pickle.dump(per_size, open(args.cache, 'wb'))

    sizes = sorted(per_size.keys())
    stems_all = [p.stem for p in paths if p.stem in img_map]
    stems_sorted = sorted(stems_all)
    foldA = set(stems_sorted[0::2])   # 交错切分，避免同序列聚在一侧
    foldB = set(stems_sorted[1::2])
    cocoA = subset_coco(coco, foldA)
    cocoB = subset_coco(coco, foldB)

    cands = [0.001, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50]

    results = []
    soft_list = [None] + list(args.soft)
    for soft in soft_list:
        for iou in args.nms_iou:
            for md in args.max_det:
                def make(confs, _soft=soft, _iou=iou, _md=md):
                    return make_dets(per_size, paths, img_map, sizes, _iou, _md, confs, _soft)

                base = fast_map(coco, make([0.001] * NCLS))
                baseA = fast_map(cocoA, make([0.001] * NCLS))
                baseB = fast_map(cocoB, make([0.001] * NCLS))

                confs, tuneA = coord_descent(make, cocoA, [0.001] * NCLS, cands,
                                             rounds=args.rounds, verbose=False)
                full = fast_map(coco, make(confs))
                holdB = fast_map(cocoB, make(confs))
                tag = 'soft=%.2f' % soft if soft else 'nms'
                print('[%s iou=%.2f md=%d] base=%.4f -> tuned=%.4f (foldA %.4f->%.4f, foldB %.4f->%.4f)'
                      % (tag, iou, md, base, full, baseA, tuneA, baseB, holdB), flush=True)
                print('    confs=%s' % [round(x, 3) for x in confs], flush=True)
                results.append((full, holdB - baseB, dict(soft=soft, iou=iou, md=md,
                                                          confs=[round(x, 4) for x in confs],
                                                          base=round(base, 4), tuned=round(full, 4),
                                                          foldB_gain=round(holdB - baseB, 4))))

    results.sort(key=lambda x: -x[0])
    print('\n=== RANK (by full val mAP) ===', flush=True)
    for r in results[:10]:
        print('  mAP=%.4f holdB_gain=%+.4f %s' % (r[0], r[1], r[2]), flush=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
