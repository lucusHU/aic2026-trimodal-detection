#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tri-modal val evaluation: one forward pass, sweep NMS/size combos, report COCO mAP.

Uses pycocotools COCOeval defaults (maxDets=[1,10,100]) -> same protocol as the
official metric (mAP@0.5:0.95).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.ops import batched_nms

def _iou1(a, b):
    ix1, iy1 = torch.max(a[0], b[0]), torch.max(a[1], b[1])
    ix2, iy2 = torch.min(a[2], b[2]), torch.min(a[3], b[3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    ua = ((a[2]-a[0]).clamp(min=0)*(a[3]-a[1]).clamp(min=0)
          + (b[2]-b[0]).clamp(min=0)*(b[3]-b[1]).clamp(min=0) - inter)
    return inter / ua.clamp(min=1e-6)


@torch.no_grad()
def wbf_merge(per_scale, num_classes=12, iou_thr=0.65, skip_thr=1e-3, max_det=100):
    dev = per_scale[0][1].device
    ldt = per_scale[0][0].dtype
    out_l, out_b, out_s = [], [], []
    for c in range(num_classes):
        bs, ss = [], []
        for lb, bx, sc in per_scale:
            m = (lb == c) & (sc >= skip_thr)
            if m.any():
                bs.append(bx[m].float().to(dev)); ss.append(sc[m].float().to(dev))
        if not bs:
            continue
        b, s = torch.cat(bs, 0), torch.cat(ss, 0)
        o = torch.argsort(s, descending=True); b, s = b[o], s[o]
        csum = b[0] * s[0]; cw = s[0].clone(); cb = b[0].clone(); cscore = s[0].clone()
        fb, fs = [], []
        for k in range(1, b.shape[0]):
            bk, sk = b[k], s[k]
            if _iou1(cb, bk) > iou_thr:
                csum = csum + bk * sk; cw = cw + sk; cb = csum / cw
                cscore = torch.maximum(cscore, sk)
            else:
                fb.append(cb); fs.append(cscore)
                csum, cw, cb, cscore = bk * sk, sk.clone(), bk.clone(), sk.clone()
        fb.append(cb); fs.append(cscore)
        out_b.append(torch.stack(fb)); out_s.append(torch.stack(fs))
        out_l.append(torch.full((len(fb),), c, dtype=ldt, device=dev))
    if not out_b:
        return (torch.zeros(0, dtype=ldt, device=dev), torch.zeros(0, 4, device=dev),
                torch.zeros(0, device=dev))
    lb, bx, sc = torch.cat(out_l), torch.cat(out_b), torch.cat(out_s)
    keep = torch.argsort(sc, descending=True)[:max_det]
    return lb[keep], bx[keep], sc[keep]


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}
AUX_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_aux(folder, stem):
    if not folder:
        return None
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
    r = base.load_state_dict(state, strict=False)
    bad = [k for k in r.missing_keys if k not in ('decoder.anchors', 'decoder.valid_mask')]
    if bad:
        print('[warn] missing %d keys e.g. %s' % (len(bad), bad[:3]), flush=True)

    class M(torch.nn.Module):
        def __init__(self, model, post):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = post.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    return M(base, cfg.postprocessor).to(args.device).eval()


def load_5ch(p, aux_folders, size):
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
    return x, (w, h)


@torch.inference_mode()
def run(model, paths, size, device, batch, aux_folders):
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ts, origs = [], []
        for p in chunk:
            x, (w, h) = load_5ch(p, aux_folders, size)
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
        lbb, bxb, scb = wbf_merge(list(zip(lbs, bxs, scs)), num_classes=12,
                                  iou_thr=0.65, skip_thr=conf, max_det=maxdet)
        k = torch.arange(lbb.shape[0])
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
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--aux', nargs='*', default=['dataset/tri/val/infrared', 'dataset/tri/val/depth'])
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:4')
    ap.add_argument('--size', type=int, nargs='+', default=[800])
    ap.add_argument('--nms-iou', type=float, nargs='+', default=[0.7])
    ap.add_argument('--max-det', type=int, nargs='+', default=[100])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {}
    for iid, info in coco.imgs.items():
        img_map[Path(info['file_name']).stem] = iid
    print('[info] val images %d, ann images %d, matched %d'
          % (len(paths), len(img_map), sum(1 for p in paths if p.stem in img_map)), flush=True)
    if args.aux:
        hit = sum(1 for p in paths if find_aux(args.aux[0], p.stem) is not None)
        print('[info] aux[0] hit %d/%d' % (hit, len(paths)), flush=True)

    state = load_state(args.resume, use_ema=not args.no_ema)
    t0 = time.time()
    per_size = {}
    for s in sorted(set(args.size)):
        model = build(args, state, s)
        per_size[s] = run(model, paths, s, args.device, args.batch_size, args.aux)
        print('[fwd] size=%d  %.0fs' % (s, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    print('=== sweep ===', flush=True)
    best = (-1, None)
    for s in sorted(set(args.size)):
        for iou in args.nms_iou:
            for md in args.max_det:
                dets = to_dt(per_size, paths, img_map, iou, md, size_filter=s)
                buf = sys.stdout
                if args.quiet:
                    sys.stdout = open(os.devnull, 'w')
                m = evaluate(coco, dets)
                sys.stdout = buf
                print('  size=%d iou=%.2f md=%d  mAP=%.4f' % (s, iou, md, m), flush=True)
                if m > best[0]:
                    best = (m, (s, iou, md))
    if len(set(args.size)) > 1:
        for iou in args.nms_iou:
            for md in args.max_det:
                dets = to_dt(per_size, paths, img_map, iou, md)
                buf = sys.stdout
                if args.quiet:
                    sys.stdout = open(os.devnull, 'w')
                m = evaluate(coco, dets)
                sys.stdout = buf
                print('  ALL-scales iou=%.2f md=%d  mAP=%.4f' % (iou, md, m), flush=True)
                if m > best[0]:
                    best = (m, ('all', iou, md))
    print('[BEST] mAP=%.4f  cfg=%s' % (best[0], best[1]), flush=True)
    print('[done] %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
