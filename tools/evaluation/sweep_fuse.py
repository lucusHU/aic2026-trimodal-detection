#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep the tri-modal fusion residual gain lambda at inference time.

fusion:  out = p + LAMBDA * (proj(a) * 2*sigmoid(gate(a)))

Rationale: the zero-init gated residual only reached |w|~0.003 after 24 epochs,
so the aux (IR+depth) branch is nearly muted. Scaling its residual at inference
is a single-model inference-parameter search (allowed), not ensembling.
"""
import argparse, json, os, sys, time, types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.ops import batched_nms
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


def prep(p, aux_folders, size):
    img = Image.open(p).convert('RGB')
    w, h = img.size
    rgb = np.asarray(img.resize((size, size), Image.BILINEAR))
    chans = [np.ascontiguousarray(rgb).transpose(2, 0, 1)]
    stem = Path(p).stem
    for folder in aux_folders:
        ap = find_aux(folder, stem)
        if ap is None:
            chans.append(np.zeros((1, size, size), dtype=np.uint8))
            continue
        im = Image.open(ap)
        if im.mode != 'L':
            im = im.convert('L')
        chans.append(np.asarray(im.resize((size, size), Image.BILINEAR))[None])
    x = np.concatenate(chans, axis=0)  # uint8, C,H,W
    return x, (w, h)


def set_lambda(model, lam):
    n = 0
    for m in model.modules():
        if m.__class__.__name__ == 'ZeroGatedFuse':
            proj = m.proj
            gate = m.gate

            def fwd(p, a, _proj=proj, _gate=gate, _lam=lam):
                return p + _lam * (_proj(a) * (2.0 * torch.sigmoid(_gate(a))))

            m.forward = types.MethodType(fwd, m)
            n += 1
    return n


@torch.inference_mode()
def run(model, prep_data, device, batch):
    out = []
    N = len(prep_data)
    for i in range(0, N, batch):
        chunk = prep_data[i:i + batch]
        xs = torch.stack([torch.from_numpy(np.ascontiguousarray(c[0])) for c in chunk],
                         0).to(device).float().div_(255.0).contiguous()
        o = torch.tensor([c[1] for c in chunk], dtype=torch.float32, device=device)
        labels, boxes, scores = model(xs, o)
        for j in range(len(chunk)):
            out.append((labels[j].cpu(), boxes[j].cpu().float(), scores[j].cpu().float()))
    return out


def to_dt(preds, paths, img_map, iou, maxdet, conf=0.001):
    dets = []
    for i, p in enumerate(paths):
        lb, bx, sc = preds[i]
        m = sc >= conf
        if not bool(m.any()):
            continue
        lb, bx, sc = lb[m], bx[m], sc[m]
        k = batched_nms(bx.float(), sc.float(), lb, iou)[:maxdet]
        iid = img_map.get(p.stem)
        if iid is None:
            continue
        for idx in k:
            x1, y1, x2, y2 = bx[idx].tolist()
            dets.append({'image_id': iid, 'category_id': int(lb[idx]),
                         'bbox': [x1, y1, x2 - x1, y2 - y1],
                         'score': float(sc[idx])})
    return dets


def evaluate(coco, dets, quiet=True):
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
    ap.add_argument('--device', default='cuda:5')
    ap.add_argument('--size', type=int, default=800)
    ap.add_argument('--nms-iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lambdas', type=float, nargs='+', default=[1.0, 2.0, 4.0, 8.0, 16.0])
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {}
    for iid, info in coco.imgs.items():
        img_map[Path(info['file_name']).stem] = iid
    print('[info] val %d, matched %d' % (len(paths), sum(1 for p in paths if p.stem in img_map)), flush=True)

    t0 = time.time()
    data = [prep(p, args.aux, args.size) for p in paths]
    print('[prep] %.0fs' % (time.time() - t0), flush=True)

    state = load_state(args.resume, use_ema=True)
    model = build(args, state, args.size)
    nf = set_lambda(model, 1.0)
    print('[info] ZeroGatedFuse modules patched: %d' % nf, flush=True)

    import io, contextlib
    best = (-1, None)
    for lam in args.lambdas:
        set_lambda(model, lam)
        preds = run(model, data, args.device, args.batch_size)
        buf = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            m = evaluate(coco, to_dt(preds, paths, img_map, args.nms_iou, args.max_det))
        finally:
            sys.stdout = buf
        print('  lambda=%-6.2g mAP=%.4f  (%.0fs)' % (lam, m, time.time() - t0), flush=True)
        if m > best[0]:
            best = (m, lam)
    print('[BEST] mAP=%.4f lambda=%s' % (best[0], best[1]), flush=True)


if __name__ == '__main__':
    main()
