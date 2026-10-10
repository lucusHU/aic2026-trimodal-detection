#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三模态最后一轮：一次前向缓存，多尺度组合 + 模态置零，全部在 val300 上评。

前向只做一次（每个尺度一遍），之后所有组合/置零变体都复用缓存 —— 40 分钟内跑得完。

用法:
  python eval_tri_multi.py -c configs/tri_final3.yml -r output/tri_final3/best_stg1.pth \
    --aux <ir> <depth> --sizes 800 928 1024 1088 1152 \
    --combos 800,928 800,1024 800,1088 800,1024,1088 800,1024,1088,1152 \
    --zero-ch 3 4
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


def build_dets(cache, combo, paths, img_map, iou, maxdet, conf):
    dets = []
    for i, p in enumerate(paths):
        lbs, bxs, scs = [], [], []
        for size in combo:
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
        for idx in k:
            x1, y1, x2, y2 = bxb[idx].tolist()
            dets.append({'image_id': iid, 'category_id': int(lbb[idx]),
                         'bbox': [x1, y1, x2 - x1, y2 - y1],
                         'score': float(scb[idx])})
    return dets


def eval_dets(coco, dets):
    if not dets:
        return 0.0
    dts = coco.loadRes(dets)
    ev = COCOeval(coco, dts, 'bbox')
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
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--sizes', type=int, nargs='+', default=[800, 928, 1024, 1088, 1152])
    ap.add_argument('--combos', nargs='+',
                    default=['800,928', '800,1024', '800,1088', '800,1024,1088',
                             '800,1024,1088,1152'])
    ap.add_argument('--zero-ch', type=int, nargs='*', default=[3, 4])
    ap.add_argument('--nms-iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {Path(v['file_name']).stem: k for k, v in coco.imgs.items()}
    print('[info] val=%d sizes=%s combos=%d' %
          (len(paths), sorted(set(args.sizes)), len(args.combos)), flush=True)

    state = load_state(args.resume, use_ema=True)
    t0 = time.time()

    def mk(s):
        return tuple(int(x) for x in s.split(','))

    combos = [mk(c) for c in args.combos]
    need = sorted({s for c in combos for s in c})

    def run_set(zero_ch, tag):
        cache = {}
        for s in need:
            model = build(args, state, s)
            cache[s] = forward_all(model, paths, s, args.device, args.batch_size, args.aux, zero_ch)
            print('[fwd] %-14s size=%d  %.0fs' % (tag, s, time.time() - t0), flush=True)
            del model
            torch.cuda.empty_cache()
        res = {}
        for combo in combos:
            m = eval_dets(coco, build_dets(cache, combo, paths, img_map,
                                           args.nms_iou, args.max_det, args.conf))
            res[combo] = m
            print('[RESULT] %-14s %-22s mAP=%.4f' %
                  (tag, '+'.join(map(str, combo)), m), flush=True)
        return res

    base = run_set(None, 'full-5ch')

    # 模态置零：只在最强的那个组合上测（这里固定为含 1024+1088 的最长组合）
    best_combo = max(base, key=lambda k: base[k])
    print('[info] best combo so far = %s (%.4f)' % ('+'.join(map(str, best_combo)), base[best_combo]),
          flush=True)

    for z in (args.zero_ch or []):
        tag = 'zero-ch%d' % z
        cache = {}
        for s in best_combo:
            model = build(args, state, s)
            cache[s] = forward_all(model, paths, s, args.device, args.batch_size, args.aux, z)
            print('[fwd] %-14s size=%d  %.0fs' % (tag, s, time.time() - t0), flush=True)
            del model
            torch.cuda.empty_cache()
        m = eval_dets(coco, build_dets(cache, best_combo, paths, img_map,
                                       args.nms_iou, args.max_det, args.conf))
        print('[RESULT] %-14s %-22s mAP=%.4f' % (tag, '+'.join(map(str, best_combo)), m), flush=True)

    print()
    print('=== 汇总（按 mAP 降序）===')
    for k, v in sorted(base.items(), key=lambda kv: -kv[1]):
        print('  %-24s %.4f' % ('+'.join(map(str, k)), v))


if __name__ == '__main__':
    main()
