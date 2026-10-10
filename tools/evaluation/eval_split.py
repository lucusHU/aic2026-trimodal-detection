#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多尺度 TTA 的「折半复现 + 配对 bootstrap」稳健性检验。

为什么做这个：我们从 ~21 个尺度组合里挑出了 val300 上的最大值，存在 winner's curse。
单看全集 mAP 无法回答"这是真提升还是挑出来的噪声"。

做法：
  1. 一次性前向所有需要的尺度；
  2. 对每个候选组合，在**全集**上算 mAP；
  3. 再做 K 次随机折半（150/150），每次分别算基线组合与候选组合的 mAP，得到配对 Δ；
  4. 报告 Δ 的均值、以及 Δ>0 的折半比例。

如果候选组合在大多数折半里都为正 → 提升是可复现的，不是单次侥幸。
"""

import argparse
import contextlib
import io
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.ops import batched_nms  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


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


@torch.inference_mode()
def forward_all(model, paths, size, device, batch):
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ts, origs = [], []
        for p in chunk:
            img = Image.open(p).convert('RGB')
            ts.append(tf(img))
            origs.append([img.size[0], img.size[1]])
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


def eval_dets(coco, dets, img_ids):
    if not dets:
        return 0.0
    dts = coco.loadRes(dets)
    ev = COCOeval(coco, dts, 'bbox')
    ev.params.imgIds = sorted(img_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--sizes', type=int, nargs='+', default=[800, 928, 1024, 1088, 1152])
    ap.add_argument('--base-combo', default='800,928')
    ap.add_argument('--combos', nargs='+',
                    default=['800,1024,1088', '800,928,1024,1088', '800,1024,1088,1152'])
    ap.add_argument('--nms-iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--folds', type=int, default=10, help='随机折半次数')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXT])
    coco = COCO(args.ann)
    img_map = {Path(v['file_name']).stem: k for k, v in coco.imgs.items()}
    all_ids = [img_map[p.stem] for p in paths if p.stem in img_map]
    print('[info] val=%d  sizes=%s  nms=%.2f md=%d' %
          (len(paths), sorted(set(args.sizes)), args.nms_iou, args.max_det), flush=True)

    state = load_state(args.resume, use_ema=True)
    t0 = time.time()
    cache = {}
    for s in sorted(set(args.sizes)):
        model = build(args, state, s)
        cache[s] = forward_all(model, paths, s, args.device, args.batch_size)
        print('[fwd] size=%d  %.0fs' % (s, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    def mk(s):
        return tuple(int(x) for x in s.split(','))

    groups = {'BASE=' + args.base_combo: mk(args.base_combo)}
    for c in args.combos:
        groups[c] = mk(c)

    # ---- 全集 ----
    full = {}
    for name, combo in groups.items():
        d = build_dets(cache, combo, paths, img_map, args.nms_iou, args.max_det, args.conf)
        full[name] = eval_dets(coco, d, all_ids)
        print('[full] %-22s mAP=%.4f' % (name, full[name]), flush=True)

    # ---- 随机折半配对 ----
    rng = random.Random(args.seed)
    deltas = {name: [] for name in groups if name not in ('BASE=' + args.base_combo,)}
    base_name = 'BASE=' + args.base_combo
    # ★ 每一半必须是 subsets 里的一个独立元素（之前误把两半打包成一个 tuple，
    #   导致 eval_dets 拿到的 imgIds 是个嵌套列表、过滤失效，10 次结果完全相同）
    subsets = []
    for _ in range(args.folds):
        ids = all_ids[:]
        rng.shuffle(ids)
        half = len(ids) // 2
        subsets.append(ids[:half])
        subsets.append(ids[half:])
    print('[info] folds=%d -> %d half-subsets (each n=%d), base=%s'
          % (args.folds, len(subsets), len(subsets[0]), args.base_combo), flush=True)

    for name, combo in groups.items():
        d = build_dets(cache, combo, paths, img_map, args.nms_iou, args.max_det, args.conf)
        if name == base_name:
            base_d = d
            base_sub = [eval_dets(coco, d, s) for s in subsets]
            continue
        sub = [eval_dets(coco, d, s) for s in subsets]
        fl = [a - b for a, b in zip(sub, base_sub)]
        deltas[name] = fl
        n_pos = sum(1 for x in fl if x > 0)
        n = len(fl)
        mean = sum(fl) / n
        print('[fold ] %-22s mean Δ=%+.4f  Δ>0 in %d/%d halves  min=%+.4f max=%+.4f'
              % (name, mean, n_pos, n, min(fl), max(fl)), flush=True)
        print('[full ] %-22s Δ vs base = %+.4f' % (name, full[name] - full[base_name]), flush=True)

    print()
    print('=== 结论判定 ===')
    print('如果 Δ>0 的折半比例 ≥ 80%，说明提升是可复现的，不是从 21 个组合里挑出来的偶然。')


if __name__ == '__main__':
    main()
