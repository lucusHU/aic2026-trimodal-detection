#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最佳模型推理 -> 提交 txt + zip

支持多尺度 TTA：--size 可给多个值，各尺度分别前向，框合并后统一 NMS。
（val 实测最优：800+928, nms 0.7, maxDet 100 -> mAP 0.5580）
输入与训练保持一致：拉伸到方形 size x size。
输出格式：class_id cx cy w h score（归一化到原图）
"""

import argparse, os, shutil, sys, time
from pathlib import Path

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.ops import batched_nms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_state(path, use_ema=True):
    ck = torch.load(path, map_location='cpu')
    if use_ema and 'ema' in ck:
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

    cfg = YAMLConfig(args.config, resume=args.resume)
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
def run_one_size(model, paths, size, device, batch, amp=False):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ts, origs = [], []
        for p in chunk:
            img = Image.open(p).convert('RGB')
            ts.append(tf(img))
            origs.append([img.size[0], img.size[1]])
        x = torch.stack(ts, 0).to(device)
        o = torch.tensor(origs, dtype=torch.float32, device=device)
        if amp and str(device).startswith('cuda'):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                labels, boxes, scores = model(x, o)
        else:
            labels, boxes, scores = model(x, o)
        for j in range(len(chunk)):
            out.append((labels[j].cpu(), boxes[j].cpu().float(), scores[j].cpu().float()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--nms-iou', type=float, default=0.7)
    ap.add_argument('--max-det', type=int, default=100)
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--submit-dir', default='submit')
    ap.add_argument('--zip-dir', default='zip')
    ap.add_argument('--zip-name', default=None)
    ap.add_argument('--size', type=int, nargs='+', default=[928])
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.input).iterdir() if p.suffix.lower() in IMG_EXT])
    sizes = sorted(args.size)
    print('[info] images: %d' % len(paths), flush=True)
    print('[info] sizes=%s  nms_iou=%.2f  max_det=%d  conf=%.3f'
          % (sizes, args.nms_iou, args.max_det, args.conf), flush=True)

    state = load_state(args.resume, use_ema=not args.no_ema)

    t0 = time.time()
    per_size = {}
    for size in sizes:
        model = build(args, state, size)
        per_size[size] = run_one_size(model, paths, size, args.device, args.batch_size, args.amp)
        print('[fwd] size=%d  %.0fs' % (size, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    name = args.zip_name or ('ms%s_nms%02d_' % ('-'.join(map(str, sizes)), int(args.nms_iou * 100))
                             + Path(args.resume).stem)
    txt_dir = Path(args.submit_dir) / name
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)

    nbox = 0
    for i, p in enumerate(paths):
        lbs, bxs, scs = [], [], []
        for size in sizes:
            lb, bx, sc = per_size[size][i]
            m = sc >= args.conf
            if not bool(m.any()):
                continue
            lbs.append(lb[m]); bxs.append(bx[m]); scs.append(sc[m])
        lines = []
        if lbs:
            lbb = torch.cat(lbs); bxb = torch.cat(bxs); scb = torch.cat(scs)
            k = batched_nms(bxb.float(), scb.float(), lbb, args.nms_iou)[:args.max_det]
            with Image.open(p) as im:
                w, h = im.size
            for idx in k:
                x1, y1, x2, y2 = bxb[idx].tolist()
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append('%d %.6f %.6f %.6f %.6f %.6f'
                             % (int(lbb[idx]), cx, cy, bw, bh, float(scb[idx])))
        nbox += len(lines)
        with open(txt_dir / (p.stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if (i + 1) % 250 == 0:
            print('  wrote %d/%d  (%.0fs)' % (i + 1, len(paths), time.time() - t0), flush=True)

    Path(args.zip_dir).mkdir(parents=True, exist_ok=True)
    created = shutil.make_archive(str(Path(args.zip_dir) / name), 'zip', txt_dir)
    print('[done] txt dir: %s  (%d files, %d boxes)' % (txt_dir, len(paths), nbox), flush=True)
    print('[done] zip     : %s' % created, flush=True)
    print('[done] elapsed : %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
