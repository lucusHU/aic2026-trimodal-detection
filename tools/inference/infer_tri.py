#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三模态（RGB+IR+Depth）推理 -> 提交 txt + zip

与 infer_best800.py 的唯一区别：输入是 5 通道，且**必须复刻训练集的辅助模态量化口径**。

★ 深度图口径（本次最大的坑）：
    训练集 dataset/tri/*/depth 是「16bit 毫米 /80 -> 8bit PNG」，
    而测试集 dataset/data/AIC2026_PHASE_2_1000/depth 是**官方原始 uint16 毫米值(0..19999)**。
    直接当 8bit 用会让辅助通道完全 OOD（且不会报错）。
    所以这里按 dtype 自动判别：uint16 -> clip(v/80,0,255).astype(uint8)；uint8 -> 原样。

★ 红外：官方三通道数值相同，取第 0 通道（与 extract_tri.py 的训练集处理一致）。
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.ops import batched_nms  # noqa: E402

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
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


def load_5ch(p, aux_folders, size, stats=None):
    img = Image.open(p).convert('RGB')
    w, h = img.size
    rgb = np.asarray(img.resize((size, size), Image.BILINEAR))
    chans = [torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)]
    stem = Path(p).stem
    for k, folder in enumerate(aux_folders):
        ap = find_aux(folder, stem)
        if ap is None:
            chans.append(torch.zeros(1, size, size, dtype=chans[0].dtype))
            continue
        a = np.asarray(Image.open(ap))
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.dtype == np.uint16:
            before = int(a.max())
            a = np.clip(a.astype(np.float32) / 80.0, 0, 255).astype(np.uint8)
            if stats is not None:
                stats.setdefault('u16', 0)
                stats['u16'] += 1
                stats.setdefault('u16_max', []).append(before)
        elif stats is not None:
            stats.setdefault('u8', 0)
            stats['u8'] += 1
        im2 = Image.fromarray(a)
        if im2.size != (size, size):
            im2 = im2.resize((size, size), Image.BILINEAR)
        chans.append(torch.from_numpy(np.ascontiguousarray(np.asarray(im2))).unsqueeze(0))
    x = torch.cat(chans, 0).to(torch.float32).div_(255.0).contiguous()
    return x, (w, h)


@torch.inference_mode()
def run_one_size(model, paths, size, device, batch, aux_folders, amp=False):
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        ts, origs = [], []
        for p in chunk:
            x, (w, h) = load_5ch(p, aux_folders, size)
            ts.append(x)
            origs.append([w, h])
        xs = torch.stack(ts, 0).to(device)
        o = torch.tensor(origs, dtype=torch.float32, device=device)
        if amp and str(device).startswith('cuda'):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                labels, boxes, scores = model(xs, o)
        else:
            labels, boxes, scores = model(xs, o)
        for j in range(len(chunk)):
            out.append((labels[j].cpu(), boxes[j].cpu().float(), scores[j].cpu().float()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('--aux', nargs='+', required=True)
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
    ap.add_argument('--no-zip', action='store_true')
    ap.add_argument('--size', type=int, nargs='+', default=[800, 928])
    args = ap.parse_args()

    assert len(args.aux) >= 1, 'aux folders required'
    paths = sorted([p for p in Path(args.input).iterdir() if p.suffix.lower() in IMG_EXT])
    sizes = sorted(args.size)
    print('[info] images: %d' % len(paths), flush=True)
    print('[info] aux   : %s' % args.aux, flush=True)
    print('[info] sizes=%s  nms_iou=%.2f  max_det=%d  conf=%.3f'
          % (sizes, args.nms_iou, args.max_det, args.conf), flush=True)

    # 先体检辅助模态的 dtype，把量化口径显式打印出来
    st = {}
    for p in paths[:50]:
        load_5ch(p, args.aux, 64, stats=st)
    print('[info] aux dtype scan(first 50): uint16=%d uint8=%d  max16=%s'
          % (st.get('u16', 0), st.get('u8', 0),
             sorted(set(st.get('u16_max', [])))[-3:] if st.get('u16_max') else '-'), flush=True)

    state = load_state(args.resume, use_ema=not args.no_ema)

    t0 = time.time()
    per_size = {}
    for size in sizes:
        model = build(args, state, size)
        per_size[size] = run_one_size(model, paths, size, args.device, args.batch_size,
                                      args.aux, args.amp)
        print('[fwd] size=%d  %.0fs' % (size, time.time() - t0), flush=True)
        del model
        torch.cuda.empty_cache()

    name = args.zip_name or ('tri_ms%s_nms%02d_%s'
                             % ('-'.join(map(str, sizes)), int(args.nms_iou * 100),
                                Path(args.resume).stem))
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
                cx = float(np.clip(((x1 + x2) / 2.0) / w, 0.0, 1.0))
                cy = float(np.clip(((y1 + y2) / 2.0) / h, 0.0, 1.0))
                bw = float(np.clip((x2 - x1) / w, 0.0, 1.0))
                bh = float(np.clip((y2 - y1) / h, 0.0, 1.0))
                lines.append('%d %.6f %.6f %.6f %.6f %.6f'
                             % (int(lbb[idx]), cx, cy, bw, bh, float(scb[idx])))
        nbox += len(lines)
        with open(txt_dir / (p.stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        if (i + 1) % 250 == 0:
            print('  wrote %d/%d  (%.0fs)' % (i + 1, len(paths), time.time() - t0), flush=True)

    print('[done] txt dir: %s  (%d files, %d boxes)' % (txt_dir, len(paths), nbox), flush=True)
    if not args.no_zip:
        Path(args.zip_dir).mkdir(parents=True, exist_ok=True)
        created = shutil.make_archive(str(Path(args.zip_dir) / name), 'zip', txt_dir)
        print('[done] zip     : %s' % created, flush=True)
    print('[done] elapsed : %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
