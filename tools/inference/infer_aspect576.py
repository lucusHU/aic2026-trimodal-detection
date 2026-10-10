#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""aspect576 模型推理 -> 提交 txt + zip

与训练保持一致：输入固定 576x1024（16:9 等比，无拉伸、无 pad）。
TTA 只做水平翻转（不改尺度，避免偏离训练分布）。
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
IN_HW = (576, 1024)  # h, w


def load_state(path):
    ck = torch.load(path, map_location='cpu')
    if 'ema' in ck:
        state = ck['ema']['module']
    elif 'model' in ck:
        state = ck['model']
    else:
        state = ck
    state = dict(state)
    state.pop('decoder.anchors', None)
    state.pop('decoder.valid_mask', None)
    return state


def build(args, state):
    repo = os.path.abspath(args.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.resume)
    cfg.yaml_cfg['eval_spatial_size'] = [IN_HW[0], IN_HW[1]]
    for key in ('HybridEncoder', 'DFINETransformer'):
        if key in cfg.yaml_cfg and 'eval_spatial_size' in cfg.yaml_cfg[key]:
            cfg.yaml_cfg[key]['eval_spatial_size'] = [IN_HW[0], IN_HW[1]]
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


def infer_batch(model, images, device, amp=False, flip=False):
    tf = T.Compose([T.Resize(IN_HW), T.ToTensor()])
    tensors, origs = [], []
    for img in images:
        w, h = img.size
        x = img
        if flip:
            x = img.transpose(getattr(Image, 'Transpose', Image).FLIP_LEFT_RIGHT)
        tensors.append(tf(x))
        origs.append([w, h])
    x = torch.stack(tensors, 0).to(device)
    o = torch.tensor(origs, dtype=torch.float32, device=device)
    with torch.inference_mode():
        if amp and str(device).startswith('cuda'):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                labels, boxes, scores = model(x, o)
        else:
            labels, boxes, scores = model(x, o)
    out = []
    for i, img in enumerate(images):
        w, h = img.size
        lb = labels[i].cpu()
        bx = boxes[i].cpu().clone()
        sc = scores[i].cpu()
        if flip and bx.numel() > 0:
            ox1, ox2 = bx[:, 0].clone(), bx[:, 2].clone()
            bx[:, 0] = w - ox2
            bx[:, 2] = w - ox1
        out.append((lb, bx, sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--nms-iou', type=float, default=0.6)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--submit-dir', default='submit')
    ap.add_argument('--zip-dir', default='zip')
    ap.add_argument('--zip-name', default=None)
    args = ap.parse_args()

    paths = sorted([p for p in Path(args.input).iterdir() if p.suffix.lower() in IMG_EXT])
    print('[info] images: %d' % len(paths), flush=True)

    state = load_state(args.resume)
    model = build(args, state)
    print('[model] loaded %s @ %dx%d' % (args.resume, IN_HW[0], IN_HW[1]), flush=True)

    name = args.zip_name or ('aspect576_' + Path(args.resume).stem)
    txt_dir = Path(args.submit_dir) / name
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(0, len(paths), args.batch_size):
        chunk = paths[i:i + args.batch_size]
        imgs = []
        for p in chunk:
            imgs.append(Image.open(p).convert('RGB'))
        res = infer_batch(model, imgs, args.device, args.amp, flip=False)
        if not args.no_flip:
            res_f = infer_batch(model, imgs, args.device, args.amp, flip=True)
            res = [(torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]), torch.cat([a[2], b[2]]))
                   for a, b in zip(res, res_f)]

        for j, p in enumerate(chunk):
            lb, bx, sc = res[j]
            keep = sc >= args.conf
            lb, bx, sc = lb[keep], bx[keep], sc[keep]
            lines = []
            if bx.numel() > 0:
                k = batched_nms(bx.float(), sc.float(), lb, args.nms_iou)
                k = k[:args.max_det]
                w, h = imgs[j].size
                for idx in k:
                    x1, y1, x2, y2 = bx[idx].tolist()
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append('%d %.6f %.6f %.6f %.6f %.6f'
                                 % (int(lb[idx]), cx, cy, bw, bh, float(sc[idx])))
            with open(txt_dir / (p.stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        if i % (args.batch_size * 20) == 0:
            print('  %d/%d  (%.0fs)' % (i, len(paths), time.time() - t0), flush=True)

    Path(args.zip_dir).mkdir(parents=True, exist_ok=True)
    created = shutil.make_archive(str(Path(args.zip_dir) / name), 'zip', txt_dir)
    print('[done] txt dir: %s' % txt_dir, flush=True)
    print('[done] zip     : %s' % created, flush=True)
    print('[done] elapsed : %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
