#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 val 上评测：stretch800 基线 vs +flip TTA(+NMS合并)，用现有 best 权重"""

import argparse, json, os, sys, time
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as T
from torchvision.ops import batched_nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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


def build(args, size, state):
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


def run(args, model, size, coco, items, img_dir, use_flip, nms_iou, max_det):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    preds = []
    B = args.batch
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        tensors, origs = [], []
        for it in chunk:
            img = Image.open(os.path.join(img_dir, it['file_name'])).convert('RGB')
            tensors.append(tf(img))
            origs.append([img.size[0], img.size[1]])
        x = torch.stack(tensors, 0).to(args.device)
        o = torch.tensor(origs, dtype=torch.float32, device=args.device)
        with torch.inference_mode():
            labels, boxes, scores = model(x, o)

        fl = None
        if use_flip:
            ft, fo = [], []
            for it in chunk:
                img = Image.open(os.path.join(img_dir, it['file_name'])).convert('RGB')
                img = img.transpose(getattr(Image, 'Transpose', Image).FLIP_LEFT_RIGHT)
                ft.append(tf(img))
                fo.append([img.size[0], img.size[1]])
            fx = torch.stack(ft, 0).to(args.device)
            fo_t = torch.tensor(fo, dtype=torch.float32, device=args.device)
            with torch.inference_mode():
                fl = model(fx, fo_t)

        for j, it in enumerate(chunk):
            lb = labels[j].cpu()
            bx = boxes[j].cpu()
            sc = scores[j].cpu()
            if fl is not None:
                w = chunk[j]['width']
                fb = fl[1][j].cpu().clone()
                if fb.numel() > 0:
                    ox1, ox2 = fb[:, 0].clone(), fb[:, 2].clone()
                    fb[:, 0] = w - ox2
                    fb[:, 2] = w - ox1
                lb = torch.cat([lb, fl[0][j].cpu()])
                bx = torch.cat([bx, fb])
                sc = torch.cat([sc, fl[2][j].cpu()])
            keep = sc >= args.conf
            lb, bx, sc = lb[keep], bx[keep], sc[keep]
            if bx.numel() == 0:
                continue
            if use_flip:
                k = batched_nms(bx.float(), sc.float(), lb, nms_iou)
            else:
                k = torch.argsort(sc, descending=True)
            k = k[:max_det]
            for idx in k:
                x1, y1, x2, y2 = bx[idx].tolist()
                preds.append({'image_id': int(it['id']), 'category_id': int(lb[idx]),
                              'bbox': [float(x1), float(y1),
                                       float(max(x2 - x1, 1.0)), float(max(y2 - y1, 1.0))],
                              'score': float(sc[idx])})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--repo', default='../D-FINE')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ann', default='dataset/annotations/instances_val.json')
    ap.add_argument('--img-dir', default='dataset/images/val')
    ap.add_argument('--size', type=int, default=800)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--nms-iou', type=float, default=0.6)
    ap.add_argument('--max-det', type=int, default=100)
    args = ap.parse_args()

    coco = COCO(args.ann)
    items = coco.loadImgs(sorted(coco.getImgIds()))
    state = load_state(args.resume)
    model = build(args, args.size, state)
    results = {}

    for tag, use_flip in [('base', False), ('flip_tta', True)]:
        t0 = time.time()
        preds = run(args, model, args.size, coco, items, args.img_dir,
                    use_flip, args.nms_iou, args.max_det)
        tmp = '/tmp/_pred_%s.json' % tag
        json.dump(preds, open(tmp, 'w'))
        dt = coco.loadRes(tmp)
        E = COCOeval(coco, dt, 'bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        st = E.stats
        results[tag] = {'mAP': round(float(st[0]), 4), 'AP50': round(float(st[1]), 4),
                        'AP75': round(float(st[2]), 4), 'APs': round(float(st[3]), 4),
                        'APm': round(float(st[4]), 4), 'APl': round(float(st[5]), 4)}
        print('%-10s mAP=%.4f AP50=%.4f AP75=%.4f APs=%.4f APm=%.4f APl=%.4f (%ds)'
              % (tag, st[0], st[1], st[2], st[3], st[4], st[5], int(time.time() - t0)), flush=True)

    print('=== SUMMARY (size=%d, nms_iou=%.2f, max_det=%d) ==='
          % (args.size, args.nms_iou, args.max_det))
    for k, v in results.items():
        print('%-10s %s' % (k, v))
    json.dump(results, open('flip_compare_%d.json' % args.size, 'w'), indent=2)


if __name__ == '__main__':
    main()
