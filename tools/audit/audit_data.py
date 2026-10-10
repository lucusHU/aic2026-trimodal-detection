import json, collections
from pycocotools.coco import COCO

coco = COCO('dataset/annotations/instances_train.json')
cats = coco.getCatIds()
print('classes:', len(cats))
print('images:', len(coco.imgs), 'anns:', len(coco.anns))

# 1. 极端/异常框
tiny = 0; huge = 0; oob = 0; badratio = 0
areas = []
ratios = []
for a in coco.anns.values():
    x, y, w, h = a['bbox']
    areas.append(w * h)
    if w < 2 or h < 2:
        tiny += 1
    im = coco.imgs[a['image_id']]
    if x < -1 or y < -1 or x + w > im['width'] + 1 or y + h > im['height'] + 1:
        oob += 1
    if w > 0 and h > 0:
        r = max(w / h, h / w)
        ratios.append(r)
        if r > 20:
            badratio += 1
    if w * h > 0.9 * im['width'] * im['height']:
        huge += 1

print('tiny(<2px):', tiny, ' out-of-bound:', oob, ' ratio>20:', badratio, ' huge(>90%img):', huge)
areas.sort()
print('area  p1=%.0f p50=%.0f p99=%.0f max=%.0f' %
      (areas[len(areas)//100], areas[len(areas)//2], areas[int(len(areas)*0.99)], areas[-1]))
ratios.sort()
print('aspect p50=%.2f p99=%.2f max=%.2f' %
      (ratios[len(ratios)//2], ratios[int(len(ratios)*0.99)], ratios[-1]))

# 2. 类别分布
cnt = collections.Counter(a['category_id'] for a in coco.anns.values())
print('\nclass counts (train):')
for c in sorted(cnt, key=lambda k: cnt[k]):
    print('  cls_%-2d %5d' % (c, cnt[c]))

# 3. 每图框数
per = collections.Counter(a['image_id'] for a in coco.anns.values())
vals = sorted(per.values())
print('\nboxes/image: min=%d p50=%d p99=%d max=%d  empty_imgs=%d' %
      (vals[0], vals[len(vals)//2], vals[int(len(vals)*0.99)], vals[-1],
       len(coco.imgs) - len(per)))

# 4. GT 拥挤度：单图内 IoU>0.5 的 GT 对数量（判断是否需要 soft-nms / 更多 query）
import itertools
ov_pairs = 0
for iid in list(coco.imgs)[:400]:
    anns = coco.imgToAnns.get(iid, [])
    if len(anns) < 2:
        continue
    bs = []
    for a in anns:
        x, y, w, h = a['bbox']
        bs.append([x, y, x + w, y + h])
    for a1, a2 in itertools.combinations(bs, 2):
        ix1, iy1 = max(a1[0], a2[0]), max(a1[1], a2[1])
        ix2, iy2 = min(a1[2], a2[2]), min(a1[3], a2[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ar1 = max(0, a1[2] - a1[0]) * max(0, a1[3] - a1[1])
        ar2 = max(0, a2[2] - a2[0]) * max(0, a2[3] - a2[1])
        u = ar1 + ar2 - inter
        if u > 0 and inter / u > 0.5:
            ov_pairs += 1
print('GT pairs with IoU>0.5 (first 400 imgs):', ov_pairs)
