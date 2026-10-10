import json, sys

BASE = [0.0038, 0.0077, 0.0206, 0.0342, 0.0481, 0.0624, 0.0784,
        0.0975, 0.1184, 0.1403, 0.1682, 0.2334]


def read(run):
    out = []
    try:
        for l in open('output/%s/log.txt' % run):
            try:
                r = json.loads(l)
            except Exception:
                continue
            m = r.get('test_coco_eval_bbox')
            if m:
                out.append((r.get('epoch', 0), m[0]))
    except FileNotFoundError:
        pass
    return out


runs = sys.argv[1:] or ['baseline_800', 'e25_800']
data = {r: dict(read(r)) for r in runs}
eps = sorted({e for d in data.values() for e in d})
print('ep    ' + '  '.join('%-14s' % r for r in runs))
for e in eps:
    row = []
    for r in runs:
        v = data[r].get(e)
        row.append('%-14s' % ('%.4f' % v if v is not None else '-'))
    b = BASE[e] if e < len(BASE) else None
    extra = ('   baseline=%.4f' % b) if b is not None else ''
    print('ep%-3d ' % e + '  '.join(row) + extra)
