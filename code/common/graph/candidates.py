from common import runtime
SHARED = {}
SHAPE_FIELDS = []
import json
import numpy as np
import cv2
import common.graph.features as f
c = f.c if hasattr(f, 'c') else f.u.c
H = runtime.GRAPH_ROOT / 'protocols'

def detect(image, base_lines, backend='fld', mode='graph', group='fine150_n2_original', ranking='combined_area', boundary_min=0.5, score_min=0.54, relative_min=0, cap=2):
    h, w = image.shape[:2]
    d = c.detector(backend)
    fast = c.fldfast if backend == 'fld' else c.airfast
    base = np.asarray(base_lines).reshape(-1, 4)
    mid = (base[:, :2] + base[:, 2:]) / 2
    half = (base[:, 2:] - base[:, :2]) / 2 / 1.25
    unique = {}
    groups = {'all': []}
    geometries = json.loads((H / 'graph.json').read_text())['geometries']
    if mode == 'graph':
        protocol = json.loads((H / 'graph.json').read_text())
        variants = [(name, factor, nms, geometries, protocol['whiteness']) for name, (factor, nms) in protocol['variants'].items()]
        candidate_cap = protocol['candidate_cap']
    else:
        real = dict(compactness_threshold=0.1, length_diff_threshold=0.2, angle_threshold=0.3, rects_min_area_factor=0.005, rects_max_area_factor=0.3)
        variants = [('125', 1.25, 4, geometries, [0.85, 0.65, 0.45]), ('150', 1.5, 4, dict(geometries, tfm_real_shape=real), [0.85, 0.65, 0.45])]
        candidate_cap = 500
        groups.update({g: [] for g in ['strict125', 'strict150', 'orig125', 'orig150', 'origboth']})
    if SHARED:
        variants = [(name, SHARED['elongation_factor'], SHARED['nms_threshold'], {key: dict(value, **{k: SHARED[k] for k in SHAPE_FIELDS}) for key, value in geos.items()}, [SHARED['whiteness_threshold']]) for name, factor, nms, geos, whites in variants]
    for variant, factor, nms, geos, whites in variants:
        d.nms_threshold = nms
        lines = base if mode == 'legacy' and factor == 1.25 else np.c_[mid - half * factor, mid + half * factor]
        pts = fast.intersections(d, lines)
        drawing = c.drawing(lines, h, w, backend)
        support = c.support(d, pts, drawing, backend)
        if mode == 'graph':
            groups[variant + '_original'] = []
            groups[variant + '_all'] = []
        for geometry, parameters in geos.items():
            for k, v in parameters.items():
                setattr(d, k, v)
            d.max_num_trapezoids = candidate_cap
            for white in whites:
                qs, _ = fast.extract(d, pts, support >= white, image)
                for q in qs:
                    key = tuple(sorted(map(tuple, np.asarray(q, dtype=int).tolist())))
                    if key not in unique:
                        unique[key] = (len(unique), (np.asarray(q) / [w, h]).tolist())
                    i = unique[key][0]
                    groups['all'].append(i)
                    if mode == 'graph':
                        groups[variant + '_all'].append(i)
                        if geometry == 'original':
                            groups[variant + '_original'].append(i)
                    elif geometry == 'original':
                        groups['orig' + variant].append(i)
                        groups['origboth'].append(i)
                        if white == 0.85:
                            groups['strict' + variant].append(i)
    qs = [q for i, q in unique.values()]
    ff = np.asarray(f.compute(image, qs, base)).reshape(-1, 5)
    bd, ep, cl, co, ar = ff.T
    aw = 0.75 + 0.25 * np.sqrt(ar)
    weights = {'boundary_area': bd * (0.5 + 0.5 * np.sqrt(ar)), 'closure_area': bd * (0.5 + 0.5 * cl) * aw, 'endpoint_area': bd * (0.35 + 0.65 * ep) * aw, 'contrast_area': bd * (0.4 + 0.6 * co) * aw, 'combined_area': bd * (0.5 + 0.25 * ep + 0.25 * co) * aw}
    sc = weights[ranking]
    ids = [i for i in sorted(set(groups[group])) if bd[i] >= boundary_min]
    chosen = f.u.select([np.asarray(qs[i]) * [w, h] for i in ids], [sc[i] for i in ids], max(cap, 10), -1)
    chosen = [ids[i] for i in chosen]
    maximum = max([sc[i] for i in chosen], default=0)
    chosen = [i for i in chosen if sc[i] >= max(score_min, relative_min * maximum)][:cap]
    polygons = [qs[i] for i in chosen]
    return dict(width=w, height=h, backend=backend, mode=mode, polygons=polygons, polygons_pixels=[(np.asarray(q) * [w, h]).tolist() for q in polygons], scores=[float(sc[i]) for i in chosen], candidate_count=len(qs), configuration=dict(group=group, ranking=ranking, boundary_min=boundary_min, score_min=score_min, relative_min=relative_min, cap=cap))

def fld_lines(image):
    d = c.detector('fld')
    processed = d.reduce_saturation_brightness(image, 0.5, 0.5)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    return np.asarray(d.extract_lines(gray), dtype=float).reshape(-1, 4)
