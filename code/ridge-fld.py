"""ridge-fld: method-specific algorithms and CLI. Shared kernels are imported."""
from __future__ import annotations
METHOD = 'ridge-fld'
import sys, json, argparse, ctypes
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
cv2.setNumThreads(1)
PARAMETERS = json.loads((ROOT / 'parameters' / f'{METHOD}.json').read_text())
CONFIG = PARAMETERS['datasets']
from common.configuration import load_shared_geometry, resolve_shared
SHARED_GEOMETRY = load_shared_geometry(ROOT)
from common import runtime
TEMP = runtime.prepare_graph(ROOT, PARAMETERS['pool_protocols'], next(iter(SHARED_GEOMETRY.values())))
from common.graph import candidates as det
f = det.f
u = f.u
c = u.c
normalized_H = TEMP / 'protocols'
import common.graph.selection as utils
normalized_c = utils.c
normalized_P = json.loads((normalized_H / 'normalized_fld.json').read_text())
cv2.setNumThreads(1)

def normalized_detect(image, variant='resize768_clahe_low', ranking='boundary_area', cap=1, threshold=0.7):
    oh, ow = image.shape[:2]
    ratio = min(1, normalized_P['max_working_side'] / max(ow, oh))
    w, h = (round(ow * ratio), round(oh * ratio))
    im = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    settings = normalized_P['variants'][variant]
    if settings['clahe']:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(1, float(otsu) * settings['canny_scale'])
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=normalized_P['minimum_line_length'], distance_threshold=1.414, canny_th1=max(1, 0.33 * high), canny_th2=high, canny_aperture_size=3, do_merge=True)
    detected = fld.detect(gray)
    raw = np.empty((0, 4)) if detected is None else np.asarray(detected).reshape(-1, 4)
    d = normalized_c.detector('fld')
    unique = {}
    d.elongation_factor = 1.25
    base = np.asarray(d.elongate_lines(raw), dtype=float).reshape(-1, 4)
    for factor in normalized_P['elongation']:
        d.elongation_factor = factor
        lines = np.asarray(d.elongate_lines(raw), dtype=float).reshape(-1, 4)
        pts = normalized_c.fldfast.intersections(d, lines)
        drawing = normalized_c.drawing(lines, h, w, 'fld')
        support = normalized_c.support(d, pts, drawing, 'fld')
        for pars in normalized_P['geometries'].values():
            for k, v in pars.items():
                setattr(d, k, v)
            d.max_num_trapezoids = normalized_P['candidate_cap_per_pool']
            for white in normalized_P['whiteness']:
                qs, _ = normalized_c.fldfast.extract(d, pts, support >= white, im)
                ss = utils.scores(qs, base, w, h)
                for q, score in zip(qs, ss):
                    key = tuple(sorted(map(tuple, np.asarray(q, dtype=int).tolist())))
                    if key not in unique or score > unique[key][1]:
                        unique[key] = (np.asarray(q), score)
    items = [(q, score) for q, score in unique.values() if score >= threshold]
    qs = [q for q, score in items]
    scores = np.array([score for q, score in items])
    areas = np.array([abs(cv2.contourArea(q.astype(np.float32))) / (w * h) for q in qs])
    weights = scores if ranking == 'boundary' else scores * (0.5 + 0.5 * np.sqrt(areas))
    indices = utils.select(qs, weights, cap, -1)
    polygons = [(qs[i] / [w, h]).tolist() for i in indices]
    return dict(width=ow, height=oh, working_width=w, working_height=h, configuration=dict(variant=variant, ranking=ranking, cap=cap, threshold=threshold), polygons_normalized=polygons, polygons_pixels=[(np.asarray(q) * [ow, oh]).tolist() for q in polygons], boundary_scores=[float(scores[i]) for i in indices])

def normalized(im, settings):
    global normalized_P
    normalized_P = json.loads((normalized_H / 'normalized_fld.json').read_text())
    return normalized_detect(im, **{k: settings[k] for k in ['variant', 'ranking', 'cap', 'threshold']})

def short_base(im):
    d = c.detector('fld')
    gray = cv2.cvtColor(d.reduce_saturation_brightness(im, 0.5, 0.5), cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi = max(float(otsu), 1.0)
    detector = cv2.ximgproc.createFastLineDetector(length_threshold=8, distance_threshold=1.414, canny_th1=max(0.33 * hi, 1.0), canny_th2=hi, canny_aperture_size=3, do_merge=True)
    z = detector.detect(gray)
    raw = np.asarray(z if z is not None else [], dtype=float).reshape(-1, 4)
    mid = (raw[:, :2] + raw[:, 2:]) / 2
    half = (raw[:, 2:] - raw[:, :2]) * 0.625
    return np.c_[mid - half, mid + half]

def centroid(im, base, backend, settings):
    h, w = im.shape[:2]
    P = json.loads((TEMP / 'protocols' / ('centroid_fld.json' if backend == 'fld' else 'centroid.json')).read_text())
    fast = c.fldfast if backend == 'fld' else c.airfast
    old = fast._cycles
    lib = ctypes.CDLL(str(TEMP / 'lib/centroid' / ('cycles_fld.so' if backend == 'fld' else 'cycles.so')))
    lib.extract_cycles.argtypes = old.extract_cycles.argtypes
    lib.extract_cycles.restype = old.extract_cycles.restype
    fast._cycles = lib
    d = c.detector(backend)
    mid = (base[:, :2] + base[:, 2:]) / 2
    half = (base[:, 2:] - base[:, :2]) / 2 / 1.25
    unique = {}
    groups = {k: [] for k in P['geometries']}
    for variant, (factor, nms) in P['variants'].items():
        d.nms_threshold = nms
        lines = np.c_[mid - half * factor, mid + half * factor]
        pts = fast.intersections(d, lines)
        draw = c.drawing(lines, h, w, backend)
        support = c.support(d, pts, draw, backend)
        for geo, pars in P['geometries'].items():
            for k, v in pars.items():
                setattr(d, k, v)
            d.max_num_trapezoids = P['candidate_cap']
            for white in P['whiteness']:
                qs, _ = fast.extract(d, pts, support >= white, im)
                for q in qs:
                    key = tuple(sorted(map(tuple, np.asarray(q, dtype=int).tolist())))
                    if key not in unique:
                        unique[key] = (len(unique), (np.asarray(q) / [w, h]).tolist())
                    groups[geo].append(unique[key][0])
    fast._cycles = old
    qs = [q for _, q in unique.values()]
    ff = np.asarray(f.compute(im, qs, base)).reshape(-1, 5)
    bd, ep, cl, co, ar = ff.T
    score = bd * (0.5 + 0.5 * cl) * (0.75 + 0.25 * np.sqrt(ar))
    ids = [i for i in sorted(set(groups[settings['group']])) if bd[i] >= settings['boundary_min']]
    kept = u.select([np.asarray(qs[i]) * [w, h] for i in ids], [score[i] for i in ids], 10, -1)
    final = [ids[i] for i in kept if score[ids[i]] >= settings['score_min']][:settings['cap']]
    return [qs[i] for i in final]

def predict(im, ds, settings, adapter=None, base=None):
    h, w = im.shape[:2]
    backend = 'fld'
    if settings['mode'] == 'normalized':
        return normalized(im, settings)['polygons_normalized']
    if base is None:
        base = short_base(im) if ds == 'isaacsim50' else det.fld_lines(im)
    if settings['mode'] == 'centroid':
        return centroid(im, base, backend, settings)
    return det.detect(im, base, backend=backend, **settings)['polygons']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(CONFIG))
    p.add_argument('--image')
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    rs = json.loads(Path(a.manifest).read_text()) if a.manifest else [dict(dataset=a.dataset, name=Path(a.image).name, path=a.image)]
    out = []
    for r in rs:
        if r['dataset'] != a.dataset:
            continue
        path = Path(r['path'])
        path = path if path.is_absolute() else ROOT / path
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(str(path))
        h, w = im.shape[:2]
        q = predict(im, a.dataset, CONFIG[a.dataset])
        out.append(dict(dataset=a.dataset, name=r['name'], width=w, height=h, polygons_px=[(np.asarray(p) * [w, h]).tolist() for p in q]))
        if len(out) % 10 == 0:
            print('PROGRESS', len(out), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out))
    print('COMPLETE', len(out), flush=True)
if __name__ == '__main__':
    main()
