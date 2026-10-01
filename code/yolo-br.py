"""yolo-br: method-specific algorithms and CLI. Shared kernels are imported."""
from __future__ import annotations
import numpy as np
import cv2
import itertools
boundary_DISTANCE = 1.414

def boundary_line_equation(segment):
    a = np.asarray(segment[:2], float)
    b = np.asarray(segment[2:], float)
    v = b - a
    n = np.linalg.norm(v)
    if n < 1e-09:
        raise ValueError('Zero-length segment')
    normal = np.array([-v[1], v[0]]) / n
    return np.array([*normal, -normal @ a])

def boundary_intersection(a, b):
    value = np.cross(a, b)
    return None if abs(value[2]) < 1e-09 else value[:2] / value[2]

def boundary_observed_support(a, b, segments, tolerance=boundary_DISTANCE):
    """Integral of maximum aligned observed-segment coverage along a side.

    Each finite segment is clipped to a perpendicular tolerance band and the
    side interval. Overlapping fragments count once; a crossing perpendicular
    segment contributes zero. No extension, drawing, or integer point merging.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length < 1e-09 or not len(segments):
        return 0.0
    direction = edge / length
    normal = np.array([-direction[1], direction[0]])
    segments = np.asarray(segments).reshape(-1, 4)
    starts = segments[:, :2]
    vectors = segments[:, 2:] - starts
    sizes = np.linalg.norm(vectors, axis=1)
    d0 = (starts - a) @ normal
    dv = vectors @ normal
    intervals = []
    for i in range(len(segments)):
        if sizes[i] < 1e-09:
            continue
        weight = float((vectors[i] @ direction / sizes[i]) ** 2)
        if weight < 1e-12:
            continue
        if abs(dv[i]) < 1e-12:
            if abs(d0[i]) > tolerance:
                continue
            lo, hi = (0.0, 1.0)
        else:
            low, high = sorted([(-tolerance - d0[i]) / dv[i], (tolerance - d0[i]) / dv[i]])
            lo = max(0.0, low)
            hi = min(1.0, high)
            if hi <= lo:
                continue
        p = (starts[i] + lo * vectors[i] - a) @ direction
        q = (starts[i] + hi * vectors[i] - a) @ direction
        start = max(0.0, min(p, q))
        end = min(length, max(p, q))
        if end > start:
            intervals.append((start, end, weight))
    if not intervals:
        return 0.0
    intervals = np.asarray(intervals)
    breaks = np.unique(intervals[:, :2])
    mid = (breaks[:-1] + breaks[1:]) / 2
    active = (mid[:, None] >= intervals[:, 0]) & (mid[:, None] <= intervals[:, 1])
    weights = np.max(active * intervals[:, 2], axis=1)
    return float(np.clip(np.sum(np.diff(breaks) * weights) / length, 0.0, 1.0))


def boundary_bbox_iou(q, box):
    a = np.asarray(q)
    b = np.asarray(box)
    al = a.min(0)
    ah = a.max(0)
    bl = b.min(0)
    bh = b.max(0)
    overlap = np.maximum(0, np.minimum(ah, bh) - np.maximum(al, bl))
    inter = float(np.prod(overlap))
    union = float(np.prod(ah - al) + np.prod(bh - bl) - inter)
    return inter / union if union > 0 else 0.0


def interval_interval_evidence(a, b, segments, tolerance=boundary_DISTANCE):
    """Weighted interval IoU: explain both the proposed side and observed continuation."""
    a = np.asarray(a)
    b = np.asarray(b)
    edge = b - a
    length = float(np.linalg.norm(edge))
    if length < 1e-09 or not len(segments):
        return 0.0
    direction = edge / length
    normal = np.array([-direction[1], direction[0]])
    segments = np.asarray(segments).reshape(-1, 4)
    starts = segments[:, :2]
    vectors = segments[:, 2:] - starts
    sizes = np.linalg.norm(vectors, axis=1)
    d0 = (starts - a) @ normal
    dv = vectors @ normal
    intervals = []
    for i in range(len(segments)):
        if sizes[i] < 1e-09:
            continue
        weight = float((vectors[i] @ direction / sizes[i]) ** 2)
        if weight < 1e-12:
            continue
        if abs(dv[i]) < 1e-12:
            if abs(d0[i]) > tolerance:
                continue
            lo, hi = (0.0, 1.0)
        else:
            low, high = sorted([(-tolerance - d0[i]) / dv[i], (tolerance - d0[i]) / dv[i]])
            lo = max(0.0, low)
            hi = min(1.0, high)
            if hi <= lo:
                continue
        p = (starts[i] + lo * vectors[i] - a) @ direction
        q = (starts[i] + hi * vectors[i] - a) @ direction
        start = min(p, q)
        end = max(p, q)
        if min(end, length) > max(start, 0.0):
            intervals.append((start, end, weight))
    if not intervals:
        return 0.0
    intervals = np.asarray(intervals)
    breaks = np.unique([0.0, length, *intervals[:, :2].ravel()])
    mid = (breaks[:-1] + breaks[1:]) / 2
    active = (mid[:, None] >= intervals[:, 0]) & (mid[:, None] <= intervals[:, 1])
    weights = np.max(active * intervals[:, 2], axis=1)
    lengths = np.diff(breaks)
    total = float(lengths @ weights)
    inside = float(np.sum(lengths * weights * (mid >= 0) * (mid <= length)))
    return inside / (length + total - inside) if length + total - inside > 0 else 0.0


selector_POLICY = dict(candidate_generation='supporting_line_four_cycles', min_side_coverage=0.5, min_bbox_affinity=0.5, line_tolerance=boundary_DISTANCE, detector_side_priors=False, parent_fallback=False, ranking='affinity times mean interval support; cyclic displacement tie-break')

def selector_supporting_lines(observed, center):
    """Stable segment IDs; inward normals allow only convex directed turns.

    Keep distinct finite fragments as evidence, even on the same infinite line.
    Only exact duplicate segments are removed; no angle-dependent clustering.
    """
    canonical = set()
    for raw in np.asarray(observed, float).reshape(-1, 4):
        if not np.isfinite(raw).all() or np.linalg.norm(raw[2:] - raw[:2]) < 1e-09:
            continue
        a, b = (tuple(raw[:2]), tuple(raw[2:]))
        canonical.add((*min(a, b), *max(a, b)))
    segments = np.asarray(sorted(canonical), float).reshape(-1, 4)
    lines = np.asarray([boundary_line_equation(s) for s in segments]).reshape(-1, 3)
    for line in lines:
        if line[:2] @ center + line[2] < 0:
            line *= -1
    return (lines, segments)

def selector_compatible_graph(lines, shape):
    """Directed intersections inside the padded crop; positive convex turns."""
    adjacent = [set() for _ in lines]
    crossings = {}
    for i, j in itertools.combinations(range(len(lines)), 2):
        turn = lines[i, 0] * lines[j, 1] - lines[i, 1] * lines[j, 0]
        if abs(turn) < 1e-09:
            continue
        point = boundary_intersection(lines[i], lines[j])
        if point is None or not np.isfinite(point).all():
            continue
        if np.any(point < 0) or point[0] > shape[1] or point[1] > shape[0]:
            continue
        a, b = (i, j) if turn > 0 else (j, i)
        adjacent[a].add(b)
        crossings[a, b] = point
    return (adjacent, crossings)

def selector_four_cycles(adjacent):
    """Each directed simple four-cycle once, starting at its smallest ID."""
    incoming = [set() for _ in adjacent]
    for a, neighbors in enumerate(adjacent):
        for b in neighbors:
            incoming[b].add(a)
    for a, neighbors in enumerate(adjacent):
        for b in sorted(neighbors):
            if b <= a:
                continue
            for c in sorted(adjacent[b]):
                if c <= a or c == b:
                    continue
                for d in sorted(adjacent[c] & incoming[a]):
                    if d > a and d != b and (d != c):
                        yield (a, b, c, d)

def selector_canonical_quad(q, ids):
    """Rotate only: side i always belongs to supporting line ids[i]."""
    start = min(range(4), key=lambda i: (float(q[i].sum()), *q[i]))
    return (np.roll(q, -start, axis=0), ids[start:] + ids[:start])

def selector_choose_observed(observed, parent, shape, diagnostic=False):
    parent = np.asarray(parent, float)
    center = parent.mean(0)
    lines, segments = selector_supporting_lines(observed, center)
    adjacent, crossings = selector_compatible_graph(lines, shape)
    audit = dict(line_nodes=len(lines), compatibility_edges=len(crossings), considered=0, geometrically_valid=0, supported=0, selection='rejected', reason='no_closed_supported_boundary')
    if len(lines) < 4:
        return (None, {**audit, 'reason': 'insufficient_observed_lines'})
    best = None
    diagnostic_candidates = []
    evidence = {}
    for ids in selector_four_cycles(adjacent):
        audit['considered'] += 1
        q = np.asarray([crossings[ids[i - 1], ids[i]] for i in range(4)])
        contour = q.astype(np.float32)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1e-08:
            continue
        if cv2.pointPolygonTest(contour, tuple(center), False) < 0:
            continue
        affinity = boundary_bbox_iou(q, parent)
        if affinity < selector_POLICY['min_bbox_affinity']:
            continue
        audit['geometrically_valid'] += 1
        coverage, support = ([], [])
        for i in range(4):
            key = (ids[i - 1], ids[i], ids[(i + 1) % 4])
            if key not in evidence:
                cover = boundary_observed_support(q[i], q[(i + 1) % 4], segments)
                value = interval_interval_evidence(q[i], q[(i + 1) % 4], segments) if cover >= selector_POLICY['min_side_coverage'] else 0.0
                evidence[key] = (cover, value)
            cover, value = evidence[key]
            coverage.append(cover)
            support.append(value)
            if cover < selector_POLICY['min_side_coverage']:
                break
        if len(coverage) != 4 or min(coverage) < selector_POLICY['min_side_coverage']:
            continue
        audit['supported'] += 1
        score = affinity * float(np.mean(support))
        if diagnostic:
            ordered_diagnostic, _ = selector_canonical_quad(q, ids)
            diagnostic_candidates.append(dict(polygon=ordered_diagnostic.tolist(), score=float(score), bbox_iou=float(affinity), coverage=[float(v) for v in coverage], support=[float(v) for v in support]))
        shift = min((float(np.linalg.norm(np.roll(q, k, axis=0) - p, axis=1).mean()) for p in (parent, parent[::-1]) for k in range(4)))
        ordered, ordered_ids = selector_canonical_quad(q, ids)
        key = (score, -shift, tuple(ordered.ravel()))
        if best is None or key > best[0]:
            offset = ids.index(ordered_ids[0])
            best = (key, ordered, dict(coverage=coverage[offset:] + coverage[:offset], support=support[offset:] + support[:offset], score=score, bbox_iou=affinity, line_ids=list(ordered_ids)))
    if best is None:
        if diagnostic:
            audit['candidate_polygons'] = diagnostic_candidates
        return (None, audit)
    detail = {**audit, **best[2], 'selection': 'observed_complete', 'reason': None, 'used_parent': False, 'observed_sides': 4}
    if diagnostic:
        detail['candidate_polygons'] = diagnostic_candidates
    return (best[1], detail)

class fld_RectangleDetector:

    def __init__(self, elongation_factor=1.25):
        self.elongation_factor = elongation_factor


    def elongate_lines(self, lines):
        extended_lines = []
        extension = (self.elongation_factor - 1) / 2
        for line in lines:
            x1, y1, x2, y2 = line
            dx = x2 - x1
            dy = y2 - y1
            new_x1 = x1 - dx * extension
            new_y1 = y1 - dy * extension
            new_x2 = x2 + dx * extension
            new_y2 = y2 + dy * extension
            extended_lines.append([new_x1, new_y1, new_x2, new_y2])
        return extended_lines


    def extract_lines(self, gray):
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_th1 = float(0.33 * otsu_val)
        canny_th2 = float(otsu_val)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=15, distance_threshold=1.414, canny_th1=canny_th1, canny_th2=canny_th2, canny_aperture_size=3, do_merge=True)
        lines = fld.detect(gray)
        if lines is None:
            return []
        lines = [line[0].tolist() for line in lines]
        return self.elongate_lines(lines)


    def reduce_saturation_brightness(self, image, saturation_scale=0.5, value_scale=0.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s *= saturation_scale
        v *= value_scale
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

import sys
import json
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def detect(image, model, c, device):
    selector_POLICY['min_side_coverage'] = c.get('min_side_coverage', 0.5)
    selector_POLICY['min_bbox_affinity'] = c.get('min_bbox_affinity', 0.5)
    oh, ow = image.shape[:2]
    w, h = (round(ow * 320 / max(oh, ow)), round(oh * 320 / max(oh, ow)))
    frame = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA) if (w, h) != (ow, oh) else image.copy()
    p = model.predict(frame, device=device, imgsz=320, conf=0.001, iou=c['nms'], max_det=300, half=False, verbose=False)[0]
    boxes = p.boxes.xyxy.detach().cpu().numpy()
    scores = p.boxes.conf.detach().cpu().numpy()
    polys = []
    for box in boxes[scores >= c['confidence']]:
        x1, y1, x2, y2 = map(float, box)
        mode = c['mode']
        if mode == 'P0':
            px = py = float(c.get('fixed_padding', 15.0))
        else:
            alpha = dict(P05=0.05, P10=0.1, P15=0.15, P20=0.2)[mode]
            px = max(5.0, alpha * (x2 - x1))
            py = max(5.0, alpha * (y2 - y1))
        a, b, x, y = (max(0, int(np.floor(x1 - px))), max(0, int(np.floor(y1 - py))), min(w, int(np.ceil(x2 + px))), min(h, int(np.ceil(y2 + py))))
        obj = fld_RectangleDetector()
        crop = obj.reduce_saturation_brightness(frame[b:y, a:x], 0.5, 0.5)
        extended = obj.extract_lines(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        observed = []
        for line in extended:
            p0 = np.asarray(line[:2])
            q0 = np.asarray(line[2:])
            center = (p0 + q0) / 2
            half = (q0 - p0) / 2.5
            observed.append([*center - half, *center + half])
        parent = (np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float) / [w, h]).tolist()
        local = np.asarray(parent) * [w, h] - [a, b]
        q, _ = selector_choose_observed(observed, local, (y - b, x - a), diagnostic=True)
        if q is not None:
            polys.append(((np.asarray(q) + [a, b]) / [w, h] * [ow, oh]).tolist())
    return dict(width=ow, height=oh, polygons_px=polys)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--image')
    p.add_argument('--manifest')
    p.add_argument('--checkpoint')
    p.add_argument('--config', default=str(ROOT / 'parameters/yolo-br.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())[a.dataset]
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    rows = json.loads(Path(a.manifest).read_text()) if a.manifest else [dict(dataset=a.dataset, name=Path(a.image).name, path=a.image)]
    models = {}
    out = []
    for r in rows:
        if r['dataset'] != a.dataset:
            continue
        if a.checkpoint:
            checkpoint = a.checkpoint
        elif 'folds' in cfg:
            matches = [f for f in cfg['folds'] if r['name'] in f['names']]
            if len(matches) != 1:
                raise ValueError('OOF replay requires an image in exactly one recorded fold; supply --checkpoint for arbitrary images.')
            checkpoint = str(ROOT / matches[0]['checkpoint'])
        else:
            checkpoint = str(ROOT / cfg['checkpoint'])
        if checkpoint not in models:
            models[checkpoint] = YOLO(checkpoint, task='detect')
        path = Path(r['path'])
        path = path if path.is_absolute() else ROOT / path
        im = cv2.imread(str(path))
        out.append(dict(dataset=a.dataset, name=r['name'], **detect(im, models[checkpoint], cfg, a.device)))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out))
    print('COMPLETE', len(out), flush=True)
if __name__ == '__main__':
    main()
