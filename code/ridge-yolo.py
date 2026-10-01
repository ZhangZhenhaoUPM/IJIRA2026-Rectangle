"""ridge-yolo: method-specific algorithms and CLI. Shared kernels are imported."""
from __future__ import annotations
import json
import itertools
import cv2
import numpy as np
ridge_YOLO_CONF = 0.25
ridge_YOLO_IOU = 0.45
ridge_YOLO_CLASSES = None
ridge_ROI_PADDING_RATIO = 0.15
ridge_ROI_PADDING_MIN = 25
ridge_ROI_PADDING_MAX = 100
ridge_SCALE = 2
ridge_LINES_MODE = 'roi'
ridge_PRE_DARKEN = True
ridge_MAX_LINES = 400
ridge_MAX_CYCLES = 60000
ridge_NMS_POINT_PX = 4
ridge_PASSES = [dict(name='estricto', fld_min_len=15, elong=1.4, white=0.85, angle=0.35, compact=0.22, len_ratio=0.25), dict(name='relajado', fld_min_len=10, elong=1.6, white=0.7, angle=0.5, compact=0.15, len_ratio=0.2)]
ridge_QUAD_AREA_RATIO_MIN = 0.3
ridge_QUAD_AREA_RATIO_MAX = 1.8
ridge_MIN_IOU_WITH_BOX = 0.4
ridge_MIN_RIDGE_SCORE = 0.45
ridge_W_IOU, ridge_W_SUP, ridge_W_ORT = (0.5, 0.3, 0.2)
ridge_MAX_QUADS_PER_ROI = 1
ridge_QUAD_NMS_IOU = 0.5

def ridge__ccw(poly):
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = (poly[:, 0], poly[:, 1])
    signed = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if signed >= 0 else poly[::-1].copy()

def ridge__area(pts):
    if len(pts) < 3:
        return 0.0
    p = np.asarray(pts, dtype=np.float64)
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))

def ridge__clip_convex(subject, clip):
    """Clip two counterclockwise convex polygons with Sutherland-Hodgman."""
    out = [tuple(p) for p in subject]
    for i in range(len(clip)):
        if not out:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % len(clip)]
        ex, ey = (bx - ax, by - ay)
        inp, out = (out, [])
        prev = inp[-1]
        sp = ex * (prev[1] - ay) - ey * (prev[0] - ax)
        for cur in inp:
            sc = ex * (cur[1] - ay) - ey * (cur[0] - ax)
            if sc >= 0:
                if sp < 0:
                    t = sp / (sp - sc)
                    out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
                out.append(cur)
            elif sp >= 0:
                t = sp / (sp - sc)
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            prev, sp = (cur, sc)
    return out

def ridge_poly_iou(a, b):
    """Compute convex-polygon IoU with Sutherland-Hodgman clipping."""
    a, b = (ridge__ccw(a), ridge__ccw(b))
    if len(a) < 3 or len(b) < 3:
        return 0.0
    inter = ridge__area(ridge__clip_convex(a.tolist(), b.tolist()))
    union = ridge__area(a) + ridge__area(b) - inter
    return float(inter / union) if union > 0 else 0.0

def ridge_box_to_quad(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

def ridge_nms_items(items, thr):
    """NMS de detecciones (dicts con 'quad' y 'score') por IoU poligonal."""
    kept = []
    for it in sorted(items, key=lambda d: d['score'], reverse=True):
        if all((ridge_poly_iou(it['quad'], k['quad']) < thr for k in kept)):
            kept.append(it)
    return kept

class ridge_RidgeFLD:
    """Vectorized RIDGE geometry in the enlarged image coordinates."""

    def __init__(self, scale=ridge_SCALE, lines_mode=ridge_LINES_MODE):
        if not hasattr(cv2, 'ximgproc'):
            raise RuntimeError('Falta ximgproc: pip install opencv-contrib-python')
        self.S = scale
        self.mode = lines_mode
        self.gray = None
        self._global_cache = {}

    @staticmethod
    def _darken(img, s_scale=0.5, v_scale=0.5):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * s_scale, 0, 255)
        v = np.clip(v * v_scale, 0, 255)
        return cv2.cvtColor(cv2.merge([h, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def set_image(self, img):
        proc = self._darken(img) if ridge_PRE_DARKEN else img
        self.gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        self._global_cache = {}

    @staticmethod
    def _fld(gray, min_len):
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th1 = max(1.0, 0.33 * float(otsu))
        th2 = max(th1 + 1.0, float(otsu))
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=int(max(2, min_len)), distance_threshold=1.414, canny_th1=th1, canny_th2=th2, canny_aperture_size=3, do_merge=True)
        det = fld.detect(gray)
        if det is None:
            return np.zeros((0, 4), dtype=np.float32)
        return det.reshape(-1, 4).astype(np.float32)

    def _lines(self, cfg, pass_idx, roi):
        """Segmentos en coordenadas LOCALES de la ROI."""
        rx1, ry1, rx2, ry2 = roi
        min_len = cfg['fld_min_len'] * self.S
        if self.mode == 'roi':
            return self._fld(self.gray[ry1:ry2, rx1:rx2], min_len)
        if pass_idx not in self._global_cache:
            self._global_cache[pass_idx] = self._fld(self.gray, min_len)
        rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)
        out = []
        for x1, y1, x2, y2 in self._global_cache[pass_idx]:
            ok, p1, p2 = cv2.clipLine(rect, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))))
            if ok and np.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= min_len:
                out.append([p1[0] - rx1, p1[1] - ry1, p2[0] - rx1, p2[1] - ry1])
        return np.asarray(out, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def _elongate(lines, factor):
        ext = (factor - 1.0) / 2.0
        d = lines[:, 2:] - lines[:, :2]
        return np.hstack([lines[:, :2] - d * ext, lines[:, 2:] + d * ext])

    @staticmethod
    def _intersections(lines, angle_thr):
        """
        Intersecciones entre segmentos elongados. Se descartan las que no son
        casi perpendiculares (|cos| > sin(angle_thr)): ese vértice fallaría igualmente
        el test de ángulos del cuadrilátero, y así el grafo es mucho más pequeño.
        """
        n = len(lines)
        if n < 2:
            return np.zeros((0, 2), dtype=np.float32)
        i, j = np.triu_indices(n, k=1)
        p = lines[i, :2].astype(np.float64)
        r = (lines[i, 2:] - lines[i, :2]).astype(np.float64)
        q = lines[j, :2].astype(np.float64)
        s = (lines[j, 2:] - lines[j, :2]).astype(np.float64)
        rxs = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
        ok = np.abs(rxs) > 1e-09
        safe = np.where(ok, rxs, 1.0)
        qp = q - p
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / safe
        u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / safe
        nr = np.linalg.norm(r, axis=1)
        ns = np.linalg.norm(s, axis=1)
        cosang = np.abs(np.sum(r * s, axis=1)) / np.maximum(nr * ns, 1e-09)
        m = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (cosang <= np.sin(angle_thr))
        return (p[m] + t[m, None] * r[m]).astype(np.float32)

    @staticmethod
    def _merge_points(pts, thr):
        if len(pts) == 0:
            return pts
        used = np.zeros(len(pts), dtype=bool)
        out = []
        for a in range(len(pts)):
            if used[a]:
                continue
            m = (np.linalg.norm(pts - pts[a], axis=1) < thr) & ~used
            used |= m
            out.append(pts[m].mean(axis=0))
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _build_graph(pts, support, white_thr, min_edge):
        n = len(pts)
        h, w = support.shape
        adj = np.zeros((n, n), dtype=bool)
        for a in range(n - 1):
            p = pts[a]
            q = pts[a + 1:]
            d = np.linalg.norm(q - p, axis=1)
            good = d >= min_edge
            if not good.any():
                continue
            idx = np.nonzero(good)[0] + a + 1
            q, d = (q[good], d[good])
            k = int(np.clip(d.max(), 16, 200))
            t = np.linspace(0.0, 1.0, k)[None, :]
            xs = np.rint(p[0] + (q[:, 0] - p[0])[:, None] * t).astype(int)
            ys = np.rint(p[1] + (q[:, 1] - p[1])[:, None] * t).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            hit = (support[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] > 64) & valid
            nv = valid.sum(axis=1)
            frac = hit.sum(axis=1) / np.maximum(nv, 1)
            ok = (nv >= 0.5 * k) & (frac >= white_thr)
            adj[a, idx[ok]] = True
        return adj | adj.T

    @staticmethod
    def _cycles4(adj):
        """Ciclos i-a-k-b de 4 nodos, cada uno una sola vez."""
        A = adj.astype(np.int32)
        common = A @ A
        ii, kk = np.nonzero(np.triu(common >= 2, 1))
        for i, k in zip(ii.tolist(), kk.tolist()):
            c = np.flatnonzero(adj[i] & adj[k])
            c = c[c > i].tolist()
            for a, b in itertools.combinations(c, 2):
                yield (i, a, k, b)

    @staticmethod
    def _geometry(P, cfg):
        """Return the acceptance mask, areas and orthogonality scores for M x 4 x 2 candidates."""
        E = np.roll(P, -1, axis=1) - P
        L = np.maximum(np.linalg.norm(E, axis=2), 1e-09)
        En = np.roll(E, -1, axis=1)
        Ln = np.roll(L, -1, axis=1)
        ratio = L.min(axis=1) / L.max(axis=1)
        x, y = (P[..., 0], P[..., 1])
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1))
        per = L.sum(axis=1)
        compact = 4.0 * np.pi * area / np.maximum(per ** 2, 1e-09)
        cross = E[..., 0] * En[..., 1] - E[..., 1] * En[..., 0]
        convex = np.all(cross > 0, axis=1) | np.all(cross < 0, axis=1)
        abscos = np.abs(np.sum(E * En, axis=2) / (L * Ln))
        sin_thr = np.sin(cfg['angle'])
        orth_ok = np.all(abscos <= sin_thr, axis=1)
        orth_score = np.clip(1.0 - abscos.mean(axis=1) / max(sin_thr, 1e-06), 0.0, 1.0)
        ok = convex & orth_ok & (ratio >= cfg['len_ratio']) & (compact >= cfg['compact'])
        return (ok, area, orth_score)

    @staticmethod
    def _edge_support(P, support, k=40):
        """Fracción de cada lado apoyada en el mapa de líneas. Devuelve S x 4."""
        h, w = support.shape
        A = P[:, :, None, :]
        B = np.roll(P, -1, axis=1)[:, :, None, :]
        t = np.linspace(0.0, 1.0, k)[None, None, :, None]
        pts = A + (B - A) * t
        xs = np.rint(pts[..., 0]).astype(int)
        ys = np.rint(pts[..., 1]).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hit = (support[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)] > 64) & valid
        nv = valid.sum(axis=2)
        frac = hit.sum(axis=2) / np.maximum(nv, 1)
        return np.where(nv >= 0.3 * k, frac, 0.0)

    def find_quads(self, roi, box_local, cfg, pass_idx):
        """
        roi: (rx1, ry1, rx2, ry2) en la imagen ampliada
        box_local: bbox YOLO en coordenadas locales de la ROI (x1, y1, x2, y2)
        Devuelve lista de dicts {quad, score, iou_box, support, orth, area} ordenada por score.
        """
        rec = dict(roi=list(map(int, roi)), box_local=list(map(float, box_local)), pass_idx=pass_idx, counts={})
        ridge_TRACE.append(rec)
        counts = rec['counts']
        rx1, ry1, rx2, ry2 = roi
        rw, rh = (rx2 - rx1, ry2 - ry1)
        bx1, by1, bx2, by2 = box_local
        area_box = max(1.0, (bx2 - bx1) * (by2 - by1))
        lines = self._lines(cfg, pass_idx, roi)
        counts['raw_lines'] = len(lines)
        rec['raw_lines_xyxy'] = lines.tolist()
        if len(lines) < 4:
            return []
        if len(lines) > ridge_MAX_LINES:
            length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
            lines = lines[np.argsort(-length)[:ridge_MAX_LINES]]
        counts['kept_lines'] = len(lines)
        rec['kept_lines_xyxy'] = lines.tolist()
        lines = self._elongate(lines, cfg['elong'])
        lw = max(2, int(round(ridge_LINE_WIDTH_FACTOR * self.S)))
        bk = 2 * lw + 1
        lines_img = np.zeros((rh, rw), dtype=np.uint8)
        for x1, y1, x2, y2 in lines:
            cv2.line(lines_img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, lw, cv2.LINE_AA)
        support = cv2.GaussianBlur(lines_img, (bk, bk), 0)
        pts = self._intersections(lines, cfg['angle'])
        counts['raw_intersections'] = len(pts)
        pts = self._merge_points(pts, ridge_NMS_POINT_PX * self.S)
        counts['merged_intersections'] = len(pts)
        rec['nodes'] = pts.tolist()
        if len(pts) < 4:
            return []
        min_edge = max(8 * self.S, 0.1 * min(bx2 - bx1, by2 - by1))
        adj = self._build_graph(pts, support, cfg['white'], min_edge)
        counts['graph_edges'] = int(adj.sum() // 2)
        cycles = list(itertools.islice(self._cycles4(adj), ridge_MAX_CYCLES))
        counts['cycles'] = len(cycles)
        if not cycles:
            return []
        P = pts[np.asarray(cycles)]
        ok, area, orth = self._geometry(P, cfg)
        counts['geometry_pass'] = int(ok.sum())
        ok &= (area >= ridge_QUAD_AREA_RATIO_MIN * area_box) & (area <= ridge_QUAD_AREA_RATIO_MAX * area_box)
        counts['area_pass'] = int(ok.sum())
        if not ok.any():
            return []
        P, area, orth = (P[ok], area[ok], orth[ok])
        if len(P) > 80:
            keep = np.argsort(np.abs(np.log(area / area_box)))[:80]
            P, area, orth = (P[keep], area[keep], orth[keep])
        counts['after_top80'] = len(P)
        sup_min = self._edge_support(P, support).min(axis=1)
        box_poly = ridge_box_to_quad(bx1, by1, bx2, by2)
        cands = []
        for q, a, o, s in zip(P, area, orth, sup_min):
            iou = ridge_poly_iou(q, box_poly)
            score = ridge_W_IOU * iou + ridge_W_SUP * float(s) + ridge_W_ORT * float(o)
            cands.append(dict(quad=q.astype(np.float32), score=float(score), iou_box=float(iou), support=float(s), orth=float(o), area=float(a)))
        rec['before_score'] = [dict(quad=((q['quad'] + np.asarray([rx1, ry1])) / self.S).tolist(), score=q['score'], iou_box=q['iou_box'], support=q['support'], orth=q['orth']) for q in cands]
        counts['iou_pass'] = sum((c['iou_box'] >= ridge_MIN_IOU_WITH_BOX for c in cands))
        cands = [c for c in cands if c['iou_box'] >= ridge_MIN_IOU_WITH_BOX and c['score'] >= ridge_MIN_RIDGE_SCORE]
        counts['score_pass'] = len(cands)
        final = ridge_nms_items(cands, ridge_QUAD_NMS_IOU)
        counts['nms_pass'] = len(final)
        return final

class ridge_YoloRidge:

    def _roi(self, box, W, H):
        x1, y1, x2, y2 = box
        pad = int(max(x2 - x1, y2 - y1) * ridge_ROI_PADDING_RATIO)
        pad = max(ridge_ROI_PADDING_MIN, min(ridge_ROI_PADDING_MAX, pad))
        pad = int(pad * ridge_SCALE)
        return (max(0, int(x1) - pad), max(0, int(y1) - pad), min(W, int(x2) + pad), min(H, int(y2) + pad))

    def process_image(self, image, name):
        H0, W0 = image.shape[:2]
        S = ridge_SCALE
        img_s = cv2.resize(image, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC) if S != 1 else image
        Hs, Ws = img_s.shape[:2]
        self.ridge.set_image(img_s)
        res = self.yolo.predict(source=image, conf=ridge_YOLO_CONF, iou=ridge_YOLO_IOU, classes=ridge_YOLO_CLASSES, verbose=False)[0]
        out = {'yolo': [], 'ridge': [], 'ridge_fb': []}
        stats = dict(yolo=0, validated=0, fallback=0)
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
        else:
            xyxy, conf, cls = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
        for idx, (b, c, k) in enumerate(zip(xyxy, conf, cls), 1):
            stats['yolo'] += 1
            cname = self.names[int(k)]
            bs = b * S
            roi = self._roi(bs, Ws, Hs)
            rx1, ry1, rx2, ry2 = roi
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            base = dict(**{'class': cname}, class_id=int(k), yolo_conf=float(c), bbox=[float(v) for v in b])
            yolo_quad = ridge_box_to_quad(*b)
            out['yolo'].append({**base, 'score': float(c), 'quad': yolo_quad, 'source': 'yolo'})
            box_local = (bs[0] - rx1, bs[1] - ry1, bs[2] - rx1, bs[3] - ry1)
            quads, used = ([], None)
            for pi, cfg in enumerate(ridge_PASSES):
                try:
                    quads = self.ridge.find_quads(roi, box_local, cfg, pi)
                except Exception as exc:
                    print(f"      ERROR Ridge ({cfg['name']}): {exc}")
                    quads = []
                if quads:
                    used = cfg['name']
                    break
            if quads:
                stats['validated'] += 1
                for q in quads[:ridge_MAX_QUADS_PER_ROI]:
                    quad = (q['quad'] + np.array([rx1, ry1], dtype=np.float32)) / S
                    det = {**base, 'score': float(c) * (0.5 + 0.5 * q['score']), 'quad': quad, 'source': 'ridge', 'ridge_pass': used, 'ridge_score': q['score'], 'iou_with_yolo_box': q['iou_box']}
                    out['ridge'].append(det)
                    out['ridge_fb'].append(det)
            else:
                stats['fallback'] += 1
                out['ridge_fb'].append({**base, 'score': float(c) * 0.5, 'quad': yolo_quad, 'source': 'yolo_fallback'})
        out['ridge'] = ridge_nms_items(out['ridge'], ridge_QUAD_NMS_IOU)
        out['ridge_fb'] = ridge_nms_items(out['ridge_fb'], ridge_QUAD_NMS_IOU)
        return (out, stats)

def ridge_export_predictions(dets):
    """Export quadrilaterals and scores in original-image pixel coordinates."""
    rows = []
    for d in dets:
        quad = np.asarray(d['quad'], dtype=np.float32)
        row = {k: v for k, v in d.items() if k not in ('quad',)}
        row['quad'] = np.round(quad, 1).tolist()
        row['score'] = round(float(d['score']), 4)
        rows.append(row)
    return rows


ridge_TRACE = []
ridge_LINE_WIDTH_FACTOR = 1.5
import argparse
import types
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / 'parameters/ridge-yolo.json').read_text())
sys.path.insert(0, str(ROOT))
ORIGINAL_PASSES = [dict(x) for x in ridge_PASSES]

class Array:

    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

class Boxes:

    def __init__(self, rs):
        self.xyxy = Array([r['bbox'] for r in rs])
        self.conf = Array([r['yolo_conf'] for r in rs])
        self.cls = Array([r['class_id'] for r in rs])
        self.n = len(rs)

    def __len__(self):
        return self.n

class Frozen:

    def __init__(self, rs):
        self.rs = rs

    def predict(self, **kw):
        return [types.SimpleNamespace(boxes=Boxes(self.rs))]

def configure(ds, c):
    global ridge_LINE_WIDTH_FACTOR, ridge_MIN_IOU_WITH_BOX, ridge_NMS_POINT_PX, ridge_PASSES, ridge_QUAD_NMS_IOU, ridge_ROI_PADDING_MAX, ridge_ROI_PADDING_MIN, ridge_ROI_PADDING_RATIO
    ridge_PASSES = [dict(x) for x in ORIGINAL_PASSES]
    for p, elong in zip(ridge_PASSES, [1.8, 2.0]):
        p['elong'] = elong
    if ds == 'isaacsim50':
        for p, a, co in zip(ridge_PASSES, [0.6, 0.8], [0.15, 0.1]):
            p.update(angle=a, compact=co, len_ratio=0.1)
    ridge_ROI_PADDING_RATIO = 0.15
    ridge_ROI_PADDING_MIN = 10
    ridge_ROI_PADDING_MAX = 100
    ridge_NMS_POINT_PX = 4
    ridge_QUAD_NMS_IOU = 0.5
    ridge_LINE_WIDTH_FACTOR = 1.5
    ridge_MIN_IOU_WITH_BOX = 0.4
    for k, v in c.items():
        if k in ('support', 'elong', 'lengths', 'angles', 'compact', 'ratios'):
            field = dict(support='white', elong='elong', lengths='fld_min_len', angles='angle', compact='compact', ratios='len_ratio')[k]
            for p, x in zip(ridge_PASSES, v):
                p[field] = x
        elif k.isupper():
            globals()['ridge_' + k] = v

    def fld(gray, min_len):
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gain = c.get('fld_gain', 1.0)
        lo = max(1.0, 0.33 * float(otsu) * gain)
        hi = max(lo + 1.0, float(otsu) * gain)
        d = cv2.ximgproc.createFastLineDetector(length_threshold=int(min_len), distance_threshold=c.get('fld_distance', 1.0 if ds == 'isaacsim50' else 1.414), canny_th1=lo, canny_th2=hi, canny_aperture_size=3, do_merge=c.get('fld_merge', ds != 'isaacsim50'))
        z = d.detect(gray)
        return np.zeros((0, 4), np.float32) if z is None else z.reshape(-1, 4).astype(np.float32)

    def fixed_roi(self, box, W, H):
        x1, y1, x2, y2 = box
        pad = int(max(x2 - x1, y2 - y1) / ridge_SCALE * ridge_ROI_PADDING_RATIO)
        pad = max(ridge_ROI_PADDING_MIN, min(ridge_ROI_PADDING_MAX, pad)) * ridge_SCALE
        return (max(0, int(x1) - pad), max(0, int(y1) - pad), min(W, int(x2) + pad), min(H, int(y2) + pad))
    ridge_RidgeFLD._fld = staticmethod(fld)
    ridge_YoloRidge._roi = fixed_roi

def detect(image, ds, c, model, device):
    global ridge_TRACE
    configure(ds, c)
    oh, ow = image.shape[:2]
    scale = c.get('working_longside', 320) / max(oh, ow)
    p = model.predict(source=image, imgsz=320, conf=c['model_conf'], iou=c['model_nms'], classes=None, half=False, max_det=300, agnostic_nms=False, rect=True, verbose=False, device=device, save=False)[0]
    props = [dict(bbox=xy.tolist(), yolo_conf=float(a), class_id=int(k)) for xy, a, k in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy(), p.boxes.cls.cpu().numpy())]
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else image
    frozen = [dict(p, bbox=(np.asarray(p['bbox']) * scale).tolist()) for p in props]
    d = ridge_YoloRidge.__new__(ridge_YoloRidge)
    d.ridge = ridge_RidgeFLD()
    d.yolo = Frozen(frozen)
    d.names = model.names
    ridge_TRACE = []
    layers, stats = d.process_image(work, 'image')
    seen = set()
    for layer in layers.values():
        for det in layer:
            if id(det) not in seen:
                det['quad'] = det['quad'] / scale
                det['bbox'] = (np.asarray(det['bbox']) / scale).tolist()
                seen.add(id(det))
    exported = {k: ridge_export_predictions(v) for k, v in layers.items()}
    return dict(width=ow, height=oh, polygons_px=[p['quad'] for p in exported['ridge']], layers=exported, stats=stats)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(CONFIG))
    p.add_argument('--image')
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    c = CONFIG[a.dataset]
    model = YOLO(a.checkpoint or str(ROOT / c['checkpoint']))
    rs = json.loads(Path(a.manifest).read_text()) if a.manifest else [dict(dataset=a.dataset, name=Path(a.image).name, path=a.image)]
    rows = []
    for r in rs:
        if r['dataset'] != a.dataset:
            continue
        path = Path(r['path'])
        path = path if path.is_absolute() else ROOT / path
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(str(path))
        rows.append(dict(dataset=a.dataset, name=r['name'], **detect(im, a.dataset, c, model, a.device)))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(rows))
    print('COMPLETE', len(rows), flush=True)
if __name__ == '__main__':
    main()
