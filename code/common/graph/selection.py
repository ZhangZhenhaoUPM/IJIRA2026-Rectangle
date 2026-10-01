import numpy as np
import cv2
import common.graph.geometry as c
cv2.setNumThreads(1)

def scores(qs, lines, w, h):
    mid = (lines[:, :2] + lines[:, 2:]) / 2
    half = (lines[:, 2:] - lines[:, :2]) / 2 / 1.25
    raw = np.c_[mid - half, mid + half]
    mask = np.zeros((h, w), np.uint8)
    for a, b, x, y in np.rint(raw).astype(int):
        cv2.line(mask, (a, b), (x, y), 255, 1)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    tol = np.clip(0.002 * np.hypot(w, h), 1.5, 6)
    t = (np.arange(128) + 0.5) / 128
    out = []
    for q in qs:
        q = np.asarray(q)
        v = q[:, None, :] * (1 - t[None, :, None]) + np.roll(q, -1, axis=0)[:, None, :] * t[None, :, None]
        ij = np.rint(v).astype(int)
        valid = (ij[:, :, 0] >= 0) & (ij[:, :, 0] < w) & (ij[:, :, 1] >= 0) & (ij[:, :, 1] < h)
        hit = dist[np.clip(ij[:, :, 1], 0, h - 1), np.clip(ij[:, :, 0], 0, w - 1)] <= tol
        coverage = (hit & valid).mean(axis=1)
        out.append(float(0.5 * coverage.mean() + 0.5 * coverage.min()))
    return out

def select(qs, ss, limit, threshold):
    order = sorted(range(len(qs)), key=lambda i: (-ss[i], i))
    keep = []
    areas = [abs(cv2.contourArea(np.asarray(q, dtype=np.float32))) for q in qs]
    for i in order:
        if ss[i] < threshold:
            break
        qi = np.asarray(qs[i], dtype=np.float32)
        suppressed = False
        for j in keep:
            inter, _ = cv2.intersectConvexConvex(qi, np.asarray(qs[j], dtype=np.float32))
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > 0.5:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
        if len(keep) >= limit:
            break
    return keep

