import numpy as np
import cv2
import common.graph.selection as u
cv2.setNumThreads(1)

def sample(a, p):
    xy = np.rint(p).astype(int)
    h, w = a.shape[:2]
    valid = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    value = a[np.clip(xy[..., 1], 0, h - 1), np.clip(xy[..., 0], 0, w - 1)]
    return (value, valid)

def compute(image, qs, extended):
    h, w = image.shape[:2]
    mid = (extended[:, :2] + extended[:, 2:]) / 2
    half = (extended[:, 2:] - extended[:, :2]) / 2 / 1.25
    raw = np.c_[mid - half, mid + half]
    mask = np.zeros((h, w), np.uint8)
    ends = mask.copy()
    for x1, y1, x2, y2 in np.rint(raw).astype(int):
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        for x, y in [(x1, y1), (x2, y2)]:
            if 0 <= x < w and 0 <= y < h:
                ends[y, x] = 255
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    endpoint = cv2.distanceTransform(255 - ends, cv2.DIST_L2, 3)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    tol = np.clip(0.002 * np.hypot(w, h), 1.5, 6)
    rows = []
    tt = (np.arange(64) + 0.5) / 64
    for q in qs:
        p = np.asarray(q) * [w, h]
        v = np.roll(p, -1, axis=0) - p
        lens = np.linalg.norm(v, axis=1)
        t = p[:, None, :] + v[:, None, :] * tt[None, :, None]
        dd, valid = sample(dist, t)
        cov = ((dd <= tol) & valid).mean(axis=1)
        boundary = 0.5 * cov.mean() + 0.5 * cov.min()
        ed, ev = sample(endpoint, p)
        ep = (np.exp(-ed / (tol + 0.025 * max(1, lens.min()))) * ev).mean()
        extra = np.r_[np.linspace(-0.12, -0.02, 16), np.linspace(1.02, 1.12, 16)]
        od, ov = sample(dist, p[:, None, :] + v[:, None, :] * extra[None, :, None])
        closure = 1 - ((od <= tol) & ov).mean()
        n = np.c_[-v[:, 1], v[:, 0]] / np.maximum(lens[:, None], 1e-08)
        center = p.mean(axis=0)
        sign = np.sign(np.sum((center - (p + v / 2)) * n, axis=1))
        n *= sign[:, None]
        offset = np.clip(0.015 * lens.min(), 2, 12)
        inside, vi = sample(lab, t + n[:, None, :] * offset)
        outside, vo = sample(lab, t - n[:, None, :] * offset)
        delta = np.linalg.norm(inside - outside, axis=-1)
        contrast = float(np.mean(np.clip(delta / 60, 0, 1) * vi * vo))
        area = abs(cv2.contourArea(p.astype(np.float32))) / (w * h)
        rows.append(list(map(float, [boundary, ep, closure, contrast, area])))
    return rows

