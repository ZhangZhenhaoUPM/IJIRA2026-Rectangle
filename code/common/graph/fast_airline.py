from common import runtime
import numpy as np

def intersections(d, lines):
    a = np.asarray(lines, dtype=float).reshape(-1, 4)
    points = []
    for i in range(len(a)):
        p = a[i, :2]
        r = a[i, 2:] - p
        q = a[i + 1:, :2]
        s = a[i + 1:, 2:] - q
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        good = den != 0
        den2 = np.where(good, den, 1)
        t = ((q - p)[:, 0] * s[:, 1] - (q - p)[:, 1] * s[:, 0]) / den2
        u = ((q - p)[:, 0] * r[1] - (q - p)[:, 1] * r[0]) / den2
        good &= (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        points.extend(np.rint(p + t[good, None] * r).astype(int).tolist())
    points = np.asarray(points, dtype=int).reshape(-1, 2)
    suppressed = np.zeros(len(points), dtype=bool)
    result = []
    for i in range(len(points)):
        if suppressed[i]:
            continue
        close = np.linalg.norm(points[i + 1:] - points[i], axis=1) < d.nms_threshold
        indices = np.flatnonzero(close) + i + 1
        suppressed[indices] = True
        mean = np.mean(points[np.r_[i, indices]], axis=0)
        result.append((int(mean[0]), int(mean[1])))
    return result


import ctypes

_cycles = ctypes.CDLL(str(runtime.GRAPH_ROOT / 'lib/cycles.so'))
_cycles.extract_cycles.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
_cycles.extract_cycles.restype = ctypes.c_int

class FailedCount:

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

def extract(d, pts, adj, image):
    pts = np.ascontiguousarray(pts, dtype=np.int64).reshape(-1, 2)
    adj = np.ascontiguousarray(adj, dtype=np.uint8)
    output = np.empty((d.max_num_trapezoids, 4, 2), dtype=np.int64)
    failed = ctypes.c_int64()
    count = _cycles.extract_cycles(pts.ctypes.data, adj.ctypes.data, len(pts), d.rects_min_area_factor * image.shape[1] * image.shape[0], d.rects_max_area_factor * image.shape[1] * image.shape[0], d.length_diff_threshold, d.compactness_threshold, d.angle_threshold, d.max_num_trapezoids, output.ctypes.data, ctypes.byref(failed))
    return (list(output[:count]), FailedCount(failed.value))
