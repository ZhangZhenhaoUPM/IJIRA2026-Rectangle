from common.graph.fld_geometry import RectangleDetector as FLDRectangleDetector
from types import SimpleNamespace
from common import runtime
import ctypes
import json
import cv2
import numpy as np
import common.graph.fast_airline as airfast
import common.graph.fast_fld as fldfast
OUT = runtime.GRAPH_ROOT / 'lib'
PROTOCOL = json.loads((runtime.GRAPH_ROOT / 'protocols/base.json').read_text())

def detector(backend):
    d = FLDRectangleDetector.__new__(FLDRectangleDetector) if backend == 'fld' else SimpleNamespace()
    for k, v in PROTOCOL['detector'].items():
        setattr(d, k, v)
    return d
lib = ctypes.CDLL(str(OUT / 'graph_support.so'))
lib.graph_support.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

def support(d, pts, img, backend):
    pts = np.ascontiguousarray(pts, dtype=np.int64).reshape(-1, 2)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    s = np.zeros((len(pts), len(pts)), dtype=np.float64)
    lib.graph_support(pts.ctypes.data, len(pts), img.ctypes.data, img.shape[0], img.shape[1], int(backend == 'fld'), s.ctypes.data)
    return s

def drawing(lines, h, w, backend):
    img = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in lines:
        coords = [int(round(v)) if backend == 'fld' else int(v) for v in [x1, y1, x2, y2]]
        cv2.line(img, tuple(coords[:2]), tuple(coords[2:]), 255, 1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (5, 5), 0)
