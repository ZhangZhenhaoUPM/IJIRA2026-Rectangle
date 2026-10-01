from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import runtime
import cv2
import numpy as np
import torch
from third_party.dexined.model import DexiNed
THETA_RESOLUTION = 6
KERNEL_SIZE = 9
DEXINED_PATH = str(Path(__file__).resolve().parents[1] / 'other-checkpoints/dexi.pth')

def init_orientation_detector(theta_resolution, kernel_size, device):
    """Initialize the orientation convolution with line kernels."""
    od = torch.nn.Conv2d(in_channels=1, out_channels=theta_resolution, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False).to(device)
    for i in range(theta_resolution):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        angle = i * 180 / theta_resolution
        length = kernel_size // 2
        dx = int(np.cos(np.deg2rad(angle)) * length)
        dy = int(np.sin(np.deg2rad(angle)) * length)
        cv2.line(img=kernel, pt1=(kernel_size // 2 - dx, kernel_size // 2 - dy), pt2=(kernel_size // 2 + dx, kernel_size // 2 + dy), color=1, thickness=1)
        od.weight.data[i, 0] = torch.from_numpy(kernel)
    return od

class RectangleDetector:

    def __init__(self, elongation_factor=1.3):
        self.elongation_factor = elongation_factor
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = DexiNed().to(self.device)
        self.model.load_state_dict(torch.load(DEXINED_PATH, map_location=self.device))
        # Keep neural initialization order consistent with the tested frontend.
        self.orientation_detector = init_orientation_detector(THETA_RESOLUTION, KERNEL_SIZE, self.device)


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


    def reduce_saturation_brightness(self, image, saturation_scale=0.5, value_scale=0.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s *= saturation_scale
        v *= value_scale
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


import argparse
import hashlib
import importlib
import json
import subprocess


class OfficialCRGAirLineAdapter:
    """Use the official AirLine DexiNed+CRG lines with released RIDGE geometry.

    ``CRG311_fixed`` contains only documented memory-safety compatibility
    corrections to the upstream C++ extension.  The checkpoint, thresholds,
    training-mode inference, and CRG growth parameters match the official demo.
    """
    method = 'RIDGE-AirLine-official-CRG-compat'

    def __init__(self, checkpoint: Path):
        self.module = sys.modules[__name__]
        self.checkpoint = checkpoint.resolve()
        self.module.DEXINED_PATH = str(self.checkpoint)
        self.detector = RectangleDetector()
        self.crg = importlib.import_module('CRG311_fixed')
        self.crg_source = ROOT / 'code/native/airline/CRGandLP_fixed.cpp'
        self.orientation_detector = self._orientation_detector()
        self.last_crg_raw_lines = 0
        self.last_crg_kept_lines = 0

    def _orientation_detector(self):
        torch = self.module.torch
        detector = torch.nn.Conv2d(1, 6, 9, 1, 4, bias=False).to(self.detector.device)
        for index in range(6):
            kernel = np.zeros((9, 9), dtype=np.float32)
            angle = index * np.pi / 6
            x = int(np.cos(angle) * 50)
            y = int(np.sin(angle) * 50)
            cv2.line(kernel, (4 - x, 4 - y), (4 + x, 4 + y), 1, 1)
            detector.weight.data[index] = torch.from_numpy(kernel)
        detector.weight.requires_grad_(False)
        return detector

    def _extract_crg_lines(self, image):
        torch = self.module.torch
        height, width = image.shape[:2]
        resized = cv2.resize(image, (width // 16 * 16, height // 16 * 16))
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(self.detector.device) / 255.0
        with torch.no_grad():
            prediction = self.detector.model(tensor)
            orientation = self.orientation_detector(prediction)
            orientation = torch.nn.functional.normalize(orientation - orientation.mean(1, keepdim=True), p=2.0, dim=1)
        edge_logits = prediction.detach().cpu().numpy()[0, 0]
        edges = (edge_logits > 0.5).astype(np.uint8) * 255
        output_map = np.zeros((*edges.shape, 3), dtype=np.uint8)
        output_lines = np.zeros((30000, 2, 3), dtype=np.float32)
        temp_1 = np.zeros((50000, 2), dtype=np.int32)
        temp_2 = np.zeros((2, 300000, 2), dtype=np.int32)
        temp_3 = np.zeros((30000, 2, 2), dtype=np.float32)
        count = self.crg.desGrow(output_map, edges, orientation[0].detach().cpu().numpy(), output_lines, 0.9, 10, temp_1, temp_2, temp_3, 6)
        lines = []
        for item in output_lines[:count]:
            y1, x1 = item[0, :2]
            y2, x2 = item[1, :2]
            if np.hypot(x2 - x1, y2 - y1) > 15:
                lines.append([x1, y1, x2, y2])
        self.last_crg_raw_lines = int(count)
        self.last_crg_kept_lines = len(lines)
        return self.detector.elongate_lines(lines)


METHOD = 'ridge-airline'
import sysconfig, ctypes
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parent.parent
cv2.setNumThreads(1)
PARAMETERS = json.loads((ROOT / 'parameters' / f'{METHOD}.json').read_text())
CONFIG = PARAMETERS['datasets']
from common.configuration import load_shared_geometry, resolve_shared
SHARED_GEOMETRY = load_shared_geometry(ROOT)
TEMP = runtime.prepare_graph(ROOT, PARAMETERS['pool_protocols'], next(iter(SHARED_GEOMETRY.values())))
from common.graph import candidates as det
f = det.f
u = f.u
c = u.c


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

def airline_model(device):
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    A = TEMP / 'airline'
    A.mkdir(exist_ok=True)
    inc = Path(torch.__file__).parent / 'include'
    target = A / ('CRG311_fixed' + sysconfig.get_config_var('EXT_SUFFIX'))
    subprocess.run(['g++', '-O2', '-shared', '-std=c++17', '-fPIC', '-I' + sysconfig.get_paths()['include'], '-I' + str(inc), '-I' + str(inc / 'pybind11'), str(ROOT / 'code/native/airline/CRGandLP_fixed.cpp'), '-o', str(target)], check=True, capture_output=True)
    sys.path.insert(0, str(A))
    mod = sys.modules[__name__]
    adapter = mod.OfficialCRGAirLineAdapter(ROOT / 'other-checkpoints/dexi.pth')
    adapter.detector.elongation_factor = 1.25
    adapter.detector.device = torch.device(device)
    adapter.detector.model.to(device)
    adapter.orientation_detector.to(device)
    return adapter

def predict(im, ds, settings, adapter=None, base=None):
    h, w = im.shape[:2]
    backend = 'crg'
    if base is None:
        raise ValueError('AirLine prediction requires CRG frontend lines.')
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
    p.add_argument('--lines-cache', help='Optional previously verified CRG frontend NPZ cache, keyed by image SHA256')
    a = p.parse_args()
    rs = json.loads(Path(a.manifest).read_text()) if a.manifest else [dict(dataset=a.dataset, name=Path(a.image).name, path=a.image)]
    adapter = None
    if METHOD == 'ridge-airline' and (not a.lines_cache):
        adapter = airline_model(a.device)
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
        base = None
        if METHOD == 'ridge-airline':
            if a.lines_cache:
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                base = np.load(Path(a.lines_cache) / ('crg__' + sha + '.npz'))['lines']
            else:
                base = np.asarray(adapter._extract_crg_lines(adapter.detector.reduce_saturation_brightness(im, 0.5, 0.5)), float).reshape(-1, 4) * np.array([w / (w // 16 * 16), h / (h // 16 * 16)] * 2)
        q = predict(im, a.dataset, CONFIG[a.dataset], adapter, base)
        out.append(dict(dataset=a.dataset, name=r['name'], width=w, height=h, polygons_px=[(np.asarray(p) * [w, h]).tolist() for p in q]))
        if len(out) % 10 == 0:
            print('PROGRESS', len(out), flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out))
    print('COMPLETE', len(out), flush=True)
if __name__ == '__main__':
    main()
