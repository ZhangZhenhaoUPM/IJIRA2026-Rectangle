"""ridge-roi: method-specific algorithms and CLI. Shared kernels are imported."""
from __future__ import annotations
SCORE_MAP = {}
SCORE_KEY = lambda q: tuple(map(tuple, q))
import json, argparse
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parent.parent
PARAMETERS = json.loads((ROOT / 'parameters/ridge-roi.json').read_text())
pipeline_PARAM = PARAMETERS['datasets']
import copy
import hashlib
roi_frontend_P = PARAMETERS['frontends']
roi_frontend_SMART_PROTOCOL = PARAMETERS['smartdoc_frontend']

def roi_frontend_legacy_raw_reference(self, gray):
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_th2 = max(float(otsu_val), 1.0)
    canny_th1 = max(0.33 * canny_th2, 1.0)
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=self.minimum_line_length, distance_threshold=1.414, canny_th1=canny_th1, canny_th2=canny_th2, canny_aperture_size=3, do_merge=True)
    lines = fld.detect(gray)
    if lines is None:
        return []
    lines = [line[0].tolist() for line in lines]
    return lines

def roi_frontend_legacy_darken(image, saturation_scale=0.5, value_scale=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_scale, 0, 255)
    v = np.clip(v * value_scale, 0, 255)
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def roi_frontend_legacy_smart_reference(image, variant='resize768_clahe_low', ranking='boundary_area', cap=1, threshold=0.7):
    oh, ow = image.shape[:2]
    ratio = min(1, roi_frontend_SMART_PROTOCOL['max_working_side'] / max(ow, oh))
    w, h = (round(ow * ratio), round(oh * ratio))
    im = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    settings = roi_frontend_SMART_PROTOCOL['variants'][variant]
    if settings['clahe']:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(1, float(otsu) * settings['canny_scale'])
    fld = cv2.ximgproc.createFastLineDetector(length_threshold=roi_frontend_SMART_PROTOCOL['minimum_line_length'], distance_threshold=1.414, canny_th1=max(1, 0.33 * high), canny_th2=high, canny_aperture_size=3, do_merge=True)
    detected = fld.detect(gray)
    raw = np.empty((0, 4)) if detected is None else np.asarray(detected).reshape(-1, 4)
    return (gray, raw)
roi_frontend_RAW_REFERENCE = roi_frontend_legacy_raw_reference
roi_frontend_DARKEN = roi_frontend_legacy_darken
roi_frontend_SMART_REFERENCE = roi_frontend_legacy_smart_reference

def roi_frontend_clip(lines, box):
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    if not len(lines):
        return lines
    p = lines[:, :2]
    d = lines[:, 2:] - p
    lo = np.zeros(len(lines))
    hi = np.ones(len(lines))
    valid = np.ones(len(lines), dtype=bool)
    for axis, (a, b) in enumerate([(box[0], box[2]), (box[1], box[3])]):
        parallel = np.abs(d[:, axis]) < 1e-12
        valid &= ~(parallel & ((p[:, axis] < a) | (p[:, axis] > b)))
        den = np.where(parallel, 1, d[:, axis])
        ta = (a - p[:, axis]) / den
        tb = (b - p[:, axis]) / den
        low = np.minimum(ta, tb)
        high = np.maximum(ta, tb)
        lo = np.maximum(lo, np.where(parallel, -np.inf, low))
        hi = np.minimum(hi, np.where(parallel, np.inf, high))
    valid &= hi >= lo
    return np.c_[p[valid] + d[valid] * lo[valid, None], p[valid] + d[valid] * hi[valid, None]]
assert np.array_equal(roi_frontend_clip([[-1, 2, 8, 2], [2, -2, 2, 9], [7, 1, 8, 2]], (0, 0, 4, 4)), [[0, 2, 4, 2], [2, 0, 2, 4]])

class roi_frontend_LegacyFrontend:

    def __init__(self, image, dataset):
        self.dataset = dataset
        self.params = roi_frontend_P[dataset]
        self.minimum_line_length = self.params['minimum_line_length']
        self.oh, self.ow = image.shape[:2]
        self.local_cache = {}
        if dataset == 'smartdoc90':
            self.gray, ref = roi_frontend_SMART_REFERENCE(image)
        else:
            self.gray = cv2.cvtColor(roi_frontend_DARKEN(image, 0.5, 0.5), cv2.COLOR_BGR2GRAY)
            ref = roi_frontend_RAW_REFERENCE(self, self.gray)
        self.h, self.w = self.gray.shape
        self.scale = np.array([self.w / self.ow, self.h / self.oh])
        self.raw, self.thresholds = self.detect(self.gray)
        assert np.array_equal(self.raw, np.asarray(ref, dtype=np.float32).reshape(-1, 4)), dataset

    def detect(self, gray):
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = max(float(otsu) * self.params['canny_gain'], 1.0)
        low = max(0.33 * hi, 1.0)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=self.minimum_line_length, distance_threshold=1.414, canny_th1=low, canny_th2=hi, canny_aperture_size=3, do_merge=True)
        det = fld.detect(gray)
        raw = np.asarray(det if det is not None else [], dtype=np.float32).reshape(-1, 4)
        return (raw, dict(otsu=float(otsu), low=low, high=hi))

    def lines_for(self, roi, dest_scale, mode):
        x1, y1, x2, y2 = map(float, roi)
        to_front = self.scale / dest_scale
        if mode == 'old_global':
            lines = self.raw
            thresholds = self.thresholds
            crop = None
        else:
            assert mode == 'old_local'
            a = np.floor(np.array([x1, y1]) * to_front).astype(int)
            b = np.ceil(np.array([x2, y2]) * to_front).astype(int)
            a = np.maximum(a, [0, 0])
            b = np.minimum(b, [self.w, self.h])
            key = tuple(a) + tuple(b)
            if key not in self.local_cache:
                gray = self.gray[a[1]:b[1], a[0]:b[0]]
                if min(gray.shape, default=0) < 3:
                    raw = np.empty((0, 4), dtype=np.float32)
                    thresholds = {}
                else:
                    raw, thresholds = self.detect(gray)
                lines = np.asarray(raw, dtype=np.float64) + np.tile(a, 2)
                self.local_cache[key] = (lines, thresholds)
            lines, thresholds = self.local_cache[key]
            crop = list(map(int, key))
        mapped = np.asarray(lines, dtype=np.float64) / np.tile(to_front, 2)
        clipped = roi_frontend_clip(mapped, (x1, y1, x2 - 1, y2 - 1))
        length = np.linalg.norm((clipped[:, 2:] - clipped[:, :2]) * to_front, axis=1)
        clipped = clipped[length >= self.minimum_line_length]
        out = (clipped - [x1, y1, x1, y1]).astype(np.float32)
        if len(out):
            assert np.all(out >= -0.001) and np.all(out[:, [0, 2]] <= x2 - x1 - 1 + 0.001) and np.all(out[:, [1, 3]] <= y2 - y1 - 1 + 0.001)
        return (out, dict(mode=mode, roi_destination=list(roi), destination_per_original=dest_scale, legacy_crop=crop, detected_before_clip=len(lines), kept_after_clip=len(out), thresholds=thresholds))


def proposal_geometry__ccw(poly):
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = (poly[:, 0], poly[:, 1])
    signed = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if signed >= 0 else poly[::-1].copy()

def proposal_geometry__area(pts):
    if len(pts) < 3:
        return 0.0
    p = np.asarray(pts, dtype=np.float64)
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))

def proposal_geometry__clip_convex(subject, clip):
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

def proposal_geometry_poly_iou(a, b):
    """Compute convex-polygon IoU with Sutherland-Hodgman clipping."""
    a, b = (proposal_geometry__ccw(a), proposal_geometry__ccw(b))
    if len(a) < 3 or len(b) < 3:
        return 0.0
    inter = proposal_geometry__area(proposal_geometry__clip_convex(a.tolist(), b.tolist()))
    union = proposal_geometry__area(a) + proposal_geometry__area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def contours_calculate_iou(rect1, rect2):
    """Compute intersection over union for two axis-aligned rectangles."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def contours_calculate_overlap_percentage(rect1, rect2):
    """Compute intersection area as a percentage of the smaller rectangle."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    smaller_area = min(area1, area2)
    return intersection_area / smaller_area if smaller_area > 0 else 0.0

def contours_filter_overlapping_rectangles(rectangulos_info, iou_threshold=0.5, overlap_threshold=0.65):
    """Filter overlapping rectangles using IoU and the overlap percentage."""
    if not rectangulos_info:
        return []
    rectangulos_sorted = sorted(rectangulos_info, key=lambda x: x['area'], reverse=True)
    filtered_rectangles = []
    for i, rect_current in enumerate(rectangulos_sorted):
        should_keep = True
        current_coords = rect_current['coordenadas_originales']
        for rect_filtered in filtered_rectangles:
            filtered_coords = rect_filtered['coordenadas_originales']
            iou = contours_calculate_iou(current_coords, filtered_coords)
            overlap_pct = contours_calculate_overlap_percentage(current_coords, filtered_coords)
            if iou > iou_threshold or overlap_pct > overlap_threshold:
                should_keep = False
                break
        if should_keep:
            filtered_rectangles.append(rect_current)
    return filtered_rectangles

def contours_non_max_suppression_rectangles(rectangulos_info, overlap_threshold=0.3):
    """Apply non-maximum suppression to rectangle proposals."""
    if not rectangulos_info:
        return []
    boxes = []
    scores = []
    for rect in rectangulos_info:
        x, y, w, h = rect['coordenadas_originales']
        boxes.append([x, y, x + w, y + h])
        scores.append(rect['area'])
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, overlap_threshold)
    filtered_rectangles = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_rectangles.append(rectangulos_info[i])
    return filtered_rectangles

class contours_RectangleDetector:

    def __init__(self, elongation_factor=1.3, rects_min_area_factor=0.001, rects_max_area_factor=0.3, compactness_threshold=0.3, length_diff_threshold=0.1, nms_threshold=10, lineImg_drawing_width=1, rectImg_drawing_width=2, draw_graph=True, draw_trapezoids=True, whiteness_threshold=0.85, angle_threshold=0.25, max_num_trapezoids=50, iou_threshold=0.35):
        self.elongation_factor = elongation_factor
        self.rects_min_area_factor = rects_min_area_factor
        self.rects_max_area_factor = rects_max_area_factor
        self.compactness_threshold = compactness_threshold
        self.length_diff_threshold = length_diff_threshold
        self.nms_threshold = nms_threshold
        self.lineImg_drawing_width = lineImg_drawing_width
        self.rectImg_drawing_width = rectImg_drawing_width
        self.draw_graph = (draw_graph,)
        self.draw_trapezoids = draw_trapezoids
        self.whiteness_threshold = whiteness_threshold
        self.angle_threshold = angle_threshold
        self.max_num_trapezoids = max_num_trapezoids
        self.iou_threshold = iou_threshold

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype='float32')
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype(int)

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

    def is_white_line(self, img, p1, p2):
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if dist == 0:
            return False
        white_count, total = (0, 0)
        for t in np.linspace(0, 1, dist):
            x = int(round(p1[0] + t * (p2[0] - p1[0])))
            y = int(round(p1[1] + t * (p2[1] - p1[1])))
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                if img[y, x] > 64:
                    white_count += 1
                total += 1
        return total > 0 and white_count / total >= self.whiteness_threshold

    def intersect(self, line1, line2):
        p = np.array([line1[0], line1[1]])
        r = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        q = np.array([line2[0], line2[1]])
        s = np.array([line2[2] - line2[0], line2[3] - line2[1]])
        rxs = r[0] * s[1] - r[1] * s[0]
        if rxs == 0:
            return None
        t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / rxs
        u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / rxs
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p + t * r
            return (int(round(intersection[0])), int(round(intersection[1])))
        return None

    def trapezoid_line_test(self, a, b, c, d):
        lengths = [np.linalg.norm(np.array(a) - np.array(b)), np.linalg.norm(np.array(b) - np.array(c)), np.linalg.norm(np.array(c) - np.array(d)), np.linalg.norm(np.array(d) - np.array(a))]
        min_len = min(lengths)
        max_len = max(lengths)
        return min_len / max_len >= self.length_diff_threshold if max_len > 0 else False

    def trapezoid_area_test(self, a, b, c, d, min_area, max_area):
        pts = np.array([a, b, c, d])
        area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
        if area < min_area or area > max_area:
            return False
        perimeter = sum((np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)))
        if perimeter == 0:
            return False
        pp = 4 * np.pi * area / (perimeter * perimeter)
        return pp >= self.compactness_threshold

    def trapezoid_angles_test(self, a, b, c, d):

        def angle(pt1, pt2, pt3):
            v1 = np.array(pt1) - np.array(pt2)
            v2 = np.array(pt3) - np.array(pt2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 == 0 or norm_v2 == 0:
                return 0
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            return np.arccos(np.clip(cos_angle, -1.0, 1.0))
        pts = [a, b, c, d]
        for i in range(4):
            ang = angle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
            if not np.pi / 2 - self.angle_threshold <= ang <= np.pi / 2 + self.angle_threshold:
                return False
        return True

    def is_rectangle(self, pts):
        return self.trapezoid_angles_test(*pts)

    def non_maximum_suppression_points(self, points):
        if not points:
            return []
        points_np = np.array(points)
        filtered_points = []
        suppressed = [False] * len(points_np)
        for i in range(len(points_np)):
            if suppressed[i]:
                continue
            current_point = points_np[i]
            close_pts = [current_point]
            for j in range(i + 1, len(points_np)):
                if suppressed[j]:
                    continue
                distance = np.linalg.norm(current_point - points_np[j])
                if distance < self.nms_threshold:
                    suppressed[j] = True
                    close_pts.append(points_np[j])
            avg_pt = np.mean(close_pts, axis=0)
            filtered_points.append((int(avg_pt[0]), int(avg_pt[1])))
        return filtered_points

    def extract_lines(self, gray):
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny_th1 = float(0.33 * otsu_val)
        canny_th2 = float(otsu_val)
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=10, distance_threshold=1.414, canny_th1=canny_th1, canny_th2=canny_th2, canny_aperture_size=3, do_merge=True)
        lines = fld.detect(gray)
        if lines is None:
            return []
        lines = [line[0].tolist() for line in lines]
        return self.elongate_lines(lines)

    def find_intersections(self, lines):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self.intersect(lines[i], lines[j])
                if pt:
                    intersections.append(pt)
        return self.non_maximum_suppression_points(intersections)

    def create_graph(self, intersections, lines_img):
        size = len(intersections)
        adj = [[False] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                if self.is_white_line(lines_img, intersections[i], intersections[j]):
                    adj[i][j] = adj[j][i] = True
        return adj

    def extract_trapezoids(self, intersections, adjacency, image):
        trapezoids = []
        seen = set()
        min_area = self.rects_min_area_factor * image.shape[1] * image.shape[0]
        max_area = self.rects_max_area_factor * image.shape[1] * image.shape[0]
        n = len(intersections)
        for i in range(n):
            for j in range(i + 1, n):
                if not adjacency[i][j]:
                    continue
                for k in range(i + 1, n):
                    if not adjacency[k][j]:
                        continue
                    for l in range(j + 1, n):
                        if not (adjacency[k][l] and adjacency[i][l]):
                            continue
                        raw_pts = np.array([intersections[x] for x in [i, j, k, l]])
                        pts = self.order_points_clockwise(raw_pts)
                        contour = np.array(pts)
                        if not self.trapezoid_line_test(*pts):
                            continue
                        if not self.trapezoid_area_test(*pts, min_area, max_area):
                            continue
                        if not self.trapezoid_angles_test(*pts):
                            continue
                        if not cv2.isContourConvex(contour):
                            continue
                        if contours_DEDUP_BEFORE_CAP:
                            canonical = tuple(sorted((tuple(p) for p in pts)))
                            if canonical in seen:
                                continue
                            seen.add(canonical)
                        trapezoids.append(pts)
                        if len(trapezoids) >= self.max_num_trapezoids:
                            return trapezoids
        return trapezoids

    def non_maximum_suppression_shapes(self, shapes):
        if not shapes:
            return []
        bboxes = []
        for shape in shapes:
            x_coords = [p[0] for p in shape]
            y_coords = [p[1] for p in shape]
            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
            bboxes.append([x1, y1, x2, y2])
        indices = cv2.dnn.NMSBoxes(bboxes, [1.0] * len(bboxes), 0.5, self.iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices[0], (list, np.ndarray)):
            indices = [i[0] for i in indices]
        filtered_shapes = [shapes[i] for i in indices]
        return filtered_shapes

    def process_roi(self, roi_image):
        gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        lines_img = np.zeros_like(gray_roi)
        lines = self.extract_lines(gray_roi)
        for line in lines:
            cv2.line(lines_img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, self.lineImg_drawing_width, cv2.LINE_AA)
        lines_img_blur = cv2.GaussianBlur(lines_img, (5, 5), 0)
        intersections = self.find_intersections(lines)
        adjacency = self.create_graph(intersections, lines_img_blur)
        trapezoids = self.extract_trapezoids(intersections, adjacency, roi_image)
        rectangles = []
        other_trapezoids = []
        for t in trapezoids:
            if self.is_rectangle(t):
                rectangles.append(t)
            else:
                other_trapezoids.append(t)
        return (lines, intersections, rectangles, other_trapezoids)

def contours_reduce_saturation_brightness(image, saturation_scale=0.5, value_scale=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s *= saturation_scale
    v *= value_scale
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def contours_detectar_rectangulos(image, expansion=10):
    image = contours_reduce_saturation_brightness(image, 0.8, 0.8)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, contours_CANNY_LOW, contours_CANNY_HIGH)
    if contours_CLOSE_KERNEL:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((contours_CLOSE_KERNEL, contours_CLOSE_KERNEL), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    output_filtrada = image.copy()
    black_img = np.zeros_like(image)
    black2_img = image.copy()
    lines_img = np.zeros_like(gray)
    detector = contours_RectangleDetector()
    lines = detector.extract_lines(gray)
    for line in lines:
        cv2.line(lines_img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), 255, 1, cv2.LINE_AA)
    lines_img_blur = cv2.GaussianBlur(lines_img, (5, 5), 0)
    intersections = detector.find_intersections(lines)
    adjacency = detector.create_graph(intersections, lines_img_blur)
    graph_black = np.zeros_like(image)
    rectangulos_info = []
    for i, cnt in enumerate(contours):
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        epsilon = contours_APPROX_EPS * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contour_area = cv2.contourArea(cnt)
        if not contours_AREA_LO < contour_area < contours_AREA_HI:
            continue
        if not 4 <= len(approx) <= contours_MAX_VERTICES:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(np.round(box))
        rect_area = cv2.contourArea(box)
        if not contours_AREA_LO < rect_area < contours_AREA_HI:
            continue
        rectangularity = contour_area / rect_area
        if rectangularity < contours_MIN_RECTANGULARITY:
            continue
        x, y, w, h = cv2.boundingRect(box)
        if w <= 1 or h <= 1:
            continue
        x_exp = max(0, x - expansion)
        y_exp = max(0, y - expansion)
        w_exp = min(width - x_exp, w + 2 * expansion)
        h_exp = min(height - y_exp, h + 2 * expansion)
        vertices = box.reshape(-1, 2)
        box_contour = box.reshape((-1, 1, 2))
        vertices_originales = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        vertices_expandidos = [(x_exp, y_exp), (x_exp + w_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), (x_exp, y_exp + h_exp)]
        for vx, vy in vertices_originales:
            cv2.circle(black_img, (int(vx), int(vy)), 2, (255, 255, 255), -1)
        for vx, vy in vertices_expandidos:
            cv2.circle(black_img, (int(vx), int(vy)), 2, (255, 255, 255), -1)
        rect_info = {'id': i, 'coordenadas_originales': (int(x), int(y), int(w), int(h)), 'coordenadas_expandidas': (int(x_exp), int(y_exp), int(w_exp), int(h_exp)), 'area': float(rect_area), 'vertices': vertices.tolist(), 'contorno': box_contour}
        rectangulos_info.append(rect_info)
        cv2.rectangle(output, (int(x_exp), int(y_exp)), (int(x_exp + w_exp), int(y_exp + h_exp)), (0, 0, 255), 1)
        cv2.putText(output, f'ID: {i}', (int(x_exp), max(0, int(y_exp - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    rectangulos_filtrados = contours_filter_overlapping_rectangles(rectangulos_info, iou_threshold=0.5, overlap_threshold=0.65)
    for rect_info in rectangulos_filtrados:
        approx = np.array(rect_info['contorno'])
        vertices = np.array(rect_info['vertices'])
        x, y, w, h = rect_info['coordenadas_originales']
        x_exp, y_exp, w_exp, h_exp = rect_info['coordenadas_expandidas']
        vertices_originales = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        vertices_expandidos = [(x_exp, y_exp), (x_exp + w_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), (x_exp, y_exp + h_exp)]
        for vx, vy in vertices_expandidos:
            cv2.circle(black_img, (vx, vy), 2, (255, 255, 255), -1)
        vertices = approx.reshape(-1, 2)
        centro_x = int(np.mean(vertices[:, 0]))
        centro_y = int(np.mean(vertices[:, 1]))
        punto_centro = (centro_x, centro_y)
        for i, vertex in enumerate(vertices):
            punto = tuple(vertex.astype(int))
            cv2.circle(output_filtrada, punto, 4, (255, 255, 255), -1)
            cv2.circle(output_filtrada, punto, 4, (0, 0, 0), 2)
        for i in range(len(vertices)):
            pt1 = tuple(vertices[i].astype(int))
            pt2 = tuple(vertices[(i + 1) % len(vertices)].astype(int))
            cv2.line(output_filtrada, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)
    all_detected_fld_rects = []
    all_detected_fld_trapezoids = []
    all_fld_lines = []
    all_fld_intersections = []
    roi_validation_records = []
    for rect_info in rectangulos_info:
        x_orig, y_orig, w_orig, h_orig = rect_info['coordenadas_originales']
        x_exp, y_exp, w_exp, h_exp = rect_info['coordenadas_expandidas']
        roi = image[y_exp:y_exp + h_exp, x_exp:x_exp + w_exp].copy()
        if roi.size == 0:
            print(f"ROI for ID {rect_info['id']} is empty. Skipping.")
            continue
        lines_roi, intersections_roi, rects_fld_roi, traps_fld_roi = detector.process_roi_at(roi, (x_exp, y_exp, w_exp, h_exp))
        roi_vertices = [[int(v[0]), int(v[1])] for v in rect_info.get('vertices', [])]
        inner_rectangles_global = []
        for r_pts in rects_fld_roi:
            inner_rectangles_global.append([[int(round(p[0] + x_exp)), int(round(p[1] + y_exp))] for p in r_pts])
        roi_validation_records.append({'roi_id': int(rect_info['id']), 'roi_pts': roi_vertices, 'roi_bbox_original': [int(x_orig), int(y_orig), int(w_orig), int(h_orig)], 'roi_bbox_expanded': [int(x_exp), int(y_exp), int(w_exp), int(h_exp)], 'validated': bool(len(rects_fld_roi) > 0), 'inner_rectangles': inner_rectangles_global, 'num_inner_rectangles': int(len(rects_fld_roi))})
        for line in lines_roi:
            all_fld_lines.append([int(line[0] + x_exp), int(line[1] + y_exp), int(line[2] + x_exp), int(line[3] + y_exp)])
        for pt in intersections_roi:
            all_fld_intersections.append((int(pt[0] + x_exp), int(pt[1] + y_exp)))
        for r_pts in rects_fld_roi:
            mapped_r_pts = [(p[0] + x_exp, p[1] + y_exp) for p in r_pts]
            all_detected_fld_rects.append(mapped_r_pts)
        for t_pts in traps_fld_roi:
            mapped_t_pts = [(p[0] + x_exp, p[1] + y_exp) for p in t_pts]
            all_detected_fld_trapezoids.append(mapped_t_pts)
    final_fld_rects = detector.non_maximum_suppression_shapes(all_detected_fld_rects)
    final_fld_trapezoids = detector.non_maximum_suppression_shapes(all_detected_fld_trapezoids)
    output_fld_image = image.copy()
    for line in all_fld_lines:
        cv2.line(output_fld_image, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 1, cv2.LINE_AA)
    half = len(lines) // 2
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = map(int, line)
        cv2.line(black2_img, (x1, y1), (x2, y2), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(black2_img, (x1, y1), 3, (255, 255, 255), -1)
        cv2.circle(black2_img, (x2, y2), 3, (255, 255, 255), -1)
        cv2.line(black_img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        if i % 2 == 0:
            xm = (x1 + x2) // 2
            ym = (y1 + y2) // 2
            cv2.circle(black_img, (xm, ym), 3, (255, 255, 255), -1)
    for line in all_fld_lines:
        cv2.line(black_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2, cv2.LINE_AA)
    for pt in all_fld_intersections:
        cv2.circle(output_fld_image, pt, 2, (0, 0, 255), -1)
        cv2.circle(black_img, pt, 2, (0, 0, 255), -1)
        for trap_pts in final_fld_trapezoids:
            pts = np.array(trap_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output_fld_image, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    return (output, output_filtrada, lines_img, rectangulos_info, black_img, output_fld_image, black2_img, rectangulos_filtrados, final_fld_rects, roi_validation_records)


contours_AREA_LO = 500
contours_AREA_HI = 50000
contours_MAX_VERTICES = 5
contours_MIN_RECTANGULARITY = 0.72
contours_CLOSE_KERNEL = 0
contours_CANNY_LOW = 20
contours_CANNY_HIGH = 150
contours_APPROX_EPS = 0.02
contours_DEDUP_BEFORE_CAP = False
cv2.setNumThreads(1)
contours_CLOSE_KERNEL = 3
trace_orig_init = contours_RectangleDetector.__init__
trace_orig_order = contours_RectangleDetector.order_points_clockwise
trace_variant = 'baseline'

def trace_init(self, *args, **kwargs):
    trace_orig_init(self, *args, **kwargs)
    if trace_variant != 'resize320':
        self.rects_max_area_factor = 0.95
    self.angle_threshold = 1.0
    self.compactness_threshold = 0.1
    self.nms_threshold = 4
    self.elongation_factor = 1.8
    self.lineImg_drawing_width = 2
    self.whiteness_threshold = 0.65
contours_RectangleDetector.__init__ = trace_init

def trace_center(self, pts):
    p = np.asarray(pts)
    c = p.mean(axis=0)
    return p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
trace_TRACE = []
trace_original_process = contours_RectangleDetector.process_roi

def trace_process(self, img):
    self._record = {'width': img.shape[1], 'height': img.shape[0], 'counts': {}}
    trace_TRACE.append(self._record)
    result = trace_original_process(self, img)
    self._record['counts']['rectangles'] = len(result[2])
    self._record['counts']['other'] = len(result[3])
    return result
contours_RectangleDetector.process_roi = trace_process

def trace_instrument(name, kind):
    original = getattr(contours_RectangleDetector, name)

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if hasattr(self, '_record'):
            c = self._record['counts']
            if kind == 'gate':
                c[name + '_tested'] = c.get(name + '_tested', 0) + 1
                c[name + '_passed'] = c.get(name + '_passed', 0) + int(result)
            elif kind == 'graph':
                c['graph_edges'] = int(np.asarray(result).sum() // 2)
            else:
                c[name] = len(result)
        return result
    setattr(contours_RectangleDetector, name, wrapped)
for trace_name in ['extract_lines', 'find_intersections', 'extract_trapezoids']:
    trace_instrument(trace_name, 'length')
trace_instrument('create_graph', 'graph')
for trace_name in ['trapezoid_line_test', 'trapezoid_area_test', 'trapezoid_angles_test']:
    trace_instrument(trace_name, 'gate')

cv2.setNumThreads(1)
candidates_CACHE = {}
candidates_ACTIVE = False
candidates_CURRENT_CAP = 50
candidates_hits = 0
candidates_original_extract = contours_RectangleDetector.extract_lines
candidates_original_process = contours_RectangleDetector.process_roi
candidates_original_init = contours_RectangleDetector.__init__

def candidates_init(self, *a, **kw):
    candidates_original_init(self, *a, **kw)
    self.max_num_trapezoids = candidates_CURRENT_CAP

def candidates_extract(self, gray):
    return candidates_original_extract(self, gray) if candidates_ACTIVE else []

def candidates_process(self, img):
    global candidates_ACTIVE, candidates_hits
    key = (img.shape, hashlib.sha256(img.tobytes()).hexdigest(), candidates_CURRENT_CAP, contours_DEDUP_BEFORE_CAP)
    if key in candidates_CACHE:
        candidates_hits += 1
        return copy.deepcopy(candidates_CACHE[key])
    candidates_ACTIVE = True
    try:
        result = candidates_original_process(self, img)
    finally:
        candidates_ACTIVE = False
    candidates_CACHE[key] = copy.deepcopy(result)
    return result
contours_RectangleDetector.__init__ = candidates_init
contours_RectangleDetector.extract_lines = candidates_extract
contours_RectangleDetector.process_roi = candidates_process
contours_RectangleDetector.order_points_clockwise = trace_center

cv2.setNumThreads(1)
pipeline_MODE = 'baseline'
pipeline_ACTIVE = False
pipeline_FRONT = None
pipeline_WORK_SCALE = 1.0
pipeline_LINE_TRACE = []
pipeline_original_elongate = contours_RectangleDetector.elongate_lines

def pipeline_elongate(self, lines):
    out = pipeline_original_elongate(self, lines)
    if pipeline_ACTIVE:
        self._line_record.update(raw_lines=copy.deepcopy(lines), extended_lines=copy.deepcopy(out))
    return out

def pipeline_extract(self, gray):
    if not pipeline_ACTIVE:
        return []
    if pipeline_MODE == 'baseline':
        return candidates_original_extract(self, gray)
    x, y, w, h = self._legacy_bbox
    lines, meta = pipeline_FRONT.lines_for((x, y, x + w, y + h), pipeline_WORK_SCALE, pipeline_MODE)
    self._line_record['frontend'] = meta
    self._record['counts']['extract_lines'] = len(lines)
    return self.elongate_lines(lines.tolist())

def pipeline_process_at(self, image, bbox):
    global pipeline_ACTIVE
    self._legacy_bbox = bbox
    self._line_record = dict(roi_working=list(bbox))
    pipeline_ACTIVE = True
    try:
        out = candidates_original_process(self, image)
    finally:
        pipeline_ACTIVE = False
    self._line_record.update(nodes=out[1], counts=self._record['counts'].copy())
    pipeline_LINE_TRACE.append(self._line_record)
    return out
contours_RectangleDetector.elongate_lines = pipeline_elongate
contours_RectangleDetector.extract_lines = pipeline_extract
contours_RectangleDetector.process_roi_at = pipeline_process_at


def corrected_nms(self, shapes):
    if not shapes:
        return []
    boxes = []
    for q in shapes:
        p = np.asarray(q)
        a = p.min(0)
        b = p.max(0)
        boxes.append([float(a[0]), float(a[1]), float(b[0] - a[0]), float(b[1] - a[1])])
    ids = np.asarray(cv2.dnn.NMSBoxes(boxes, [1.0] * len(boxes), 0.5, self.iou_threshold)).reshape(-1)
    return [shapes[int(i)] for i in ids]

def detect(im, ds):
    global candidates_CACHE, candidates_CURRENT_CAP, contours_APPROX_EPS, contours_AREA_HI, contours_AREA_LO, contours_CANNY_HIGH, contours_CANNY_LOW, contours_CLOSE_KERNEL, contours_DEDUP_BEFORE_CAP, contours_MAX_VERTICES, contours_MIN_RECTANGULARITY, pipeline_FRONT, pipeline_LINE_TRACE, pipeline_MODE, pipeline_WORK_SCALE, trace_TRACE
    cfg = pipeline_PARAM[ds]
    oh, ow = im.shape[:2]
    scale = cfg.get('working_longside', 320) / max(oh, ow)
    work = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pipeline_MODE = 'old_local' if ds == 'robotathome80' else 'baseline'
    pipeline_FRONT = roi_frontend_LegacyFrontend(im, ds)
    pipeline_WORK_SCALE = scale
    candidates_CURRENT_CAP = cfg['candidate_cap']
    contours_DEDUP_BEFORE_CAP = cfg['deduplicate_before_cap']
    contours_CANNY_LOW, contours_CANNY_HIGH = cfg['canny']
    contours_CLOSE_KERNEL = cfg['closing']
    contours_APPROX_EPS = cfg['epsilon']
    contours_AREA_LO = cfg['area_min']
    contours_AREA_HI = cfg['area_max']
    contours_MAX_VERTICES = cfg['max_vertices']
    contours_MIN_RECTANGULARITY = cfg['min_rectangularity']
    candidates_CACHE = {}
    trace_TRACE = []
    pipeline_LINE_TRACE = []
    v = contours_detectar_rectangulos(work, expansion=cfg['expansion'])
    chosen = []
    for q in v[9]:
        parent = np.asarray(q['roi_pts']) / scale
        qs = [np.asarray(z) / scale for z in q['inner_rectangles']]
        if qs:
            scores = [proposal_geometry_poly_iou(z, parent) for z in qs]
            j = int(np.argmax(scores))
            if scores[j] >= cfg['parent_iou_min']:
                chosen.append(np.asarray(qs[j], dtype=np.int32))
                SCORE_MAP.setdefault(SCORE_KEY(np.asarray(qs[j], dtype=np.int32)), float(scores[j]))
    if cfg['nms_metric'] == 'axis_aligned_bbox':
        d = contours_RectangleDetector()
        d.iou_threshold = cfg['nms_iou']
        kept = corrected_nms(d, chosen)
    else:
        kept = []
        for q in chosen:
            if all((proposal_geometry_poly_iou(q, z) <= cfg['nms_iou'] for z in kept)):
                kept.append(q)
    return dict(width=ow, height=oh, polygons_px=[np.asarray(q).tolist() for q in kept])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--image')
    p.add_argument('--manifest')
    p.add_argument('--output', required=True)
    p.add_argument('--nms-iou', type=float)
    a = p.parse_args()
    cv2.setNumThreads(1)
    if a.nms_iou is not None:
        pipeline_PARAM[a.dataset]['nms_iou'] = a.nms_iou
    records = json.loads(Path(a.manifest).read_text()) if a.manifest else [dict(dataset=a.dataset, name=Path(a.image).name, path=a.image)]
    out = []
    for v in records:
        if v['dataset'] != a.dataset:
            continue
        path = Path(v['path'])
        path = path if path.is_absolute() else ROOT / path
        im = cv2.imread(str(path))
        out.append(dict(dataset=a.dataset, name=v['name'], **detect(im, a.dataset)))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out))
    print('COMPLETE', len(out), flush=True)
if __name__ == '__main__':
    main()
