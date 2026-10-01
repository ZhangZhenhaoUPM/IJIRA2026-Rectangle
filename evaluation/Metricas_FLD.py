import csv
import json
import os
import sys

import numpy as np
from PIL import Image
from shapely.geometry import Polygon


# ============================================================
# CONFIGURACIÓN
# ============================================================

def clean_path(path):
    return str(path).replace("\u202a", "").replace("\u202c", "").strip()


# Standalone defaults; the release entrypoint is scripts/evaluate_results.py.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_FOLDER = os.path.join(BASE_DIR, "images")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "raw_data", "annotations.json")

DETECTOR_OUTPUT_DIR = os.path.join(BASE_DIR, "resul_detector")
CANDIDATES_JSON = os.path.join(
    DETECTOR_OUTPUT_DIR,
    "candidates_combined.json"
)

EVAL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "metricas_detector"
)

OUTPUT_CSV = os.path.join(
    EVAL_OUTPUT_DIR,
    "metricas_por_iou.csv"
)

DATASET_NAME = "Robot@Home 2"
METHOD_NAME = "Rectangle Detector"

# Se calculan TODOS estos IoU
IOU_THRESHOLDS = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

# Solo se marca visualmente este IoU
SELECTED_IOU = 0.5

# Filtro de detecciones aceptadas
MIN_AREA_RATIO = 0.0005
PRED_NMS_IOU_THRESHOLD = 0.50

# Filtro de TN:
# varios rectángulos rechazados muy solapados cuentan como un solo TN
TN_DEDUP_THRESHOLD = 0.50
TN_DEDUP_METRIC = "min"

INCLUDE_EMPTY_IMAGES = False


# ============================================================
# UTILIDADES
# ============================================================

def safe_div(a, b):
    return a / b if b else 0.0


def fmt(value):
    return f"{value:.4f}"


def print_table(headers, rows):
    widths = [
        max(
            len(str(headers[i])),
            max(len(str(row[i])) for row in rows)
        )
        for i in range(len(headers))
    ]

    line = "+-" + "-+-".join(
        "-" * width
        for width in widths
    ) + "-+"

    row_format = "| " + " | ".join(
        f"{{:<{width}}}"
        for width in widths
    ) + " |"

    print(line)
    print(row_format.format(*headers))
    print(line)

    for row in rows:
        print(row_format.format(*row))

    print(line)


# ============================================================
# GEOMETRÍA
# ============================================================

def points_to_polygon(points):
    try:
        polygon = Polygon(points)

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty or polygon.area <= 0:
            return None

        return polygon

    except Exception:
        return None


def bbox_to_polygon(bbox):
    x, y, w, h = bbox

    return points_to_polygon([
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h),
    ])


def calculate_iou(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0.0

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    if not poly1.intersects(poly2):
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union <= 0:
        return 0.0

    return intersection / union


def overlap_measure(poly1, poly2, metric="iou"):
    if poly1 is None or poly2 is None:
        return 0.0

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    if not poly1.intersects(poly2):
        return 0.0

    intersection = poly1.intersection(poly2).area

    if metric == "min":
        smaller_area = min(poly1.area, poly2.area)

        if smaller_area <= 0:
            return 0.0

        return intersection / smaller_area

    union = poly1.union(poly2).area

    if union <= 0:
        return 0.0

    return intersection / union


def nms_indices(
    polygons,
    indices,
    threshold,
    metric="iou"
):
    if not indices:
        return []

    keep = []

    for index in sorted(
        indices,
        key=lambda i: polygons[i].area,
        reverse=True
    ):
        duplicate = False

        for selected in keep:
            overlap = overlap_measure(
                polygons[index],
                polygons[selected],
                metric
            )

            if overlap > threshold:
                duplicate = True
                break

        if not duplicate:
            keep.append(index)

    return keep


# ============================================================
# GROUND TRUTH
# ============================================================

def coco_annotation_to_polygon(annotation):
    segmentation = annotation.get("segmentation")

    if (
        isinstance(segmentation, list)
        and segmentation
        and isinstance(segmentation[0], list)
        and len(segmentation[0]) >= 6
    ):
        flat = segmentation[0]

        points = [
            (flat[i], flat[i + 1])
            for i in range(0, len(flat) - 1, 2)
        ]

        polygon = points_to_polygon(points)

        if polygon is not None:
            return polygon

    bbox = annotation.get("bbox")

    if isinstance(bbox, list) and len(bbox) == 4:
        return bbox_to_polygon(bbox)

    return None


def get_image_size(filename, sizes):
    width, height = sizes.get(filename, (0, 0))

    if width and height:
        return width, height

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    if os.path.exists(image_path):
        with Image.open(image_path) as image:
            return image.size

    return None


def get_image_area(filename, sizes):
    size = get_image_size(
        filename,
        sizes
    )

    if size is None:
        return 0

    return size[0] * size[1]


def load_ground_truth(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    ground_truth = {}
    sizes = {}

    # --------------------------------------------------------
    # FORMATO COCO
    # --------------------------------------------------------

    if (
        isinstance(data, dict)
        and "images" in data
        and "annotations" in data
    ):
        id_to_name = {}

        for image_info in data["images"]:
            name = (
                (image_info.get("extra") or {}).get("name")
                or image_info.get("file_name")
            )

            if name is None:
                continue

            name = os.path.basename(str(name))

            id_to_name[
                image_info["id"]
            ] = name

            sizes[
                name
            ] = (
                image_info.get("width", 0),
                image_info.get("height", 0)
            )

            if INCLUDE_EMPTY_IMAGES:
                ground_truth.setdefault(
                    name,
                    []
                )

        for annotation in data["annotations"]:
            name = id_to_name.get(
                annotation.get("image_id")
            )

            if name is None:
                continue

            polygon = coco_annotation_to_polygon(
                annotation
            )

            if polygon is not None:
                ground_truth.setdefault(
                    name,
                    []
                ).append(
                    polygon
                )

    # --------------------------------------------------------
    # FORMATO DICT / QUADS
    # --------------------------------------------------------

    elif isinstance(data, dict):

        for filename, entry in data.items():
            filename = os.path.basename(
                str(filename)
            )

            if isinstance(entry, dict):
                quads = entry.get("quads", [])
            else:
                quads = entry

            quads = [
                quad
                for quad in quads
                if (
                    isinstance(quad, list)
                    and len(quad) >= 3
                )
            ]

            if not quads:
                if INCLUDE_EMPTY_IMAGES:
                    ground_truth.setdefault(
                        filename,
                        []
                    )
                continue

            all_values = [
                value
                for quad in quads
                for point in quad
                for value in point
            ]

            if max(all_values) <= 1.0 + 1e-6:
                size = get_image_size(
                    filename,
                    sizes
                )

                if size is None:
                    continue

                width, height = size

                quads = [
                    [
                        (
                            x * width,
                            y * height
                        )
                        for x, y in quad
                    ]
                    for quad in quads
                ]

            valid = []

            for quad in quads:
                polygon = points_to_polygon(quad)

                if polygon is not None:
                    valid.append(polygon)

            if valid:
                ground_truth[filename] = valid

    else:
        raise ValueError(
            "Formato de anotaciones no reconocido."
        )

    return ground_truth, sizes


# ============================================================
# CANDIDATOS
# ============================================================

def load_candidates(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "candidates_combined.json no tiene formato válido."
        )

    return {
        os.path.basename(str(name)): candidates
        for name, candidates in data.items()
    }


# ============================================================
# MATCHING 1 A 1
# ============================================================

def match_predictions(
    iou_matrix,
    threshold
):
    n_predictions, n_gt = iou_matrix.shape

    pairs = []

    for pred_index in range(n_predictions):
        for gt_index in range(n_gt):
            iou = iou_matrix[
                pred_index,
                gt_index
            ]

            if iou >= threshold:
                pairs.append(
                    (
                        float(iou),
                        pred_index,
                        gt_index
                    )
                )

    pairs.sort(reverse=True)

    matched_predictions = set()
    matched_gt = set()
    matches = []

    for (
        iou,
        pred_index,
        gt_index
    ) in pairs:

        if (
            pred_index in matched_predictions
            or gt_index in matched_gt
        ):
            continue

        matched_predictions.add(
            pred_index
        )

        matched_gt.add(
            gt_index
        )

        matches.append(
            (
                pred_index,
                gt_index,
                iou
            )
        )

    return matches


# ============================================================
# PREPARAR DATOS
# ============================================================

def build_records(
    ground_truth,
    sizes,
    candidates_by_image
):
    records = []

    for filename in sorted(ground_truth):
        gt_objects = ground_truth[filename]

        candidates = candidates_by_image.get(
            filename,
            []
        )

        polygons = [
            points_to_polygon(
                candidate.get("pts", [])
            )
            for candidate in candidates
        ]

        image_area = get_image_area(
            filename,
            sizes
        )

        min_area = (
            MIN_AREA_RATIO
            * image_area
        )

        # ----------------------------------------------------
        # ACEPTADOS
        # ----------------------------------------------------

        accepted_indices = []

        for index, candidate in enumerate(candidates):
            if not candidate.get(
                "accepted",
                False
            ):
                continue

            polygon = polygons[index]

            if polygon is None:
                continue

            if polygon.area < min_area:
                continue

            accepted_indices.append(index)

        if (
            PRED_NMS_IOU_THRESHOLD is not None
            and accepted_indices
        ):
            accepted_indices = nms_indices(
                polygons,
                accepted_indices,
                PRED_NMS_IOU_THRESHOLD,
                metric="iou"
            )

        accepted_set = set(
            accepted_indices
        )

        accepted = [
            polygons[index]
            for index in accepted_indices
        ]

        # ----------------------------------------------------
        # RECHAZADOS
        # ----------------------------------------------------

        rejected = []

        for index in range(len(candidates)):
            if index in accepted_set:
                continue

            polygon = polygons[index]

            if polygon is None:
                continue

            rejected.append(polygon)

        # ----------------------------------------------------
        # DEDUPLICACIÓN TN
        # ----------------------------------------------------

        if (
            rejected
            and TN_DEDUP_THRESHOLD is not None
        ):
            unique_indices = set(
                nms_indices(
                    rejected,
                    list(range(len(rejected))),
                    TN_DEDUP_THRESHOLD,
                    metric=TN_DEDUP_METRIC
                )
            )

            tn_mask = np.array(
                [
                    index in unique_indices
                    for index in range(len(rejected))
                ],
                dtype=bool
            )

        else:
            tn_mask = np.ones(
                len(rejected),
                dtype=bool
            )

        # ----------------------------------------------------
        # IoU ACEPTADOS vs GT
        # ----------------------------------------------------

        iou_accepted = np.zeros(
            (
                len(accepted),
                len(gt_objects)
            ),
            dtype=float
        )

        for pred_index, prediction in enumerate(accepted):
            for gt_index, gt in enumerate(gt_objects):
                iou_accepted[
                    pred_index,
                    gt_index
                ] = calculate_iou(
                    prediction,
                    gt
                )

        # ----------------------------------------------------
        # IoU RECHAZADOS vs GT
        # ----------------------------------------------------

        iou_rejected = np.zeros(
            (
                len(rejected),
                len(gt_objects)
            ),
            dtype=float
        )

        for pred_index, prediction in enumerate(rejected):
            for gt_index, gt in enumerate(gt_objects):
                iou_rejected[
                    pred_index,
                    gt_index
                ] = calculate_iou(
                    prediction,
                    gt
                )

        records.append({
            "iou_acc": iou_accepted,
            "iou_rej": iou_rejected,
            "tn_mask": tn_mask,
        })

    return records


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    records,
    threshold
):
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for record in records:
        iou_accepted = record["iou_acc"]
        iou_rejected = record["iou_rej"]

        number_predictions = iou_accepted.shape[0]
        number_gt = iou_accepted.shape[1]

        matches = match_predictions(
            iou_accepted,
            threshold
        )

        matched_gt = {
            gt_index
            for _, gt_index, _ in matches
        }

        TP += len(matches)

        FP += (
            number_predictions
            - len(matches)
        )

        FN += (
            number_gt
            - len(matches)
        )

        # ----------------------------------------------------
        # TN
        # ----------------------------------------------------

        number_rejected = iou_rejected.shape[0]

        if number_rejected:
            unmatched_gt = [
                gt_index
                for gt_index in range(number_gt)
                if gt_index not in matched_gt
            ]

            if unmatched_gt:
                could_be_missing_gt = (
                    iou_rejected[
                        :,
                        unmatched_gt
                    ]
                    >= threshold
                ).any(axis=1)

            else:
                could_be_missing_gt = np.zeros(
                    number_rejected,
                    dtype=bool
                )

            TN += int(
                (
                    (~could_be_missing_gt)
                    & record["tn_mask"]
                ).sum()
            )

    accuracy = safe_div(
        TP + TN,
        TP + TN + FP + FN
    )

    precision = safe_div(
        TP,
        TP + FP
    )

    recall = safe_div(
        TP,
        TP + FN
    )

    specificity = safe_div(
        TN,
        TN + FP
    )

    f1_score = safe_div(
        2 * precision * recall,
        precision + recall
    )

    return {
        "IoU": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1_score,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(
            f"No existe la carpeta: {IMAGE_FOLDER}"
        )

    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(
            f"No existe: {ANNOTATIONS_FILE}"
        )

    if not os.path.exists(CANDIDATES_JSON):
        raise FileNotFoundError(
            f"No existe: {CANDIDATES_JSON}"
        )

    os.makedirs(
        EVAL_OUTPUT_DIR,
        exist_ok=True
    )

    ground_truth, sizes = load_ground_truth(
        ANNOTATIONS_FILE
    )

    candidates_by_image = load_candidates(
        CANDIDATES_JSON
    )

    common = (
        set(ground_truth)
        & set(candidates_by_image)
    )

    if not common:
        raise RuntimeError(
            "No coinciden los nombres de imagen "
            "entre anotaciones y candidatos."
        )

    records = build_records(
        ground_truth,
        sizes,
        candidates_by_image
    )

    # --------------------------------------------------------
    # CALCULAR 0.1 - 0.9
    # --------------------------------------------------------

    results = [
        calculate_metrics(
            records,
            threshold
        )
        for threshold in IOU_THRESHOLDS
    ]

    # --------------------------------------------------------
    # TABLA COMPLETA: IoU 0.1 - 0.9
    # --------------------------------------------------------

    headers = [
        "IoU",
        "Dataset",
        "Method",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1 Score"
    ]

    rows = []

    for result in results:

        rows.append([
            f"{result['IoU']:.1f}",
            DATASET_NAME,
            METHOD_NAME,
            fmt(result["Accuracy"]),
            fmt(result["Precision"]),
            fmt(result["Recall"]),
            fmt(result["Specificity"]),
            fmt(result["F1 Score"]),
        ])

    print()
    print(
        "=" * 92
    )

    print(
        "RESULTADOS POR IoU"
    )

    print(
        "=" * 92
    )

    print_table(
        headers,
        rows
    )


    # --------------------------------------------------------
    # RESUMEN APARTE: SOLO IoU = 0.5
    # --------------------------------------------------------

    selected_result = next(
        (
            result
            for result in results
            if abs(
                result["IoU"]
                - SELECTED_IOU
            ) < 1e-9
        ),
        None
    )

    if selected_result is None:

        print(
            f"\nNo se encontró resultado "
            f"para IoU = {SELECTED_IOU:.1f}"
        )

    else:

        print()
        print(
            "=" * 62
        )

        print(
            "RESUMEN — IoU = 0.5"
        )

        print(
            "=" * 62
        )

        print(
            f"Dataset      : {DATASET_NAME}"
        )

        print(
            f"Method       : {METHOD_NAME}"
        )

        print(
            f"IoU          : "
            f"{selected_result['IoU']:.1f}"
        )

        print(
            f"Accuracy     : "
            f"{selected_result['Accuracy']:.4f}"
        )

        print(
            f"Precision    : "
            f"{selected_result['Precision']:.4f}"
        )

        print(
            f"Recall       : "
            f"{selected_result['Recall']:.4f}"
        )

        print(
            f"Specificity  : "
            f"{selected_result['Specificity']:.4f}"
        )

        print(
            f"F1 Score     : "
            f"{selected_result['F1 Score']:.4f}"
        )

        print(
            "=" * 62
        )


    # --------------------------------------------------------
    # CSV: TODOS LOS IoU, SIN DATOS EXTRA
    # --------------------------------------------------------

    csv_headers = [
        "IoU",
        "Dataset",
        "Method",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1 Score"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_headers
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                "IoU": result["IoU"],
                "Dataset": DATASET_NAME,
                "Method": METHOD_NAME,
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "Specificity": result["Specificity"],
                "F1 Score": result["F1 Score"],
            })


if __name__ == "__main__":
    main()
