"""Read the three annotation files using the fixed image manifest as an index."""
from pathlib import Path
import hashlib
import json
import math

DATASETS = ("robotathome80", "isaacsim50", "smartdoc90")


def load_dataset_records(data_dir, verify_images=False):
    """Return aligned image records, quadrilaterals and available semantic metadata.

    Geometry comes from each dataset's annotations.json. The manifest fixes the
    image order and checks that annotations still match the published benchmark.
    """
    data_dir = Path(data_dir)
    index = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    annotations = {}
    for dataset in DATASETS:
        path = data_dir / "images" / dataset / "annotations.json"
        annotations[dataset] = json.loads(path.read_text(encoding="utf-8"))
        expected = [row["name"] for row in index if row["dataset"] == dataset]
        if len(expected) != len(set(expected)) or set(annotations[dataset]) != set(expected):
            raise ValueError(f"Image/annotation correspondence mismatch: {dataset}")
    records = []
    for row in index:
        dataset, name = row["dataset"], row["name"]
        annotation = annotations[dataset][name]
        for field in ("width", "height", "image_sha256"):
            if annotation[field] != row[field]:
                raise ValueError(f"Annotation {field} mismatch: {dataset}/{name}")
        quads = annotation["quads"]
        if quads != row["gt"]:
            raise ValueError(f"Geometry differs from published benchmark: {dataset}/{name}")
        for quad in quads:
            if len(quad) != 4 or any(len(point) != 2 or not all(math.isfinite(v) for v in point) for point in quad):
                raise ValueError(f"Invalid quadrilateral: {dataset}/{name}")
        if "categories" in annotation and len(annotation["categories"]) != len(quads):
            raise ValueError(f"Category/quadrilateral correspondence mismatch: {dataset}/{name}")
        if dataset == "robotathome80" and "categories" not in annotation:
            raise ValueError(f"Missing RobotHome80 semantic labels: {name}")
        expected_path = f"dataset/images/{dataset}/{name}"
        if row["path"] != expected_path:
            raise ValueError(f"Unexpected image path: {row['path']}")
        image_path = data_dir / "images" / dataset / name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if verify_images and hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {dataset}/{name}")
        record = dict(row, gt=quads)
        for field in ("categories", "occlusions", "conditions", "reviewed", "category_kind", "document_group"):
            if field in annotation:
                record[field] = annotation[field]
        records.append(record)
    return records


def inference_records(records, release_root):
    """Build an image-only manifest; neither geometry nor semantics reach inference."""
    root = Path(release_root)
    return [
        {**{key: row[key] for key in ("dataset", "name", "width", "height", "image_sha256")},
         "path": str((root / row["path"]).resolve())}
        for row in records
    ]
