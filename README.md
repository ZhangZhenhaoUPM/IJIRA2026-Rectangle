# Rectangle Detection — Version 1.0

Version 1.0 reproduces six rectangle detection methods on RobotHome80, IsaacSim50 and SmartDoc90. The four RIDGE variants share eight numerical geometry settings within each dataset, with distinct backends and method-specific candidate selection.

## Source layout

The main project algorithm for each method is in its corresponding Python file, together with its command-line interface. Functions and classes are defined directly in those files.

| File | Algorithm code in that file |
|---|---|
| `code/ridge-fld.py` | FLD preprocessing, normalized candidate generation, centroid branch, selection and inference flow |
| `code/ridge-airline.py` | AirLine frontend and CRG adaptation, centroid branch, selection and inference flow |
| `code/ridge-roi.py` | ROI frontend, contour proposals, geometry checks, refinement, candidate caps and selection |
| `code/ridge-yolo.py` | Proposal handling, RIDGE geometry, cascade passes, scoring and selection |
| `code/yolo-br.py` | Supporting lines, interval support, compatible cycles, quadrilateral selection and inference |
| `code/mlsd.py` | Dataset settings, official M-LSD invocation and selected-output handling |

`code/common/graph/` retains the graph tools used by both FLD and AirLine: geometric primitives, C++ bindings, candidate pools, features and selection utilities. `code/common/runtime.py` builds native graph kernels and prepares temporary protocol files. `code/common/datasets.py` reads the per-dataset annotations and builds image-only inference records. `code/common/configuration.py` resolves references to the shared geometry settings.

Installed libraries (NumPy, OpenCV, PyTorch, SciPy, Ultralytics and TensorFlow) are imported normally. `third_party/dexined/` and `third_party/mlsd/` each retain one copy of the supplied upstream implementation tied to the recorded models. C++ kernels are in `code/native/graph/` and `code/native/airline/`. Runtime settings and frontend protocols are stored in each method's file under `parameters/`.

Separate RIDGE backends, frontend resolutions and selected parameters are preserved. `scripts/reproduce.py` runs each detector in a fresh process and applies the paper configuration. All 18 cases were rerun: predicted coordinates exactly match the references, all exported scores match, and all five metrics match at four decimals. AirLine used fresh neural frontend inference on all 220 images. Graph kernels and the AirLine CRG extension were compiled from `code/native/`. See `VALIDATION.json`.

## What is reproduced

Six methods on 220 images: RobotHome80 (80 images, 146 objects), IsaacSim50 (50, 76), and SmartDoc90 (90, 90). Reference predictions, native scores and the five-metric table are versioned. All dataset images and annotations are included in `dataset/`. Model weights and the optional AirLine frontend cache are available from [Release v1.0](https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/tag/v1.0).

| Method | RobotHome80 F1 | IsaacSim50 F1 | SmartDoc90 F1 |
|---|---:|---:|---:|
| RIDGE-FLD | 0.6833 | 0.6667 | 0.9162 |
| RIDGE-AirLine | 0.6537 | 0.7059 | 0.8712 |
| RIDGE-ROI | 0.6692 | 0.7273 | 0.9112 |
| RIDGE-YOLO | 0.8028 | 0.7259 | 0.9944 |
| YOLO-BR | 0.7107 | 0.8400 | 0.9535 |
| M-LSD | 0.4538 | 0.6232 | 0.9889 |

Full precision values: `reference/current_all_methods.csv`. Metric definitions and limitations appear below.

## Obtain the repository and models

Clone this repository, or use GitHub's **Code > Download ZIP** and extract it:

```bash
git clone https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle.git
cd IJIRA2026-Rectangle
python scripts/verify_bundle.py --datasets
```

The repository includes all 220 images, 312 annotated quadrilaterals, and one `annotations.json` for each dataset:

```text
dataset/
  manifest.json
  images/
    robotathome80/   # 80 images + annotations.json
    isaacsim50/      # 50 images + annotations.json
    smartdoc90/      # 90 images + annotations.json
```

Only RobotHome80 has object semantic labels. IsaacSim50 and SmartDoc90 provide geometry annotations; SmartDoc document groups are separate metadata. See [the annotation guide](dataset/README_annotations.md). Table reproduction from saved predictions needs no model download.

For detector inference, download [models.zip](https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/download/v1.0/models.zip) and, for cached AirLine inference, [frontend_cache.zip](https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/download/v1.0/frontend_cache.zip). The model archive contains 15 YOLO checkpoints, DexiNed and the official M-LSD TFLite checkpoint. The optional cache contains 220 image-hash-keyed AirLine/CRG line arrays. It contains no ground truth or final rectangle predictions.

Install the environment below, then extract the downloads into an external assets directory and run the preparation script:

```bash
mkdir -p ../ridge_assets
curl -L --fail -o ../ridge_assets/models.zip https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/download/v1.0/models.zip
unzip ../ridge_assets/models.zip -d ../ridge_assets
curl -L --fail -o ../ridge_assets/frontend_cache.zip https://github.com/ZhangZhenhaoUPM/IJIRA2026-Rectangle/releases/download/v1.0/frontend_cache.zip
unzip ../ridge_assets/frontend_cache.zip -d ../ridge_assets
python scripts/prepare_assets.py --assets ../ridge_assets
```

The archives contain their own `models/` and `frontend_cache/` folders. The preparation script verifies the included dataset and downloaded files before copying model weights and cache arrays into the runtime layout. It refuses to replace different existing files. To omit the optional cache, skip its download and extraction commands and run inference with `--fresh-airline`. Archive links and checksums are in `configs/downloads.json`.

## Environment

Full inference targets Linux, Python 3.12, `g++`, Python development headers, and an NVIDIA GPU. FLD/AirLine compile the C++ sources in `code/native/` into temporary directories. [The native code guide](code/native/README.md) describes each kernel. TensorFlow/M-LSD uses a separate environment to keep its dependencies isolated.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python -m pip install --no-deps opencv-contrib-python-headless==4.12.0.88
python -c "import cv2,numpy,torch,shapely; assert hasattr(cv2,'ximgproc'); print(cv2.__version__,numpy.__version__,torch.__version__,shapely.__version__)"

python3.12 -m venv .venv-mlsd
.venv-mlsd/bin/python -m pip install -r requirements-mlsd.txt
```

The selected RIDGE score replay was checked with NumPy 1.26.4 and OpenCV contrib 4.12.0.88. OpenCV is deliberately installed with `--no-deps` to preserve that tested NumPy combination rather than follow its NumPy 2 metadata. M-LSD uses its original OpenCV 4.11/TensorFlow 2.18.1 environment.

For table reproduction without neural inference, install `requirements-eval.txt` and the same OpenCV contrib wheel with `--no-deps`; Torch and TensorFlow are unnecessary.

## 1. Recompute all table values from saved predictions

```bash
python scripts/evaluate_results.py --output outputs/reference_evaluation
```

This reads the fixed GT and saved predictions, reruns area filtering, polygon NMS and matching, recalculates both pixel metrics, and checks all 18 rows against the paper at four decimals. It does not run the detectors. Reports and full precision numbers are written to the requested new folder.

## 2. Run the selected detectors again

```bash
python scripts/reproduce.py --output outputs/inference --device cuda:0 --mlsd-python .venv-mlsd/bin/python
```

Each method reads one file in `parameters/`. The four RIDGE runs also read `parameters/shared_geometry.json`, which defines the eight geometry thresholds for each dataset. The method files already contain the final candidate-selection settings. See [the parameter guide](parameters/README.md). Each RIDGE run checks exact output coordinates against the selected predictions. The final evaluator checks all five metrics for all 18 method/dataset pairs. Mismatches fail visibly; no saved prediction is substituted for inference.

By default AirLine reuses the distributed, verified RGB frontend line cache; rectangle reconstruction and selection are rerun. To rerun AirLine's neural line frontend too:

```bash
python scripts/reproduce.py --output outputs/fresh_inference --device cuda:0 --fresh-airline --mlsd-python .venv-mlsd/bin/python
```

To run one method, append e.g. `--methods ridge-yolo`. Outputs must be new directories. Fresh GPU outputs can depend on the hardware/software stack; exact-coordinate checks intentionally expose drift. `--device cpu` is available but CPU numerical parity has not been certified.

Use these entrypoints for the manuscript results. Each paper method has its main implementation in `code/` and imports shared tools where needed. The reproduction entrypoint applies the common geometry to the four RIDGE backends; directly calling a RIDGE engine alone omits these hooks.

## Shared parameters and scope

`parameters/` contains six method JSON files and `shared_geometry.json`. `configs/` contains Release download links and asset checksums. `reference/` contains 18 per-image prediction JSON files and the paper results table for recomputing metrics and checking inference outputs.

| Parameter | RobotHome80 | IsaacSim50 | SmartDoc90 |
|---|---:|---:|---:|
| elongation_factor | 1.8 | 1.7 | 2.0 |
| whiteness_threshold | 0.85 | 0.85 | 0.70 |
| angle_threshold (rad) | 0.5 | 1.0 | 0.5 |
| compactness_threshold | 0.22 | 0.10 | 0.15 |
| length_diff_threshold | 0.25 | 0.10 | 0.20 |
| rects_min_area_factor | 0.001 | 0.0005 | 0.001 |
| rects_max_area_factor | 0.95 | 0.90 | 0.95 |
| nms_threshold (working pixels) | 4 | 3 | 4 |

Backends, frontend resolutions, candidate construction, ranking and caps remain method-specific. Equal pixel thresholds are not equal physical image scales; RIDGE-YOLO internally scales point distances by two. Area fractions refer to each backend's reconstruction image/crop. RIDGE-YOLO applies the shared area bounds relative to the reconstruction crop. The evaluated variants apply angular checks.

## Evaluation and limits

- Precision/recall/F1: instance-level, polygon IoU >= 0.5, one-to-one matching in descending IoU order using the original matching implementation in `Metricas_FLD.py`. Unmatched duplicate predictions count as false positives.
- Accuracy/specificity: pooled **pixel-level** confusion counts on 256x256 masks. They are not candidate-level TN metrics. The five table columns do not use one shared confusion matrix.
- These datasets were reused for parameter/selection tuning; results are development-set comparisons.
- IsaacSim50 YOLO-BR uses ten image-level OOF checkpoints with fixed fold assignments in `parameters/yolo-br.json`. RIDGE-YOLO uses a fixed checkpoint. This is not a controlled comparison of training protocols. OOF routing is defined only for the listed images.
- This package supports detector inference and quantitative evaluation. Model training and timing benchmarks are outside the reproduction scope.

## Validation and licensing

`VALIDATION.json` records checks actually performed for this release. `SHA256SUMS.json` covers the released repository files, including the dataset and download configuration. `configs/*_sha256.json` provides the dataset, model and cache file checksums. The evaluator and the selected shared-parameter configuration are included. Licenses and attribution are under `licenses/`.
