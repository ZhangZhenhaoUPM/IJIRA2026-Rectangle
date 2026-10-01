"""Official M-LSD square construction with dataset-level numeric settings.
Apache-2.0 original source is in third_party/mlsd/utils.py. No GT is passed to inference.
"""
from pathlib import Path
import os, sys, json, argparse, time, hashlib
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np, cv2, tensorflow as tf
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
cv2.setNumThreads(1)
sys.path.insert(0, str(ROOT))
from third_party.mlsd.utils import pred_squares

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--config', default=str(ROOT / 'parameters/mlsd.json'))
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', default=str(ROOT / 'dataset/manifest.json'))
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())[a.dataset]
    interpreter = tf.lite.Interpreter(model_path=str(ROOT / 'other-checkpoints/M-LSD_512_large_fp16.tflite'), num_threads=1)
    interpreter.allocate_tensors()
    inputs, outputs = (interpreter.get_input_details(), interpreter.get_output_details())
    result = []
    for row in json.loads(Path(a.manifest).read_text()):
        if row['dataset'] != a.dataset:
            continue
        path = ROOT / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['image_sha256']
        im = np.asarray(Image.open(path).convert('RGB'))
        h, w = im.shape[:2]
        if max(h, w) > 1024:
            im = cv2.resize(im, (int(w * 1024 / max(h, w)), int(h * 1024 / max(h, w))), interpolation=cv2.INTER_AREA)
        start = time.perf_counter()
        segments, qs, scores, intersections = pred_squares(im, interpreter, inputs, outputs, [512, 512], params=config['params'])
        elapsed = time.perf_counter() - start
        qs = np.asarray(qs).reshape(-1, 4, 2)
        scores = np.asarray(scores)
        assert len(qs) == len(scores) and np.isfinite(qs).all()
        k = config['top_k']
        polygons = qs[:k] / [im.shape[1], im.shape[0]] * [row['width'], row['height']]
        result.append(dict(name=row['name'], dataset=a.dataset, polygons_px=polygons.tolist(), scores=scores[:k].tolist(), candidate_count=len(qs), seconds=elapsed))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(a.dataset, a.config, len(result), flush=True)
if __name__ == '__main__':
    main()
