"""Verify versioned files and optionally the extracted dataset."""
from pathlib import Path
import hashlib,json,ast,argparse,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--datasets',action='store_true',help='Also verify images and annotations in dataset/')
args=parser.parse_args()
hashes=json.loads((ROOT/'SHA256SUMS.json').read_text())
for rel,digest in hashes.items():
 p=ROOT/rel
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,rel
for rel in hashes:
 if rel.endswith('.py'):ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'))
sys.path.insert(0,str(ROOT/'code'))
from common.configuration import load_shared_geometry,resolve_shared
shared=load_shared_geometry(ROOT)
methods=['ridge-fld','ridge-airline','ridge-roi','ridge-yolo','yolo-br','mlsd']
assert set(shared)=={'robotathome80','isaacsim50','smartdoc90'}
assert {p.name for p in (ROOT/'parameters').glob('*.json')}=={m+'.json' for m in methods}|{'shared_geometry.json'}
for method in methods:
 config=json.loads((ROOT/'parameters'/f'{method}.json').read_text())
 datasets=config['datasets'] if method in ['ridge-fld','ridge-airline','ridge-roi'] else config
 assert set(datasets)==set(shared),method
 if 'pool_protocols' in config:
  for geometry in shared.values():resolve_shared(config['pool_protocols'],geometry)
print(f'PASS: {len(hashes)} file hashes; six method configurations and one shared geometry configuration.')
if args.datasets:
 expected=json.loads((ROOT/'configs/datasets_sha256.json').read_text())
 for rel,digest in expected.items():
  p=ROOT/rel
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError(f'Dataset checksum mismatch: {p}')
 sys.path.insert(0,str(ROOT/'code'))
 from common.datasets import load_dataset_records
 rows=load_dataset_records(ROOT/'dataset')
 print(f'PASS: {len(rows)} images, {sum(len(r["gt"]) for r in rows)} quadrilaterals and three per-dataset annotation files.')
