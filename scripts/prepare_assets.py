"""Verify downloaded assets, then copy them into the engine layout."""
from pathlib import Path
import argparse,json,hashlib,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from common.datasets import load_dataset_records
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--datasets-only',action='store_true');a=p.parse_args()
 parts=['datasets'] if a.datasets_only else ['datasets','models','frontend_cache']
 for part in parts:
  source=a.assets.resolve()/part
  if part=='frontend_cache' and not source.exists():
   print('frontend_cache: not supplied; use --fresh-airline for neural frontend inference')
   continue
  expected=json.loads((ROOT/'configs'/f'{part}_sha256.json').read_text())
  if part=='datasets':
   source=next((folder for folder in [source,a.assets.resolve(),ROOT] if (folder/'dataset/manifest.json').is_file()),source)
  for rel,digest in expected.items():
   src=source/rel
   if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise ValueError(f'Asset checksum mismatch: {src}')
  for rel in expected:
   dest=ROOT/('results/raw/'+rel if part=='frontend_cache' else rel);dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected[rel]:raise ValueError(f'Refusing to replace a different file: {dest}')
   else:shutil.copy2(source/rel,dest)
  print(f'{part}: verified and installed {len(expected)} files')
  if part=='datasets':
   rows=load_dataset_records(ROOT/'dataset',verify_images=True)
   print(f'Annotations: {len(rows)} images; {sum(len(r["gt"]) for r in rows)} quadrilaterals; per-dataset annotation files verified')
if __name__=='__main__':main()
