"""Run all six methods with the fixed paper configuration into a new folder."""
from pathlib import Path
import argparse,sys,subprocess,json,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from common.datasets import load_dataset_records,inference_records
from common.configuration import load_shared_geometry
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--mlsd-python',default=sys.executable);p.add_argument('--fresh-airline',action='store_true');p.add_argument('--methods',nargs='+',choices=['ridge-fld','ridge-airline','ridge-roi','ridge-yolo','yolo-br','mlsd'],default=['ridge-fld','ridge-airline','ridge-roi','ridge-yolo','yolo-br','mlsd']);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);pred=out/'predictions';pred.mkdir();jobs=[(ds,ds) for ds in load_shared_geometry(ROOT)]
 manifest=out/'inference_manifest.json'
 manifest.write_text(json.dumps(inference_records(load_dataset_records(ROOT/'dataset'),ROOT)),encoding='utf-8')
 for m in a.methods:
  if m.startswith('ridge-'):
   cmd=[sys.executable,str(ROOT/'scripts/run_shared_ridge.py'),'--release',str(ROOT),'--output',str(out/m),'--method',m,'--device',a.device]
   if a.fresh_airline:cmd+=['--fresh-airline']
   subprocess.run(cmd,check=True)
   for profile,ds in jobs:shutil.copy2(out/m/f'{profile}__{ds}'/'predictions.json',pred/f'{m}__{ds}.json')
  else:
   for _,ds in jobs:
    cmd=[a.mlsd_python if m=='mlsd' else sys.executable,str(ROOT/'code'/(m+'.py')),'--dataset',ds,'--manifest',str(manifest),'--output',str(pred/f'{m}__{ds}.json')]
    if m=='yolo-br':cmd+=['--device',a.device]
    subprocess.run(cmd,check=True,cwd=ROOT)
 if len(a.methods)==6:subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_results.py'),'--predictions',str(pred),'--output',str(out/'evaluation')],check=True)
 else:print('Subset inference finished. Automatic 18-row comparison is only run for all six methods.')
if __name__=='__main__':main()
