"""Recompute the five paper metrics and compare to the selected reference."""
from pathlib import Path
import argparse,json,csv,importlib.util
ROOT=Path(__file__).resolve().parents[1]
METHODS=['ridge-fld','ridge-airline','ridge-roi','ridge-yolo','yolo-br','mlsd'];DATASETS=['robotathome80','isaacsim50','smartdoc90']
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,default=ROOT/'reference/predictions');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(f'Choose a new output directory: {a.output}')
 s=importlib.util.spec_from_file_location('evaluation',ROOT/'evaluation/evaluate.py');ev=importlib.util.module_from_spec(s);s.loader.exec_module(ev)
 expected=list(csv.DictReader((ROOT/'reference/current_all_methods.csv').open(encoding='utf-8-sig')));rows=[];checks=[];effective={}
 names={'ridge-fld':'RIDGE-FLD','ridge-airline':'RIDGE-AirLine','ridge-roi':'RIDGE-ROI','ridge-yolo':'RIDGE-YOLO','yolo-br':'YOLO-BR','mlsd':'M-LSD'}
 fields={'Accuracy':'accuracy_pixel','Precision':'precision','Recall':'recall','Specificity':'specificity_pixel','F1 Score':'f1score'}
 for ds in DATASETS:
  for m in METHODS:
   key=m+'__'+ds;raw=json.loads((a.predictions/(key+'.json')).read_text());r,_,eff=ev.evaluate(ds,m,{v['name']:v for v in raw});effective[key]=eff
   row={'Dataset':ds,'Method':names[m],**{k:r[v] for k,v in fields.items()}};rows.append(row)
   ref=next(v for v in expected if v['Dataset']==ds and v['Method']==names[m]);ok=all(f'{row[k]:.4f}'==f'{float(ref[k]):.4f}' for k in fields)
   checks.append(dict(dataset=ds,method=m,all_five_metrics_match_4dp=ok,tp=r['tp'],fp=r['fp'],fn=r['fn']))
   print(ds,m,'PASS' if ok else 'DIFF',f"F1={r['f1score']:.8f}",flush=True)
 a.output.mkdir(parents=True)
 with (a.output/'results.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for name,data in [('results',rows),('verification',checks),('effective_predictions',effective)]: (a.output/(name+'.json')).write_text(json.dumps(data,indent=2))
 if not all(r['all_five_metrics_match_4dp'] for r in checks):raise SystemExit('Metric differences detected; see verification.json')
if __name__=='__main__':main()
