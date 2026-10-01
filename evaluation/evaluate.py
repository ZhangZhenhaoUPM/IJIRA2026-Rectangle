from pathlib import Path
import json,sys,importlib.util,csv,hashlib
import numpy as np,cv2
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
from common.datasets import load_dataset_records
spec=importlib.util.spec_from_file_location('david',ROOT/'evaluation/Metricas_FLD.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
MANIFEST=load_dataset_records(ROOT/'dataset')
def evaluate(ds,method,items):
 rs=[r for r in MANIFEST if r['dataset']==ds];assert set(items)=={r['name'] for r in rs}
 gt={r['name']:[d.points_to_polygon(np.asarray(q)*[r['width'],r['height']]) for q in r['gt']] for r in rs}
 sizes={r['name']:(r['width'],r['height']) for r in rs}
 preds={n:[dict(pts=q,accepted=True) for q in v['polygons_px']] for n,v in items.items()}
 prepared=d.build_records(gt,sizes,preds);tp=fp=fn=0;pix=np.zeros(4,np.int64);per=[];effective={}
 for r,rec in zip(rs,prepared):
  # build_records iterates sorted names; resolve by explicit one-image preparation.
  name=r['name'];rec=d.build_records({name:gt[name]},{name:sizes[name]},{name:preds[name]})[0]
  pairs=d.match_predictions(rec['iou_acc'],.5);t=len(pairs);f=rec['iou_acc'].shape[0]-t;n=rec['iou_acc'].shape[1]-t
  tp+=t;fp+=f;fn+=n
  polygons=[d.points_to_polygon(q['pts']) for q in preds[name]]
  ids=[i for i,p in enumerate(polygons) if p is not None and p.area>=d.MIN_AREA_RATIO*r['width']*r['height']]
  if d.PRED_NMS_IOU_THRESHOLD is not None and ids:ids=d.nms_indices(polygons,ids,d.PRED_NMS_IOU_THRESHOLD,metric='iou')
  qs=[list(polygons[i].exterior.coords)[:-1] for i in ids];assert len(qs)==rec['iou_acc'].shape[0]
  gm=np.zeros((256,256),np.uint8);pm=gm.copy()
  for q in r['gt']:cv2.fillPoly(gm,[np.rint(np.asarray(q)*255).astype(np.int32)],1)
  for q in qs:cv2.fillPoly(pm,[np.rint(np.asarray(q)/[r['width'],r['height']]*255).astype(np.int32)],1)
  pp=np.array([((gm==1)&(pm==1)).sum(),((gm==0)&(pm==1)).sum(),((gm==1)&(pm==0)).sum(),((gm==0)&(pm==0)).sum()],np.int64);pix+=pp
  per.append(dict(dataset=ds,method=method,name=name,tp=t,fp=f,fn=n,matched_gt=[int(j) for _,j,_ in pairs],pixel_tp=int(pp[0]),pixel_fp=int(pp[1]),pixel_fn=int(pp[2]),pixel_tn=int(pp[3])))
  effective[name]=dict(width=r['width'],height=r['height'],polygons_px=qs,matched_prediction_indices=[int(i) for i,_,_ in pairs])
 reference=d.calculate_metrics(prepared,.5)
 assert abs(reference['Precision']-(tp/(tp+fp) if tp+fp else 0.))<1e-12
 assert abs(reference['Recall']-(tp/(tp+fn) if tp+fn else 0.))<1e-12
 assert abs(reference['F1 Score']-(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))<1e-12
 p=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.
 result=dict(dataset=ds,method=method,images=len(rs),gt=tp+fn,tp=tp,fp=fp,fn=fn,accuracy_pixel=float((pix[0]+pix[3])/pix.sum()),precision=p,recall=re,specificity_pixel=float(pix[3]/(pix[3]+pix[1])) if pix[3]+pix[1] else None,f1score=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,accuracy_david_candidate=None,specificity_david_candidate=None)
 return result,per,effective
def main():
 rows=[];per=[];eff={}
 for p in sorted((ROOT/'results/raw').glob('*__*.json')):
  method,ds=p.stem.split('__',1)
  if method not in ['ridge','ridge-fld','ridge-airline','ridge-roi','ridge-yolo','yolo-br','mlsd']:continue
  rr=json.loads(p.read_text());items={v['name']:v for v in rr};m,pi,eq=evaluate(ds,method,items);rows.append(m);per+=pi;eff[method+'__'+ds]=eq;print(m,flush=True)
 (ROOT/'results/tables/best_results.json').write_text(json.dumps(rows,indent=2))
 with (ROOT/'results/tables/best_results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (ROOT/'results/raw/per_image_metrics.json').write_text(json.dumps(per))
 (ROOT/'results/raw/evaluation_effective_predictions.json').write_text(json.dumps(eff))
if __name__=='__main__':main()
