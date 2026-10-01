"""Apply shared geometry and run one RIDGE method on the three datasets.

Each method reads its final settings from parameters/. Pixel thresholds remain
in the backend's working coordinates; frontends and candidate selection are
method-specific.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,sys,time
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common.datasets import load_dataset_records,inference_records
from common.configuration import FIELDS,load_shared_geometry,resolve_shared


def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--fresh-airline',action='store_true');a=parser.parse_args()
    a.output=a.output.resolve();a.release=a.release.resolve()
    a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output);cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    shared=load_shared_geometry(a.release);method=a.method
    plan={'profiles':[dict(id=ds,**values) for ds,values in shared.items()],
          'datasets':list(shared),'jobs':[[ds,ds] for ds in shared]}
    module=load('original_detector',a.release/'code'/f'{method}.py')
    evaluator=load('original_evaluator',a.release/'evaluation/evaluate.py')
    active={};assertions={'geometry_calls':0,'point_calls':0,'extract_cache_hits':0};extract_cache={}
    records=[]
    for r in inference_records(load_dataset_records(a.release/'dataset'),a.release):
        p=Path(r['path']);p=p if p.is_absolute() else a.release/p
        records.append({k:r[k] for k in ['dataset','name','width','height']}|{'path':str(p)})
    save(a.output/'inference_manifest.json',records)
    # This manifest has no GT; original evaluator alone reads the frozen GT.
    images={r['path']:cv2.imread(r['path']) for r in records}
    assert all(x is not None for x in images.values())
    if method in ['ridge-fld','ridge-airline']:
        detector_original=module.c.detector
        def detector(backend):
            d=detector_original(backend)
            for k in FIELDS:
                # All frontend line caches are extended by exactly 1.25.
                # Graph variants, not base extraction, apply the shared elongation.
                if k!='elongation_factor':setattr(d,k,active[k])
            d.elongation_factor=1.25
            return d
        module.c.detector=detector
        # Normalize every constituent geometry, including the legacy hard-coded
        # additional profile; preserve group names and candidate/scoring rules.
        module.det.SHARED=active
        module.det.SHAPE_FIELDS=FIELDS[2:7]
        protocols=copy.deepcopy(module.PARAMETERS['pool_protocols'])
        for fast in [module.c.fldfast,module.c.airfast]:
            original_extract=fast.extract;original_intersections=fast.intersections
            def extract(d,pts,adj,image,_original=original_extract,_fast=fast):
                for key in FIELDS[2:7]:assert getattr(d,key)==active[key],(key,getattr(d,key),active[key])
                assertions['geometry_calls']+=1
                key=(_fast._cycles._name,image.shape[:2],d.max_num_trapezoids,tuple(getattr(d,k) for k in FIELDS[2:7]),np.ascontiguousarray(pts,dtype=np.int64).tobytes(),np.ascontiguousarray(adj,dtype=np.uint8).tobytes())
                if key not in extract_cache:extract_cache[key]=_original(d,pts,adj,image)
                else:assertions['extract_cache_hits']+=1
                qs,failed=extract_cache[key]
                return [q.copy() for q in qs],failed
            def intersections(d,lines,_original=original_intersections):
                d.nms_threshold=active['nms_threshold'];assertions['point_calls']+=1
                return _original(d,lines)
            fast.extract=extract;fast.intersections=intersections
        base_lines={}
        if method=='ridge-airline':
            if a.fresh_airline:
                adapter=module.airline_model(a.device)
                for r in records:
                    im=images[r['path']];h,w=im.shape[:2]
                    base_lines[r['path']]=np.asarray(adapter._extract_crg_lines(adapter.detector.reduce_saturation_brightness(im,.5,.5)),float).reshape(-1,4)*np.array([w/(w//16*16),h/(h//16*16)]*2)
            else:
                for r in records:
                    digest=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
                    base_lines[r['path']]=np.load(a.release/'results/raw/airline_frontend_cache'/f'crg__{digest}.npz')['lines']
    elif method=='ridge-roi':
        original_init=module.contours_RectangleDetector.__init__
        def init(self,*args,**kw):
            original_init(self,*args,**kw)
            for k in FIELDS:setattr(self,k,active[k])
        module.contours_RectangleDetector.__init__=init
        original_angle=module.contours_RectangleDetector.trapezoid_angles_test
        def angle(self,*args,**kw):
            for k in FIELDS:assert getattr(self,k)==active[k],(k,getattr(self,k),active[k])
            assertions['geometry_calls']+=1
            return original_angle(self,*args,**kw)
        module.contours_RectangleDetector.trapezoid_angles_test=angle
    else:
        original_configure=module.configure
        def configure(ds,c):
            original_configure(ds,c)
            for p in module.ridge_PASSES:
                p.update(elong=active['elongation_factor'],white=active['whiteness_threshold'],
                         angle=active['angle_threshold'],compact=active['compactness_threshold'],len_ratio=active['length_diff_threshold'])
            module.ridge_NMS_POINT_PX=active['nms_threshold']
            for p in module.ridge_PASSES:
                assert (p['elong'],p['white'],p['angle'],p['compact'],p['len_ratio'])==tuple(active[k] for k in FIELDS[:5])
        module.configure=configure
        original_find=module.ridge_RidgeFLD.find_quads
        def find_quads(self,roi,box_local,cfg,pass_idx):
            self._shared_region_area=(roi[2]-roi[0])*(roi[3]-roi[1])
            return original_find(self,roi,box_local,cfg,pass_idx)
        module.ridge_RidgeFLD.find_quads=find_quads
        original_geometry=module.ridge_RidgeFLD._geometry
        def geometry(self,p,cfg):
            assert cfg['angle']==active['angle_threshold'] and cfg['compact']==active['compactness_threshold'] and cfg['len_ratio']==active['length_diff_threshold']
            assertions['geometry_calls']+=1
            ok,area,orth=original_geometry(p,cfg)
            ok&=(area>=active['rects_min_area_factor']*self._shared_region_area)&(area<=active['rects_max_area_factor']*self._shared_region_area)
            return ok,area,orth
        module.ridge_RidgeFLD._geometry=geometry
        class CachedModel:
            def __init__(self,model):self.model=model;self.names=model.names;self.cache={}
            def predict(self,**kw):
                key=hashlib.sha256(kw['source'].tobytes()).hexdigest()
                if key not in self.cache:self.cache[key]=self.model.predict(**kw)
                return self.cache[key]
        from ultralytics import YOLO
        models={ds:CachedModel(YOLO(str(a.release/c['checkpoint']))) for ds,c in module.CONFIG.items()}

    score_map={}
    def score_key(q):return tuple(sorted(tuple(np.round(v,4)) for v in q))
    if method in ['ridge-fld','ridge-airline']:
        original_compute=module.f.compute
        feature_pool=[]
        def compute(im,qs,base):
            ff=original_compute(im,qs,base);feature_pool.clear();feature_pool.append((qs,np.asarray(ff).reshape(-1,5)));return ff
        module.f.compute=compute
        if method=='ridge-fld':
            original_normalized=module.normalized
            def normalized(im,settings):
                z=original_normalized(im,settings)
                for q,b in zip(z['polygons_normalized'],z['boundary_scores']):
                    area=abs(cv2.contourArea(np.asarray(q,dtype=np.float32)))
                    score_map[score_key(np.asarray(q)*[im.shape[1],im.shape[0]])]=float(b*(.5+.5*np.sqrt(area)))
                return z
            module.normalized=normalized
        original_predict=module.predict
        def predict(im,ds,settings,*args,**kw):
            cfg=dict(settings)
            score_map.clear();feature_pool.clear();qs=original_predict(im,ds,cfg,*args,**kw)
            if feature_pool:
                pool,ff=feature_pool[0];bd,ep,cl,co,ar=ff.T;aw=.75+.25*np.sqrt(ar)
                ranks={'combined_area':bd*(.5+.25*ep+.25*co)*aw,'closure_area':bd*(.5+.5*cl)*aw,'boundary_area':bd*(.5+.5*np.sqrt(ar))}
                score_map.update({score_key(np.asarray(q)*[im.shape[1],im.shape[0]]):float(s) for q,s in zip(pool,ranks[cfg['ranking']])})
            return qs
        module.predict=predict
    elif method=='ridge-roi':
        module.SCORE_MAP=score_map
        module.SCORE_KEY=score_key
        original_detect=module.detect
        def detect(im,ds,*args,**kw):
            score_map.clear();return original_detect(im,ds,*args,**kw)
        module.detect=detect
    else:
        original_detect=module.detect
        def detect(*args,**kw):
            z=original_detect(*args,**kw);score_map.clear()
            score_map.update({score_key(v['quad']):float(v['score']) for v in z['layers']['ridge']});return z
        module.detect=detect

    results=[]
    for spec in plan['profiles']:
        active.clear();active.update({k:spec[k] for k in FIELDS})
        if method in ['ridge-fld','ridge-airline']:
            for path,obj in protocols.items():save(module.TEMP/path,resolve_shared(obj,active))
        for ds in plan['datasets']:
            if 'jobs' in plan and [spec['id'],ds] not in plan['jobs']:continue
            started=time.perf_counter();out=[];case=a.output/f"{spec['id']}__{ds}"
            case.mkdir(exist_ok=False)
            save(case/'effective_thresholds.json',active)
            selected=[r for r in records if r['dataset']==ds]
            before=copy.deepcopy(assertions)
            for index,r in enumerate(selected):
                extract_cache.clear()
                im=images[r['path']]
                if method in ['ridge-fld','ridge-airline']:
                    q=module.predict(im,ds,module.CONFIG[ds],base=base_lines.get(r['path']) if method=='ridge-airline' else None)
                    polygons=[(np.asarray(x)*[r['width'],r['height']]).tolist() for x in q]
                elif method=='ridge-roi':polygons=module.detect(im,ds)['polygons_px']
                else:polygons=module.detect(im,ds,module.CONFIG[ds],models[ds],a.device)['polygons_px']
                out.append(dict(dataset=ds,name=r['name'],width=r['width'],height=r['height'],polygons_px=polygons,scores=[score_map[score_key(q)] for q in polygons]))
                if index%20==0:save(a.output/'progress.json',dict(profile=spec['id'],dataset=ds,done=index+1,total=len(selected)))
            expected={r['name']:r for r in json.loads((a.release/'reference/predictions'/f'{method}__{ds}.json').read_text())}
            for rec in out:
                assert np.array_equal(np.asarray(rec['polygons_px']).reshape(-1,4,2),np.asarray(expected[rec['name']]['polygons_px']).reshape(-1,4,2)),(method,ds,rec['name'],'prediction mismatch')
            save(case/'predictions.json',out)
            metric,per,_=evaluator.evaluate(ds,method,{r['name']:r for r in out})
            assert assertions['geometry_calls']>before['geometry_calls'],(method,spec['id'],ds,'No geometry assertion executed')
            save(case/'runtime_assertions.json',{k:assertions[k]-before[k] for k in assertions})
            save(case/'metrics.json',metric);save(case/'per_image.json',per)
            results.append(dict(profile=spec['id'],seconds=time.perf_counter()-started,**metric))
            save(a.output/'metrics.json',results)
            print('RESULT',method,spec['id'],ds,metric['tp'],metric['fp'],metric['fn'],round(metric['f1score'],5),flush=True)
    save(a.output/'complete.json',dict(status='complete',combinations=len(results)))

if __name__=='__main__':main()
