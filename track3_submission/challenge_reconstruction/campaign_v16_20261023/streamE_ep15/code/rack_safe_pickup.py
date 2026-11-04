"""Pickup CEM (physics_pickup_search_v2.PickupExperiment) with an explicit penalty on the OTHER object's mean error above a
tolerance (the dish rack must stay <= ~1 cm) and on the held object's tracking rotation; optional resume from a previous
search's progress.json. Same bounded robot commands, immutable object targets/clock.
usage: rack_safe_pickup.py --spec SPEC --output DIR [--resume PROGRESS_JSON --samples 96 --iterations 10 --workers 8 --seed 45
       --rack-tol-cm 0.8 --rack-weight 20 --rot-weight 1.5]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from physics_pickup_search_v2 import PickupExperiment
from pickup_optimizer import optimize

class RackSafePickup:
    def __init__(self,spec):
        self.inner=PickupExperiment(spec); self.m=self.inner.m
        for key in ('ref','fps','dim','bound','scale'): setattr(self,key,getattr(self.inner,key))
        self.tol=float(spec.get('rack_tol_cm',.8)); self.w=float(spec.get('rack_weight',20.)); self.wr=float(spec.get('rot_weight',1.5))
    def commands(self,x): return self.inner.commands(x)
    def evaluate(self,x,return_trace=False):
        r=self.inner.evaluate(x,return_trace=return_trace)
        if r.get('unstable') or 'mean_other_object_error_cm' not in r: return r
        r=dict(r); r['base_cost']=r['cost']
        r['cost']=r['cost']+self.w*max(0.,r['mean_other_object_error_cm']-self.tol)+self.wr*np.minimum(r['mean_tracking_rotation_deg']/30.,4.)
        return r

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--spec',required=True); p.add_argument('--output',required=True); p.add_argument('--resume')
    p.add_argument('--samples',type=int,default=96); p.add_argument('--iterations',type=int,default=10); p.add_argument('--workers',type=int,default=8); p.add_argument('--seed',type=int,default=45)
    p.add_argument('--rack-tol-cm',type=float,default=.8); p.add_argument('--rack-weight',type=float,default=20.); p.add_argument('--rot-weight',type=float,default=1.5); a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text()); spec.update(rack_tol_cm=a.rack_tol_cm,rack_weight=a.rack_weight,rot_weight=a.rot_weight)
    r=optimize(RackSafePickup,spec,a.output,samples=a.samples,iterations=a.iterations,workers=a.workers,seed=a.seed,resume=a.resume)
    m=r['metrics']['official_style_reference_proxy']; print('RESULT',Path(a.output).name,m['mp_sr'],{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['objects'].items()},flush=True)
