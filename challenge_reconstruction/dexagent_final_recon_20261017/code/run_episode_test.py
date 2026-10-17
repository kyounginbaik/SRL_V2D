"""One reproducible, fixed-time DexAgent floating-Sharpa MPC experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]
sys.path.insert(0,str(ROOT/'third_party/DexAgent'))
from floating_runtime import execute_reference
from floating_tracking import refine
from evaluate_mp_sr import evaluate_files

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)

def run_episode(episode,seed=44,workers=4,samples=16,max_minutes=10.,tag='pilot'):
    name=f'episode_{int(episode):06d}'
    src=B/'episodes'/name
    manifest=json.loads((src/'input_manifest.json').read_text())
    out=B/'runs'/tag/name/f'seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    status={'episode':name,'seed':seed,'method':'DexAgent predictive-sampling MPC, floating-Sharpa runtime port',
        'upstream_commit':'e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633',
        'started_utc':datetime.now(timezone.utc).isoformat(),'stage':'starting',
        'native_full_pipeline':False,'source':manifest['source'],'source_manifest':str(src/'input_manifest.json'),
        'settings':{'samples':samples,'workers':workers,'max_minutes':max_minutes,
        'horizon_s':.4,'replan_s':.2,'seed':seed},
        'code_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((B/'code').glob('*.py'))}}
    dump(out/'status.json',status)
    scene=src/'stage2/scene.xml';reference=src/'tracking_reference.npz'
    task={'scope':'Track original reconstructed object trajectories over the complete original timeline.',
        'evaluation':'Reference-based MP-SR: every object mean position <0.03m and symmetry-aware mean angle <30degrees.',
        'subgoals':[{'index':0,'motion':'track_reference','acting_hand':'both',
        'object_refs':[o['id'] for o in manifest['objects']]}]}
    dump(out/'tracking_task.json',task)
    duration=(manifest['frames']-1)/manifest['fps']
    start=time.monotonic()
    try:
        status['stage']='baseline_physics';dump(out/'status.json',status)
        baseline=execute_reference(scene,reference,out/'baseline',expected_duration_s=duration)
        bm=evaluate_files(manifest['source'],baseline['trajectory'],src/'input_manifest.json',baseline['scene'],src/'metric_geometry.json')
        dump(out/'baseline/mp_sr.json',bm)
        status.update(stage='mpc_search',baseline_mp_sr=bm['reference_proxy_mp_sr']);dump(out/'status.json',status)
        search=refine(workspace=str(out),scene_semantic_json=str(out/'tracking_task.json'),
            scene_recon_json=str(src/'stage2/scene_recon.json'),reference=str(reference),scene=baseline['scene'],
            out_dir=str(out/'mpc'),samples=samples,workers=workers,max_minutes=max_minutes,seed=seed)
        status.update(stage='refined_physics',search_improved=search['improved']);dump(out/'status.json',status)
        executed=execute_reference(baseline['scene'],search['reference'],out/'refined',expected_duration_s=duration)
        em=evaluate_files(manifest['source'],executed['trajectory'],src/'input_manifest.json',executed['scene'],src/'metric_geometry.json')
        dump(out/'refined/mp_sr.json',em)
        status.update(stage='complete',baseline_execution=baseline,refined_execution=executed,
            baseline_metrics=bm,refined_metrics=em,search=search,
            reference_proxy_mp_sr=em['reference_proxy_mp_sr'],
            physically_recorded_mp_sr=float(em['reference_proxy_mp_sr'] if executed['physics_verified'] and executed['execution_completed'] else 0),
            elapsed_seconds=time.monotonic()-start,
            completed_utc=datetime.now(timezone.utc).isoformat())
        dump(out/'result.json',status);dump(out/'status.json',status)
        return status
    except Exception as error:
        import traceback
        status.update(stage='error',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        dump(out/'status.json',status)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True)
    p.add_argument('--seed',type=int,default=44);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--samples',type=int,default=16);p.add_argument('--max-minutes',type=float,default=10.)
    p.add_argument('--tag',default='pilot');a=p.parse_args()
    r=run_episode(**vars(a));print(json.dumps({k:r[k] for k in ['episode','stage','reference_proxy_mp_sr','elapsed_seconds']},indent=2))
