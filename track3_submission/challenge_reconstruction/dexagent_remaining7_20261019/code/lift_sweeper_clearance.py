#!/usr/bin/env python3
"""Physical robot wrist clearance proposal for the observed rack-hooking event."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,json,sys
import mujoco,numpy as np
from prepare_seven import BASE,ROOT,LEGACY,sha,write
sys.path[:0]=[str(LEGACY/'code'),str(ROOT/'third_party/DexAgent')]
from tool_library.common import tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

def blend(x):
    x=np.clip(x,0.,1.);return x*x*x*(10-15*x+6*x*x)

def run(height,start):
    work=BASE/'physics_collision_v2';inp=work/'episodes/episode_000030';source=work/'sweeper_transfer/donor3_park0_v1/pickup_search_v1'
    out=work/'sweeper_transfer'/f'liftclear_{height:g}_s{start:g}_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);previous=json.loads((source/'result.json').read_text());scene=previous['execution']['scene']
    model=mujoco.MjModel.from_xml_path(scene);ref=TR.load_reference(source/'reference.npz',model);q=ref['qpos'].copy()
    t=np.arange(len(q))/ref['fps'];w=blend((t-start)/.2)*blend((3.5-t)/.4)
    a=int(model.joint('robot_right_right_pos_z').qposadr[0]);q[:,a]+=height*w
    if not np.array_equal(q[:,np.arange(model.nq)!=a],ref['qpos'][:,np.arange(model.nq)!=a]):raise AssertionError('Unrelated commands changed')
    reference=TR.save_reference(out/'reference.npz',model,q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Early smooth robot wrist lift clears rack prongs before transport; finger/object reference columns unchanged',floating_runtime=True,no_retiming=True)
    execution=execute_reference(scene,reference,out/'execution');manifest=json.loads((inp/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode='episode_000030',input_dir=str(inp),reference=str(reference),execution=execution,metrics=metrics,source_run=str(source),modified_joint='robot_right_right_pos_z',height_m=height,window_seconds=[start,start+.2,3.1,3.5],all_other_commands_unchanged=True,original_object_targets_unchanged=True,learned_policy=False,producer_script_sha256=sha(__file__))
    write(out/'result.json',result);print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--height',type=float,required=True);p.add_argument('--start',type=float,required=True);run(**vars(p.parse_args()))
