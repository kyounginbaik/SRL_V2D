#!/usr/bin/env python3
"""Briefly curl ring/pinky during the measured rack-hooking lift interval."""
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
    x=np.clip(x,0.,1.);return x*x*(3-2*x)

def run(digits,angle,start):
    work=BASE/'physics_collision_v2';inp=work/'episodes/episode_000030';source=work/'sweeper_transfer/donor3_park0_v1/pickup_search_v1'
    out=work/'sweeper_transfer'/f'liftfold_{digits}_{angle:g}_s{start:g}_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);previous=json.loads((source/'result.json').read_text());scene=previous['execution']['scene']
    model=mujoco.MjModel.from_xml_path(scene);ref=TR.load_reference(source/'reference.npz',model);q=ref['qpos'].copy()
    t=np.arange(len(q))/ref['fps'];w=blend((t-start)/.25)*blend((3.4-t)/.3);names=[]
    for digit in (['ring','pinky'] if digits=='both' else [digits]):
        for suffix in ['MCP_FE','PIP','DIP']:
            name=f'robot_right_right_{digit}_{suffix}';j=model.joint(name).id;a=int(model.jnt_qposadr[j]);target=np.clip(angle,*model.jnt_range[j]);q[:,a]=(1-w)*q[:,a]+w*target;names.append(name)
    changed=[int(model.jnt_qposadr[model.joint(name).id]) for name in names];other=np.setdiff1d(np.arange(model.nq),changed)
    if not np.array_equal(q[:,other],ref['qpos'][:,other]) or not np.array_equal(q[0],ref['qpos'][0]):raise AssertionError('Unrelated commands changed')
    reference=TR.save_reference(out/'reference.npz',model,q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Brief ring/pinky lift clearance for measured rack hooking; wrist, thumb/index and object references unchanged',floating_runtime=True,no_retiming=True)
    execution=execute_reference(scene,reference,out/'execution');manifest=json.loads((inp/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode='episode_000030',input_dir=str(inp),reference=str(reference),execution=execution,metrics=metrics,source_run=str(source),modified_joint_names=names,target_angle_rad=angle,window_seconds=[start,start+.25,3.1,3.4],all_other_commands_unchanged=True,original_object_targets_unchanged=True,learned_policy=False,producer_script_sha256=sha(__file__))
    write(out/'result.json',result);print(json.dumps(dict(output=str(out),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--digits',choices=['both','pinky','ring'],required=True);p.add_argument('--angle',type=float,required=True);p.add_argument('--start',type=float,required=True);run(**vars(p.parse_args()))
