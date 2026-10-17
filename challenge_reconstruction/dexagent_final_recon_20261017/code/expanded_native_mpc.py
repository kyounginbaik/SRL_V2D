#!/usr/bin/env python3
"""Released DexAgent MPC module on calibrated floating-hand new-episode scenes."""
from pathlib import Path
import argparse
import json
import sys
import time
import hashlib
from prepare_expanded_batch import B,OUT,NEW,read,write,sha
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from floating_tracking import refine
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files


def run(episode):
    if episode not in NEW:raise ValueError('Expanded-episode scope only')
    name=f'episode_{episode:06d}';inp=B/'episodes'/name;out=OUT/'native_mpc'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return read(out/'result.json')
    baseline=read(OUT/'baseline'/name/'result.json');manifest=read(inp/'input_manifest.json')
    if not baseline['execution']['physics_verified']:raise ValueError('Baseline physical validation failed')
    task=dict(scope='Track every original reconstructed object over unchanged full timeline',
        subgoals=[dict(index=0,motion='track_reference',acting_hand='both',object_refs=[x['id'] for x in manifest['objects']])])
    task_path=out/'tracking_task.json';write(task_path,task)
    begin=time.monotonic()
    write(out/'status.json',dict(episode=name,stage='released_module_MPC_search',samples=16,workers=4,max_minutes=10,
        baseline_proxy_mp_sr=baseline['metrics']['reference_proxy_mp_sr'],learned_policy=False))
    search=refine(workspace=str(out),scene_semantic_json=str(task_path),scene_recon_json=str(inp/'stage2/scene_recon.json'),
        reference=str(inp/'tracking_reference.npz'),scene=baseline['execution']['scene'],out_dir=str(out/'search'),
        samples=16,workers=4,max_minutes=10.,seed=44)
    write(out/'status.json',dict(episode=name,stage='full_horizon_physical_replay',search=search,learned_policy=False))
    execution=execute_reference(baseline['execution']['scene'],search['reference'],out/'execution',
        expected_duration_s=(manifest['frames']-1)/manifest['fps'])
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=name,method='Released DexAgent predictive-sampling MPC module; explicit floating-Sharpa runtime port',
        native_full_LLM_pipeline=False,learned_policy=False,upstream_commit='e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633',
        search=search,execution=execution,metrics=metrics,baseline_metrics=baseline['metrics'],
        scene_sha256=sha(execution['scene']),input_manifest_sha256=sha(inp/'input_manifest.json'),
        source_object_poses_sha256=manifest['source_object_poses_sha256'],full_original_timing=True,
        wall_seconds=time.monotonic()-begin,submission_ready=False)
    write(out/'result.json',result);write(out/'status.json',dict(episode=name,stage='complete',reference_proxy_mp_sr=metrics['reference_proxy_mp_sr']))
    print(json.dumps({'episode':name,'reference_proxy_mp_sr':metrics['reference_proxy_mp_sr'],'wall_seconds':result['wall_seconds']}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);run(p.parse_args().episode)
