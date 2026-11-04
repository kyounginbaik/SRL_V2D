#!/usr/bin/env python3
"""Independently score and render calibrated-table raw versus native-MPC physics."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

B=Path(__file__).resolve().parents[1]
LIB=B/'.software_gl/usr/lib64'
if __name__=='__main__':
    os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
    if str(LIB) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        os.execve(sys.executable,[sys.executable,*sys.argv],dict(os.environ,LD_LIBRARY_PATH=str(LIB)+':'+os.environ.get('LD_LIBRARY_PATH','')))

import cv2
import numpy as np
from prepare_expanded_batch import OUT,read,write
from evaluate_mp_sr import evaluate_files
from grasp_stability import audit_grasp_stability


def run(episode):
    name=f'episode_{episode:06d}';inp=B/'episodes'/name;run=OUT/'native_mpc'/name
    r=read(run/'result.json');base=read(OUT/'baseline'/name/'result.json');manifest=read(inp/'input_manifest.json')
    ex=r['execution'];out=run/'audit';out.mkdir(exist_ok=True)
    metrics=evaluate_files(manifest['source'],ex['trajectory'],inp/'input_manifest.json',ex['scene'],inp/'metric_geometry.json')
    write(out/'independent_mp_sr.json',metrics)
    write(out/'grasp_stability_frames.json',audit_grasp_stability(ex['scene'],ex['trajectory'],frames_only=True))
    videos=run/'videos';args=dict(episode_dir=str(inp),baseline_report=str(Path(base['execution']['trajectory']).parent/'execution_report.json'),
        refined_report=str(Path(ex['trajectory']).parent/'execution_report.json'),out_dir=str(videos),
        width=512,height=384,stride=1,include_source=True)
    write(videos/'render_args.json',args)
    subprocess.run([sys.executable,str(B/'code/render_rollouts.py'),'--args',str(videos/'render_args.json')],check=True)
    cap=cv2.VideoCapture(str(videos/'reference_baseline_refined.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    snapshots=videos/'review_frames';snapshots.mkdir(exist_ok=True)
    for t in sorted(set(np.arange(0.,ex['duration_s']+1e-8,.5).tolist()+[ex['duration_s']])):
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*fps));ok,img=cap.read()
        if not ok:raise ValueError(f'Video missing original timestamp{t}')
        cv2.imwrite(str(snapshots/f'time_{round(t*1000):06d}ms.jpg'),img)
    cap.release()
    write(out/'completed.json',dict(episode=name,reference_proxy_mp_sr=metrics['reference_proxy_mp_sr'],
        physics_replay_verified=ex['physics_verified'],full_horizon=ex['execution_completed'],
        video=str(videos/'reference_baseline_refined.mp4'),learned_policy=False,official_score=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);run(p.parse_args().episode)
