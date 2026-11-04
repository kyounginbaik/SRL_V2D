"""Distil a teacher recording (execution/trajectory.npz + execution/scene_physics.xml) into the registered RBF actor and evaluate it
fresh in the SAME scene against a given episode dir (no shared_policy_handoff scene check, unlike distill_teacher.py).
usage: distill_direct.py --episode E --teacher DIR --episode-dir DIR --out DIR [--fg 0.03 --centers 1024 --vg 0.0005 --seed 43]"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, sys
from pathlib import Path
R=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=[str(R/'code'),'/data1/SRL_V2D/third_party/DexAgent']
from learned_feedback_policy import train
from evaluate_policy_component import evaluate
p=argparse.ArgumentParser(); p.add_argument('--episode',type=int,required=True); p.add_argument('--teacher',required=True); p.add_argument('--episode-dir',required=True); p.add_argument('--out',required=True)
p.add_argument('--fg',type=float,default=.03); p.add_argument('--centers',type=int,default=1024); p.add_argument('--vg',type=float,default=.0005); p.add_argument('--seed',type=int,default=43)
a=p.parse_args(); te=Path(a.teacher); scene=str(te/'execution/scene_physics.xml'); traj=str(te/'execution/trajectory.npz'); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
rep=train(scene,traj,out/'training',centers=a.centers,feedback_gain=a.fg,velocity_gain=a.vg,seed=a.seed)
rep=dict(rep) if isinstance(rep,dict) else {}
tr=out/'training/training_report.json'
if tr.exists():
    d=json.loads(tr.read_text()); d['teacher']=str(te); tr.write_text(json.dumps(d,indent=2,default=str))
evaluate(a.episode,scene,str(out/'training/policy.npz'),str(out/'evaluation'),a.episode_dir)
r=json.loads((out/'evaluation/result.json').read_text()); o=r['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(out=str(out),mp_sr=r['metrics']['reference_proxy_mp_sr'],physics=r['execution']['physics_verified'],replay=r['execution'].get('independent_control_replay',{}).get('ok'),objects={k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in o.items()})),flush=True)
