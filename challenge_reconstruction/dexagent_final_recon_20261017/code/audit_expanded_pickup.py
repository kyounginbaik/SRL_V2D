#!/usr/bin/env python3
"""Independent metric, contact, joint-quality and visual review of pickup trials."""
from pathlib import Path
import argparse,json,os,subprocess,sys
B=Path(__file__).resolve().parents[1]
LIB=B/'.software_gl/usr/lib64'
if __name__=='__main__':
    os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa'
    if str(LIB) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        os.execve(sys.executable,[sys.executable,*sys.argv],dict(os.environ,LD_LIBRARY_PATH=str(LIB)+':'+os.environ.get('LD_LIBRARY_PATH','')))
from evaluate_mp_sr import evaluate_files
from grasp_stability import audit_grasp_stability
from audit_joint_limits import audit


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def run(episode,path):
    name=f'episode_{episode:06d}';inp=B/'episodes'/name;r=json.loads((path/'result.json').read_text());ex=r['execution']
    manifest=json.loads((inp/'input_manifest.json').read_text())
    spec=r.get('spec',dict(source=manifest['source'],manifest=str(inp/'input_manifest.json'),geometry=str(inp/'metric_geometry.json')))
    out=path/'audit';out.mkdir(exist_ok=True)
    metric=evaluate_files(spec['source'],ex['trajectory'],spec['manifest'],ex['scene'],spec['geometry'])
    write(out/'independent_mp_sr.json',metric)
    write(out/'grasp_stability.json',audit_grasp_stability(ex['scene'],ex['trajectory'],frames_only=False))
    write(out/'actual_joint_limits.json',audit(ex['scene'],ex['trajectory']))
    base=B/'refinement_20261018/expanded_batch/native_mpc'/name/'execution/execution_report.json'
    videos=path/'videos';args=dict(episode_dir=str(inp),baseline_report=str(base),
        refined_report=str(Path(ex['trajectory']).parent/'execution_report.json'),out_dir=str(videos),
        width=640,height=480,stride=1,include_source=True)
    write(videos/'render_args.json',args)
    subprocess.run([sys.executable,str(B/'code/render_rollouts.py'),'--args',str(videos/'render_args.json')],check=True)
    write(out/'completed.json',dict(episode=name,reference_proxy_mp_sr=metric['reference_proxy_mp_sr'],
        physics_replay_verified=ex['physics_verified'],full_horizon=ex['execution_completed'],
        video=str(videos/'reference_baseline_refined.mp4'),learned_policy=bool(r.get('learned_policy',False)),official_score=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.episode,a.run)
