"""Train and independently execute a learned phase/proprioception36 actor."""
from pathlib import Path
import argparse,json
from learned_feedback_policy import train
from evaluate_policy_component import evaluate
from audit_full_joint_limits import main as joint_audit
from audit_object_support import audit as support_audit

B=Path(__file__).resolve().parents[1]

def main(teacher,output,gain):
    teacher=Path(teacher);output=Path(output);scene=teacher/'execution/scene_physics.xml'
    report=json.loads((teacher/'result.json').read_text())
    if report['metrics']['reference_proxy_mp_sr']!=1:raise ValueError('Teacher has not passed full source MP-SR')
    support=json.loads((teacher/'audit/object_support.json').read_text())['rows']
    final=[r['objects']['wooden_spoon'] for r in support if r['time_s']>19.8]
    if not final or any(sum(r['fingers'].values())>.05 or sum(r['support'].values())<.2 for r in final):raise ValueError('Teacher does not finish hands-free and supported')
    train(scene,teacher/'execution/trajectory.npz',output/'training',centers=512,feedback_gain=gain,seed=43)
    evaluate(36,scene,output/'training/policy.npz',output/'evaluation',B/'episodes/episode_000036')
    joint_audit(output/'evaluation');support_audit(output/'evaluation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gain',type=float,required=True);main(**vars(p.parse_args()))
