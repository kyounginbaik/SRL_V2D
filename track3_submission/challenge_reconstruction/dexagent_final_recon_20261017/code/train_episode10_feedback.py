"""Fit and freshly execute a phase/proprioception policy for successful cups."""
from pathlib import Path
import argparse
from learned_feedback_policy import train
from evaluate_policy_component import evaluate
from audit_full_joint_limits import main as joint_audit
from audit_object_support import audit

B=Path(__file__).resolve().parents[1];R=B/'refinement_20261018/episode_000010'

def run(gain):
    teacher=R/'calibrated_nest_release4';out=R/'learned_feedback'/f'gain_{gain:g}'
    scene=teacher/'execution/scene_physics.xml'
    train(scene,teacher/'execution/trajectory.npz',out/'training',centers=512,feedback_gain=gain,seed=43)
    evaluate(10,scene,out/'training/policy.npz',out/'evaluation',R/'calibrated_inputs/episodes/episode_000010')
    joint_audit(out/'evaluation');audit(out/'evaluation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gain',type=float,required=True);run(p.parse_args().gain)
