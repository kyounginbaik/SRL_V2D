#!/usr/bin/env python3
"""Train and freshly evaluate actual learned weights from a physical teacher."""
from pathlib import Path
import argparse,json
from learned_feedback_policy import train
from evaluate_learned_feedback import evaluate
from audit_joint_limits import audit


def run(episode,teacher,output,feedback_gain):
    result=json.loads((teacher/'result.json').read_text());ex=result['execution']
    if not ex['physics_verified'] or result['metrics']['reference_proxy_mp_sr']!=1.:
        raise ValueError('Teacher must first pass full physical replay and proxy MP-SR')
    train(ex['scene'],ex['trajectory'],output/'training',feedback_gain=feedback_gain)
    result=evaluate(episode,ex['scene'],output/'training/policy.npz',output/'evaluation')
    out=output/'evaluation/actual_joint_limits.json'
    out.write_text(json.dumps(audit(result['execution']['scene'],result['execution']['trajectory']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--feedback-gain',type=float,required=True);run(**vars(p.parse_args()))
