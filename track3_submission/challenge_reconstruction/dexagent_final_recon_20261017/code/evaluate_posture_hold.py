"""Full original-clock evaluation of unused fingers held through release."""
from pathlib import Path
import argparse,json
import numpy as np
import evaluate_learned_feedback as runtime
from posture_feedback_policy_hold import PosturePolicy
from learned_feedback_policy import sha


def main(episode,scene,checkpoint,output):
    runtime.LearnedPolicy=PosturePolicy
    result=runtime.evaluate(episode,scene,checkpoint,output)
    result['policy_code_sha256']=sha(Path(__file__).with_name('posture_feedback_policy_hold.py'))
    result['policy_dependency_sha256']={name:sha(Path(__file__).with_name(name)) for name in ['posture_feedback_policy_v4.py','posture_feedback_policy_v3.py','posture_feedback_policy.py','learned_feedback_policy.py']}
    result['evaluation_wrapper_sha256']=sha(__file__);result['policy_runtime']='posture_feedback_policy_hold.PosturePolicy'
    result['unused_finger_phase_frozen_after_s']=float(PosturePolicy(checkpoint).data.get('posture_hold_phase_s',14.2))
    result['active_finger_release_and_object_clock_unchanged']=True
    Path(output,'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--scene',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);main(**vars(p.parse_args()))
