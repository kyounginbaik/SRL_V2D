"""Fresh physical evaluation of a learned gated-posture actor."""
from pathlib import Path
import argparse,json
import evaluate_learned_feedback as runtime
from posture_feedback_policy import PosturePolicy
from learned_feedback_policy import sha

def main(episode,scene,checkpoint,output):
 runtime.LearnedPolicy=PosturePolicy
 result=runtime.evaluate(episode,scene,checkpoint,output)
 result['policy_code_sha256']=sha(Path(__file__).with_name('posture_feedback_policy.py'))
 result['evaluation_wrapper_sha256']=sha(__file__)
 result['policy_runtime']='posture_feedback_policy.PosturePolicy'
 result['algorithm']='radial_basis_behavior_cloning_with_gated_posture_head'
 Path(output,'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--scene',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);main(**vars(p.parse_args()))
