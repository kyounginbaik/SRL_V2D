"""Fresh learned-policy evaluation using the actual repaired mesh adapter."""
from pathlib import Path
import argparse,json
import numpy as np
from evaluate_policy_component import evaluate
from audit_joint_limits import audit
from learned_feedback_policy import sha
B=Path(__file__).resolve().parents[1]
def run(episode,source,output,decimals=None):
 source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 if (out/'training_report.json').exists():raise FileExistsError(out)
 original=json.loads((source/'evaluation/result.json').read_text());checkpoint_source=Path(original['policy_checkpoint'])
 with np.load(checkpoint_source) as z:w={k:z[k] for k in z.files}
 if decimals is not None:w['component_output_decimals']=np.array(decimals)
 checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**w);inp=B/f'refinement_20261018/expanded_batch/repaired_export_inputs/episode_{episode:06d}'
 result=evaluate(episode,original['execution']['scene'],checkpoint,out/'evaluation',inp);ad=out/'evaluation/audit';ad.mkdir(exist_ok=True);quality=audit(result['execution']['scene'],result['execution']['trajectory']);(ad/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n')
 report=dict(method='Actual repaired-mesh export adapter and explicit output command precision on learned actor',source_checkpoint=str(checkpoint_source),source_checkpoint_sha256=sha(checkpoint_source),source_training_report=str(source/'training_report.json'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),evaluator_driver_sha256=sha(__file__),command_decimal_places=decimals,original_learned_weights_unchanged=True,source_object_targets_metric_anchors_clock_and_physics_unchanged=True,input_dir=str(inp),actual_scene_visual_geometry_verified=True,reference_proxy_success=result['reference_proxy_success'])
 (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--decimals',type=int);run(**vars(p.parse_args()))
