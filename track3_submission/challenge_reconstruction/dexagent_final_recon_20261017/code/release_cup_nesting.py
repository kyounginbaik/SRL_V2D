"""Execute contact-normal opening after the upper cup gains real support."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np
from supported_release import plan_supported_release,TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_object_support import audit
from audit_full_joint_limits import main as joint_audit

B=Path(__file__).resolve().parents[1];R=B/'refinement_20261018/episode_000010'

def run(start,output):
    source=R/'calibrated_supported_nest';out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    scene=source/'execution/scene_physics.xml';trajectory=source/'execution/trajectory.npz';m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(source/'reference.npz',m)
    Q,release=plan_supported_release(scene,trajectory,source/'reference.npz',side='right',object_name='beige_cup',start_s=start,open_complete_s=start+.35,clear_s=start+1.1,opening_m=.012,maximum_joint_increment_rad=.5,withdrawal_horizontal_m=.12,withdrawal_up_m=.08)
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Robot-only supported nesting release with observed contact-normal finger opening and clear withdrawal',arms_coupled=False,floating_runtime=True)
    execution=execute_reference(scene,reference,out/'execution');inp=R/'calibrated_inputs/episodes/episode_000010';manifest=json.loads((inp/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    with np.load(trajectory) as z:old=z['substep_qpos'].copy()
    with np.load(execution['trajectory']) as z:err=float(abs(z['substep_qpos'][:round(start/.002)]-old[:round(start/.002)]).max())
    assert err==0.
    result=dict(episode='episode_000010',input_dir=str(inp),reference=str(reference),execution=execution,metrics=metrics,protected_prefix_s=start,protected_prefix_max_qpos_error=err,release=release,source_targets_unchanged=True,learned_policy=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');joint_audit(out);audit(out);print(json.dumps(metrics['declared_frame_reference_proxy'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,required=True);p.add_argument('--output',type=Path,required=True);run(**vars(p.parse_args()))
