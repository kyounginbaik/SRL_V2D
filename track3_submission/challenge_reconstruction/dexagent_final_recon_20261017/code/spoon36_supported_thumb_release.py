"""Open the actual thumb contact before withdrawing from a supported spoon."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from physics_pickup_search import B,TR,dump,execute_reference,evaluate_files
from supported_release import plan_supported_release
from audit_object_support import audit as audit_support
from audit_full_joint_limits import main as audit_joints

def main(opening):
    source=B/'refinement_20261018/episode_000036/right_only_upright/variant_01'
    out=source.parent.parent/'right_only_thumb_release'/f'open_{opening:.3f}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    scene=source/'execution/scene_physics.xml';trajectory=source/'execution/trajectory.npz';m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(source/'reference.npz',m)
    Q,meta=plan_supported_release(scene,trajectory,source/'reference.npz',side='right',object_name='wooden_spoon',start_s=18.5,open_complete_s=18.8,clear_s=19.5,opening_m=opening,withdrawal_horizontal_m=.16,withdrawal_up_m=.10,maximum_joint_increment_rad=.7)
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Open actual thumb contact on supported spoon, then clear withdrawal; alternate right-only carry and immutable original object goals',floating_runtime=True)
    inp=B/'episodes/episode_000036';manifest=json.loads((inp/'input_manifest.json').read_text());execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    n=round(18.5/m.opt.timestep)
    with np.load(execution['trajectory']) as new,np.load(trajectory) as old:assert np.array_equal(new['substep_qpos'][:n],old['substep_qpos'][:n])
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(inp),strategy='right_only_alternative_not_handover',source_reference=str(source/'reference.npz'),protected_prefix_s=18.5,release=meta,reference=str(reference),execution=execution,metrics=metrics,learned_policy=False))
    audit_support(out);audit_joints(out);print(json.dumps(dict(opening=opening,objects=metrics['declared_frame_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--opening',type=float,required=True);main(p.parse_args().opening)
