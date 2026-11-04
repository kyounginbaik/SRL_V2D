"""Compare identical original command strategies in the measured gravity frame."""
from pathlib import Path
import json,sys
import mujoco,numpy as np
from scipy.spatial.transform import Rotation

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR
from floating_runtime import _freeze,execute_reference
from calibrate_controller import controller_scene
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz


def main():
    inp=B/'refinement_20261018/episode_000010/calibrated_inputs/episodes/episode_000010'
    out=B/'refinement_20261018/episode_000010/calibrated_baselines';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inp/'input_manifest.json').read_text())
    frozen=out/'frozen_initial';frozen.mkdir(exist_ok=True)
    _,base_scene=_freeze(inp/'stage2/scene.xml',frozen)
    scene=out/'controller_scene.xml';controller_scene(base_scene,scene,'kp5000_r200_d5')
    model=mujoco.MjModel.from_xml_path(str(scene));native=TR.load_reference(inp/'tracking_reference.npz',model)
    old=B/'refinement_20261017/hand_combinations/episode_000010/original_left_cem_right'
    old_model=mujoco.MjModel.from_xml_path(str(old/'execution/scene_physics.xml'));ref=TR.load_reference(old/'reference.npz',old_model)
    old_manifest=json.loads((B/'episodes/episode_000010/input_manifest.json').read_text())
    delta=np.array(manifest['setup_from_source'])@np.array(old_manifest['source_from_setup'])
    rr=Rotation.from_matrix(delta[:3,:3]);q=ref['qpos'].copy()
    assert [model.joint(j).name for j in range(model.njnt)]==[old_model.joint(j).name for j in range(old_model.njnt)]
    for side in ['left','right']:
        cols=np.array([int(model.joint(f'robot_{side}_{side}_{n}').qposadr[0]) for n in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']])
        q[:,cols[:3]]=rr.apply(q[:,cols[:3]])+delta[:3,3]
        q[:,cols[3:]]=continuous_euler_xyz(rr*Rotation.from_euler('XYZ',q[:,cols[3:]]),native['qpos'][0,cols[3:]])
    for j in range(model.njnt):
        if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
            a=int(model.jnt_qposadr[j]);q[:,a:a+7]=native['qpos'][:,a:a+7]
    q[0]=native['qpos'][0]
    mixed=TR.save_reference(out/'mixed_reference.npz',model,q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Original-left/CEM-right robot commands rigidly transformed with the calibrated table frame; object targets unchanged.',floating_runtime=True)
    reports={}
    for name,reference in [('native',inp/'tracking_reference.npz'),('mixed',mixed)]:
        target=out/name
        if (target/'result.json').exists():raise FileExistsError(target)
        target.mkdir(exist_ok=True)
        execution=execute_reference(scene,reference,target/'execution')
        metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
        result=dict(episode='episode_000010',strategy=name,reference=str(reference),input_manifest=str(inp/'input_manifest.json'),execution=execution,metrics=metrics,learned_policy=False,source_targets_unchanged=True)
        (target/'result.json').write_text(json.dumps(result,indent=2)+'\n');reports[name]=result
        print(json.dumps(dict(strategy=name,mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['declared_frame_reference_proxy']['objects'])),flush=True)
    (out/'comparison.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
