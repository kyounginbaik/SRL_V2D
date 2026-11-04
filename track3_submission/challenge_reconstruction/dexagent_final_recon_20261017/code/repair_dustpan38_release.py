"""Test a supported, contact-normal opening of the acquired episode-38 grasp.

Only robot release commands change. The original object targets, full source
clock, physical model, successful pickup and other hand are preserved.
"""
from pathlib import Path
import argparse,json,sys
import mujoco,numpy as np
from supported_release import plan_supported_release
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_joint_limits import audit

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import tracking as TR


def run(start_s,opening_m,output):
    source=B/'refinement_20261018/expanded_batch/dustpan_continue/episode_000038/seed_45'
    inp=B/'episodes/episode_000038';scene=source/'execution/scene_physics.xml'
    trajectory=source/'execution/trajectory.npz';old_ref=source/'reference.npz'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise FileExistsError(output)
    Q,meta=plan_supported_release(scene,trajectory,old_ref,side='right',object_name='mini_dust_pan',
        start_s=start_s,open_complete_s=start_s+.35,clear_s=start_s+1.30,opening_m=opening_m)
    m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(old_ref,m)
    reference=TR.save_reference(output/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Measured contact-normal opening after actual table support; robot-only release refinement with immutable object targets and original source clock.',floating_runtime=True)
    execution=execute_reference(scene,reference,output/'execution')
    old=np.load(trajectory);new=np.load(execution['trajectory']);steps=round(start_s/m.opt.timestep)
    err=float(np.max(np.abs(old['substep_qpos'][:steps]-new['substep_qpos'][:steps])))
    if err!=0.:raise ValueError(f'Protected physical prefix changed: {err}')
    manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    limits=audit(execution['scene'],execution['trajectory'])
    result=dict(episode='episode_000038',reference=str(reference),execution=execution,metrics=metrics,
        release=meta,actual_joint_limits=limits,protected_prefix_max_qpos_error=err,
        object_targets_unchanged=True,source_clock_unchanged=True,learned_policy=False)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['declared_frame_reference_proxy']['objects'],
        max_hinge_excursion_deg=limits['hinge_max_violation_degrees'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start-s',type=float,required=True)
    p.add_argument('--opening-m',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
