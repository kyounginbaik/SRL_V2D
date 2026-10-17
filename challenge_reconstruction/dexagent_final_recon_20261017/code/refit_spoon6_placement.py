"""Finite setdown residual correction on the physically optimized learned actor."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from evaluate_mp_sr import load_reference,apply_world_transform,rotation_matrices,validate_geometry
from evaluate_policy_component import evaluate
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit
from learned_feedback_policy import sha
B=Path(__file__).resolve().parents[1]
def run(case):
 source=B/'refinement_20261018/expanded_batch/optimized_release6/seed64';old=json.loads((source/'evaluation/result.json').read_text());out=B/f'refinement_20261018/expanded_batch/refit_release6/case_{case}';out.mkdir(parents=True,exist_ok=True)
 if (out/'training_report.json').exists():raise FileExistsError(out)
 with np.load(source/'policy.npz') as z:w={k:z[k] for k in z.files}
 p=str(w['multi_release_prefixes'][-1]);m=mujoco.MjModel.from_xml_path(old['execution']['scene']);ep=Path(old['input_dir']);manifest=json.loads((ep/'input_manifest.json').read_text());ref=apply_world_transform(load_reference(manifest['source']),np.asarray(manifest['setup_from_source']));anchors,_,_=validate_geometry(ref['object_names'],json.loads((ep/'metric_geometry.json').read_text()));oi=ref['object_names'].index('wooden_spoon');anchor=anchors[oi];targetR=rotation_matrices(ref['object_quaternions_wxyz'])[-1,0,oi];target=ref['object_positions'][-1,0,oi]+targetR@anchor
 with np.load(old['execution']['trajectory']) as z:q=z['qpos'][-1];initial=z['substep_qpos'][round(float(w[p+'start_s'])/.002)-1]
 oc=int(m.joint('wooden_spoon_free').qposadr[0]);actualR=Rotation.from_quat(q[oc+3:oc+7],scalar_first=True).as_matrix();actual=q[oc:oc+3]+actualR@anchor;gain=[.5,.8,1.0][case%3];delta=gain*(target-actual);delta[2]=0.;w[p+'place_delta_m']=w[p+'place_delta_m']+delta
 rotation_fraction=0. if case<3 else .3
 if rotation_fraction:
  ra=w[p+'rotation_actuator_ids'];wrist=Rotation.from_euler('XYZ',initial[w['qadr'][ra]]);r0=Rotation.from_rotvec(w[p+'place_rotation_vector']);extra=Rotation.from_rotvec(rotation_fraction*Rotation.from_matrix(targetR@actualR.T).as_rotvec());w[p+'place_rotation_vector']=(wrist.inv()*extra*wrist*r0).as_rotvec()
 checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**w);result=evaluate(6,old['execution']['scene'],checkpoint,out/'evaluation',ep);ad=out/'evaluation/audit';ad.mkdir(exist_ok=True);quality=joint_audit(result['execution']['scene'],result['execution']['trajectory']);(ad/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n');support_audit(out/'evaluation')
 report=dict(method='Finite placement residual correction learned from physical final pose versus immutable demonstrated target',source_checkpoint=str(source/'policy.npz'),source_checkpoint_sha256=sha(source/'policy.npz'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),translation_residual_m=delta.tolist(),translation_gain=gain,rotation_fraction=rotation_fraction,original_learned_weights_unchanged=True,reference_targets_scene_limits_clock_unchanged=True,runtime_has_no_object_targets_or_command_table=True,reference_proxy_success=result['reference_proxy_success']);(out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=range(6),required=True);run(**vars(p.parse_args()))
