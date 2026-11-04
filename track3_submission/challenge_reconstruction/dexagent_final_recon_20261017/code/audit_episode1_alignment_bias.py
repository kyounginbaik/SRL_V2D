"""Explain, without changing it, the original initial-scene-aligned proxy."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from evaluate_mp_sr import (load_reference,load_trajectory,validate_poses,resample_to_reference,
 apply_world_transform,validate_geometry,rotation_matrices,align_initial_scene,angular_errors,sha256)


def audit(result):
    p=Path(result).resolve();report=json.loads(p.read_text());metric=report['metrics'];provenance=metric['provenance']
    ref=validate_poses(load_reference(Path(provenance['reference']).parent))
    raw=validate_poses(load_trajectory(provenance['trajectory']))
    raw=resample_to_reference(raw,ref)
    raw=apply_world_transform(raw,np.asarray(metric['recording_to_reference_transform']))
    anchors,axes,kinds=validate_geometry(ref['object_names'],metric['geometry'])
    for x in (ref,raw):x['object_positions']+=np.einsum('twoij,oj->twoi',rotation_matrices(x['object_quaternions_wxyz']),anchors)
    aligned,transforms=align_initial_scene(raw,ref,axes,kinds)
    t=ref['timestamps_seconds'];normal=np.array([-.0390194052,-.1545787589,.9872096501]);normal/=np.linalg.norm(normal)
    errors={};displacement={}
    for label,value in [('declared_source_frame',raw),('original_initial_scene_aligned',aligned)]:
        dv=value['object_positions']-ref['object_positions']
        errors[label]=(np.linalg.norm(dv,axis=-1),np.degrees(angular_errors(value['object_quaternions_wxyz'],ref['object_quaternions_wxyz'],axes,kinds)),dv@normal)
    displacement=aligned['object_positions']-raw['object_positions']
    phases=[('initial_and_approach',0.,1.5),('first_lid_transfer',1.5,4.5),('pot_acquisition',4.5,5.5),('pot_shake_lid_rest',5.5,11.),('second_lid_transfer',11.,14.5),('supported_setdown',14.5,15.15),('release',15.15,16.700001)]
    rows=[]
    for phase,a,b in phases:
        mask=(t>=a)&(t<b);objects={}
        for i,name in enumerate(ref['object_names']):
            vals={}
            for label,(pos,ang,vertical) in errors.items():
                vals[label]=dict(mean_position_cm=float(100*pos[mask,0,i].mean()),mean_orientation_deg=float(ang[mask,0,i].mean()),mean_signed_table_normal_error_cm=float(100*vertical[mask,0,i].mean()))
            vals['alignment_only_displacement_cm']=float(100*np.linalg.norm(displacement[mask,0,i],axis=-1).mean())
            vals['alignment_only_signed_table_normal_displacement_cm']=float(100*(displacement[mask,0,i]@normal).mean())
            objects[name]=vals
        rows.append(dict(phase=phase,original_time_range_s=[a,min(b,16.7)],frames=int(mask.sum()),objects=objects))
    for label,key in [('declared_source_frame','declared_frame_reference_proxy'),('original_initial_scene_aligned','official_style_reference_proxy')]:
        for i,name in enumerate(ref['object_names']):
            if not np.allclose(100*errors[label][0][:,0,i].mean(),metric[key]['objects'][name]['mean_position_error_cm'][0],atol=1e-10,rtol=0):raise ValueError('Audit does not reproduce saved metric')
    out=dict(scope='Diagnostic decomposition; no reference, trajectory, thresholds, or metric formula changed',
       result=str(p),result_sha256=sha256(p),metric_provenance=provenance,
       recording_to_source_transform=metric['recording_to_reference_transform'],original_initial_scene_transform=transforms[0],
       initial_alignment_rotation_degrees=float(np.degrees(Rotation.from_matrix(np.asarray(transforms[0])[:3,:3]).magnitude())),
       initial_alignment_translation_m=np.asarray(transforms[0])[:3,3].tolist(),
       independently_fitted_table_normal_source=normal.tolist(),
       table_evidence=str(Path(__file__).resolve().parents[1]/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json'),
       whole_episode=dict(original_aligned=metric['official_style_reference_proxy'],declared_frame=metric['declared_frame_reference_proxy']),phases=rows,
       interpretation='The official-style common transform matches the repaired physical pot initial orientation back to the old reconstructed pot orientation, and applies that same rotation to the lid. It therefore introduces height/position offsets for the lid resting beside the pot. This explains a metric difference, not an exemption or official success. The original proxy remains unchanged and must be reported.',
       allowed_reference_revision='Only independently source-image/stereo-supported reconstruction changes; never fitted to achieved physics or selected to cross the score threshold.')
    target=p.parent/'audit/alignment_bias.json';target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(path=str(target),rotation_deg=out['initial_alignment_rotation_degrees'],phases=rows),indent=2))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);audit(**vars(p.parse_args()))
