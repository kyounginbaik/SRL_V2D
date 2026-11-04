"""Two bounded acquisition comparisons: thumb closed or opened by 2 mm.

The thumb posture is blended during the existing approach, rather than trying
to close a 10 mm gap after a transient index touch. Limits and physics persist.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial import cKDTree
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from lid_collision_grasps import collision_surface
from contact_seek_runtime import execute_seeking
from evaluate_mp_sr import evaluate_files
from audit_acquisition import audit


def run(task):
    opening=(0.,.002)[task]
    source=B/'refinement_20261017/lid_collision_feedback/episode_000001/candidate_00_open10'
    config=json.loads((source/'config.json').read_text())
    config.update(seeking_finger='index',seeking_speed_m_s=.02,seeking_travel_limit_m=.01,
                  requested_thumb_opening_m=opening,preposition_thumb_during_existing_approach=True)
    scene=source/'execution/scene_physics.xml'
    if opening==0:
        config['open_thumb_targets']={k:v for k,v in config['closed_finger_targets'].items() if H.finger_of_joint(k)=='thumb'}
        config['actual_thumb_outward_travel_m']=0.
        config['thumb_opening_ik_residual_m']={'thumb':0.}
    else:
        model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
        data.qpos[:]=np.load(config['collision_planned_pose'])['qpos'];mujoco.mj_forward(model,data)
        rig=H.HandRig.build(model,'robot_right_','right');body=model.body(config['object_name']).id
        op=data.xpos[body].copy();R=data.xmat[body].reshape(3,3).copy()
        points,normals,_=collision_surface(model,data,body)
        p=data.site_xpos[rig.fingertip_sites['thumb']].copy()
        index=cKDTree(points).query((p-op)@R)[1];normal=R@normals[index]
        residual=H.finger_ik(model,data,rig,{'thumb':p+opening*normal},iters=100)
        config['open_thumb_targets']={k:float(data.qpos[a]) for k,a in rig.finger_qpos.items() if H.finger_of_joint(k)=='thumb'}
        config['actual_thumb_outward_travel_m']=float((data.site_xpos[rig.fingertip_sites['thumb']]-p)@normal)
        config['thumb_opening_ik_residual_m']=residual
    out=B/f'refinement_20261017/lid_thumb_preposition/episode_000001/thumb_open{round(opening*1000)}'
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(config,indent=2))
    execution=execute_seeking(scene,config['source_reference'],out/'execution',config)
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    retention=audit(out/'execution');report=dict(config=config,execution=execution,metrics=metrics,retention=retention)
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(metrics['official_style_reference_proxy'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,choices=(0,1),required=True)
    run(parser.parse_args().task)
