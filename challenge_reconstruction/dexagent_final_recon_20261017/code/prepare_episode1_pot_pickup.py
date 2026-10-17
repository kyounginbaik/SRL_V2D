"""Build left-pot pickup proposals while retaining an optimized right-lid plan."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import mujoco
from physics_pickup_search import PickupExperiment,dump

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR


def prepare(search_dir,output=None):
    search_dir=Path(search_dir)
    out=Path(output or B/'refinement_20261018/episode_000001/pot_inputs')
    out.mkdir(parents=True,exist_ok=True)
    settings=json.loads((search_dir/'settings.json').read_text())
    progress=json.loads((search_dir/'progress.json').read_text())
    exp=PickupExperiment(settings['spec'])
    commands,timing=exp.commands(progress['best_parameters'])
    right_reference=TR.save_reference(out/'right_lid_reference.npz',exp.m,commands,
        exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],
        source='Fixed snapshot of right-lid physical pickup CEM; immutable object targets',
        arms_coupled=False,floating_runtime=True)
    dump(out/'right_lid_snapshot.json',dict(search_dir=str(search_dir),
        parameters=progress['best_parameters'],report=progress['history'][-1]['best'],timing=timing))
    native=json.loads((B/'refinement_20261017/geometry_v2/episode_000001/native_grasp_pose_seeds.json').read_text())[0]
    rig=H.HandRig.build(exp.m,'robot_left_','left')
    wcols=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    fcols=np.array(list(rig.finger_qpos.values()))
    original=TR.load_reference(settings['spec']['original_reference'],exp.m)['qpos']
    for candidate in range(2):
        pose=original[34].copy()
        if candidate:
            pose[wcols]=np.r_[native['pos'],native['euler']]
            pose[fcols]=[native['finger_targets'][k] for k in rig.finger_qpos]
        anchor=out/f'pot_anchor_{candidate:02d}.npz'
        np.savez_compressed(anchor,qpos=pose)
        spec=dict(settings['spec'],scene=settings['spec']['scene'],reference=str(right_reference),
                  side='left',object_name='white_pot',anchor_pose=str(anchor),
                  open_fingers=original[20,fcols].tolist(),finger_names=list(rig.finger_qpos),
                  hover_offset=[0.,0.,.06],
                  times=dict(start=.75,hover=1.1,contact=1.45,closed=1.8,transport=2.1,release=15.1),
                  score_hold_start_s=5.8,score_track_start_s=5.8,horizon_s=8.5,
                  scale=np.r_[[.010]*3,[.12]*3,[.015]*3,[.10]*3,
                              [.12]*22,[.16]*22,[.015]*3,[.10]*3,[.10,.08,.12]].tolist())
        dump(out/f'pot_{candidate:02d}.json',spec)
    print(str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--search-dir',required=True);p.add_argument('--output')
    prepare(**vars(p.parse_args()))
