"""Episode 25 board 1: pickup-search specs anchored on the teacher's own left pinch (wrist at contact time from the base
reference, fingers = the reference closure after the lift), so the CEM can find a firmer closure / pose; hold until RELEASE.
usage: prepare_board1_pickup.py BASE_RUN OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from physics_pickup_search import dump
R7=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); EP=R7/'physics_slots_v3/episodes/episode_000025'
base,out=Path(sys.argv[1]),Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
r=json.loads((base/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
m=mujoco.MjModel.from_xml_path(scene); rig=H.HandRig.build(m,'robot_left_','left'); ref=TR.load_reference(base/'reference.npz',m); Q=ref['qpos']; fps=ref['fps']
fcols=list(rig.finger_qpos.values()); opened=rig.finger_targets(.20); specs=[]
for name,ts,closeat,hz,sc in (('t20_h8',dict(start=1.2,hover=1.6,contact=1.95,closed=2.3,transport=2.7,release=15.5),3.0,8.0,1.0),
                            ('t20_h8_s05',dict(start=1.2,hover=1.6,contact=1.95,closed=2.3,transport=2.7,release=15.5),3.0,8.0,0.5),
                            ('t22_h10',dict(start=1.4,hover=1.8,contact=2.1,closed=2.45,transport=2.8,release=15.5),3.2,10.0,1.0),
                            ('t20_h7',dict(start=1.2,hover=1.6,contact=1.95,closed=2.3,transport=2.7,release=15.5),3.5,7.0,0.75)):
    pose=Q[round(ts['contact']*fps)].copy(); pose[fcols]=Q[round(closeat*fps)][fcols]
    anchor=out/f'anchor_{name}.npz'; np.savez_compressed(anchor,qpos=pose)
    spec=dict(scene=scene,reference=str(base/'reference.npz'),original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
              manifest=str(EP/'input_manifest.json'),object_name='wooden_piece_1',side='left',anchor_pose=str(anchor),open_fingers=[opened[k] for k in rig.finger_qpos],
              finger_names=list(rig.finger_qpos),times=ts,horizon_s=hz,hover_offset=[0.,0.,.05],grasp_generator='teacher pinch anchor (wrist at contact, fingers = post-lift closure)',
              previous_teacher=str(base),scale=(np.r_[[.012]*3,[.15]*3,[.02]*3,[.12]*3,[.15]*22,[.20]*22,[.015]*3,[.10]*3,[.10,.08,.12]]*sc).tolist())
    dump(out/f'{name}.json',spec); specs.append(str(out/f'{name}.json')); print(name,'ok')
json.dump(specs,open(out/'all_specs.json','w'),indent=1)
