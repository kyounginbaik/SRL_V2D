"""Right-hand rigid-follow pickup spec on top of a two-hand base (left lateral pinch already in the base reference): anchor =
the base's own right pinch (wrist at 1.1 s, fingers = 1.5 s closure), so after the search BOTH hands follow the object track with
fixed offsets. usage: prepare_right_follow.py BASE_RUN EPISODE_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
LEG=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0]=['/data1/SRL_V2D/third_party/DexAgent',str(LEG/'code')]
from tool_library.common import hand as H, tracking as TR
from physics_pickup_search_v2 import dump
base,EP,out=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True)
r=json.loads((base/'result.json').read_text()); scene=r['execution']['scene']; man=json.loads((EP/'input_manifest.json').read_text())
m=mujoco.MjModel.from_xml_path(scene); rig=H.HandRig.build(m,'robot_right_','right'); ref=TR.load_reference(base/'reference.npz',m); Q=ref['qpos']; fps=ref['fps']
fcols=list(rig.finger_qpos.values()); opened=rig.finger_targets(.20); specs=[]
for name,ts,closeat in (('rf_a',dict(start=0.3,hover=0.6,contact=0.95,closed=1.2,transport=1.5,release=19.5),1.5),('rf_b',dict(start=0.3,hover=0.6,contact=1.0,closed=1.3,transport=1.6,release=19.5),1.8)):
    pose=Q[round(1.1*fps)].copy(); pose[fcols]=Q[round(closeat*fps)][fcols]; anchor=out/f'anchor_{name}.npz'; np.savez_compressed(anchor,qpos=pose)
    spec=dict(scene=scene,reference=str(base/'reference.npz'),original_reference=str(EP/'tracking_reference.npz'),source=man['source'],geometry=str(EP/'metric_geometry.json'),
              manifest=str(EP/'input_manifest.json'),object_name='wooden_spoon',side='right',anchor_pose=str(anchor.resolve()),open_fingers=[opened[k] for k in rig.finger_qpos],
              finger_names=list(rig.finger_qpos),times=ts,horizon_s=8.0,hover_offset=[0.,0.,.08],grasp_generator='right pinch anchor from the two-hand base (wrist at 1.1 s, fingers = closure)',previous_teacher=str(base),
              scale=np.r_[[.012]*3,[.15]*3,[.02]*3,[.12]*3,[.15]*22,[.20]*22,[.015]*3,[.10]*3,[.10,.08,.12]].tolist())
    dump(out/f'{name}.json',spec); specs.append(str((out/f'{name}.json').resolve())); print(name,'ok')
json.dump(specs,open(out/'all_specs.json','w'),indent=1)
