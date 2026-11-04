"""Grasp diagnosis of an executed trajectory: per-frame contacts between the object and robot fingers, object pose in the
carrying palm frame (in-hand slip/roll after pickup), and object error vs the episode-dir tracking reference.
usage: diag_grasp.py TRAJ_NPZ SCENE_XML EPISODE_DIR OBJECT SIDE [OUT_JSON]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
traj,scene,epdir,obj,side=sys.argv[1:6]; out=sys.argv[6] if len(sys.argv)>6 else None
m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); z=np.load(traj); fps=float(z['fps']); Q=z['qpos']; n=len(Q)
ref=TR.load_reference(Path(epdir)/'tracking_reference.npz',m)['qpos']; oc=int(m.joint(obj+'_free').qposadr[0])
ob=m.body(obj).id; palm=m.body(f'robot_{side}_{side}_base_yaw').id
ogeoms={g for g in range(m.ngeom) if m.geom_bodyid[g]==ob}
def body_of_geom(g): return m.body(m.geom_bodyid[g]).name
rows=[]; rel=[]
for k in range(n):
    d.qpos[:]=Q[k]; mujoco.mj_forward(m,d)
    Tw_o=np.eye(4); Tw_o[:3,:3]=d.xmat[ob].reshape(3,3); Tw_o[:3,3]=d.xpos[ob]
    Tw_p=np.eye(4); Tw_p[:3,:3]=d.xmat[palm].reshape(3,3); Tw_p[:3,3]=d.xpos[palm]
    Tp_o=np.linalg.inv(Tw_p)@Tw_o; rel.append(Tp_o)
    cs={}
    for i in range(d.ncon):
        c=d.contact[i]; g1,g2=c.geom1,c.geom2
        if g1 in ogeoms and body_of_geom(g2).startswith('robot_'): b=body_of_geom(g2); nrm=c.frame[:3]
        elif g2 in ogeoms and body_of_geom(g1).startswith('robot_'): b=body_of_geom(g1); nrm=-c.frame[:3]
        else: continue
        part='_'.join(b.split('_')[2:4]); cs.setdefault(part,[]).append(nrm)
    p=d.xpos[ob]; e=np.linalg.norm(p-ref[k,oc:oc+3])*100
    er=np.degrees((R.from_matrix(Tw_o[:3,:3])*R.from_quat(ref[k,oc+3:oc+7],scalar_first=True).inv()).magnitude())
    rows.append(dict(t=k/fps,pos_cm=float(e),rot_deg=float(er),contacts={a:len(b) for a,b in cs.items()},normals={a:np.mean(b,0).round(2).tolist() for a,b in cs.items()}))
rel=np.array(rel)
# in-hand drift: relative pose vs. a reference frame taken at the first second-long window where >=2 finger parts contact
held=[k for k in range(n) if len(rows[k]['contacts'])>=2]
print(f'frames {n}, fps {fps}; frames with >=2 contact parts: {len(held)}')
if held:
    k0=held[0]; print(f'first 2-part contact at {k0/fps:.2f}s')
for s in range(0,n,10):
    r=rows[s]; k=min(n-1,s)
    if held:
        kb=held[0]; dT=np.linalg.inv(rel[kb])@rel[k]; dp=np.linalg.norm(dT[:3,3])*100; dr=np.degrees(R.from_matrix(dT[:3,:3]).magnitude())
    else: dp=dr=0
    print(f'{r["t"]:5.2f}s err {r["pos_cm"]:5.2f}cm {r["rot_deg"]:5.1f}deg | in-hand drift vs first hold {dp:5.2f}cm {dr:5.1f}deg | contacts {r["contacts"]}')
if out: json.dump(dict(rows=rows,held_first=held[0] if held else None),open(out,'w'))
