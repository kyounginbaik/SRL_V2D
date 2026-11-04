"""Fine-grained tail of an execution: per-frame object error vector, object height, wrist (base_yaw) position, finger contacts.
usage: release_tail.py TEACHER_DIR [T_FROM=18.0] [STEP_FRAMES=2]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
src=Path(sys.argv[1]); t_from=float(sys.argv[2]) if len(sys.argv)>2 else 18.; step=int(sys.argv[3]) if len(sys.argv)>3 else 2
prev=json.loads((src/'result.json').read_text()); scene=prev['execution']['scene']; m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m)
epd=Path(prev['input_dir']); ref=TR.load_reference(epd/'tracking_reference.npz',m)['qpos']; cmd=TR.load_reference(src/'reference.npz',m)['qpos']
tr=np.load(prev['execution']['trajectory']); Q=tr['qpos']; n=len(Q); obj='wooden_spoon'; oc=int(m.joint(obj+'_free').qposadr[0]); ob=m.body(obj).id
palm=m.body('robot_right_right_base_yaw').id; og={g for g in range(m.ngeom) if m.geom_bodyid[g]==ob}
wc=[int(m.joint('robot_right_right_pos_'+x).qposadr[0]) for x in 'xyz']
print('t    err_vec(cm) x y z |rot deg| obj z(cm) ref z | wrist z cmd z | contacts')
for k in range(int(t_from*20),n,step):
    d.qpos[:]=Q[k]; mujoco.mj_forward(m,d); p=d.xpos[ob]; e=(p-ref[k,oc:oc+3])*100
    er=np.degrees((R.from_matrix(d.xmat[ob].reshape(3,3))*R.from_quat(ref[k,oc+3:oc+7],scalar_first=True).inv()).magnitude())
    cs={}
    for i in range(d.ncon):
        c=d.contact[i]; b=None
        if c.geom1 in og and m.body(m.geom_bodyid[c.geom2]).name.startswith('robot_'): b=m.body(m.geom_bodyid[c.geom2]).name
        elif c.geom2 in og and m.body(m.geom_bodyid[c.geom1]).name.startswith('robot_'): b=m.body(m.geom_bodyid[c.geom1]).name
        elif c.geom1 in og or c.geom2 in og: b='other:'+m.body(m.geom_bodyid[c.geom2 if c.geom1 in og else c.geom1]).name
        if b: cs['_'.join(b.split('_')[2:4]) if b.startswith('robot_') else b]=cs.get(b,0)+1
    print(f'{k/20:5.2f} [{e[0]:5.2f} {e[1]:5.2f} {e[2]:5.2f}] {er:5.1f} | {p[2]*100:5.1f} {ref[k,oc+2]*100:5.1f} | {d.xpos[palm][2]*100:5.1f} {cmd[k,wc[2]]*100:5.1f} | {cs}')
