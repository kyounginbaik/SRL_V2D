"""Episode-28 pot diagnostics: frame-0 support of the pot, reference pot height/tilt over time, where the executed pot ends up vs the reference.
usage: diag_pot.py TRAJECTORY_NPZ EPISODE_DIR SCENE_XML"""
import os; os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0]=['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code','/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
traj, epdir, scene = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); ref=TR.load_reference(epdir/'tracking_reference.npz', m)['qpos']; tr=np.load(traj)
names=[str(x) for x in tr['object_names']]; ip=names.index('white_pot'); ir=names.index('plastic_dish_rack')
oc=int(m.joint('white_pot_free').qposadr[0]); rc=int(m.joint('plastic_dish_rack_free').qposadr[0])
def bodyname(g): return m.body(int(m.geom_bodyid[g])).name
def contacts(q, obj):
    d.qpos[:]=q; mujoco.mj_forward(m,d); out={}
    for c in d.contact[:d.ncon]:
        b1,b2=bodyname(c.geom1),bodyname(c.geom2)
        if obj in (b1,b2):
            o=b1 if b2==obj else b2; n,dd=out.get(o,(0,0.)); out[o]=(n+1,min(dd,c.dist*1000))
    return out
# pot geom extent at frame 0: lowest collision vertex z vs table top
d.qpos[:]=ref[0]; mujoco.mj_forward(m,d)
tb=[b for b in range(m.nbody) if 'table' in m.body(b).name.lower()]
print('table bodies', [m.body(b).name for b in tb])
zmin=1e9
for g in range(m.ngeom):
    if m.body(int(m.geom_bodyid[g])).name=='white_pot' and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g]; v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        w=(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]); zmin=min(zmin,w[:,2].min())
ztab=None
for g in range(m.ngeom):
    if 'table' in m.body(int(m.geom_bodyid[g])).name.lower():
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX: ztab=d.geom_xpos[g][2]+m.geom_size[g][2]
        elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE: ztab=d.geom_xpos[g][2]
print(f'frame0: pot lowest collision vertex z={zmin:.4f} m, table top z={ztab}, gap mm={(zmin-ztab)*1000 if ztab is not None else None}')
print('frame0 pot contacts', contacts(ref[0],'white_pot'), ' rack contacts', contacts(ref[0],'plastic_dish_rack'))
print('reference pot: t  z(cm)  tilt-from-frame0(deg)  speed  rack-contact(mm)')
R0=R.from_quat(ref[0,oc+3:oc+7],scalar_first=True)
for k in range(0,len(ref),10):
    Rk=R.from_quat(ref[k,oc+3:oc+7],scalar_first=True); sp=np.linalg.norm(ref[min(k+1,len(ref)-1),oc:oc+3]-ref[k,oc:oc+3])*20*100
    print(f'  {k/20:5.1f}  {ref[k,oc+2]*100:6.2f}  {np.degrees((Rk*R0.inv()).magnitude()):6.1f}  {sp:5.1f}  {contacts(ref[k],"white_pot")}')
n=min(len(ref),len(tr['qpos']))
print('executed pot vs reference, 7.5-11 s every 0.1 s: t pos_err(cm) dxyz(cm) rot_err(deg) tilt-axis contacts')
for k in range(150,n,2):
    p=tr['object_positions'][k,ip]; pr=ref[k,oc:oc+3]; Ra=R.from_quat(tr['object_quaternions_wxyz'][k,ip][[1,2,3,0]]); Rr=R.from_quat(ref[k,oc+3:oc+7],scalar_first=True)
    dr=(Ra*Rr.inv()); print(f'  {k/20:5.2f} {np.linalg.norm(p-pr)*100:5.2f} {np.round((p-pr)*100,2)} {np.degrees(dr.magnitude()):5.1f} {np.round(dr.as_rotvec()/max(dr.magnitude(),1e-9),2)} {contacts(tr["qpos"][k],"white_pot")}')
