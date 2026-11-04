"""Who touches the rack, when: per-frame contact census of an executed trajectory (kinematic replay + mj_forward collision pass).
usage: contact_timeline.py TRAJECTORY_NPZ SCENE_XML [STEP]"""
import os; os.environ['OPENBLAS_NUM_THREADS']='1'
import sys, numpy as np, mujoco
tr=np.load(sys.argv[1]); m=mujoco.MjModel.from_xml_path(sys.argv[2]); step=int(sys.argv[3]) if len(sys.argv)>3 else 5; d=mujoco.MjData(m)
names=[str(x) for x in tr['object_names']]; j=names.index('plastic_dish_rack'); p0=tr['object_positions'][0,j]
def grp(b):
    if b.startswith('robot_left'): return 'Lhand'
    if b.startswith('robot_right'): return 'Rhand'
    return b
print('t     rack_disp_cm  contacts with rack: group count deepest_mm (and brush-rack, rack-table)')
for k in range(0,len(tr['qpos']),step):
    d.qpos[:]=tr['qpos'][k]; mujoco.mj_forward(m,d); cnt={}
    for c in d.contact[:d.ncon]:
        b1,b2=grp(m.body(int(m.geom_bodyid[c.geom1])).name),grp(m.body(int(m.geom_bodyid[c.geom2])).name)
        if 'plastic_dish_rack' in (b1,b2):
            o=b1 if b2=='plastic_dish_rack' else b2; n,dd=cnt.get(o,(0,0.)); cnt[o]=(n+1,min(dd,c.dist*1000))
    print(f'{k/20:5.2f}s {np.linalg.norm(tr["object_positions"][k,j]-p0)*100:6.2f}  '+'  '.join(f'{o}:{n}/{dd:.1f}' for o,(n,dd) in sorted(cnt.items())))
