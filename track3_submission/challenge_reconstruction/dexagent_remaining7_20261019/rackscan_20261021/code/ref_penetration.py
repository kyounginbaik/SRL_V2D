"""Min signed distance between the partner object (at its reference pose, every frame) and the rack, old vs new scene."""
import numpy as np, json, mujoco, sys
from scipy.spatial.transform import Rotation as R
W='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021'
def scan(scene, T, names, other):
    m=mujoco.MjModel.from_xml_path(scene); d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0)
    j=m.joint(other+'_free').qposadr[0]; jr=m.joint('plastic_dish_rack_free').qposadr[0]; io=names.index(other); ir=names.index('plastic_dish_rack')
    rack=m.body('plastic_dish_rack').id; ob=m.body(other).id; out=[]
    for k in range(len(T)):
        for jj,ii in ((j,io),(jr,ir)):
            q=R.from_matrix(T[k,ii,:3,:3]).as_quat(); d.qpos[jj:jj+3]=T[k,ii,:3,3]; d.qpos[jj+3:jj+7]=[q[3],q[0],q[1],q[2]]
        mujoco.mj_forward(m,d); best=0.05
        for i in range(d.ncon):
            c=d.contact[i]; bs={m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]}
            if bs=={rack,ob}: best=min(best,c.dist)
        out.append(best)
    return np.array(out)
for ep in map(int, sys.argv[1:]):
    r=json.load(open(f'{W}/retarget/episode_{ep:06d}/result.json')); z=np.load(r['spec']['source']+'/object_poses.npz'); T=z['world_from_object']; names=[str(x) for x in z['names']]
    other=[n for n in names if n!='plastic_dish_rack'][0]
    o=scan(r['rack_swap']['source_scene'],T,names,other); n=scan(r['rack_swap']['scene'],T,names,other)
    print(f'ep{ep} {other}: OLD rack min dist {o.min()*1000:.1f} mm (frames <-3mm: {(o<-.003).sum()}) | NEW rack min {n.min()*1000:.1f} mm (frames <-3mm: {(n<-.003).sum()}, worst at t={n.argmin()/20:.2f}s)')
