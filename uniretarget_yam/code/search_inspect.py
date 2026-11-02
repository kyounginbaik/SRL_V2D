import os;os.environ['MUJOCO_GL']='osmesa'
import sys,json,numpy as np,mujoco,cv2
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from grasp_search import Task
root=Path(__file__).parent.parent/'runs';t=Task(root/'surface_anchor_fit_v7/native_model.mjb',root/'cpu_baseline_jacobians_v1/physics.npz',Path(__file__).parents[3]/'runtime_canonical_rest_anchor_v1/episode_000036/runs/plan_v1/plan.npz')
m=t.m;m.vis.global_.offwidth=480;m.vis.global_.offheight=360;rng=np.random.default_rng(5);r=mujoco.Renderer(m,360,480);cam=mujoco.MjvCamera();cam.distance=.45;cam.elevation=-30;cam.azimuth=115;cam.lookat[:]=t.center
right={int(m.jnt_bodyid[j]) for j in t.joints};f6=np.zeros(6);tiles=[]
print('open q',np.round(t.q_open,2).tolist());print('table/object: center',np.round(t.center,3).tolist())
samples=[t.initial(rng,'across') for _ in range(4)]+[t.initial(rng,'demo') for _ in range(2)]
samples[0]['subset']=3;samples[1]['subset']=3
for k,s in enumerate(samples):
    Q,T=t.compile(s,60);t.s.reset(Q[0]);log={}
    for f in range(55):
        t.s.step(T[f]);d=t.s.d
        if f in (20,27,33,38,45):
            hand={};table=0.;up=0.
            for i in range(d.ncon):
                c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]]);mujoco.mj_contactForce(m,d,i,f6)
                if (t.obj in (a,b)) and (a in right or b in right):
                    o=a if b==t.obj else b;fw=c.frame.reshape(3,3).T@f6[:3];fo=fw if b==t.obj else -fw;nm=m.body(o).name.replace('right_sharpa_wave_right_','');hand[nm]=round(hand.get(nm,0)+float(np.linalg.norm(fo)),1);up+=float(fo@t.up)
                elif (a in right or b in right) and (m.body(a).name=='support' or m.body(b).name=='support' or a==0 or b==0):table+=float(f6[0])
            log[f]=dict(hand=hand,up=round(up,2),table=round(table,1),z_mm=round(1e3*float(d.qpos[t.qo:t.qo+3]@t.up-t.o0@t.up),1))
        if f==33:
            r.update_scene(d,cam);im=r.render().copy();cv2.putText(im,f"#{k} subset{s['subset']} {s['mode']}",(8,20),0,.5,(255,255,0),1);tiles.append(im)
    print(k,s['mode'],'subset',s['subset'],'gain',round(s['gain'],2));[print('   ',f,v) for f,v in log.items()]
cv2.imwrite(sys.argv[1],cv2.cvtColor(np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:6])]),cv2.COLOR_RGB2BGR))
