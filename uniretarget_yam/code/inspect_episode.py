"""Render fresh proposals at closure with their contact forces, for any episode."""
import os;os.environ['MUJOCO_GL']='osmesa'
import sys,numpy as np,mujoco,cv2
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
e=Path(sys.argv[1]);side=sys.argv[3] if len(sys.argv)>3 else 'right';k=int(sys.argv[4]) if len(sys.argv)>4 else 0
t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz',side,k);t.stage='lift';t.set_rung('lift');m=t.m;m.vis.global_.offwidth=480;m.vis.global_.offheight=360;r=mujoco.Renderer(m,360,480);cam=mujoco.MjvCamera();cam.distance=.5;cam.elevation=-30;cam.azimuth=115;cam.lookat[:]=t.center;rng=np.random.default_rng(1);tiles=[];f6=np.zeros(6)
print('onset',t.onset,'object mass',float(m.body_mass[t.obj]),'extent',np.ptp(t.verts,axis=0).round(3).tolist(),'com',t.com.round(3).tolist(),'low',round(t.low,3),'friction',m.geom_friction[np.flatnonzero(m.geom_bodyid==t.obj)[0]][0])
n=0;tries=0
while n<6 and tries<200:
    tries+=1;g=t.propose(rng)
    if g is None:continue
    s=t.solve_hand(g,rng)
    if s['residual_m']>.006 or s['penetration_m']>.006:continue
    n+=1;Q,T=t.compile_tape(s,60);t.s.reset(Q[0]);log=[]
    for f in range(55):
        t.s.step(T[f]);d=t.s.d
        if f in (20,26,30,36,44):
            hand={};up=0.;floor=0.
            for i in range(d.ncon):
                c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]]);mujoco.mj_contactForce(m,d,i,f6);fw=c.frame.reshape(3,3).T@f6[:3]
                if t.obj in (a,b):
                    o=a if b==t.obj else b
                    if o in t.right:nm=m.body(o).name.split('_')[-2]+m.body(o).name.split('_')[-1];hand[nm]=round(hand.get(nm,0)+float(np.linalg.norm(fw)),1);up+=float((fw if b==t.obj else -fw)@t.up)
                    else:floor+=float(f6[0])
            log.append((f,hand,round(up/t.mg,2),round(floor/t.mg,2),round(1e3*float((d.qpos[t.qo:t.qo+3]-t.o0)@t.up),1)))
        if f==30:
            r.update_scene(d,cam);im=r.render().copy();cv2.putText(im,f"#{n} cover {s['coverage']:.2f}",(8,20),0,.5,(255,255,0),1);tiles.append(im)
    print(n,'cov',round(s['coverage'],2),'res',round(1e3*s['residual_m'],1),'pen',round(1e3*s['penetration_m'],1));[print('   ',x) for x in log]
if tiles:
    while len(tiles)<6:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(sys.argv[2],cv2.cvtColor(np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:6])]),cv2.COLOR_RGB2BGR))
