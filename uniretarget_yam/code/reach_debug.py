"""Planned versus achieved fingertips at closure for fresh proposals."""
import sys,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
e=Path(sys.argv[1]);t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz',sys.argv[2],int(sys.argv[3]));t.stage='lift';t.set_rung('lift');m=t.m;rng=np.random.default_rng(2);n=0
z0=t.s.poses[0,t.s.object_index];print('onset',t.onset,'reset obj',z0[:3].round(3),'settled',t.o0.round(3),'obj at f10-f40 drift mm',[round(1e3*float(np.linalg.norm(t.s.poses[f,t.s.object_index,:3]-t.o0)),1) for f in (10,20,30,40,44)])
while n<3:
    g=t.propose(rng)
    if g is None:continue
    s=t.solve_hand(g,rng)
    if s['residual_m']>=.006 or s['penetration_m']>=.006 or t.sweep(s)>=.006:continue
    n+=1;Q,T=t.compile_tape(s,t.onset+6);d0=t.scratch;goals={x:np.asarray(g[x])+.007*np.asarray(g['n_'+x]) for x in ('thumb','index','middle')}
    d0.qpos[:]=t.s.native(Q[0]);d0.qpos[t.slide_q]=np.asarray(s['p'])-t.slide_offset;d0.qpos[t.ball_q:t.ball_q+4]=(t.fixed.inv()*R.from_rotvec(s['r'])).as_quat()[[3,0,1,2]];d0.qpos[t.finger_q]=s['closed'];mujoco.mj_kinematics(m,d0);print('proposal',n,'cov',round(s['coverage'],2),'planned tip-goal mm',{x:round(1e3*float(np.linalg.norm(t.tip_world(d0,x)-goals[x])),1) for x in goals})
    t.s.reset(Q[0])
    for f in range(t.onset+3):
        t.s.step(T[f]);d=t.s.d
        if f in (t.onset-20,t.onset-12,t.onset-6,t.onset-3,t.onset+1):
            pos=d.qpos[t.qo:t.qo+3];ro=R.from_quat(d.qpos[t.qo+3:t.qo+7][[1,2,3,0]]);moved=ro*t.r0.inv();world={x:pos+moved.apply(goals[x]-t.o0) for x in goals}
            fq=[t.s.qs[t.s.jmap[j]] for j in t.fingers];sl=[t.s.qs[t.s.jmap[j]] for j in t.slides]
            print('  f',f,'wrist err cm',round(100*float(np.linalg.norm(d.qpos[t.slide_q]-Q[f+1][sl])),2),'finger err max',round(float(np.abs(d.qpos[t.finger_q]-Q[f+1][fq]).max()),2),'tip-to-goal mm',{x:round(1e3*float(np.linalg.norm(t.tip_world(d,x)-world[x])),0) for x in goals},'obj moved mm',round(1e3*float(np.linalg.norm(pos-t.o0)),1))
