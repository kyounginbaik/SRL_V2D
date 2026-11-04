import sys,json,collections
sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,yam_refine as F,yam_loop as Y
from scipy.spatial.transform import Rotation as R
Y.ONSET_FLOOR=1.4;ep,side,k=sys.argv[1],sys.argv[2],int(sys.argv[3]);t=Y.YamTask(f'/uniretarget_yam/episodes/{ep}',side,k);t.set_stage('real');ref=f'/uniretarget_yam/episodes/{ep}/yam/flash/mink/reference.parquet';F.SEED['parquet']=ref
tips,pp,pr=F.reference_tips(t,ref);F.SEED.update(tips=tips,palm_p=pp,palm_r=pr)
bid=t.m.body(f'{side}_arm_base_link').id;d=t.p;d.qpos[:]=t.base_q;import mujoco;mujoco.mj_kinematics(t.m,d);b=d.xpos[bid].copy()
g=t.gp_raw;print(ep,'onset',t.onset,'base',b.round(2).tolist(),'object rest rel base',(t.o0-b).round(2).tolist(),'goal rel base range',(g-b).min(0).round(2).tolist(),(g-b).max(0).round(2).tolist(),'goal dist max',round(float(np.linalg.norm(g-b,axis=1).max()),2),'turn deg',round(float(np.degrees((t.gr_raw*t.gr_raw[0].inv()).magnitude()).max()),0))
print(' tips to object cm',{x:round(100*float(np.linalg.norm(t.verts-tips[x],axis=1).min()),1) for x in Y.DIGITS},'palm-up',round(float(pr.apply(t.local_normal)@t.natural_palm),2),'extent',np.ptp(t.verts,axis=0).round(2).tolist(),'mass',round(t.mg/9.81,2))
for label,seedp,prop in (('palm+seed',True,F.propose),('noPalm+seed',False,F.propose),('unseeded',False,Y.YamTask.propose)):
    rng=np.random.default_rng(0);c=collections.Counter();car=[]
    for i in range(40):
        if prop is F.propose:
            g_=F.propose(t,rng,tries=30)
            if not seedp:t.palm_seed=None
        else:
            t.palm_seed=None;g_=prop(t,rng)
        if g_ is None:c['no_contacts']+=1;continue
        t.fail=None;s=t.solve_hand(g_,rng)
        if s is None:c[t.fail[0] if t.fail else 'other']+=1;continue
        if s['residual_m']>=.008:c['finger_residual']+=1
        elif s['penetration_m']>=.008:c['penetration']+=1
        elif t.reach_speed(s)>Y.ARM_SPEED:c['approach_speed']+=1
        else:c['ok']+=1;car.append(t.carry_score(s))
    print(' ',label,dict(c),'carry p50/max',(round(float(np.median(car)),2),round(float(max(car)),2)) if car else None,flush=True)
