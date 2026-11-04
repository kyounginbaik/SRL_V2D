"""For each scheduled hold: did the assigned hand really hold its object? Uses one recorded world from video/run npz. Usage: hold_check.py EP_DIR clip.npz"""
import sys,json,numpy as np,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/uniretarget_yam/code')
import yam_loop as Y
Y.ONSET_FLOOR=1.4
E,clip=sys.argv[1:3];M=json.load(open(E+'/recon_meta.json'));z=np.load(clip)
if 'objs' not in z:print('clip has one object only');sys.exit()
t=Y.YamTask(E,M['holds'][0]['side'],M['holds'][0]['index']);m=t.ik_m;d=mujoco.MjData(m)
names=[str(n) for n in z['names']];mp=[]
for n in names:
    j=(("right" if n[0]=="R" else "left")+"_arm_joint"+n[-1]) if n.startswith(("R_arm_j","L_arm_j")) else (("right" if n.startswith("right") else "left")+"_arm_"+n)
    mp.append(int(m.jnt_qposadr[m.joint(j).id]))
F=len(z['q']);O=z['objs']
for h in M['holds']:
    k,side=h['index'],h['side'];q_=t.object_q[k];pb=m.body(('right' if side=='right' else 'left')+'_arm_palm') if False else None
    T=Y.YamTask(E,side,k);palm=T.palm;on=T.onset;end=min(int(h['end']),F-1);rel=[];rr=[];cf=np.zeros(F,bool);pen=[]
    obj=int(T.obj);hand={b for b in range(m.nbody) if m.body(b).name.startswith(side)}
    for f in range(0,F):
        d.qpos[:]=T.base_q;d.qpos[mp]=z['q'][f]
        for kk,qk in enumerate(t.object_q):d.qpos[qk:qk+3]=O[f,kk,:3];d.qpos[qk+3:qk+7]=O[f,kk][[6,3,4,5]]
        mujoco.mj_forward(m,d);P=d.xpos[palm].copy();Rp=R.from_matrix(d.xmat[palm].reshape(3,3).copy())
        rel.append(Rp.inv().apply(O[f,k,:3]-P));rr.append(Rp.inv()*R.from_quat(O[f,k,3:7]))
        for i in range(d.ncon):
            c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
            if (a==obj and b in hand) or (b==obj and a in hand):
                if c.dist<.003:cf[f]=True
                pen.append(float(c.dist))
    rel=np.array(rel);s=min(on+10,end);w=np.arange(s,end+1);drift=np.linalg.norm(rel[w]-rel[s],axis=1);turn=np.array([(rr[f]*rr[s].inv()).magnitude() for f in w])
    g=z['goals'][:,k];gm=np.linalg.norm(g[s:end+1]-g[s],axis=1).max();om=np.linalg.norm(O[s:end+1,k,:3]-O[s,k,:3],axis=1).max();rise=(O[s:end+1,k,2]-O[s,k,2]).max()
    print(json.dumps(dict(object=h['object'],hand=side,frames=[int(s),int(end)],contact_frac=round(float(cf[w].mean()),3),rel_drift_cm_max=round(100*float(drift.max()),1),rel_turn_deg_max=round(float(np.degrees(turn.max())),1),goal_move_cm=round(100*float(gm),1),object_move_cm=round(100*float(om),1),object_rise_cm=round(100*float(rise),1),deepest_cm=round(100*min(pen),2) if pen else None)),flush=True)
