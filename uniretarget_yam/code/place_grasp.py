"""Among the reachable, collision-free workspace shifts, pick the one where the robot can actually grasp and carry every held object."""
import argparse,json,sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,yam_loop as Y
T={}
def task(a,i,h):
    if i in T:return T[i]
    Y.JOIN=dict(start=int(h['start'])) if h.get('join') else None;t=Y.YamTask(a,h['side'],int(h['index']));t.set_stage('real');s={}
    for name,m in (('m',t.m),('ik',t.ik_m)):
        robot=np.array([m.body(b).name.startswith(('left_','right_')) for b in range(m.nbody)]);roots=[b for b in range(m.nbody) if robot[b] and not robot[m.body_parentid[b]]];rigs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==0 and (m.geom(g).name or '').startswith('yam_')];s[name]=(m,roots,rigs,m.body_pos.copy(),m.geom_pos.copy())
    t.shift=s;T[i]=t;return t
def job(x):
    ep,i,h,off,seed,tries=x;t=task(ep,i,h);Y.JOIN=dict(start=int(h['start'])) if h.get('join') else None
    for m,roots,rigs,bp,gp in t.shift.values():
        m.body_pos[:]=bp;m.geom_pos[:]=gp
        for b in roots:m.body_pos[b]=bp[b]-np.array(off)
        for g in rigs:m.geom_pos[g]=gp[g]-np.array(off)
    rng=np.random.default_rng(seed);ok=0;carry=[]
    for _ in range(tries):
        g=t.propose(rng)
        if g is None:continue
        s=t.solve_hand(g,rng)
        if s is not None and s['residual_m']<.008 and s['penetration_m']<.008 and t.reach_speed(s)<=Y.ARM_SPEED:
            ok+=1
            if len(carry)<2:carry.append(t.carry_score(s))
    return i,tuple(off),ok,carry
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--place',required=True);p.add_argument('--top',type=int,default=12);p.add_argument('--tries',type=int,default=16);p.add_argument('--workers',type=int,default=30);a=p.parse_args();Y.ONSET_FLOOR=1.4
    M=json.load(open(Path(a.episode)/'recon_meta.json'));H=M['holds'];D=json.load(open(a.place));C=sorted(D['candidates'],key=lambda c:c[2])[:a.top]
    jobs=[(a.episode,i,h,[c[0],c[1],0.],1000*n+i+w,a.tries) for n,c in enumerate(C) for i,h in enumerate(H) for w in range(8)]
    with Pool(a.workers) as P:res=P.map(job,jobs,chunksize=1)
    tab={}
    for i,off,ok,car in res:
        r=tab.setdefault(off[:2],{}).setdefault(i,[0,[]]);r[0]+=ok;r[1]+=car
    rows=[]
    for off,v in tab.items():
        per=[(v[i][0]/(8*a.tries))*(.1+float(np.mean(v[i][1])) if v[i][1] else 0.) for i in range(len(H))];rows.append((min(per),off,[round(x,3) for x in per]))
    rows.sort(key=lambda r:-r[0]);best=rows[0];reach={(round(c[0],2),round(c[1],2)):c[2] for c in C};chosen=best[1] if best[0]>0 else (C[0][0],C[0][1])
    D.update(offset=[float(chosen[0]),float(chosen[1]),0.],grasp_table=[[r[1][0],r[1][1],r[0],r[2]] for r in rows]);Path(a.place).write_text(json.dumps(D))
    print(json.dumps(dict(place_grasp=[round(float(x),2) for x in chosen],score=round(best[0],3),per_hold=best[2],zero_grasp_everywhere=best[0]==0,table=[[round(float(r[1][0]),2),round(float(r[1][1]),2),round(r[0],3)] for r in rows[:6]])),flush=True)
