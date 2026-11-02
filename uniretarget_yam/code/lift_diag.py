"""Why can't this episode lift? Compare object section, table clearance and tip reach."""
import sys,numpy as np,mujoco
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from frontier_loop import Loop
for ep in sys.argv[1:]:
    e=Path(ep);t=Loop(e/'native_model.mjb',e/'baseline/physics.npz',e/'plan.npz','right',0);m=t.m;P,N=t.surface(np.random.default_rng(0),1500);h=P@t.up-t.low;com_d=np.linalg.norm(P-t.com,axis=1);near=com_d<.06
    print(e.name,'onset',t.onset,'up',t.up.round(3).tolist(),'objZrange along up mm',round(1e3*float(np.ptp(t.verts@t.up)),1),'surface kept',len(P),'near COM',int(near.sum()),'side-facing near COM',int((near&(np.abs(N@t.up)<.5)).sum()),'height of those mm pct',np.percentile(1e3*h[near],[10,50,90]).round(1).tolist())
    d=mujoco.MjData(m);d.qpos[:]=t.s.q0;mujoco.mj_forward(m,d);print('  contacts on object at f0',sorted({m.body(int(m.geom_bodyid[c.geom[0]])).name[:12]+'|'+m.body(int(m.geom_bodyid[c.geom[1]])).name[:12]+f'|{c.dist*1e3:.1f}' for c in d.contact[:d.ncon] if t.obj in m.geom_bodyid[c.geom]})
    g=0;ok=0;rng=np.random.default_rng(3);cov=[]
    for _ in range(60):
        x=t.propose(rng,tries=10)
        if x is None:continue
        g+=1;s=t.solve_hand(x,rng);ok+=s['residual_m']<.006 and s['penetration_m']<.006;cov.append(x['coverage'])
    print('  proposals',g,'/60 IK ok',ok,'coverage mean',round(float(np.mean(cov)),2) if cov else None,'axial spread of object',round(float(np.ptp(t.axial)),3))
