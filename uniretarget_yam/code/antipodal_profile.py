import sys,time,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from antipodal_search import Antipodal
from flash_chord.retargeting.surface_contact_search import demand_coverage
root=Path(__file__).parent.parent/'runs';t0=time.monotonic();t=Antipodal(root/'surface_anchor_fit_v7/native_model.mjb',root/'cpu_baseline_jacobians_v1/physics.npz',Path(__file__).parents[3]/'runtime_canonical_rest_anchor_v1/episode_000036/runs/plan_v1/plan.npz')
print('init s',round(time.monotonic()-t0,2),'tris',len(t.tri),'motion frames',len(t.motion),'peak F',round(float(np.linalg.norm(t.motion[:,:3],axis=1).max()),2),'peak tau',round(float(np.linalg.norm(t.motion[:,3:],axis=1).max()),4))
rng=np.random.default_rng(0);t0=time.monotonic();P,N=t.surface(rng,1500);print('surface s',round(time.monotonic()-t0,4),'kept',len(P),'of 1500')
pairs=tri=0;cov=[];tt=0.;t0=time.monotonic()
for _ in range(60):
    i=int(rng.integers(len(P)));a,na=P[i],N[i];line=P-a;dist=np.linalg.norm(line,axis=1)+1e-9;ok=(N@na<-.85)&(dist>.008)&(dist<.07)&((line/dist[:,None])@(-na)>.85)
    if not ok.any():continue
    pairs+=1;j=int(rng.choice(np.flatnonzero(ok)));sep=np.linalg.norm(P-a,axis=1);near=(N@na>.7)&(sep>.015)&(sep<.05)
    if not near.any():continue
    tri+=1;k=int(rng.choice(np.flatnonzero(near)));ps=[t.patch(a,na,0),t.patch(P[k],N[k],1),t.patch(P[j],N[j],2)];lim={g:8*t.mg for g in range(3)};mn={g:t.mg for g in range(3)};s=time.monotonic();cov.append(demand_coverage(ps,lim,mn,t.motion,t.o0));tt+=time.monotonic()-s
print('60 tries s',round(time.monotonic()-t0,2),'pairs',pairs,'triples',tri,'LP sec each set',round(tt/max(1,tri),3),'coverage pct',np.percentile(cov,[10,50,90]).round(2).tolist() if cov else [],'>=.75',sum(c>=.75 for c in cov))
t0=time.monotonic();gs=[]
while len(gs)<12 and time.monotonic()-t0<40:
    g=t.propose(rng,tries=20)
    if g:gs.append(g)
print('proposals',len(gs),'sec each',round((time.monotonic()-t0)/max(1,len(gs)),2),'coverage',[round(g['coverage'],2) for g in gs])
res=[];t0=time.monotonic()
for g in gs:
    for _ in range(4):
        s=t.solve_hand(g,rng);res.append((round(1e3*s['residual_m'],1),round(1e3*s['penetration_m'],1)))
print('solve s each',round((time.monotonic()-t0)/max(1,len(res)),3),'residual pct',np.percentile([r[0] for r in res],[10,50,90]).tolist(),'penetration pct',np.percentile([r[1] for r in res],[10,50,90]).tolist(),'accepted',sum(r[0]<6 and r[1]<6 for r in res),'of',len(res))
