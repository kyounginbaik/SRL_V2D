"""G12 A: consensus ep32 roll = per-frame circular mean of the official-consistent, ball-near-optimal family members
(s4, s6, s8, s10, d4_4, d6_4, d4_7, m8), 5-frame smoothed; frame 0 = ours. Approximate E via a32b_E matrices."""
import numpy as np, json
G='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12'
Z=np.load(f'{G}/out/a32b_E.npz'); Ub=Z['Ub']; inf=Z['inf']; E={k:Z[k] for k in ('v24','v25','v26')}; nf=Ub.shape[0]
S=np.load(f'{G}/out/a32c_smooth_rolls.npz'); Dl=np.load(f'{G}/out/a32c_delta_rolls.npz'); M=np.load(f'{G}/out/a32c_rolls.npz')
fam={'s4':S['s4'],'s6':S['s6'],'s8':S['s8'],'s10':S['s10'],'d4_4':Dl['d4_4'],'d6_4':Dl['d6_4'],'d4_7':Dl['d4_7'],'m8':M['m8']}
A=np.radians(np.stack(list(fam.values())))
mean=np.degrees(np.arctan2(np.sin(A).mean(0),np.cos(A).mean(0))); R=np.hypot(np.sin(A).mean(0),np.cos(A).mean(0))
un=np.degrees(np.unwrap(np.radians(mean))); sm=un.copy()
for t in range(nf): sm[t]=un[max(0,t-2):t+3].mean()
sm[0]=0
w=lambda a:(a+180)%360-180
def ev(r):
    j=np.round((r%360)/5).astype(int)%72
    return {k:round(float(E[k][np.arange(nf),j].mean()),1) for k in E}, round(float(np.mean(Ub[inf][np.arange(inf.sum()),j[inf]]*4)),2)
print('consensus E, ballpx', ev(sm)); print('dist to members', {k:round(float(np.mean(np.abs(w(sm-v)))),1) for k,v in fam.items()})
print('member spread R (1=agree): median', round(float(np.median(R)),2), 'frames R<0.7:', int((R<0.7).sum()))
print('roll blocks', [int(np.median(w(sm[i:i+10]))) for i in range(0,259,10)])
np.savez(f'{G}/out/a32_consensus.npz', consensus=sm, R=R, **fam)
json.dump(dict(members=list(fam), E_ball=ev(sm), roll=np.round(sm,1).tolist(), agreement=np.round(R,2).tolist()), open(f'{G}/out/a32_consensus.json','w'))
