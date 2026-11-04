"""G12 A, frontier from saved E matrices (a32b stage 1): lowest-ball-cost roll track satisfying the official ep32 constraints
with a given margin (deg). Saves out/a32c_smooth_rolls.npz + out/a32c_smooth.json."""
import numpy as np, json
G='/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12'
Z=np.load(f'{G}/out/a32b_E.npz'); th=Z['th']; tw=Z['tw']; inf=Z['inf']; Ub=Z['Ub']; E={k:Z[k] for k in ('v24','v25','v26')}
nf,nt=Ub.shape; thw=np.where(th>180,th-360,th); D=np.abs(((th[:,None]-th[None])+180)%360-180)
CAP={'v24':90.,'v25':55.,'v26':55.}
CUP={'v24':2.9,'v25':2.4,'v26':3.0}; SPL={k:2*np.degrees(0.5)-v for k,v in CUP.items()}
def vit(lam,wc,mu):
    U=Ub.copy()
    if wc>0: U[~inf]+=wc*np.abs(((th[None]-tw[~inf,None])+180)%360-180)
    for k,m in zip(('v24','v25','v26'),mu): U-=m*np.minimum(E[k],CAP[k])/4.
    cost=U[0].copy(); cost[thw!=0]+=1e4; bp=np.zeros((nf,nt),int)
    for f in range(1,nf):
        tot=cost[:,None]+lam*D; bp[f]=np.argmin(tot,0); cost=tot[bp[f],np.arange(nt)]+U[f]
    p=np.zeros(nf,int); p[-1]=int(np.argmin(cost))
    for f in range(nf-1,0,-1): p[f-1]=bp[f,p[f]]
    return p
def st(p):
    s={k:float(E[k][np.arange(nf),p].mean()) for k in E}; b=float(np.mean(Ub[inf][np.arange(inf.sum()),p[inf]]*4)); return s,b
C=[]
for lam in (0.05,0.1,0.2):
  for wc in (0,0.005,0.02):
    for m24 in (0,0.01,0.03,0.1):
      for m25 in (0,0.01,0.03,0.1):
        for m26 in (0,0.01,0.03,0.1,0.3):
          p=vit(lam,wc,(m24,m25,m26)); s,b=st(p)
          mg=min(s['v24']-SPL['v24'],s['v25']-30,SPL['v25']-s['v25'],s['v26']-30,SPL['v26']-s['v26'])
          C.append(dict(lam=lam,wc=wc,mu=[m24,m25,m26],E={k:round(v,1) for k,v in s.items()},ball=round(b,2),margin=round(mg,1),p=p))
b0=st(vit(0.05,0,(0,0,0)))
print('unconstrained ball track', b0)
rolls={}; out=[]
for ml in (0,2,4,6,8,10,12,15):
    g=[c for c in C if c['margin']>=ml]
    if not g: print('none at',ml); continue
    c=min(g,key=lambda c:c['ball']); lab=f's{ml}'
    rolls[lab]=np.degrees(np.unwrap(np.radians(thw[c['p']].astype(float))))
    r=rolls[lab]; segs=[]; t=0
    while t<nf:
        if abs((r[t]+180)%360-180)>45:
            a=t
            while t<nf and abs((r[t]+180)%360-180)>45: t+=1
            segs.append([a,t-1,round(float(np.median(r[a:t])))])
        else: t+=1
    tv=float(np.abs(np.diff(r)).sum()); out.append(dict(label=lab,tv=round(tv),**{k:v for k,v in c.items() if k!='p'},segments=segs)); print(lab,out[-1])
rolls['pure_ball']=np.degrees(np.unwrap(np.radians(thw[vit(0.05,0,(0,0,0))].astype(float))))
np.savez(f'{G}/out/a32c_smooth_rolls.npz',**rolls); json.dump(dict(spl=SPL,unconstrained=b0,frontier=out),open(f'{G}/out/a32c_smooth.json','w'),indent=1)
