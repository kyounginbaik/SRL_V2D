"""Scoring-only: best temporal shift per time window (moving frames only). Usage EP [TAG]"""
import sys, numpy as np, kitx, json
from tshift_lib import shift
ep=int(sys.argv[1]); tag=sys.argv[2] if len(sys.argv)>2 else 'r8_base'; X=kitx.load_export(ep,tag)
base=kitx.score(ep,{n:X['W'][:,i] for i,n in enumerate(X['names'])},X['meshes'])
regs={n:(base['cas'][s],base['rots'][s]) for s,n in enumerate(base['objs'])}
ds=np.arange(-1,3.01,0.25); P=[]
for d in ds:
    o=kitx.score(ep,{n:shift(X['W'][:,i],d) for i,n in enumerate(X['names'])},X['meshes'],regs=regs); P.append(o['pos'])
P=np.array(P) # [D,F,n]
R=base['R'][:,:,:3]; sp=np.r_[[np.zeros(R.shape[1])],np.linalg.norm(np.diff(R,axis=0),axis=2)]*20*100
F=P.shape[1]; W=60; out=[]
for s,n in enumerate(base['objs']):
    for a in range(0,F,W):
        m=(sp[a:a+W,s]>5)
        if m.sum()<8: continue
        e=P[:,a:a+W,s][:,m].mean(1); out.append((n,a,int(m.sum()),float(ds[e.argmin()]),float(e[ds==0][0]),float(e.min())))
        print(ep,n,a,m.sum(),'best',ds[e.argmin()],'err0 %.2f best %.2f'%(e[ds==0][0],e.min()),flush=True)
json.dump(out,open(f'../out/tshiftwin_{tag}_{ep}.json','w'))
