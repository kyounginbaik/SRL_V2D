"""Scoring-only: error vs temporal shift of our tracks (SE3 interpolation). Usage tshift.py EP"""
import sys, numpy as np, kitx, json
from scipy.spatial.transform import Rotation as Rot, Slerp
ep=int(sys.argv[1]); tag=sys.argv[2] if len(sys.argv)>2 else 'r8_base'; X=kitx.load_export(ep,tag)
def shift(T,d):
    F=len(T); t=np.clip(np.arange(F)+d,0,F-1); i0=np.floor(t).astype(int); i1=np.minimum(i0+1,F-1); a=t-i0
    out=T.copy(); out[:,:3,3]=(1-a)[:,None]*T[i0,:3,3]+a[:,None]*T[i1,:3,3]
    R0=Rot.from_matrix(T[i0,:3,:3]); R1=Rot.from_matrix(T[i1,:3,:3]); rv=(R0.inv()*R1).as_rotvec()
    out[:,:3,:3]=(R0*Rot.from_rotvec(rv*a[:,None])).as_matrix(); return out
base=kitx.score(ep,{n:X['W'][:,i] for i,n in enumerate(X['names'])},X['meshes'])
regs={n:(base['cas'][s],base['rots'][s]) for s,n in enumerate(base['objs'])}
res={}
for d in ([float(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else np.arange(-2,2.01,0.25)):
    o=kitx.score(ep,{n:shift(X['W'][:,i],d) for i,n in enumerate(X['names'])},X['meshes'],regs=regs)
    res[float(d)]={n:(float(o['pos'][:,s].mean()),float(o['ori'][:,s].mean())) for s,n in enumerate(o['objs'])}
    print(ep,f"{d:+.2f}",' '.join(f"{n} {v[0]:.3f}/{v[1]:.2f}" for n,v in res[float(d)].items()),flush=True)
json.dump(res,open(f'../out/tshift_{tag}_{ep}.json','w'))
