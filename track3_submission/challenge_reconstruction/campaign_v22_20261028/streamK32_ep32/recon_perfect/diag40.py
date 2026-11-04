import sys,numpy as np
sys.argv=['merge.py','spoon','x','0','4'];src=open("merge.py").read().rsplit("\nif __name__",1)[0];exec(src)
Tm=np.load('merged_spoon.npy');B=np.load('mg_spoon_0.npz')['B']
def rollrel(a,b):
    d=R.from_matrix(a[:3,:3].T@b[:3,:3]).as_rotvec();return np.degrees(d@LONG),np.degrees(np.linalg.norm(d-(d@LONG)*LONG))
def spk(T,p):
    det=DETS[p]
    if not det:return np.nan
    ub=proj(B@T[:3,:3].T+T[:3,3]);return float(np.median([min(30,np.min(np.linalg.norm(ub-q,axis=1))) for q in det]))
print('frame | roll merged->hansol (deg, other) | sil cost merged/hansol | depth mm | spike px | mark cost')
for p in list(range(38,56,3))+list(range(156,166,3)):
    dat=getdat(p);a=parts(Tm[p],*dat);b=parts(han[p],*dat);r,o=rollrel(Tm[p],han[p])
    print(f'{p:3d} | {r:6.1f} {o:5.1f} | {a[0]:.2f} {b[0]:.2f} | {1e3*a[2]:5.1f} {1e3*b[2]:5.1f} | {spk(Tm[p],p):5.1f} {spk(han[p],p):5.1f} | {markcost(Tm[p],p+1):.3f} {markcost(han[p],p+1):.3f}  mark={ANN.get(p+1)}')
