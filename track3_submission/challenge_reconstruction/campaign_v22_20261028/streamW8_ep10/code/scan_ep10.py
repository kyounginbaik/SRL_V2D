"""Image evidence vs nest depth for ep10 (no GT). Beige translated along the blue axis (camera frame) from its v5 pose.
mode 'shift': keep v5 beige orientation; mode 'coax': beige axis aligned to blue axis, centred on it (spin kept)."""
import sys, json, numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from nestlib import *
R='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000010'
mode=sys.argv[1]; EPS=float(__import__('os').environ.get('EPS','0.006')); frames=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else list(range(80,161,2))
P=np.load(R+'/exports/final_v5/object_poses.npz'); names=[str(n) for n in P['names']]; C=P['camera_from_object']
ib,ie=names.index('blue_cup'),names.index('beige_cup')
mb=load_mesh(R+'/objects/blue_cup/mesh.obj'); me=load_mesh(R+'/objects/beige_cup/mesh.obj')
ab,cb,_=container_axis(mb); ae,ce,_=container_axis(me)
hb=(np.asarray(mb.vertices)-cb)@ab; he=(np.asarray(me.vertices)-ce)@ae
K,Wd,Ht=intrinsics(R); rend=Renderer([mb,me],K,Wd,Ht); rB=Renderer([mb],K,Wd,Ht); rE=Renderer([me],K,Wd,Ht)
def beige_pose(Tb,Te,hrr):
    """beige pose with rim-to-rim height hrr (beige rim above blue rim, along blue axis)."""
    rel=np.linalg.inv(Tb)@Te
    if mode=='coax':
        a=rel[:3,:3]@ae; Ralign=Rot.align_vectors([ab],[a])[0].as_matrix(); rel=rel.copy(); rel[:3,:3]=Ralign@rel[:3,:3]
        # centre: beige rim centre on blue axis
        rim=rel[:3,:3]@(ce+ae*he.max()); rel[:3,3]=cb+ab*(hb.max()+hrr)-rim
    else:
        rim=rel[:3,:3]@(ce+ae*he.max())+rel[:3,3]; cur=(rim-cb)@ab-hb.max(); rel=rel.copy(); rel[:3,3]+=ab*(hrr-cur)
    return Tb@rel
H=np.round(np.arange(1.0,8.01,0.25),3)/100
out={}
for f in frames:
    Tb,Te=C[f,ib],C[f,ie]
    rel=np.linalg.inv(Tb)@Te; rim=rel[:3,:3]@(ce+ae*he.max())+rel[:3,3]; h5=(rim-cb)@ab-hb.max()
    mB=read_mask(R,'blue_cup',f); mE=read_mask(R,'beige_cup',f)
    hand=np.zeros_like(mB)
    for n in ('blue_cup','beige_cup'):
        for l in '23':
            x=read_mask(R,n,f,l)
            if x is not None: hand|=x
    hand=cv2.dilate(hand.astype(np.uint8),np.ones((5,5),np.uint8))>0
    dep=read_depth(R,f)
    ys,xs=np.nonzero(mB|mE); x0,x1=max(0,xs.min()-60),min(Wd,xs.max()+60); y0,y1=max(0,ys.min()-60),min(Ht,ys.max()+60)
    roi=(x0,y0,x1,y1); sl=(slice(y0,y1),slice(x0,x1))
    MB,ME,HD,DP=mB[sl],mE[sl],hand[sl],dep[sl]
    rows=[]
    for h in [h5]+list(H):
        Te2=beige_pose(Tb,Te,h); lab,z=render_nested(rB,rE,Tb,Te2,roi,ab,cb,hb.max(),EPS)
        RB,RE=(lab==0)&~HD,(lab==1)&~HD
        iouB=(RB&MB).sum()/max(1,(RB|(MB&~HD)).sum()); iouE=(RE&ME).sum()/max(1,(RE|(ME&~HD)).sum())
        # 3-class agreement on the union of rendered/masked cup pixels
        U=(RB|RE|MB|ME)&~HD; gt=np.where(ME,1,np.where(MB,0,-1)); acc=((np.where(RE,1,np.where(RB,0,-1))==gt)&U).sum()/max(1,U.sum())
        rz=lambda Rm,Mm: np.nanmedian((z-DP)[Rm&Mm&np.isfinite(DP)]) if (Rm&Mm&np.isfinite(DP)).sum()>20 else np.nan
        dE,dB=rz(RE,ME),rz(RB,MB)
        rows.append(dict(h=float(h),iouB=float(iouB),iouE=float(iouE),acc=float(acc),dzE=float(dE),dzB=float(dB)))
    out[f]=dict(h_v5=float(h5),rows=rows)
    best=max(rows[1:],key=lambda r:r['acc']); bE=max(rows[1:],key=lambda r:r['iouE'])
    r0=rows[0]
    print(f"f{f} v5 h={h5*100:.2f} acc {r0['acc']:.3f} iouE {r0['iouE']:.3f} iouB {r0['iouB']:.3f} dzE-dzB {(r0['dzE']-r0['dzB'])*1000:.1f}mm | best-acc h={best['h']*100:.2f} acc {best['acc']:.3f} | best-iouE h={bE['h']*100:.2f} iouE {bE['iouE']:.3f}",flush=True)
json.dump(out,open(f'../out/scan_{mode}_nested{int(EPS*1000)}.json','w'))
