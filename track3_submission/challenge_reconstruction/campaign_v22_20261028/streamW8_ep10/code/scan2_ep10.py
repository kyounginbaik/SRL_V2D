"""Joint 2-D image fit per frame (no GT): blue axial shift sb (along its own axis) x beige rim-to-rim height h
(coaxial beige, spin kept), container-opaque rendering (render_nested). Score = 3-class pixel agreement with the SAM2
masks (hands excluded). Reports best (sb,h), and the profile max_sb acc(h)."""
import sys, json, numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from nestlib import *
R='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000010'
frames=[int(x) for x in sys.argv[1].split(',')]; tag=sys.argv[2]; EPS=0.015
P=np.load(R+'/exports/final_v5/object_poses.npz'); names=[str(n) for n in P['names']]; C=P['camera_from_object']
ib,ie=names.index('blue_cup'),names.index('beige_cup')
mb=load_mesh(R+'/objects/blue_cup/mesh.obj'); me=load_mesh(R+'/objects/beige_cup/mesh.obj')
ab,cb,_=container_axis(mb); ae,ce,_=container_axis(me)
hb=(np.asarray(mb.vertices)-cb)@ab; he=(np.asarray(me.vertices)-ce)@ae
K,Wd,Ht=intrinsics(R); rB=Renderer([mb],K,Wd,Ht); rE=Renderer([me],K,Wd,Ht)
def rel_h(Tb,Te):
    rel=np.linalg.inv(Tb)@Te; rim=rel[:3,:3]@(ce+ae*he.max())+rel[:3,3]; return (rim-cb)@ab-hb.max()
def coax(Tb,Te,h):
    rel=np.linalg.inv(Tb)@Te; a=rel[:3,:3]@ae; Ra=Rot.align_vectors([ab],[a])[0].as_matrix(); rel=rel.copy(); rel[:3,:3]=Ra@rel[:3,:3]
    rim=rel[:3,:3]@(ce+ae*he.max()); rel[:3,3]=cb+ab*(hb.max()+h)-rim; return Tb@rel
SB=np.arange(-0.03,0.0301,0.005); H=np.arange(0.01,0.0801,0.005)
out={}
for f in frames:
    Tb0,Te=C[f,ib],C[f,ie]; h5=rel_h(Tb0,Te)
    mB=read_mask(R,'blue_cup',f); mE=read_mask(R,'beige_cup',f); hand=np.zeros_like(mB)
    for n in ('blue_cup','beige_cup'):
        for l in '23':
            x=read_mask(R,n,f,l)
            if x is not None: hand|=x
    hand=cv2.dilate(hand.astype(np.uint8),np.ones((5,5),np.uint8))>0
    ys,xs=np.nonzero(mB|mE); roi=(max(0,xs.min()-70),max(0,ys.min()-70),min(Wd,xs.max()+70),min(Ht,ys.max()+70)); sl=(slice(roi[1],roi[3]),slice(roi[0],roi[2]))
    gt=np.where(mE[sl],1,np.where(mB[sl],0,-1)); HD=hand[sl]
    def score(Tb,Te2):
        lab,_=render_nested(rB,rE,Tb,Te2,roi,ab,cb,hb.max(),EPS); U=((lab>=0)|(gt>=0))&~HD
        return float(((lab==gt)&U).sum()/max(1,U.sum()))
    a5=score(Tb0,Te)
    A=np.zeros((len(SB),len(H)))
    for i,sb in enumerate(SB):
        Tb=Tb0.copy(); Tb[:3,3]+=sb*(Tb0[:3,:3]@ab)
        Te_s=Te.copy(); Te_s[:3,3]+=sb*(Tb0[:3,:3]@ab)   # beige rides with blue so h is relative
        for j,h in enumerate(H): A[i,j]=score(Tb,coax(Tb,Te_s,h))
    k=np.unravel_index(A.argmax(),A.shape); prof=A.max(0)
    out[f]=dict(h_v5=float(h5),acc_v5=a5,SB=SB.tolist(),H=H.tolist(),A=A.tolist())
    near=lambda x: prof[np.abs(H-x).argmin()]
    print(f"f{f} v5 h={h5*100:.2f} acc {a5:.3f} | best sb={SB[k[0]]*100:+.1f} h={H[k[1]]*100:.2f} acc {A.max():.3f} | profile acc@h2.5 {near(.025):.3f} @4.5 {near(.045):.3f} @6.5 {near(.065):.3f}",flush=True)
json.dump(out,open(f'../out/scan2_{tag}.json','w'))
