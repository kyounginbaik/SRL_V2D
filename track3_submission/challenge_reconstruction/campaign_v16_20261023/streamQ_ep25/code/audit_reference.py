"""Stream Q ep25: per-frame physical-consistency audit of OUR reconstructed reference (object_poses.npz of the slots_v3 source).
Per frame: each board's lowest mesh point above the table plane z=0 (neg = penetration, pos while unsupported = floating), long-axis
tilt from horizontal, mid-axis (board width) tilt from vertical; relative pose of board 2 in board 1's principal frame
(thin, mid, long cm), angle between the long axes, and the board-board overlap of the 6 mm slabs (face-plane thickness from the mesh,
feet excluded) measured as the SAT penetration depth of the two oriented slabs.
usage: audit_reference.py OBJECT_POSES.npz OUT.json"""
import sys, json, numpy as np, trimesh
from pathlib import Path
A=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025/assets')
src=Path(sys.argv[1]); out=Path(sys.argv[2])
z=np.load(src); T=z['world_from_object']; names=[str(x) for x in z['names']]
F={}
for n in names:
    m=trimesh.load(A/f'{n}_visual.obj',force='mesh'); v=np.asarray(m.vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0))
    if np.linalg.det(U)<0: U[:,0]*=-1
    q=(v-c0)@U; h,e=np.histogram(q[:,0],bins=80); pk=np.sort(e[np.argsort(h)[-2:]]+np.diff(e)[0]/2)
    lo=np.percentile(q,0.5,axis=0); hi=np.percentile(q,99.5,axis=0)
    F[n]=dict(v=v,c0=c0,U=U,slab_c=np.array([pk.mean(),(lo[1]+hi[1])/2,(lo[2]+hi[2])/2]),slab_h=np.array([(pk[1]-pk[0])/2,(hi[1]-lo[1])/2,(hi[2]-lo[2])/2]))
def sat_depth(c1,R1,h1,c2,R2,h2):
    axes=[R1[:,i] for i in range(3)]+[R2[:,i] for i in range(3)]+[np.cross(R1[:,i],R2[:,j]) for i in range(3) for j in range(3)]
    best=1e9
    for a in axes:
        n=np.linalg.norm(a)
        if n<1e-6: continue
        a=a/n; r1=np.sum(h1*np.abs(R1.T@a)); r2=np.sum(h2*np.abs(R2.T@a)); d=r1+r2-abs((c2-c1)@a); best=min(best,d)
    return best   # >0 overlap depth (m), <0 separation
rows=[]
for k in range(len(T)):
    r=dict(t=k/20)
    W={}
    for i,n in enumerate(names):
        Rw=T[k,i,:3,:3]; tw=T[k,i,:3,3]; f=F[n]; P=f['v']@Rw.T+tw; Fr=Rw@f['U']; c=Rw@(f['c0']+f['U']@f['slab_c'])+tw
        W[n]=(c,Fr,f['slab_h'])
        r[n]=dict(z_low_cm=float(P[:,2].min()*100),centre_z_cm=float(c[2]*100),centre_xy_cm=(c[:2]*100).round(2).tolist(),
                  long_tilt_deg=float(np.degrees(np.arcsin(abs(Fr[2,2])))),mid_from_vertical_deg=float(np.degrees(np.arccos(min(1,abs(Fr[2,1]))))),
                  thin_from_vertical_deg=float(np.degrees(np.arccos(min(1,abs(Fr[2,0]))))))
    c1,F1,h1=W[names[0]]; c2,F2,h2=W[names[1]]
    rel=F1.T@(c2-c1)
    r['rel_2_in_1_cm']=(rel*100).round(2).tolist()
    r['long_axes_angle_deg']=float(np.degrees(np.arccos(min(1,abs(F1[:,2]@F2[:,2])))))
    r['mid_axes_angle_deg']=float(np.degrees(np.arccos(min(1,abs(F1[:,1]@F2[:,1])))))
    r['slab_overlap_mm']=float(sat_depth(c1,F1,h1,c2,F2,h2)*1000)
    rows.append(r)
json.dump(dict(source=str(src),slabs={n:dict(half_extents_cm=(F[n]['slab_h']*100).round(3).tolist()) for n in names},frames=rows),open(out,'w'),indent=0)
for k in list(range(0,356,10))+[355]:
    r=rows[k]; a,b=r[names[0]],r[names[1]]
    print(f"{r['t']:5.2f} | b1 zlow {a['z_low_cm']:5.1f} cz {a['centre_z_cm']:5.1f} longtilt {a['long_tilt_deg']:4.0f} midV {a['mid_from_vertical_deg']:4.0f} | b2 zlow {b['z_low_cm']:5.1f} cz {b['centre_z_cm']:5.1f} longtilt {b['long_tilt_deg']:4.0f} midV {b['mid_from_vertical_deg']:4.0f} | rel {r['rel_2_in_1_cm']} longang {r['long_axes_angle_deg']:4.0f} midang {r['mid_axes_angle_deg']:4.0f} ovl {r['slab_overlap_mm']:6.1f}")
