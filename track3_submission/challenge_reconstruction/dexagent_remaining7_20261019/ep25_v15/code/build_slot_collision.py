"""Episode 25: slot-preserving collision for our own reconstructed boards (NO scans used).
Evidence: our reconstruction's assembled cross (frames >= 17 s) - each board's slot is where the other board passes through it.
Each board = slab (our mesh's principal-frame extents, face-to-face thickness from the mesh's two face planes) minus a slot
(width = other board thickness + clearance, depth = half the board height, opening on the edge found from the cross:
board1 opens upward, board2 downward). Collision boxes are written as convex OBJ parts in the board body frame, plus inertia."""
import numpy as np, trimesh, json, sys
from pathlib import Path
R7=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); A=R7/'collision_recon_slots_v3/episode_000025/assets'; OUT=R7/'ep25_v15/collision'
CLEAR=float(sys.argv[1]) if len(sys.argv)>1 else 0.003
z=np.load(R7/'collision_recon_slots_v3/episode_000025/object_poses.npz'); T=z['world_from_object']; names=[str(x) for x in z['names']]
info={}
for n in names:
    m=trimesh.load(A/f'{n}_visual.obj',force='mesh'); v=np.asarray(m.vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0))
    if np.linalg.det(U)<0: U[:,0]*=-1
    P=trimesh.sample.sample_surface(m,200000,seed=0)[0]; q=(P-c0)@U
    h,e=np.histogram(q[:,0],bins=80); pk=np.sort(e[np.argsort(h)[-2:]]+np.diff(e)[0]/2)
    lo=np.percentile(q,1,axis=0); hi=np.percentile(q,99,axis=0)
    info[n]=dict(c0=c0,U=U,face=(pk[0],pk[1]),mid=(lo[1],hi[1]),long=(lo[2],hi[2]))
# slot centres from the assembled cross (mean over frames >= 17 s): other board's centre expressed in this board's principal frame
ks=[k for k in range(len(T)) if k/20>=17.0]
def other_in(n_self,n_other):
    i,j=names.index(n_self),names.index(n_other); s,o=info[n_self],info[n_other]; out=[]
    for k in ks:
        cw=T[k,j,:3,:3]@o['c0']+T[k,j,:3,3]; cb=T[k,i,:3,:3].T@(cw-T[k,i,:3,3]); out.append((cb-s['c0'])@s['U'])
    up=np.mean([s['U'].T@(T[k,i,:3,:3].T@[0,0,1]) for k in ks],0)
    return np.mean(out,0), up
rep={}
for n,o in (('wooden_piece_1','wooden_piece_2'),('wooden_piece_2','wooden_piece_1')):
    s=info[n]; c_other,up=other_in(n,o); th_o=info[o]['face'][1]-info[o]['face'][0]
    width=th_o+CLEAR; slot_l=c_other[2]
    # slot opens on the mid-edge that points up for board1 (other comes from above) / down for board2 (it descends onto board1)
    up_sign=np.sign(up[1]); open_sign=up_sign if n=='wooden_piece_1' else -up_sign
    m0,m1=s['mid']; l0,l1=s['long']; t0,t1=s['face']; mc=0.5*(m0+m1)
    slot_m=(mc,m1) if open_sign>0 else (m0,mc); solid_m=(m0,mc) if open_sign>0 else (mc,m1)
    boxes=[((t0,t1),solid_m,(l0,l1)),((t0,t1),slot_m,(l0,slot_l-width/2)),((t0,t1),slot_m,(slot_l+width/2,l1))]
    d=OUT/f'{n}_slotbox'; d.mkdir(parents=True,exist_ok=True); parts=[]
    for b,(tt,mm,ll) in enumerate(boxes):
        ext=np.array([tt[1]-tt[0],mm[1]-mm[0],ll[1]-ll[0]]); ctr=np.array([np.mean(tt),np.mean(mm),np.mean(ll)])
        box=trimesh.creation.box(extents=ext); box.apply_translation(ctr); box.vertices=box.vertices@s['U'].T+s['c0']; box.export(d/f'part_{b:03d}.obj'); parts.append(box)
    rep[n]=dict(thickness_cm=float((t1-t0)*100),mid_cm=[float(m0*100),float(m1*100)],long_cm=[float(l0*100),float(l1*100)],slot_centre_long_cm=float(slot_l*100),
                slot_width_cm=float(width*100),slot_open='+mid' if open_sign>0 else '-mid',other_centre_in_frame_cm=(c_other*100).round(2).tolist(),world_up_in_frame=up.round(3).tolist())
json.dump(rep,open(OUT/'slot_collision_report.json','w'),indent=1); print(json.dumps(rep,indent=1))
