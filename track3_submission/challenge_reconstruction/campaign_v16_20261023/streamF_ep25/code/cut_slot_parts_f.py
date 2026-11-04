"""Episode 25: cut the assembly slot out of the EXISTING (validated) collision parts of our boards, leaving every part outside
the slot untouched. Slot placement from our reconstruction's assembled cross (build_slot_collision.py report). Slot width =
the other board's actual collision thickness across its crossing band + clearance. Convex parts are clipped against the slot
planes with trimesh.slice_plane (cap=True); each output piece stays convex.
usage: cut_slot_parts.py CLEARANCE_M OUT_TAG"""
import numpy as np, trimesh, json, sys, glob
from pathlib import Path
R7=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019'); A=R7/'collision_recon_slots_v3/episode_000025/assets'; W=R7/'ep25_v15'; WOUT=Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamF_ep25')
CLEAR=float(sys.argv[1]); TAG=sys.argv[2]; rep0=json.load(open(W/'collision/slot_collision_report.json'))
def frame(n):
    v=np.asarray(trimesh.load(A/f'{n}_visual.obj',force='mesh').vertices); c0=v.mean(0); w,U=np.linalg.eigh((v-c0).T@(v-c0))
    if np.linalg.det(U)<0: U[:,0]*=-1
    return c0,U
def parts(n): return [trimesh.load(f,force='mesh').convex_hull for f in sorted(glob.glob(str(A/f'{n}_convex_repaired/part_*.obj')))]
def clip(mesh, nrm, point):
    """keep the part of a convex mesh with nrm.(x-point) >= 0 (trimesh slice_mesh_plane keep convention)"""
    nrm=np.asarray(nrm,float); d=(mesh.vertices-point)@nrm; keep=list(mesh.vertices[d>=-1e-12])
    for a,b in mesh.edges_unique:
        da,db=d[a],d[b]
        if (da<0)!=(db<0) and abs(da-db)>1e-15:
            t=da/(da-db); keep.append(mesh.vertices[a]+t*(mesh.vertices[b]-mesh.vertices[a]))
    keep=np.array(keep)
    if len(keep)<4: return None
    try: h=trimesh.convex.convex_hull(keep)
    except Exception: return None
    return h if h.volume>1e-9 else None
out={}
for n,o in (('wooden_piece_1','wooden_piece_2'),('wooden_piece_2','wooden_piece_1')):
    c0,U=frame(n); oc0,oU=frame(o); r=rep0[n]; ro=rep0[o]
    # other board thickness across its crossing band (its long within +-1.5 cm of where THIS board crosses it = ro slot centre)
    pts=np.vstack([p.vertices for p in parts(o)]); q=(pts-oc0)@oU; band=np.abs(q[:,2]-ro['slot_centre_long_cm']/100)<0.015
    th=np.ptp(q[band,0]); tcen=0.5*(q[band,0].max()+q[band,0].min())
    width=th+CLEAR; sl=r['slot_centre_long_cm']/100; mc=0.5*(r['mid_cm'][0]+r['mid_cm'][1])/100
    lo_l,hi_l=sl-width/2,sl+width/2; open_neg=r['slot_open']=='-mid'
    # slot region in principal coords: long in [lo_l,hi_l], mid on the open side of mc (extended past the edge)
    newp=[]
    for p in parts(n):
        pq=trimesh.Trimesh((p.vertices-c0)@U,p.faces)
        l=pq.vertices[:,2]; m=pq.vertices[:,1]
        in_m=(m.max()>mc) if not open_neg else (m.min()<mc)
        if l.max()<=lo_l or l.min()>=hi_l or not in_m: newp.append(pq); continue
        # pieces: left of slot, right of slot, and the solid side of the middle band
        pieces=[]
        a=clip(pq,[0,0,-1],[0,0,lo_l]); pieces.append(a)
        b=clip(pq,[0,0,1],[0,0,hi_l]); pieces.append(b)
        mid=(lambda x: clip(x,[0,0,-1],[0,0,hi_l]) if x is not None else None)(clip(pq,[0,0,1],[0,0,lo_l]))
        keep=clip(mid,[0,1,0] if open_neg else [0,-1,0],[0,mc,0]) if mid is not None else None; pieces.append(keep)
        for x in pieces:
            if x is not None and len(x.faces) and x.volume>1e-9: newp.append(x.convex_hull)
    d=WOUT/'collision'/f'{n}_{TAG}'; d.mkdir(parents=True,exist_ok=True)
    for k,p in enumerate(newp):
        pw=trimesh.Trimesh(p.vertices@U.T+c0,p.faces); pw.export(d/f'part_{k:03d}.obj')
    out[n]=dict(parts_before=len(parts(n)),parts_after=len(newp),other_thickness_cm=float(th*100),slot_width_cm=float(width*100),slot_long_cm=[float(lo_l*100),float(hi_l*100)],slot_open=r['slot_open'],mid_centre_cm=float(mc*100))
json.dump(out,open(WOUT/'collision'/f'cut_report_{TAG}.json','w'),indent=1); print(json.dumps(out,indent=1))
