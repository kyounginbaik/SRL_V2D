"""Apply the cup opening-axis continuity fix to the eval cup tracks that feed the current submission, report flips,
and write recon_v15/episode_XXXXXX/object_poses.npz (only cup columns changed) where a flip is found."""
import sys, json, os, shutil, numpy as np, trimesh
sys.path.insert(0,os.path.dirname(__file__)); import cup_continuity as CC
from scipy.spatial.transform import Rotation as R
FR='/data1/SRL_V2D/challenge_reconstruction/final_recon'; R7='/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon'
V14='/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022'; OUT='/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022'
SRC={8:FR,10:FR,32:FR,33:FR,14:R7,20:R7}
report={}
for ep,root in SRC.items():
    d=f'{root}/episode_{ep:06d}'; z=np.load(d+'/object_poses.npz',allow_pickle=True); names=[str(x) for x in z['names']]; W=z['world_from_object'].copy(); changed=False
    for i,name in enumerate(names):
        if 'cup' not in name: continue
        mp=f'{d}/assets/{name}_visual.obj'; v=np.asarray(trimesh.load(mp,force='mesh').vertices)
        ax,_=CC.body_axis(v); oa=np.einsum('nij,j->ni',W[:,i,:3,:3],ax); jump=np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',oa[1:],oa[:-1]),-1,1)))
        Tf,path=CC.fix(W[:,i],v); up=np.degrees(np.arccos(np.clip(oa[:,2],-1,1)))
        r=dict(source=d, twin_frames=int(path.sum()), frames=len(path), max_frame_axis_jump_deg=float(jump.max()), opening_tilt_from_up_deg_p50_max=[float(np.median(up)),float(up.max())])
        if path.any():
            W[:,i]=Tf; changed=True; seg=np.flatnonzero(path); r['twin_span_s']=[float(seg[0]/20),float(seg[-1]/20)]
        report[f'ep{ep}:{name}']=r; print(ep,name,json.dumps(r))
    # also flag the v14 alternative for 32/33 blue cup
    if changed:
        o=f'{OUT}/episode_{ep:06d}'; os.makedirs(o,exist_ok=True); np.savez_compressed(o+'/object_poses.npz',**{k:(W if k=='world_from_object' else z[k]) for k in z.files})
json.dump(report,open(OUT+'/cup_continuity_report.json','w'),indent=1)
