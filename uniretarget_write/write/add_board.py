"""Fit the writing plane from the mocap pen tip, express it in SPIDER's scene frame, and add one static box to scene.xml and scene_eq.xml, with explicit contact pairs against the pen and hand colliders.
The surface sits where the pen's collision mesh touches at the writing height, so the reference pen does not start inside the board."""
import numpy as np, json, re, sys, trimesh, pickle, xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R
H=Path('/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write'); DS=Path('/scratch2/exaFLOPs/spider_wide/ds/processed/oakinkv2'); T='write_wb'; P=DS/f'sharpa/right/{T}'
RT='/lustre/exaflops/retarget_bench/data/oakink2_root'; TOK='scene_02__A002++seq__e9aab4c73ce9d2c16b85__2023-04-22-09-37-57'; PEN='O02@0029@00011'
a=pickle.load(open(f'{RT}/anno_preview/{TOK}.pkl','rb')); ti=json.load(open(DS/f'mano/right/{T}/task_info.json')); print('window',ti['oakink_window'],'n',ti['n_frames'])
d=np.load(H/'tip.npz'); fs=d['fs']; bi=int(d['thin']); face=float(d['face'][1]); h=d['local_plus'][:,bi]-face
wr=(h<-0.002)  # pen on the board: the lifts at +0.3 cm and the approach are out
print('on-board frames',wr.sum(),'of',len(h))
# Rigid map raw mocap frame to SPIDER's scene frame, from the processed pen trajectory.
K=np.load(DS/f'mano/right/{T}/0/trajectory_keypoints.npz'); oc=K['qpos_obj_right']; idx=np.linspace(0,len(fs)-1,len(oc)).round().astype(int)
Tr=np.stack([np.asarray(a['obj_transf'][PEN][int(fs[i])]) for i in idx])
Rs=R.from_quat(oc[:,3:][:,[1,2,3,0]]).as_matrix(); A=Rs@Tr[:,:3,:3].transpose(0,2,1); U,_,Vt=np.linalg.svd(A.mean(0)); Rg=U@Vt; tg=(oc[:,:3]-(Rg@Tr[:,:3,3,None])[...,0]).mean(0)
res=np.linalg.norm((Rg@Tr[:,:3,3,None])[...,0]+tg-oc[:,:3],axis=1).max(); print('map R',np.round(Rg,3).tolist(),'t',tg.round(4),'residual cm',res*100); assert res<.002
pen=trimesh.load(f'{RT}/object_repair/align_ds/{PEN}/model.obj',force='mesh'); v=np.asarray(pen.vertices); ax=np.linalg.svd(v-v.mean(0))[2][0]; s=(v-v.mean(0))@ax; tip_l=v[s>s.max()-.003].mean(0)
TF=np.stack([np.asarray(a['obj_transf'][PEN][int(f)]) for f in fs]); tip=(Rg@((TF[:,:3,:3]@tip_l)+TF[:,:3,3])[...,None])[...,0]+tg
c=tip[wr].mean(0); n=np.linalg.svd(tip[wr]-c)[2][2]; off=(tip[wr]-c)@n; print('plane fit residual cm std',off.std()*100,'max',np.abs(off).max()*100)
# normal points to where the pen comes from
if ((tip[~wr]-c)@n).mean()<0: n=-n
ang=np.degrees(np.arccos(abs(n[2]))); print('normal',n.round(3),'tilt from horizontal plane deg',round(ang,1))
# Offset from the pen's lowest collider point: use the visual mesh's furthest vertex along the pen axis in each on-board frame.
Vw=np.einsum('tij,nj->tni',TF[wr][::10,:3,:3],v)+TF[wr][::10,None,:3,3]; Vs=Vw@Rg.T+tg; low=((Vs-c)@n).min(1); print('pen mesh below tip plane cm: median',np.median(low)*100,'p5',np.percentile(low,5)*100,'p95',np.percentile(low,95)*100)
surf=c+n*np.median(low)
th=.02; half=[.3,.3,th/2]; ez=n; ex=np.cross([0,0,1.],ez); ex=ex/np.linalg.norm(ex) if np.linalg.norm(ex)>1e-6 else np.array([1.,0,0]); ey=np.cross(ez,ex); q=R.from_matrix(np.c_[ex,ey,ez]).as_quat()[[3,0,1,2]]; pos=surf-n*th/2
json.dump(dict(pos=pos.tolist(),quat=q.tolist(),normal=n.tolist(),surface_point=surf.tolist(),half=half,tip_local=tip_l.tolist(),map_R=Rg.tolist(),map_t=tg.tolist(),tilt_deg=ang),open(H/'board.json','w'),indent=1)
# Frame-level reference for scoring: tip in scene frame at 50 Hz, height above surface, writing mask.
t50=tip[idx]; hh=(t50-surf)@n; onb=wr[idx]; np.savez(H/'tip_ref.npz',tip=t50,height=hh,on_board=onb); print('50 Hz frames',len(idx),'on board',onb.sum(),'height while on board cm',np.round(np.percentile(hh[onb],[5,50,95])*100,2))
for name in ('scene.xml','scene_eq.xml'):
    p=P/name; s=p.read_text()
    if 'name="board"' in s: print('already',name); continue
    Path(str(p)+'.noboard').write_text(s)
    r=ET.fromstring(s); g=[e.get('name') for e in r.iter('geom') if (e.get('name') or '').startswith(('right_object_','collision_hand_')) and e.get('name')!='right_object_visual']
    geom=f'    <geom name="board" type="box" size="{half[0]} {half[1]} {half[2]}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" rgba="0.92 0.92 0.88 1"/>\n'
    s=s.replace('<geom name="floor"',geom.strip()+'\n    <geom name="floor"',1)
    pairs=''.join(f'    <pair geom1="board" geom2="{x}" name="board_{x}" friction="{".3 .3 .1 0 0" if x.startswith("right_object") else "1 1 .1 0 0"}"/>\n' for x in g)
    assert '</contact>' in s; s=s.replace('</contact>',pairs+'  </contact>',1); p.write_text(s); print(name,'pairs',len(g))
