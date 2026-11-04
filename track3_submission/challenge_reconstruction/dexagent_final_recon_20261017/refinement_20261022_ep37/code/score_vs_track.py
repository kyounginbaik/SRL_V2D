"""Score a rollout parquet (achieved poses) against an object_poses.npz track with the unchanged kit scorers.
Reference geometry = our own submitted mesh (identity registration), i.e. the local proxy, but with any reference track.
usage: score_vs_track.py ROLLOUT_PARQUET TRACK_NPZ MESH_OBJ [MESH_OBJ...]"""
import sys, io, base64, json, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, trimesh
from scipy.spatial.transform import Rotation as Rot
KIT='/data1/SRL_V2D/challenge_submissions_SRL/track3_v13_20261022/all_20_episodes/submission_kit'; sys.path.insert(0,KIT)
from v2dlb.mesh_budget import budget_mesh
def load(n):
    s=importlib.util.spec_from_file_location('k'+n.replace('-','_'),f'{KIT}/metric_code/track_3/{n}.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
mods={n:load(n) for n in ['MP-SR','SP-SR','AUC','MPPE']}; M=mods['MP-SR']
SYM={'beige_cup':360,'blue_cup':360,'white_pot_lid':2}
def enc(**a): b=io.BytesIO(); np.savez_compressed(b,**a); return base64.b64encode(b.getvalue()).decode()
def axis_of(v):
    c=v-v.mean(0); w,U=np.linalg.eigh(c.T@c/len(c)); g=[abs(w[i]-w[(i+1)%3])+abs(w[i]-w[(i+2)%3])-abs(w[(i+1)%3]-w[(i+2)%3]) for i in range(3)]; a=U[:,int(np.argmax(g))]; return a/np.linalg.norm(a)
roll, track, meshes = sys.argv[1], sys.argv[2], sys.argv[3:]
d=pd.read_parquet(roll); z=np.load(track); names=[str(x) for x in z['names']]; T=z['world_from_object']; ep=int(d.get('episode_index',pd.Series([37])).iloc[0]) if 'episode_index' in d else 37
geo=dict(names=np.array(names)); sol=[]; sub=[]; PROV=dict(eval_script_sha256='0'*64,eval_code_sha256='0'*64,checkpoint_sha256='0'*64,packer_script_sha256='0'*64,uploader_script_sha256='0'*64,code_commit_url='https://github.com/x/y/commit/'+'0'*40)
for slot,name in enumerate(names):
    mp=[m for m in meshes if m.endswith(name+'.obj')][0]; mv,mf=budget_mesh(mp,4096,4096); v=np.asarray(mv,float); f=np.asarray(mf,np.int64)
    anc,_=M._surface_moments(v,f); geo.update({f'v{slot}':v,f'f{slot}':f,f'anchor{slot}':anc,f'points{slot}':M._sample_mesh_surface(v,f,10000,0),f'axis{slot}':axis_of(v) if name in SYM else np.zeros(3),f'order{slot}':np.int64(SYM.get(name,1))})
    x=d[d.object_body_name==name].sort_values('step'); q=Rot.from_matrix(T[x.step.to_numpy(),slot,:3,:3]).as_quat(); p=T[x.step.to_numpy(),slot,:3,3]
    for i,(k,r) in enumerate(x.iterrows()):
        rid=f'track_3/e{ep:06d}/w000/f{int(r.step):06d}/o{slot}'
        sol.append(dict(row_id=rid,episode_index=ep,world_index=0,frame_index=int(r.step),object_slot=slot,object_name=name,ref_pos_x=p[i,0],ref_pos_y=p[i,1],ref_pos_z=p[i,2],ref_quat_x=q[i,0],ref_quat_y=q[i,1],ref_quat_z=q[i,2],ref_quat_w=q[i,3],reference_geometry_b64=''))
        sub.append(dict(row_id=rid,pos_x=r.achieved_position_x,pos_y=r.achieved_position_y,pos_z=r.achieved_position_z,quat_x=r.achieved_quaternion_x,quat_y=r.achieved_quaternion_y,quat_z=r.achieved_quaternion_z,quat_w=r.achieved_quaternion_w,**PROV,object_mesh_b64=enc(vertices=v,faces=f.astype(float)) if i==0 else ''))
sol=pd.DataFrame(sol); sub=pd.DataFrame(sub); sol.loc[0,'reference_geometry_b64']=enc(**geo)
sc={k:m.score(sol.copy(),sub.copy(),'row_id') for k,m in mods.items()}
sys.path.insert(0,'/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts'); import public_kit_score as K
err=K.aligned_errors(M,sol,sub)
print(json.dumps(dict(scores={k:round(float(v),4) for k,v in sc.items()}, objects={k:(round(a,2),round(b,1)) for k,(a,b) in err.items()})))
