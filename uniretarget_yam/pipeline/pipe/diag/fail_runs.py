"""Failed full-length episodes: the Mink reference kinematically, and the best grasp the search found replayed in MuJoCo with no early stop."""
import sys,json,glob
from pathlib import Path
sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,mujoco,pyarrow.parquet as pq,yam_loop as Y,yam_refine as F
from scipy.spatial.transform import Rotation as R
Y.ONSET_FLOOR=1.4;n=int(sys.argv[1]);LIVE=len(sys.argv)>2 and sys.argv[2]=='live';E=f'/uniretarget_yam/episodes/{"" if LIVE else ".fail_"}recon_{n:06d}';V=Path(f'/vid/ep{n}');V.mkdir(parents=True,exist_ok=True);M=json.load(open(E+'/recon_meta.json'));mv=max(M['movers'],key=lambda x:x['move_cm']);side,k=mv['side'],mv['index'];t=Y.YamTask(E,side,k);t.set_stage('real');t.nominal_physics();m=t.m;P0=t.poses[:,None].copy();K=P0.shape[2]
def save(name,Q,P):
    (V/name).mkdir(exist_ok=True);np.savez_compressed(V/name/'physics.npz',native_qpos=Q[:,None],object_pose=P[:,None],native_capture_world_ids=np.array([0]),fps=np.array(20.))
r=pq.read_table(E+'/yam/flash/mink/reference.parquet').to_pylist()[0];q=np.array(r['robot_joint_positions']);mp=F.mapping(m,r['robot_joint_names']);N=min(len(q),len(P0));Q=np.tile(t.base_q,(N,1));Q[:,mp]=q[:N];P=P0[:N,0].copy()
for j,a in enumerate(t.object_q):
    o=P[:,j];al=R.from_quat(o[0,3:7])*R.from_quat(o[0,10:14]).inv();gp=al.apply(o[:,7:10])+o[0,:3]-al.apply(o[0,7:10]);gq=(al*R.from_quat(o[:,10:14])).as_quat();Q[:,a:a+3]=gp;Q[:,a+3:a+7]=gq[:,[3,0,1,2]];P[:,j,:3]=gp;P[:,j,3:7]=gq
save('mink',Q,P);print('mink',N,flush=True)
tops=[]
for f in glob.glob(f'/search/{"" if LIVE else ".old3_"}recon_{n:06d}/s*/top.json'):tops.append((json.load(open(f))[0],f))
info=dict(n=n,side=side,obj=k,frames=t.frames,mink=True,attempt=False)
if tops:
    b,f=max(tops,key=lambda x:(x[0]['frontier'],x[0]['lift_mm']));C,_=t.compile(b['sample'],t.frames-1);real=t.rollout(C,t.frames-1,np.zeros(3));t.stage='show';s=t.rollout(C,t.frames-1,np.zeros(3),record=True);Qx=s['qpos'];N=len(Qx);P=P0[:N,0].copy()
    for j,a in enumerate(t.object_q):P[:,j,:3]=Qx[:,a:a+3];P[:,j,3:7]=Qx[:,a+3:a+7][:,[1,2,3,0]]
    save('attempt',Qx,P);info.update(attempt=True,tape=f,frontier=real['frontier'],cause=real['cause'],grip_lost=real['grip_lost'],lift_mm=round(s['lift_mm'],1),edits=len(b['sample'].get('edits',())))
# Later stage of the contact schedule: its best attempt with the earlier hands' saved motion playing.
cf=Path(f'/search/recon_{n:06d}/stage_current.json')
if LIVE and cf.exists():
    st=json.loads(cf.read_text());B2=glob.glob(f"/search/recon_{n:06d}/stage{st['stage']}_s*/top.json")
    if B2:
        Y.JOIN=json.load(open(st['join']));h=M['holds'][st['stage']-1];base=np.load(st['base_ctrl'])['ctrl'];t2=Y.YamTask(E,h['side'],h['index']);t2.set_stage('real');t2.nominal_physics();b=max((json.load(open(f))[0] for f in B2),key=lambda x:(x['frontier'],x['lift_mm']));C,_=t2.compile(b['sample'],t2.frames-1,base);t2.stage='show';s=t2.rollout(C,t2.frames-1,np.zeros(3),record=True);Qx=s['qpos'];N=len(Qx);P=P0[:N,0].copy()
        for j,a_ in enumerate(t2.object_q):P[:,j,:3]=Qx[:,a_:a_+3];P[:,j,3:7]=Qx[:,a_+3:a_+7][:,[1,2,3,0]]
        save('attempt',Qx,P);info.update(attempt=True,obj=h['index'],tape=f"stage {st['stage']}",frontier=b['frontier'],cause=b['cause'],lift_mm=round(s['lift_mm'],1))
(V/'info.json').write_text(json.dumps(info));print(json.dumps(info))
