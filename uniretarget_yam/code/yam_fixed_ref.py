"""Baseline: the source reference held fixed on YAM. Arms follow its palms by IK under the speed limit; vendor PD tracks it; nothing refines it."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as Y
p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--side',default='right');p.add_argument('--output',type=Path,required=True);p.add_argument('--trials',type=int,default=48);a=p.parse_args();e=Path(a.episode)
t=Y.YamTask(e,a.side,a.object);t.set_stage('real');m=t.m;n=mujoco.MjModel.from_binary_path(str(e/'native_model.mjb'));nd=mujoco.MjData(n);plan=np.load(e/'plan.npz');J=plan['joint_q'][0];T=plan['joint_target'][0];F=min(t.frames-1,len(J))
# Native layout per side: x y z, ball quaternion, fingers in q; x y z, rotvec, fingers in target.
nq=n.nq-7*t.poses.shape[1];C=np.tile(t.base_ctrl,(F,1));miss={};start=t.base_q.copy()
# Only the hand that moves this object follows the reference; the other arm rests, so it cannot be blamed for a failure.
for side,task in ((a.side,t),):
    b=n.body(f'{side}_sharpa_wave_{side}_hand_C_MC').id;P=[];R_=[]
    for f in range(F):
        nd.qpos[:nq]=J[f];mujoco.mj_kinematics(n,nd);P.append(nd.xpos[b].copy());R_.append(R.from_matrix(nd.xmat[b].reshape(3,3).copy()))
    # The arm starts on the reference's first pose: no reach is charged against the baseline.
    # Same start as UniRetarget, the ready pose; physics decides any contact on the way.
    q=Y.READY.copy();QA=np.zeros((F,6));er=[];lim=Y.ARM_SPEED/Y.FPS*.9
    for f in range(F):
        q,e1,e2=task.arm_ik(q,P[f],R_[f],steps=8,limit=lim);QA[f]=q;er.append((e1,e2))
    V=np.gradient(QA,1/Y.FPS,axis=0);C[:,task.act_arm]=QA+task.kd_over_kp*np.clip(V,-Y.ARM_SPEED,Y.ARM_SPEED);miss[side]=np.array(er)
    for j,act in zip(task.fingers,task.act_f):
        nm=m.joint(j).name.split('_arm_',1)[1];nj=n.joint(f'{side}_sharpa_wave_{nm}').id;i=int(n.jnt_qposadr[nj]);ti=i-(1 if side=='left' else 2);C[:,act]=np.clip(T[:F,ti],*m.jnt_range[j])
crashed=[]
def run(rng=None):
    if rng is None:t.nominal_physics();jit=np.zeros(3)
    else:t.randomize(rng);jit=rng.uniform(-.003,.003,3)*np.array([1,1,0])
    d=t.d;mujoco.mj_resetData(m,d);d.qpos[:]=start;d.qpos[t.qo:t.qo+7]=t.reset_pose;d.qpos[t.qo:t.qo+3]+=jit;mujoco.mj_forward(m,d);X=[]
    for f in range(F):
        d.ctrl[:]=C[f]
        try:
            for _ in range(getattr(t,'substeps',Y.SUBSTEPS)):mujoco.mj_step(m,d)
        except mujoco.FatalError:  # The simulator gave up: a failed run, holding its last state for the rest of the episode.
            X+=[X[-1] if X else d.qpos.copy()]*(F-len(X));crashed.append(f);break
        X.append(d.qpos.copy())
    return np.array(X)
def score(X):
    ep=np.linalg.norm(X[:,t.qo:t.qo+3]-t.gp_raw[:F],axis=1);er=(R.from_quat(X[:,t.qo+3:t.qo+7][:,[1,2,3,0]])*t.gr_raw[:F].inv()).magnitude();return ep,er
tr=[score(run(np.random.default_rng(900000000+i))) for i in range(a.trials)];ok=[100*x.mean()<100*Y.TOL_P and y.mean()<Y.TOL_R for x,y in tr]
X=run();ep,er=score(X);P=np.zeros((F,1,t.poses.shape[1],14))
for k,qi in enumerate(t.object_q):P[:,0,k,:3]=X[:,qi:qi+3];P[:,0,k,3:7]=X[:,qi+3:qi+7][:,[1,2,3,0]];P[:,0,k,7:14]=t.poses[:F,k,7:14]
a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'physics.npz',native_qpos=X[:,None],object_pose=P,native_capture_world_ids=np.array([0]),fps=np.array(Y.FPS))
print(json.dumps(dict(crashed_runs=len(crashed),first_crash_s=[round(c/Y.FPS,2) for c in crashed[:5]],pass_rate=float(np.mean(ok)),trials=a.trials,mean_cm_across=round(float(np.mean([100*x.mean() for x,_ in tr])),2),mean_deg_across=round(float(np.mean([np.degrees(y.mean()) for _,y in tr])),1),nominal_cm=round(100*float(ep.mean()),2),nominal_deg=round(float(np.degrees(er.mean())),1),worst_cm=round(100*float(ep.max()),1),spoon_lift_cm=round(100*float(((X[:,t.qo:t.qo+3]-t.o0)@t.up).max()),1),goal_lift_cm=round(100*float(((t.gp_raw[:F]-t.o0)@t.up).max()),1),ik_miss={k:dict(cm_mean=round(100*float(v[:,0].mean()),1),cm_max=round(100*float(v[:,0].max()),1),deg_mean=round(float(np.degrees(v[:,1].mean())),1)) for k,v in miss.items()})))
