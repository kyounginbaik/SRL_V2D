"""Baseline with the same physics budget as UniRetarget: the source reference stays fixed, only bounded low-level residuals are optimized."""
import argparse,json,sys,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
import yam_loop as Y
# FlashCHORD's residual hand-pose action limits (configs/action/residual_hand_pose.yaml).
BOUND=(.05,.15,.15);KNOT=10;T=None
class Lower:
    def __init__(s,episode,side,k):
        e=Path(episode);s.t=t=Y.YamTask(e,side,k);t.set_stage('real');n=mujoco.MjModel.from_binary_path(str(e/'native_model.mjb'));nd=mujoco.MjData(n);p=np.load(e/'plan.npz');J=p['joint_q'][0];U=p['joint_target'][0];s.F=F=t.frames-1;nq=n.nq-7*t.poses.shape[1];b=n.body(f'{side}_sharpa_wave_{side}_hand_C_MC').id;s.P=np.zeros((F,3));s.R=[]
        for f in range(F):
            nd.qpos[:nq]=J[f];mujoco.mj_kinematics(n,nd);s.P[f]=nd.xpos[b];s.R.append(R.from_matrix(nd.xmat[b].reshape(3,3).copy()))
        s.U=np.zeros((F,len(t.fingers)));s.lo=t.m.jnt_range[t.fingers,0];s.hi=t.m.jnt_range[t.fingers,1]
        for i,j in enumerate(t.fingers):
            nm=t.m.joint(j).name.split('_arm_',1)[1];q=int(n.jnt_qposadr[n.joint(f'{side}_sharpa_wave_{nm}').id]);s.U[:,i]=U[:F,q-(1 if side=='left' else 2)]
        s.knots=np.arange(0,F+KNOT,KNOT);s.size=(len(s.knots),6+len(t.fingers))
        s.bound=np.r_[[BOUND[0]]*3,[BOUND[1]]*3,[BOUND[2]]*len(t.fingers)]
    def tape(s,res):
        t=s.t;x=np.arange(s.F);r=np.column_stack([np.interp(x,s.knots,res[:,c]) for c in range(res.shape[1])]);C=np.tile(t.base_ctrl,(s.F,1));q=Y.READY.copy();Q=np.zeros((s.F,6));lim=Y.ARM_SPEED/Y.FPS*.9;t.p.qpos[:]=t.base_q
        for f in range(s.F):
            q,_,_=t.arm_ik(q,s.P[f]+r[f,:3],s.R[f]*R.from_rotvec(r[f,3:6]),steps=8,limit=lim);Q[f]=q
        t.last_Q=Q;C[:,t.act_arm]=Q+t.kd_over_kp*np.clip(np.gradient(Q,1/Y.FPS,axis=0),-Y.ARM_SPEED,Y.ARM_SPEED);C[:,t.act_f]=np.clip(s.U+r[:,6:],s.lo,s.hi);return C
def init(a):
    global T;T=Lower(*a)
def job(j):
    res,seed=j;rng=np.random.default_rng(seed);t=T.t;t.stage='real';t.randomize(rng);C=T.tape(res);r=t.rollout(C,T.F,rng.uniform(-.003,.003,3)*np.array([1,1,0]));r['sample']=dict(res=res);r['rung']='real';return r
def child(j):
    e,seed=j;rng=np.random.default_rng(seed);res=e['sample']['res'].copy();b=T.bound
    if e.get('dense_frames') and rng.random()<.3:
        # The same traced error signal the upper level gets, within the low level's limits.
        i=int(rng.integers(len(e['dense_frames'])));k=int(np.clip(round(e['dense_frames'][i]/KNOT),0,len(res)-1));res[k,:6]+=float(rng.uniform(.2,1.))*np.asarray(e['dense_error'][i])
    else:
        c=int(rng.integers(len(res)));w=int(rng.integers(1,5));g=np.exp(-.5*((np.arange(len(res))-c)/w)**2)[:,None];res+=g*rng.normal(0,.3,res.shape[1])*b
    return np.clip(res,-b,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--side',required=True);p.add_argument('--object',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--population',type=int,default=160);p.add_argument('--seed',type=int,default=7);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);log=(a.output/'generations.jsonl').open('a');rng=np.random.default_rng(a.seed);args=(a.episode,a.side,a.object);base=Lower(*args);c=a.seed*1000000+1;used=0;t0=time.monotonic()
    def ids(n):
        nonlocal c;r=range(c,c+n);c+=n;return r
    with Pool(a.workers,init,(args,)) as pool:
        zero=np.zeros(base.size);hist=pool.map(job,[(zero,int(i)) for i in ids(a.population)],chunksize=2);used=a.population;g=0
        while used<a.budget:
            hist.sort(key=Y.rank);hist=hist[:600];top=hist[:12];chk=pool.map(job,[(r['sample']['res'],int(i)) for r in top for i in ids(3)],chunksize=1);ver=[]
            for k,r in enumerate(top):
                w=min(chk[k*3:(k+1)*3],key=lambda x:(x['frontier'],x['lift_mm']));w['sample']=r['sample'];ver.append(w)
            hist=ver+hist[12:];hist.sort(key=Y.rank);b=hist[0];row=dict(generation=g,rollouts=used,seconds=round(time.monotonic()-t0),frontier=b['frontier'],hold=b['hold'],lift_mm=round(b['lift_mm'],1),cm=round(b['mean_cm'],2),deg=round(b['mean_deg'],1));print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
            el=hist[:max(6,len(hist)//20)];kids=pool.map(child,[(el[int(rng.integers(len(el)))],int(i)) for i in ids(a.population)],chunksize=4);hist+=pool.map(job,[(s,int(i)) for s,i in zip(kids,ids(len(kids)))],chunksize=2);used+=36+a.population;g+=1
        hist.sort(key=Y.rank);best=hist[0]['sample']['res'];np.save(a.output/'best_residual.npy',best);tr=pool.map(job,[(best,900000000+i) for i in range(96)],chunksize=1);H=base.F;ok=[t['frontier']>=H-1 and t['mean_cm']<100*Y.TOL_P and t['mean_deg']<np.degrees(Y.TOL_R) for t in tr]
    t=base.t;t.nominal_physics();t.stage='video';C=base.tape(best);r=t.rollout(C,H,np.zeros(3),record=True);X=r['qpos'];P=np.zeros((len(X),1,t.poses.shape[1],14))
    for k,qi in enumerate(t.object_q):P[:,0,k,:3]=X[:,qi:qi+3];P[:,0,k,3:7]=X[:,qi+3:qi+7][:,[1,2,3,0]];P[:,0,k,7:14]=t.poses[:len(X),k,7:14]
    np.savez_compressed(a.output/'physics.npz',native_qpos=X[:,None],object_pose=P,native_capture_world_ids=np.array([0]),fps=np.array(Y.FPS));print(json.dumps(dict(stage='finished',rollouts=used,held_out_pass=float(np.mean(ok)),trials=len(ok),held_out_frontier_median=float(np.median([x['frontier'] for x in tr])),held_out_lift_cm_median=round(float(np.median([x['lift_mm'] for x in tr]))/10,1),nominal_cm=round(r['mean_cm'],2),nominal_deg=round(r['mean_deg'],1),nominal_lift_cm=round(r['lift_mm']/10,1),nominal_hold=r['hold'])),flush=True)
if __name__=='__main__':main()
