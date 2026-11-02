"""Fast diagnostic: can a placed grasp carry the real object goal in plain MuJoCo?

The hand starts ON the fitted pose, so this separates holding from reaching and
is never task evidence. Same exported model, native PD and ball-drive law; the
canonical Warp evaluator remains the judge of anything that survives here.
"""
import argparse,json,glob,time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from flash_chord.retargeting.native_inverse_dynamics import NativeScalarPDInverse
from flash_chord.retargeting.contact_wrench import ContactPatch
from compile_grasp_execution import minimum_jerk,balanced_forces,goal_motion_onset,reverse_closure

WRIST_KP,WRIST_TAU=100.,90.;FPS=20.;SUBSTEPS=20


class Lab:
    def __init__(self,model_path,goal_path,rest):
        self.m=m=mujoco.MjModel.from_binary_path(str(model_path));m.opt.timestep=1/(FPS*SUBSTEPS);self.d=mujoco.MjData(m)
        free=[j for j in range(m.njnt) if m.jnt_type[j]==0][0];self.obj=int(m.jnt_bodyid[free]);self.qo=int(m.jnt_qposadr[free]);self.vo=int(m.jnt_dofadr[free])
        self.up=-m.opt.gravity/np.linalg.norm(m.opt.gravity);self.mg=float(m.body_mass[self.obj]*np.linalg.norm(m.opt.gravity))
        names=[m.body(m.jnt_bodyid[j]).name for j in range(m.njnt)]
        self.joints=[j for j,n in enumerate(names) if n.startswith('right_')]
        self.slides=[j for j in self.joints if m.jnt_type[j]==2];self.ball=[j for j in self.joints if m.jnt_type[j]==1][0];self.fingers=[j for j in self.joints if m.jnt_type[j]==3]
        self.lball=[j for j,n in enumerate(names) if n.startswith('left_') and m.jnt_type[j]==1][0]
        self.body=int(m.jnt_bodyid[self.ball])
        self.pos={};self.vel=[]
        for a in range(m.nu):
            j=int(m.actuator_trnid[a,0])
            if m.actuator_gainprm[a,0]>0 and abs(m.actuator_biasprm[a,1]+m.actuator_gainprm[a,0])<1e-5:self.pos[j]=a
        self.inverse=NativeScalarPDInverse(m,ball_drives={self.ball:(WRIST_KP,WRIST_TAU)})
        with np.load(goal_path) as z:pose=z['object_pose'][:,0,0].copy()
        init=pose[0];al=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();off=init[:3]-al.apply(init[7:10])
        gp=al.apply(pose[:,7:10])+off;gr=al*R.from_quat(pose[:,10:14]);q=(gr[rest].inv()*gr).as_quat()
        for i in range(1,len(q)):
            if q[i]@q[i-1]<0:q[i]=-q[i]
        self.gp_raw,self.gr_raw=gp,gr
        self.gp=gaussian_filter1d(gp,1.5,axis=0,mode='nearest');q=gaussian_filter1d(q,1.5,axis=0,mode='nearest')
        self.gr=gr[rest]*R.from_quat(q/np.linalg.norm(q,axis=1,keepdims=True));self.rest=rest;self.onset=goal_motion_onset(gp,gr,rest)

    def ball_torque(self,joint,target_rotvec):
        m,d=self.m,self.d;qi=int(m.jnt_qposadr[joint]);di=int(m.jnt_dofadr[joint])
        err=(R.from_quat(d.qpos[qi:qi+4][[1,2,3,0]]).inv()*R.from_rotvec(target_rotvec)).as_rotvec()
        d.qfrc_applied[di:di+3]=np.clip(WRIST_KP*err,-WRIST_TAU,WRIST_TAU)

    def run(self,star,points,normals,geoms,grip,horizon=86,settle=.1,squeeze=.4,overclose=0.):
        m,d=self.m,self.d;mujoco.mj_resetData(m,d);d.qpos[:]=star;d.qvel[:]=0;mujoco.mj_forward(m,d)
        wp0=d.xpos[self.body].copy();wr0=R.from_matrix(d.xmat[self.body].reshape(3,3).copy());slide_q=[int(m.jnt_qposadr[j]) for j in self.slides];ball_q=int(m.jnt_qposadr[self.ball])
        fixed=wr0*R.from_quat(star[ball_q:ball_q+4][[1,2,3,0]]).inv();op=star[self.qo:self.qo+3].copy();orr=R.from_quat(star[self.qo+3:self.qo+7][[1,2,3,0]])
        pre=int(round((settle+squeeze)*FPS));frames=pre+horizon;jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);lball_target=R.from_quat(star[int(m.jnt_qposadr[self.lball]):int(m.jnt_qposadr[self.lball])+4][[1,2,3,0]]).as_rotvec()
        for j,a in self.pos.items():
            if j not in self.joints:d.ctrl[a]=star[int(m.jnt_qposadr[j])]
        g0=self.onset;errors=[];rel0=None;contact_frames=0;lift=0.;n_pts=len(points)
        geometry=np.tile(star,(frames+1,1))
        ps=np.zeros((frames+1,3));rs=[None]*(frames+1)
        for f in range(frames+1):
            k=min(g0+max(0,f-pre),len(self.gp)-1);w=minimum_jerk(max(0,f-pre)/4.)
            rel=R.from_rotvec(w*(self.gr[k]*orr.inv()).as_rotvec());ps[f]=op+w*(self.gp[k]-op)+rel.apply(wp0-op);rs[f]=rel*wr0
            geometry[f,slide_q]=ps[f]-(wp0-star[slide_q]);geometry[f,ball_q:ball_q+4]=(fixed.inv()*rs[f]).as_quat()[[3,0,1,2]]
        vel=np.zeros((frames+1,m.nv))
        for f in range(frames+1):
            a,b=max(0,f-1),min(frames,f+1);mujoco.mj_differentiatePos(m,vel[f],(b-a)/FPS,geometry[a],geometry[b])
        vel[:,self.vo:self.vo+6]=0;acc=np.gradient(vel,1/FPS,axis=0)
        scratch=mujoco.MjData(m)
        # Closing targets continue past the fit, so a finger crosses its residual gap.
        finger_q=[int(m.jnt_qposadr[j]) for j in self.fingers];close=np.zeros(len(finger_q))
        if overclose>0:close=reverse_closure(m,scratch,star,m.geom_bodyid[geoms],points,normals,self.fingers,-overclose)[0]-star[finger_q]
        for f in range(frames):
            weight=minimum_jerk((f-settle*FPS)/max(1,squeeze*FPS));load=np.zeros(m.nv)
            if weight>0:
                scratch.qpos[:]=geometry[f];mujoco.mj_kinematics(m,scratch);mujoco.mj_comPos(m,scratch)
                rel=rs[f]*wr0.inv();P=ps[f]+rel.apply(points-wp0);N=rel.apply(normals);patches=[]
                for i,(pp,nn) in enumerate(zip(P,N)):
                    ax=np.eye(3)[np.argmin(np.abs(nn))];t1=np.cross(nn,ax);t1/=np.linalg.norm(t1);patches.append(ContactPatch(tuple(pp),tuple(nn),tuple(t1),tuple(np.cross(nn,t1)),(.7,.7),i))
                F,_=balanced_forces(patches,N,np.r_[self.mg*self.up,np.zeros(3)],grip*self.mg*weight,8*self.mg*max(1.,grip),P.mean(0))
                for geom,pp,ff in zip(geoms,P,F):
                    mujoco.mj_jac(m,scratch,jp,jr,pp,int(m.geom_bodyid[geom]));load+=jp.T@ff
            out=self.inverse.targets(geometry[f],vel[f],acc[f],contact_load=load)
            for j,v in out.native_joint_position_targets.items():
                if j in self.joints:d.ctrl[self.pos[j]]=v+(weight*close[self.fingers.index(j)] if j in self.fingers else 0.)
            target=out.native_ball_rotvec_targets[self.ball]
            for _ in range(SUBSTEPS):
                d.qfrc_applied[:]=0;self.ball_torque(self.ball,target);self.ball_torque(self.lball,lball_target);mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all():return dict(exploded=True,frame=f)
            if f>=pre:
                k=min(g0+f-pre,len(self.gp_raw)-1);o=d.qpos[self.qo:self.qo+3];ob=R.from_quat(d.qpos[self.qo+3:self.qo+7][[1,2,3,0]])
                errors.append((float(np.linalg.norm(o-self.gp_raw[k])),float((ob.inv()*self.gr_raw[k]).magnitude())))
                lift=max(lift,float((o-op)@self.up))
                touching=sum(1 for c in d.contact[:d.ncon] if self.obj in m.geom_bodyid[c.geom] and any(m.geom_bodyid[g] in set(int(m.jnt_bodyid[j]) for j in self.joints) for g in c.geom))
                contact_frames+=touching>0
        e=np.asarray(errors);good=(e[:,0]<.03)&(e[:,1]<np.radians(30));held=np.flatnonzero(~good)
        return dict(mean_cm=float(100*e[:,0].mean()),mean_deg=float(np.degrees(e[:,1].mean())),first_break=int(held[0]) if len(held) else -1,within=float(good.mean()),lift_mm=lift*1e3,contact_frames=int(contact_frames))


_lab=None
def _init(a):
    global _lab;_lab=Lab(*a)
def _job(job):
    name,star,P,N,G,grip=job;t=time.monotonic();r=_lab.run(star,P,N,G,grip);r.update(name=name,grip=grip,seconds=round(time.monotonic()-t,1));return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--goal',type=Path,required=True);p.add_argument('--grip',type=float,action='append');p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path)
    a=p.parse_args();rest=json.loads((a.fit/'result.json').read_text())['rest_frame'];jobs=[]
    for f in sorted(glob.glob(str(a.fit/'candidate_*.npz'))):
        z=np.load(f)
        for g in a.grip or [1.,3.,8.]:jobs.append((Path(f).stem,z['qpos'].copy(),z['patch_positions'].copy(),z['patch_normals'].copy(),z['hand_geom_ids'].copy(),g))
    with Pool(a.workers,_init,((a.fit/'native_model.mjb',a.goal/'physics.npz',rest),)) as pool:
        rows=[]
        for r in pool.imap(_job,jobs):rows.append(r);print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()}),flush=True)
    if a.output:a.output.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
