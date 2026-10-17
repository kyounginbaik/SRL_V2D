"""Physical margin search preserving accurate tracking and fully clear release.

This strengthens the search objective, never the reported MP-SR definition.
Objects remain free and every proposal uses the same scene and original clock.
"""
import numpy as np
import mujoco
from search_repaired_hand_margins_evaluator import Search as BaseSearch
from refine_grasp_cem import angular_errors
from scipy.spatial.transform import Rotation

class Search(BaseSearch):
    def __init__(self,spec):
        super().__init__(spec)
        d=self.policy.data;self.base_weights=d['phase_weights'].copy()
        self.wrist_ids=np.array([self.m.actuator('robot_right_right_'+name+'_position').id for name in ['pos_x','pos_y','pos_z']])
        times=d['centers']*self.policy.duration
        def smooth(x):
            x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))
        self.envelopes=[]
        for a,b,c,e in [(2.45,2.8,4.8,5.4),(12.65,13.,15.45,16.3)]:
            self.envelopes.append(smooth((times-a)/(b-a))*(1-smooth((times-c)/(e-c))))
        state=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,state,0)
        p,q=self.metric.object_poses(state.qpos[None]);self.align_R=(Rotation.from_quat(self.metric.tq[0,0],scalar_first=True)*Rotation.from_quat(q[0,0],scalar_first=True).inv()).as_matrix()
        self.align_p=self.metric.tp[0,0]-self.align_R@p[0,0]
    def set_parameters(self,x):
        x=np.asarray(x).reshape(2,3);d=self.policy.data;w=self.base_weights.copy()
        for env,delta in zip(self.envelopes,x):w[:,self.wrist_ids]+=env[:,None]*delta[None]/d['action_scale'][self.wrist_ids]
        d['phase_weights']=w
    def evaluate(self,x):
        self.set_parameters(x)
        d=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,d,0);mujoco.mj_forward(self.m,d)
        Q=np.empty_like(self.metric.base);Q[0]=d.qpos;limit=0.
        pot=self.m.body('white_pot').id;lid=self.m.body('white_pot_lid').id;table=self.m.body('table').id
        robot={b for b in range(self.m.nbody) if (self.m.body(b).name or '').startswith('robot_')}
        hand_force=0.;pot_support=np.inf;lid_support=np.inf;object_speed=0.;f=np.zeros(6)
        for k in range(1,len(Q)):
            for _ in range(self.steps):
                d.ctrl[:]=self.policy(d.time+self.dt,d.qpos,d.qvel);mujoco.mj_step(self.m,d)
                limit=max(limit,float(np.max(np.maximum(self.frange[:,0]-d.qpos[self.fq],d.qpos[self.fq]-self.frange[:,1]))))
                if d.time>=16.3:
                    hf=ps=ls=0.
                    for ci,contact in enumerate(d.contact):
                        a,b=(int(self.m.geom_bodyid[gid]) for gid in contact.geom)
                        if not ({a,b}&{pot,lid}):continue
                        mujoco.mj_contactForce(self.m,d,ci,f);force=max(0.,float(f[0]))
                        if (a in robot and b in {pot,lid}) or (b in robot and a in {pot,lid}):hf+=force
                        if {a,b}=={pot,table}:ps+=force
                        if {a,b}=={pot,lid}:ls+=force
                    hand_force=max(hand_force,hf);pot_support=min(pot_support,ps);lid_support=min(lid_support,ls)
                    for body in [pot,lid]:
                        adr=int(self.m.jnt_dofadr[self.m.body_jntadr[body]])
                        object_speed=max(object_speed,float(np.linalg.norm(d.qvel[adr:adr+3])))
            if not np.isfinite(d.qpos).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:return dict(cost=1e6,unstable=True)
            Q[k]=d.qpos
        p,q=self.metric.object_poses(Q);ep=np.linalg.norm(p-self.metric.tp,axis=-1)
        er=angular_errors(q[:,None],self.metric.tq[:,None],self.metric.axes,self.metric.kinds)[:,0]
        pos=ep.mean(0);ang=er.mean(0);ratio=np.maximum(pos/.03,ang/np.deg2rad(30));ld=float(np.degrees(limit))
        pa=p@self.align_R.T+self.align_p
        qa=Rotation.from_matrix(self.align_R@Rotation.from_quat(q.reshape(-1,4),scalar_first=True).as_matrix()).as_quat(scalar_first=True).reshape(q.shape)
        aligned_pos=np.linalg.norm(pa-self.metric.tp,axis=-1).mean(0)
        aligned_ang=angular_errors(qa[:,None],self.metric.tq[:,None],self.metric.axes,self.metric.kinds)[:,0].mean(0)
        aligned_ratio=np.maximum(aligned_pos/.03,aligned_ang/np.deg2rad(30))
        tracking=max(pos[0]/.016,pos[1]/.015,float(ratio.max()))
        support_ratio=max(hand_force/.02,.5/max(pot_support,1e-9),.1/max(lid_support,1e-9),object_speed/.02)
        preserved=bool(tracking<=1. and support_ratio<=1.)
        excess=max(0.,ld/1.8-1.,tracking-1.,min(support_ratio,20.)-1.)
        cost=float(aligned_ratio.max()+.15*aligned_ratio.mean()+100*excess+.002*ld)
        return dict(cost=cost,mean_position_cm=(100*pos).tolist(),mean_orientation_deg=np.degrees(ang).tolist(),
            original_aligned_mean_position_cm=(100*aligned_pos).tolist(),original_aligned_mean_orientation_deg=np.degrees(aligned_ang).tolist(),
            mp_sr=float(np.all(aligned_ratio<1)),declared_frame_mp_sr=float(np.all(ratio<1)),all_finger_limit_max_degrees=ld,
            preservation_pass=preserved,quality_preserves_baseline=bool(ld<=1.5),quality_requires_review=bool(1.5<ld<=1.8),
            max_final_hand_object_force_n=hand_force,min_final_pot_table_force_n=pot_support,
            min_final_lid_pot_force_n=lid_support,max_final_object_speed_m_s=object_speed,unstable=False)

WORK=None
def init(spec):
    global WORK
    WORK=Search(spec)
def evaluate(x):return WORK.evaluate(x)
