"""Physical margin search preserving accurate tracking and fully clear release.

This strengthens the search objective, never the reported MP-SR definition.
Objects remain free and every proposal uses the same scene and original clock.
"""
import numpy as np
import mujoco
from search_repaired_hand_margins_evaluator import Search as BaseSearch
from refine_grasp_cem import angular_errors

class Search(BaseSearch):
    def evaluate(self,x):
        self.policy.data['margin_bias_rad']=np.asarray(x)
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
        # Preserve the accurate baseline within an explicit search budget:
        # pot <=1.6cm, lid <=1.5cm, against unchanged original targets.
        tracking=max(pos[0]/.016,pos[1]/.015,float(ratio.max()))
        support_ratio=max(hand_force/.02,.5/max(pot_support,1e-9),.1/max(lid_support,1e-9),object_speed/.02)
        preserved=tracking<=1. and support_ratio<=1.
        excess=max(0.,ld/self.spec['tolerance_deg']-1.,tracking-1.,min(support_ratio,20.)-1.)
        cost=float(ratio.max()+.25*ratio.mean()+100*excess)
        return dict(cost=cost,mean_position_cm=(100*pos).tolist(),mean_orientation_deg=np.degrees(ang).tolist(),
            mp_sr=float(np.all(ratio<1)),all_finger_limit_max_degrees=ld,quality_pass=ld<=self.spec['tolerance_deg'] and preserved,
            preservation_pass=preserved,max_final_hand_object_force_n=hand_force,min_final_pot_table_force_n=pot_support,
            min_final_lid_pot_force_n=lid_support,max_final_object_speed_m_s=object_speed,unstable=False)

WORK=None
def init(spec):
    global WORK
    WORK=Search(spec)
def evaluate(x):return WORK.evaluate(x)
