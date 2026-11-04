"""Refine actual receiver contact geometry before releasing the donor.

Version4 reaches frequent two-digit contact but no force-closure geometry.
This version adds a continuous normal/friction-cone defect computed from
actual force-bearing contact patches, and warm-starts its physical search.
The strict original capture gate is unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
import physics_handover_search as core
from physics_handover_search_v4 import QualifiedReleaseHandover

# Install instrumentation only in this versioned process. Historical source
# files and running processes retain their previous evaluator.
Base=next(c for c in QualifiedReleaseHandover.__mro__ if c.__name__=='HandoverExperiment')
base_evaluate=Base.evaluate

def evaluate_geometry(self,x,return_trace=False):
    result=base_evaluate(self,x,return_trace)
    if result.get('unstable'):return result
    Q,_=self.commands(x);d=mujoco.MjData(self.m);mujoco.mj_copyData(d,self.m,self.prefix)
    force=np.zeros(6);rows=[]
    for step in range(self.start_step,min(self.end_step,round(10.7/self.dt))):
        d.ctrl[:]=self.control(Q,step);mujoco.mj_step(self.m,d)
        if step%5!=4 or d.time<9.7:continue
        patches=[]
        for i,c in enumerate(d.contact):
            b1,b2=int(self.m.geom_bodyid[c.geom1]),int(self.m.geom_bodyid[c.geom2])
            other=b2 if b1==self.ob else b1 if b2==self.ob else -1
            finger=self.body_finger.get(other)
            if finger is None:continue
            mujoco.mj_contactForce(self.m,d,i,force)
            if force[0]<=.1:continue
            patches.append((finger,np.array(c.pos),np.array(c.frame[:3])*(1 if b1==self.ob else -1),float(c.friction[0])))
        defect=3.;best=None
        for fa,pa,na,ma in patches:
            if fa!='thumb':continue
            for fb,pb,nb,mb in patches:
                if fb=='thumb':continue
                delta=pa-pb;length=np.linalg.norm(delta)
                if length<1e-5:continue
                line=delta/length;threshold=1/np.sqrt(1+min(ma,mb)**2)
                dot=float(na@nb);a=float(na@line);b=float(nb@(-line))
                loss=max(dot+.5,0)+max(threshold-a,0)+max(threshold-b,0)
                if loss<defect:defect=loss;best=(dot,a,b)
        rows.append([float(d.time),defect,*(best or (1.,-1.,-1.))])
    rows=np.asarray(rows)
    result['actual_contact_geometry_defect']=float(rows[:,1].mean())
    result['actual_normal_dot_best_mean']=float(rows[:,2].mean())
    result['actual_contact_line_alignment_mean']=rows[:,3:].mean(0).tolist()
    return result

Base.evaluate=evaluate_geometry

class ContinuousGeometryHandover(QualifiedReleaseHandover):
    def __init__(self,spec):
        super().__init__(spec)
        self.center=np.asarray(spec['initial_absolute_parameters'],float)
        assert self.center.shape==(60,)
        self.scale[:28]*=.65;self.scale[28:56]*=.35

    def commands(self,x):
        if getattr(self,'_fixed_trial',False):return super().commands(x)
        absolute=np.asarray(x)+self.center
        if getattr(self,'_resolved_key',None)!=absolute.tobytes():self.evaluate(x)
        return super().commands(absolute)

    def evaluate(self,x,return_trace=False):
        result=super().evaluate(np.asarray(x)+self.center,return_trace)
        if not result.get('unstable'):result['cost']+=8*result['actual_contact_geometry_defect']
        return result

core.HandoverExperiment=ContinuousGeometryHandover

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--warmstart',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for key,value in [('samples',128),('iterations',22),('workers',16),('seed',3650)]:p.add_argument('--'+key,type=int,default=value)
    a=vars(p.parse_args());spec=json.loads(a['spec'].read_text());previous=json.loads(a.pop('warmstart').read_text());spec['initial_absolute_parameters']=previous['parameters'];a['spec']=spec;core.optimize(**a)
