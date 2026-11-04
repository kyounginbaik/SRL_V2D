"""Handover search with an explicit collision-clearing receiving approach.

The first version's straight interpolation crossed the tabletop. Start the
robot-only approach from its attained state, lift above the support, traverse,
then descend to the same candidate grasp. Original clock and objects are fixed.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import physics_handover_search as core
from physics_pickup_search import blend
from wrist_reference import continuous_euler_xyz

class ClearApproachHandover(core.HandoverExperiment):
    def commands(self,x):
        Q,timing=super().commands(x)
        t=self.times;start=self.ts['start'];contact=timing['contact'];lift_end=start+.20;traverse_end=max(lift_end+.15,contact-.20)
        qstart=self.prefix.qpos.copy();p0=qstart[self.wcols[:3]];r0=Rotation.from_euler('XYZ',qstart[self.wcols[3:]])
        # Parent's closing interval is a fixed grasp pose, so a sample at the
        # contact boundary gives the corresponding full candidate end pose.
        gp=self.local_p+x[:3];gr=self.local_R*Rotation.from_rotvec(x[3:6])
        endp=self.anchor_p+self.anchor_R.apply(gp);endr=self.anchor_R*gr
        opened=np.clip(self.opened+x[6:28]*.3,self.limits[6:,0],self.limits[6:,1])
        lift=p0.copy();lift[2]=max(p0[2]+.13,.21)
        hover=endp.copy();hover[2]=max(endp[2]+.10,lift[2])
        mats=Rotation.from_euler('XYZ',Q[:,self.wcols[3:]]).as_matrix()
        for a,b,pa,pb,ra,rb,fa,fb in [
            (start,lift_end,p0,lift,r0,r0,qstart[self.fcols],opened),
            (lift_end,traverse_end,lift,hover,r0,endr,opened,opened),
            (traverse_end,contact,hover,endp,endr,endr,opened,opened)]:
            mask=(t>a)&(t<=b);w=blend((t[mask]-a)/(b-a))
            Q[np.ix_(mask,self.wcols[:3])]=(1-w[:,None])*pa+w[:,None]*pb
            mats[mask]=(ra*Rotation.from_rotvec((ra.inv()*rb).as_rotvec()*w[:,None])).as_matrix()
            Q[np.ix_(mask,self.fcols)]=(1-w[:,None])*fa+w[:,None]*fb
        Q[:,self.wcols[3:]]=continuous_euler_xyz(Rotation.from_matrix(mats),self.Q[0,self.wcols[3:]])
        Q[:self.start_frame+1]=self.Q[:self.start_frame+1]
        timing.update(receiving_lift_end=lift_end,receiving_traverse_end=traverse_end,approach_clearance_wrist_z=float(lift[2]))
        return Q,timing

core.HandoverExperiment=ClearApproachHandover

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for key,value in [('samples',128),('iterations',28),('workers',16),('seed',3620)]:p.add_argument('--'+key,type=int,default=value)
    a=vars(p.parse_args());a['spec']=json.loads(a['spec'].read_text());core.optimize(**a)
