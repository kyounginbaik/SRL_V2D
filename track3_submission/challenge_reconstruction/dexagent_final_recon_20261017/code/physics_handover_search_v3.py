"""Clear the donor's noncontact fingers before the receiving hand approaches."""
import argparse,json
from pathlib import Path
import numpy as np
import physics_handover_search as core
from physics_handover_search_v2 import ClearApproachHandover
from physics_pickup_search import blend

class ClearFingersHandover(ClearApproachHandover):
    def commands(self,x):
        Q,timing=super().commands(x)
        active=blend((self.times-self.ts['start'])/.35)*(1-blend((self.times-timing['right_release_start'])/(timing['right_release_end']-timing['right_release_start'])))
        for key,value in self.spec.get('giver_unused_curl',{}).items():
            col=self.right.finger_qpos[key];index=list(self.right.finger_qpos).index(key)
            # The candidate still controls every donor finger around this
            # natural clearance posture; grasping thumb/index are untouched.
            desired=value+x[28+6+index]
            Q[:,col]=(1-active)*Q[:,col]+active*desired
            aid=self.right.finger_act[key];Q[:,col]=np.clip(Q[:,col],*self.m.actuator_ctrlrange[aid])
        Q[:self.start_frame+1]=self.Q[:self.start_frame+1]
        return Q,timing

core.HandoverExperiment=ClearFingersHandover

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for key,value in [('samples',128),('iterations',28),('workers',16),('seed',3630)]:p.add_argument('--'+key,type=int,default=value)
    a=vars(p.parse_args());a['spec']=json.loads(a['spec'].read_text());core.optimize(**a)
