"""Release the donor only after the receiver demonstrates opposed contact.

The first pass holds the donor and finds the earliest30ms receiver-contact
event. A second pass releases after that event on the original20Hz grid. The
two passes have identical commands before release, so the event is causal;
the exported reference is then independently executed over the full episode.
"""
import argparse,json
from pathlib import Path
import numpy as np
import physics_handover_search as core
from physics_handover_search_v3 import ClearFingersHandover

class QualifiedReleaseHandover(ClearFingersHandover):
    def commands(self,x):
        if getattr(self,'_fixed_trial',False):return super().commands(x)
        key=np.asarray(x).tobytes()
        if getattr(self,'_resolved_key',None)!=key:self.evaluate(np.asarray(x))
        return super().commands(self._resolved_parameters)

    def evaluate(self,x,return_trace=False):
        x=np.asarray(x);held=x.copy();held[59]=self.spec['horizon_s']+1-self.spec['original_right_release_start_s']
        self._fixed_trial=True
        try:
            before=super().evaluate(held,True)
            if before.get('unstable'):return before
            trace=before['trace'];qual=(trace[:,0]>=9.7)&(trace[:,0]<=10.75)&(trace[:,3]>.5)&(trace[:,4]>.5)
            run=0;trigger=None
            for row,ok in zip(trace,qual):
                run=run+1 if ok else 0
                if run>=3:trigger=float(row[0]);break
            resolved=held.copy();release=None
            if trigger is not None:
                # Leave a full source frame before changing commands. This
                # makes the original linear20Hz control interpolation causal.
                release=float(np.ceil((trigger+.05)*self.fps)/self.fps)
                resolved[59]=release-self.spec['original_right_release_start_s']
                result=super().evaluate(resolved,True)
                if result.get('unstable'):return result
                a=result['trace'];prefix=a[:,0]<release-1/self.fps
                if not np.array_equal(a[prefix],trace[prefix]):raise AssertionError('Contact-trigger prefix changed')
            else:result=before;a=trace
        finally:self._fixed_trial=False
        near=(a[:,0]>=9.7)&(a[:,0]<=10.7);hold=a[:,0]>=10.7
        leftalone=(a[:,3]>.5)&(a[:,5]<.1)&(a[:,6]<.1);qualified=leftalone&(a[:,4]>.5)
        pos=np.minimum(a[hold,7]/.03,8).mean();ang=np.minimum(a[hold,8]/.5236,4).mean()
        cost=5*pos+2*ang+2*np.minimum((a[near,1]+a[near,2])/.03,4).mean()
        # These acquisition signals distinguish a useful incipient pinch from
        # mere nearby fingertips before donor withdrawal becomes possible.
        cost+=5*(1-a[near,3].mean())+4*(1-a[near,4].mean())
        cost+=8*(1-leftalone[hold].mean())+4*(1-qualified[hold].mean())+2*np.minimum(a[hold,5]/5.,4).mean()
        cost+=.2*np.minimum(a[:,9]/30,4).mean()+10*np.maximum(a[:,10]-.002,0).mean()/.002+10*np.maximum(a[:,11]-.003,0).mean()/.003
        cost+=5*np.maximum(np.rad2deg(a[:,12])-1.1,0).mean()+.002*np.mean((x/self.bound)**2)
        if trigger is None:cost+=4
        result.update(cost=float(cost),acquisition_both_fraction=float(a[near,3].mean()),acquisition_opposed_fraction=float(a[near,4].mean()),receiver_contact_trigger_s=trigger,donor_release_start_s=release,causal_event_prefix_verified=trigger is not None,donor_release_requires_receiver_contact=True)
        self._resolved_key=x.tobytes();self._resolved_parameters=resolved.copy()
        if not return_trace:result.pop('trace',None)
        return result

core.HandoverExperiment=QualifiedReleaseHandover

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for key,value in [('samples',128),('iterations',28),('workers',16),('seed',3640)]:p.add_argument('--'+key,type=int,default=value)
    a=vars(p.parse_args());a['spec']=json.loads(a['spec'].read_text());core.optimize(**a)
