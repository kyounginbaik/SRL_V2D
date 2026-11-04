"""Optimize a physical clearance lift/tilt before the lid's lateral movement."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from physics_pickup_clearance import ClearanceExperiment,blend,rotations_between,dump
from wrist_reference import continuous_euler_xyz
from pickup_optimizer import optimize

B=Path(__file__).resolve().parents[1]


class LidExtractionExperiment(ClearanceExperiment):
    def __init__(self,spec):
        super().__init__(spec)
        self.incumbent=np.asarray(spec['incumbent_parameters'])
        if self.incumbent.shape!=(65,):raise ValueError('expected original65-parameter grasp search')
        # Local grasp6, closed fingers22, liftXYZ, peak/end timing, lift tilt3.
        self.dim=36
        self.scale=np.r_[[.004]*3,[.05]*3,[.08]*22,[.015,.015,.025],[.07,.10],[.12]*3]
        self.bound=np.r_[[.025]*3,[.35]*3,[.35]*22,[.06,.06,.10],[.20,.30],[.65]*3]

    def commands(self,y):
        y=np.asarray(y);x=self.incumbent.copy();x[:6]+=y[:6];x[34:56]+=y[6:28]
        Q,timing=super().commands(x);close=timing['closed']
        end=max(close+.4,self.spec.get('extraction_end_s',2.85)+y[32])
        peak=np.clip(self.spec.get('extraction_peak_s',2.35)+y[31],close+.12,end-.15)
        p0=self.anchor_p+self.anchor_R.apply(self.local_p+x[:3])
        r0=self.anchor_R*self.local_R*Rotation.from_rotvec(x[3:6])
        lift=np.array([y[28],y[29],max(.02,self.spec.get('extraction_height_m',.10)+y[30])])
        p1=p0+lift;r1=r0*Rotation.from_rotvec(y[33:36])
        first=(self.times>close)&(self.times<=peak);w=blend((self.times[first]-close)/(peak-close))
        Q[np.ix_(first,self.wcols[:3])]=p0+w[:,None]*lift
        Q[np.ix_(first,self.wcols[3:])]=continuous_euler_xyz(rotations_between(r0,r1,w),Q[0,self.wcols[3:]])
        second=(self.times>peak)&(self.times<=end);w=blend((self.times[second]-peak)/(end-peak))
        goalp=Q[np.ix_(second,self.wcols[:3])].copy()
        goalR=Rotation.from_euler('XYZ',Q[np.ix_(second,self.wcols[3:])])
        Q[np.ix_(second,self.wcols[:3])]=p1+w[:,None]*(goalp-p1)
        rs=r1*Rotation.from_rotvec((r1.inv()*goalR).as_rotvec()*w[:,None])
        Q[np.ix_(second,self.wcols[3:])]=continuous_euler_xyz(rs,Q[0,self.wcols[3:]])
        timing.update(extraction_peak=float(peak),extraction_end=float(end),extraction_lift_m=lift.tolist())
        return Q,timing

    def evaluate(self,x,return_trace=False):
        report=super().evaluate(x,return_trace=True)
        trace=report.get('trace')
        if trace is not None:
            # The lid cannot improve by dragging the pot out of place.
            extra=15.*np.maximum(trace[:,11]-.025,0.).mean()/.03
            report['cost']+=float(extra);report['additional_other_object_penalty']=float(extra)
            if not return_trace:report.pop('trace')
        return report


def run(smoke=False,samples=128,iterations=24,workers=24,seed=248,tag='lid_extraction_v1'):
    parent=B/'refinement_20261018/episode_000001/pickup_cem_v1/pose_01'
    spec=json.loads((parent/'settings.json').read_text())['spec']
    spec['incumbent_parameters']=json.loads((parent/'result.json').read_text())['parameters']
    spec.update(extraction_height_m=.10,extraction_peak_s=2.35,extraction_end_s=2.85)
    out=B/f'refinement_20261018/episode_000001/{tag}'
    if smoke:
        exp=LidExtractionExperiment(spec);report=exp.evaluate(np.zeros(exp.dim),return_trace=True)
        out.mkdir(parents=True,exist_ok=True);trace=report.pop('trace',None)
        if trace is not None:np.savez_compressed(out/'trace.npz',trace=trace)
        dump(out/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)
    else:optimize(LidExtractionExperiment,spec,out,samples=samples,iterations=iterations,workers=workers,seed=seed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int,default=128)
    p.add_argument('--iterations',type=int,default=24);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--seed',type=int,default=248);p.add_argument('--tag',default='lid_extraction_v1')
    run(**vars(p.parse_args()))
