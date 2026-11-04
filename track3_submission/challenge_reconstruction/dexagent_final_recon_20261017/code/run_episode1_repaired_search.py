"""Bounded physical refinement on the versioned repaired episode1 candidate."""
import argparse,json,time
import numpy as np
from repair_episode1_registration import OUT
from physics_pickup_clearance_gravity import ClearanceExperiment,optimize,dump

def run(smoke=False,samples=64,iterations=12,workers=16):
    root=OUT/'patch_height094';spec=json.loads((root/'search_input/spec.json').read_text())
    if smoke:
        start=time.monotonic();exp=ClearanceExperiment(spec);Q,timing=exp.commands(np.zeros(exp.dim))
        for col in exp.objects.values():assert np.array_equal(Q[:,col:col+7],exp.original[:,col:col+7])
        assert np.array_equal(Q[:exp.start_frame+1],exp.Q[:exp.start_frame+1])
        report=exp.evaluate(np.zeros(exp.dim),return_trace=True);trace=report.pop('trace',np.empty((0,14)))
        report.update(elapsed_s=time.monotonic()-start,original_targets_exact=True,fixed_prefix_exact=True,world_up=exp.up.tolist())
        np.savez_compressed(root/'search_input/smoke_trace.npz',trace=trace);dump(root/'search_input/smoke.json',report);print(json.dumps(report,indent=2),flush=True)
    else:
        optimize(spec,root/'pickup_cem_v1',samples=samples,iterations=iterations,workers=workers,seed=341)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int,default=64)
    p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=16);run(**vars(p.parse_args()))
