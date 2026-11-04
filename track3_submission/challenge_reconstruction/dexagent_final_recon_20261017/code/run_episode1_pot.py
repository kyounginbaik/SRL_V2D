"""Search the pot hold/lift while keeping the selected lid command proposal."""
from pathlib import Path
import argparse,json,time
import numpy as np
from physics_pickup_clearance import ClearanceExperiment as PickupExperiment,optimize,dump

B=Path(__file__).resolve().parents[1]


def run(task,smoke=False,tag='pot_cem_v1',samples=128,iterations=24,workers=24,seed=141,input_tag='pot_inputs'):
    spec=json.loads((B/f'refinement_20261018/episode_000001/{input_tag}/pot_{task:02d}.json').read_text())
    spec['times']['start']=.25
    spec['times']['hover']=1.2
    spec.update(clearance_time_s=.85,clearance_min_wrist_z_m=.30)
    out=B/f'refinement_20261018/episode_000001/{tag}/pose_{task:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if smoke:
        exp=PickupExperiment(spec);start=time.monotonic()
        report=exp.evaluate(np.zeros(exp.dim),return_trace=True)
        trace=report.pop('trace',None)
        if trace is not None:np.savez_compressed(out/'trace.npz',trace=trace)
        report['elapsed_s']=time.monotonic()-start
        dump(out/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)
    else:
        (out/'search_physics').mkdir(parents=True,exist_ok=True)
        optimize(spec,out,samples=samples,iterations=iterations,workers=workers,seed=seed+task)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,choices=range(4),required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--tag',default='pot_cem_v1')
    p.add_argument('--samples',type=int,default=128);p.add_argument('--iterations',type=int,default=24)
    p.add_argument('--workers',type=int,default=24);p.add_argument('--seed',type=int,default=141)
    p.add_argument('--input-tag',default='pot_inputs')
    run(**vars(p.parse_args()))
