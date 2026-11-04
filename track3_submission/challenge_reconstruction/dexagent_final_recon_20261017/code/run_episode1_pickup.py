"""Episode-specific entry point for physical lid pickup CEM, original clock."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from physics_pickup_search import PickupExperiment, optimize, dump

B=Path(__file__).resolve().parents[1]


def run(task,smoke=False,samples=96,iterations=12,workers=16,tag='pickup_cem_v1',seed=44):
    spec_path=B/f'refinement_20261018/episode_000001/pickup_inputs/pose_{task:02d}.json'
    spec=json.loads(spec_path.read_text())
    spec['manifest']=str(B/'episodes/episode_000001/input_manifest.json')
    spec['scale']=np.r_[[.008]*3,[.10]*3,[.02]*3,[.12]*3,
                        [.12]*22,[.12]*22,[.012]*3,[.08]*3,[.10,.08,.12]].tolist()
    out=B/f'refinement_20261018/episode_000001/{tag}/pose_{task:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if smoke:
        started=time.monotonic();exp=PickupExperiment(spec)
        report=exp.evaluate(np.zeros(exp.dim),return_trace=True)
        trace=report.pop('trace',None)
        if trace is not None:np.savez_compressed(out/'smoke_trace.npz',trace=trace)
        report['elapsed_seconds']=time.monotonic()-started
        dump(out/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)
    else:
        (out/'search_physics').mkdir(parents=True,exist_ok=True)
        optimize(spec,out,samples=samples,iterations=iterations,workers=workers,seed=seed+task)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,choices=range(4),required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--samples',type=int,default=96)
    p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--tag',default='pickup_cem_v1');p.add_argument('--seed',type=int,default=44)
    run(**vars(p.parse_args()))
