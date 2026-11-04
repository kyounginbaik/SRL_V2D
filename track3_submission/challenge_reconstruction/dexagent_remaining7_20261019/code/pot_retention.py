#!/usr/bin/env python3
"""Keep the acquired left-hand pot grip while refining a later local window."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,sys
from prepare_seven import BASE,LEGACY,ROOT
sys.path[:0]=[str(LEGACY/'code'),str(ROOT/'third_party/DexAgent')]
from refine_expanded_retention import SmallStepRetention
from pickup_optimizer import optimize

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--stop',type=float,required=True);p.add_argument('--seed',type=int,default=65);p.add_argument('--scale',type=float,default=.75);p.add_argument('--samples',type=int,default=96);p.add_argument('--iterations',type=int,default=12);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
 inp=BASE/'physics_collision_v2/episodes/episode_000028';r=json.loads((a.source/'result.json').read_text());m=json.loads((inp/'input_manifest.json').read_text())
 spec=dict(scene=r['execution']['scene'],reference=str(a.source/'reference.npz'),original_reference=str(inp/'tracking_reference.npz'),source=m['source'],geometry=str(inp/'metric_geometry.json'),manifest=str(inp/'input_manifest.json'),start_s=a.start,stop_s=a.stop,side='left',protect_object='plastic_dish_rack',ramp_s=.3,exploration_multiplier=a.scale,episode=inp.name)
 optimize(SmallStepRetention,spec,a.output,samples=a.samples,iterations=a.iterations,workers=a.workers,seed=a.seed)
if __name__=='__main__':main()
