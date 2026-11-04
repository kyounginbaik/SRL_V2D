#!/usr/bin/env python3
"""Explicit small-step full-horizon continuation of an acquired grasp."""
from pathlib import Path
import argparse,json
from physics_retention_search import RetentionExperiment
from pickup_optimizer import optimize


class SmallStepRetention:
    def __init__(self,spec):
        self.inner=RetentionExperiment(spec);self.m=self.inner.model
        for key in ['ref','fps','dim','bound']:setattr(self,key,getattr(self.inner,key))
        self.scale=self.inner.scale*spec['exploration_multiplier']
    def commands(self,x):return self.inner.commands(x),{}
    def evaluate(self,x,return_trace=False):return self.inner.evaluate(x)


def run(source,episode_dir,output,start_s,stop_s,protect_object,seed,scale=.25,samples=96,iterations=8,workers=16,side='right'):
    a_side=side
    prev=json.loads((source/'result.json').read_text());manifest=json.loads((episode_dir/'input_manifest.json').read_text())
    spec=dict(scene=prev['execution']['scene'],reference=str((source/'reference.npz').resolve()),
        original_reference=str((episode_dir/'tracking_reference.npz').resolve()),source=manifest['source'],
        geometry=str((episode_dir/'metric_geometry.json').resolve()),manifest=str((episode_dir/'input_manifest.json').resolve()),
        start_s=start_s,stop_s=stop_s,side=a_side,protect_object=protect_object,ramp_s=.3,exploration_multiplier=scale,
        episode=manifest['episode'])
    optimize(SmallStepRetention,spec,output,samples=samples,iterations=iterations,workers=workers,seed=seed)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['source','episode-dir','output']:p.add_argument('--'+key,type=Path,required=True)
    for key in ['start-s','stop-s']:p.add_argument('--'+key,type=float,required=True)
    p.add_argument('--protect-object',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--scale',type=float,default=.25);p.add_argument('--samples',type=int,default=96)
    p.add_argument('--iterations',type=int,default=8);p.add_argument('--workers',type=int,default=16);p.add_argument('--side',default='right')
    run(**vars(p.parse_args()))
