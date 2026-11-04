#!/usr/bin/env python3
"""Prepare requested final source variants when their verified marker appears."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]

def one(episode,timeout):
    name=f'episode_{episode:06d}'
    marker=BASE/'collision_recon'/name/'collision_repair_provenance.json'
    deadline=time.monotonic()+timeout
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError(str(marker))
        time.sleep(5)
    log=BASE/'logs'/f'final{episode}.log';log.parent.mkdir(exist_ok=True)
    print(json.dumps(dict(episode=name,stage='prepare_and_baseline',log=str(log))),flush=True)
    with log.open('a') as handle:
        subprocess.run([sys.executable,str(BASE/'code/prepare_variant.py'),'--episode',str(episode),
            '--source-root',str(BASE/'collision_recon'),'--workspace',str(BASE/'physics_collision_v2'),
            '--baseline'],stdout=handle,stderr=subprocess.STDOUT,check=True)
    path=BASE/'physics_collision_v2/runs/baseline_v1'/name/'baseline/result.json'
    report=json.loads(path.read_text())
    print(json.dumps(dict(episode=name,stage='baseline_completed',result=str(path),
        mp_sr=report['metrics']['reference_proxy_mp_sr'],
        objects=report['metrics']['official_style_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    p.add_argument('--timeout',type=float,default=900.)
    a=p.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(one,e,a.timeout) for e in a.episodes]
        for f in futures:f.result()
