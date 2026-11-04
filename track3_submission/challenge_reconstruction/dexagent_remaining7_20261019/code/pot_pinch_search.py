#!/usr/bin/env python3
"""Focused physical thumb/index search around a table-clearing pot-handle grip."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
from prepare_seven import ROOT,LEGACY
sys.path[:0]=[str(LEGACY/'code'),str(ROOT/'third_party/DexAgent')]
from physics_pickup_search_v2 import PickupExperiment
from pickup_optimizer import optimize
class PotPinchSearch(PickupExperiment):
 def __init__(self,spec):
  super().__init__(spec)
  self.keep=np.r_[np.arange(9),np.arange(12,21),np.arange(34,43),np.arange(56,65)]
  self.scale=self.scale[self.keep];self.bound=self.bound[self.keep];self.dim=len(self.keep)
 def commands(self,x):
  full=np.zeros(65);full[self.keep]=np.asarray(x);return super().commands(full)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=61);p.add_argument('--workers',type=int,default=16);p.add_argument('--samples',type=int,default=96);p.add_argument('--iterations',type=int,default=12);a=p.parse_args();optimize(PotPinchSearch,json.loads(a.spec.read_text()),a.output,samples=a.samples,iterations=a.iterations,workers=a.workers,seed=a.seed)
