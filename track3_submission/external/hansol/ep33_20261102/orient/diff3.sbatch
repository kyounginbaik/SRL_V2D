#!/bin/bash
#SBATCH -p debug
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/diff3_%j.log
/home/hansol/v2d/.venvs/v2d-host/bin/python -B - <<'PY'
import numpy as np
O='/data1/hansol/ep33_20261102/orient/fit2'
for m in ('HGA','HGB'):
    A=np.load(f'{O}/recon_E14_best_B2A_S1_{m}Ys15/episode_000033/object_poses.npz')
    for v in 'ab':
        B=np.load(f'{O}/recon_E14_best_B2A_S1_{m}Y{v}s15/episode_000033/object_poses.npz'); k=[str(n) for n in A['names']].index('wooden_spoon')
        Ra, Rb = A['world_from_object'][:, k, :3, :3], B['world_from_object'][:, k, :3, :3]
        d=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f', Ra, Rb)-1)/2, -1, 1))); ch=np.where(d>0.01)[0]
        print(m, v, "max deg stored f<230: %.2f, f230-245: %.2f, f246-270: %.1f" % (d[:230].max(), d[230:246].max(), d[246:271].max()))
PY
