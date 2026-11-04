#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/fit4mk_%j.log
set -e
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit4; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033
for m in HGA HGB; do S=$V/recon_E14_best_B2A_S1_$m/$E; B=$O/recon_E14_best_B2A_S1_${m}Zr6/$E; rm -rf $(dirname $B) $W/k33x/targets/HGAZr6_$m
  $PY $W/code/ep33_rollcurve_target.py $S $W/k33x/targets/HGAZr6_$m/$E "0:-6,22:-6,34:0" --bundle-out $B
  mv $B/ROLLCURVE.json $O/ROLLCURVE_p33h_$m.json; cp -p $S/ROLLCURVE.json $B/ROLLCURVE.json; done
cd /data1/hansol/retarget33 && rm -rf targets/R33_HGA_r6 && $PY code/cwtgt33.py R33_HGA_r6 $O/recon_E14_best_B2A_S1_HGAZr6/$E | tail -2
echo MK_DONE
