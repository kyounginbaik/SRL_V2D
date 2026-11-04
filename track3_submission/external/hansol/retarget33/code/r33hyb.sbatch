#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=R33-hyb
#SBATCH --output=/data1/hansol/retarget33/logs/%x-%j.log
# usage: sbatch r33hyb.sbatch NAME t0 t1 ramp   (base R7_C final_0, donor R2_C final_0; eval vs R33_HGA_r6; cup-spoon metric)
set -e; R=/data1/hansol/retarget33; N=$1; P=/home/hansol/v2d/.venvs/v2d-host/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$R/tmp E31_TGT=$R/targets/R33_HGA_r6
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
$P $R/code/r33hybrid.py $R/runs/cw/R7_C/final_0.npz $R/runs/cw/R2_C/final_0.npz $2 $3 $4 $R/runs/hyb/$N.npz
cd $TMPDIR; $P -B -u $R/code/mresH.py eval --ep 33 --actor /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor \
  --meshes /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033 \
  --out $R/runs/eval_$N --table $R/runs/hyb/$N.npz 2>&1 | grep -v -E "Warn|warn\(" | tail -25
$P $R/code/r33cupspoon_any.py "R7=eval_R7=R33_HGA_r6" "$N=eval_$N=R33_HGA_r6"
