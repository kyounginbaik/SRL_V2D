#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=K32-retro
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32/logs/%x-%j.log
# K32 retrodiction: kit-faithful score_vs_compose of officially failing ep32 rollouts vs the K32 targets (ep32 only)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32; R=/data1/SRL_V2D/challenge_submissions_SRL; C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
cd /tmp
for p in track3_v29_20261030 track3_v26_20261029 ablation_swap_v29_20261102/SRL_sw_ep32_hansol track3_v31a_HGA_20261101; do
 for t in $K/targets/K32_jisu $K/targets/K32_jisuW $C22/target_final_v14_g17; do
  echo "$p $t"
 done
done | xargs -P 6 -L 1 sh -c 'o='$K'/out/retro/$(echo $0 | tr / _)__$(basename $1); mkdir -p $o; '$PY' -B -u '$S' '$R'/$0/all_20_episodes/track3_submission.parquet '$R'/$0/all_20_episodes/results $1 $o/ep32.json 32 > $o/ep32.log 2>&1'
echo DONE
