#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH -o /data1/hansol/retarget33/logs/%x-%j.log
#SBATCH -J R33-score
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
cd /tmp; /home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/score_vs_compose.py /data1/hansol/retarget33/packages/track3_FINAL_ep33R33R2_20261103/all_20_episodes/track3_submission.parquet /data1/hansol/retarget33/packages/track3_FINAL_ep33R33R2_20261103/all_20_episodes/results /data1/hansol/ep33_20261102/k33x/targets/HGAYc2 /data1/hansol/retarget33/packages/track3_FINAL_ep33R33R2_20261103/score_vs_7b.json 33
