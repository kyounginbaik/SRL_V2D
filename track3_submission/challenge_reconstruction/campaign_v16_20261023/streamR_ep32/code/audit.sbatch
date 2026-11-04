#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=R-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/logs/audit-%j.log
# usage: sbatch audit.sbatch OUT_JSON STEP [POSES]
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamR_ep32/code/${AUDIT:-audit_ep32.py} $1 --step $2 ${3:+--poses $3}
