#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=ep37-legacy-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/ep37_legacy/logs/stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd /tmp && $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/ep37_legacy/code/stage_ep37_legacy.py --episode 37 \
  --distilled $R/refinement_20261018/expanded_batch/learned/episode_000037/gain_0.03 \
  --base /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/ep37_legacy/staging \
  --episode-dir $R/episodes/episode_000037 \
  --original-base $R/refinement_20261018/policy_handoff/episode_000037/portable/original \
  --lineage-file $R/refinement_20261018/policy_handoff/episode_000037/readiness.json \
  --lineage-file $R/refinement_20261018/policy_handoff/episode_000037/validation/reference_proxy_mp_sr.json
