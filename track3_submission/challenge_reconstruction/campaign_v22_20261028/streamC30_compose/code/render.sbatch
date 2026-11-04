#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 2 --mem=16G -t 00:50:00 -J C30-render -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/videos/logs/render-%j.log
# usage: sbatch render.sbatch RENDER_ARGS.json   (v28 render.sbatch body)
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; cd /tmp
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py --args $1
