#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 10
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH -J T33-distill
#SBATCH -o /data1/hansol/teacher33/logs/%x-%j.log
# usage: [T33EPD=<snapped episode dir>] distill33.sbatch TEACHER_DIR OUT [GRID...]  -- team streamC actor_grid_c.py (as streamB distill.sbatch / retarget32 T1), official evaluator, snapped c18 W1 world
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/teacher33/tmp PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
T=$1; O=$2; shift 2; G=${@:-1536,0.0035,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 1536,0.002,0.0001 1024,0.001,0.00005 1536,0.005,0.0002 2048,0.001,0.00005 512,0.003,0.0001 1024,0.01,0.0005 3072,0.003,0.0001}
cd /data1/hansol/teacher33/tmp; /home/hansol/v2d/.venvs/v2d-host/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --episode 33 --teacher $T --output $O --episode-dir ${T33EPD:-/data1/hansol/teacher33/worlds/R33c18_W1s/ep33/episode_corrected/episode_000033} --workers 10 --seed 44 --grid $G 2>&1 | grep -v -E "Warn|warn\("
