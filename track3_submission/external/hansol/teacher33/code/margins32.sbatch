#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH -J R32-margins
#SBATCH -o /data1/hansol/retarget32/logs/%x-%j.log
# snapped W1 world -> official re-evaluation of the distilled actor there -> team finger-margin CEM (search_margins_k.py)
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/retarget32/tmp PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; cd /data1/hansol/retarget32/tmp
[ -d /data1/hansol/retarget32/worlds/R32_HGA_W1s/ep32/episode_corrected/episode_000032 ] || $PY -B /data1/hansol/retarget32/code/snapf0_32.py /data1/hansol/retarget32/worlds/R32_HGA_W1/ep32/episode_corrected/episode_000032 /data1/hansol/retarget32/worlds/R32_HGA_W1s/ep32/episode_corrected/episode_000032 /data1/hansol/retarget32/runs/teacher/fcem_R32_W1_b/execution/scene_physics.xml
mkdir -p /data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001; [ -d /data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001/training ] || cp -r /data1/hansol/retarget32/runs/distill/T1/actor_c1024_p0.003_v0.0001/training /data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001/
[ -f /data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001/evaluation/result.json ] || $PY -B -c "from evaluate_policy_component import evaluate; evaluate(32, \"/data1/hansol/retarget32/runs/teacher/fcem_R32_W1_b/execution/scene_physics.xml\", \"/data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001/training/policy.npz\", \"/data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001/evaluation\", \"/data1/hansol/retarget32/worlds/R32_HGA_W1s/ep32/episode_corrected/episode_000032\")" | tail -1 | cut -c1-300
$PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/code/search_margins_k.py --episode 32 --actor /data1/hansol/retarget32/runs/distill/T1s/actor_c1024_p0.003_v0.0001 --output /data1/hansol/retarget32/runs/distill/T1sm_c1024_p0.003 --samples 64 --iterations 12 --workers 12 --seed 3232 2>&1 | grep -v -E "Warn|warn\("
