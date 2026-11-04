#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=sB-carry
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/carry-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python; cd /tmp
T3=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261020_pot/power_grasp/release_grid3_ep3/open2.5_frac0.6_handle_rs0.6_rm0.08
T8=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261020_pot/ilc2/episode_000008/fb0.3_gp0.8/iter2/evaluation/execution/trajectory.npz
N3=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/calibrated_inputs/episodes/episode_000003; N8=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/calibrated_inputs/episodes/episode_000008
case $SLURM_ARRAY_TASK_ID in
0) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/carry_over.py --episode 3 --old-epdir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000003 --new-epdir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000003 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/controls/ep3_same_world_noclip --reference $T3/reference.npz --no-clip --scene /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/controls/ep3_same_world/scene/controller_scene.xml ;;
1) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/carry_over.py --episode 3 --old-epdir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000003 --new-epdir $N3 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/carry/teacher_noclip --reference $T3/reference.npz --no-clip ;;
2) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/carry_over.py --episode 3 --old-epdir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000003 --new-epdir $N3 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/carry/teacher_clip --reference $T3/reference.npz --scene /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/carry/teacher_noclip/scene/controller_scene.xml ;;
3) $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/carry_over.py --episode 8 --old-epdir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000008 --new-epdir $N8 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep8/carry/teacher_ctrl --ctrl-trajectory $T8 --no-clip ;;
esac
