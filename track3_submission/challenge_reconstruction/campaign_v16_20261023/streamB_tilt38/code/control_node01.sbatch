#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=sB-ctl01
#SBATCH --nodelist=node01
#SBATCH --array=0
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/control-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
R0=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd /tmp
if [ $SLURM_ARRAY_TASK_ID = 0 ]; then
  # ep3: replay the shipped teacher reference in the regenerated old-world scene (determinism / recipe check; expect pot 0.88/5.8 spoon 1.99/19.7)
  $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/carry_over.py --episode 3 --old-epdir $R0/episodes/episode_000003 --new-epdir $R0/episodes/episode_000003 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/controls/ep3_same_world_node01 --reference $R0/refinement_20261020_pot/power_grasp/release_grid3_ep3/open2.5_frac0.6_handle_rs0.6_rm0.08/reference.npz
else
  # ep8: replay the ilc2 closed-loop commands as a 20 Hz open-loop reference in the old world (fidelity check; closed loop gave cup 1.87/8.2 pot 1.42/8.6)
  $PY -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/code/carry_over.py --episode 8 --old-epdir $R0/episodes/episode_000008 --new-epdir $R0/episodes/episode_000008 --out /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/controls/ep8_same_world_ctrl --ctrl-trajectory $R0/refinement_20261020_pot/ilc2/episode_000008/fb0.3_gp0.8/iter2/evaluation/execution/trajectory.npz
fi
