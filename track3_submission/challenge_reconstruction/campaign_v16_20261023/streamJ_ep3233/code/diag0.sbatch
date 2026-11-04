#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --job-name=J-diag0
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/diag0-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python; J=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233
export PYTHONPATH=$R/refinement_20261021_ep32/code:$R/code:/data1/SRL_V2D/third_party/DexAgent
B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_baseline; FR=/data1/SRL_V2D/challenge_reconstruction/final_recon
cd /tmp
echo "### ep32 v17 actor (3c3c shortened-recon world)"; $PY -B $J/code/diag_timeline.py $B/episode_000032/backend/evaluation/execution/trajectory.npz $R/episodes/episode_000032 $B/episode_000032/backend/evaluation/execution/scene_physics.xml | tee $J/ep32/diag/v17_actor_timeline.txt
echo "### ep32 v17 teacher tail/c1_8.3_11.5"; $PY -B $J/code/diag_timeline.py $R/refinement_20261021_ep32/search/tail/c1_8.3_11.5/execution/trajectory.npz $R/episodes/episode_000032 $R/refinement_20261021_ep32/search/tail/c1_8.3_11.5/execution/scene_physics.xml | tee $J/ep32/diag/v17_teacher_timeline.txt
echo "### ep33 v17 actor (recon world)"; $PY -B $J/code/diag_timeline.py $B/episode_000033/backend/evaluation/execution/trajectory.npz $R/episodes/episode_000033 $B/episode_000033/backend/evaluation/execution/scene_physics.xml | tee $J/ep33/diag/v17_actor_timeline.txt
echo "### ep33 v17 teacher v12_segE/c2_R_9.8_12.3"; $PY -B $J/code/diag_timeline.py $R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3/execution/trajectory.npz $R/episodes/episode_000033 $R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3/execution/scene_physics.xml | tee $J/ep33/diag/v17_teacher_timeline.txt
echo "### ep33 chainB/c1_7.3_9.0 teacher (cup 1.24)"; $PY -B $J/code/diag_timeline.py $R/refinement_20261021_ep33/search/chainB/c1_7.3_9.0/execution/trajectory.npz $R/episodes/episode_000033 $R/refinement_20261021_ep33/search/chainB/c1_7.3_9.0/execution/scene_physics.xml | tee $J/ep33/diag/chainB_c1_teacher_timeline.txt
echo "### ep32 probe: shipped teacher reference replayed with the full 57bb recon cup (visual + 32 recon convex parts, final_recon urdf)"
$PY -B $J/code/probe_world.py --teacher $R/refinement_20261021_ep32/search/tail/c1_8.3_11.5 --out $J/ep32/probe_cup57bb/teacher_tail --episode-dir $R/episodes/episode_000032 --mesh-dir $R/episodes/episode_000032/stage2/assets/blue_cup --urdf $FR/episode_000032/assets/blue_cup.urdf | tee $J/ep32/diag/probe_cup57bb_teacher.txt
$PY -B $J/code/diag_timeline.py $J/ep32/probe_cup57bb/teacher_tail/execution/trajectory.npz $R/episodes/episode_000032 $J/ep32/probe_cup57bb/teacher_tail/scene_recon.xml | tee $J/ep32/diag/probe_cup57bb_teacher_timeline.txt
echo "### ep32 control: same replay in the unchanged 3c3c scene (determinism / node check vs the recorded teacher)"
$PY -B $J/code/probe_world.py --no-swap --teacher $R/refinement_20261021_ep32/search/tail/c1_8.3_11.5 --out $J/ep32/probe_cup57bb/control_3c3c --episode-dir $R/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032 --mesh-dir /dev/null --urdf /dev/null | tee $J/ep32/diag/probe_control_3c3c.txt
