#!/bin/bash
# ep33 retarget pipeline on recon K33_RP: world build -> teacher port -> fixed-point scene export -> two-hand fcem2 CEM (4 variants). usage: t33_pipeline.sh (run on node01)
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; T=$C22/streamT32_teacher; V16=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; DX=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$DX/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$DX/code:/data1/SRL_V2D/third_party/DexAgent
[ -f $C22/streamK33_ep33/targets/K33_RP2/episode_000033/wooden_spoon_trajectory.npz ] || { echo "K33_RP target missing"; exit 2; }
M=$V16/common_v2_candidates/P_ep33_regrasp_cem3/episode_000033/meshes/episode_000033; W=$T/ep33/W_RP33c; NEW=$W/episode_corrected/episode_000033; SC=$W/scene/scene_physics.xml
cd /tmp
[ -d $W ] || $PY -B -u $T/code/build_world_b.py 33 $W $V16/streamP_ep33/staging_C/scene/scene_physics.xml --base-epd $V16/streamP_ep33/staging_C/episode_compose/episode_000033 --meshes $M --short 0.0178 2>&1 | grep -vi warn | tail -n 3 | cut -c1-240
SEED=$V16/streamP_ep33/regrasp/tilc7/co5,7.6,8.8_gp0.4_fz6.8,9.4/iter2/reference.npz; OLD=$V16/streamP_ep33/staging_C/episode_compose/episode_000033; B=$T/ep33/port/pg_rg0_b0.5c
[ -f $B/result.json ] || $PY -B -u $T/code/delta_port.py "$SEED" $B $SC $OLD $NEW --blend 0.5 --right-gain 0 2>&1 | grep -vi warn | tail -n 1 | cut -c1-200
# fixed-point scene: export once (iterations 0), then use its execution scene for every CEM rollout (ep33 _freeze is not idempotent)
FP=$T/ep33/fcem2_fpc; [ -f $FP/execution/scene_physics.xml ] || $PY -B -u $T/code/fcem2.py 33 $B $NEW $FP --t-lock 3.6 --t-off 14.0 --ramp 0.15 --knots 10 --samples 4 --iterations 0 --workers 2 --seed 1 2>&1 | grep -E "^INIT|^RESULT|Traceback" | cut -c1-240
FS=$FP/execution/scene_physics.xml
: > $T/ep33/fcem2_33g.list
echo "33 $B $NEW $T/ep33/fcem2_33g_sc3_s51 --t-lock 3.2 --t-off 15.0 --ramp 0.2 --knots 10 --scoop-w 3 --samples 128 --iterations 50 --workers 48 --seed 51 --scene $FS --init /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT32_teacher/ep33/fcem2_33d_scoop3/progress.json" >> $T/ep33/fcem2_33g.list
echo "33 $B $NEW $T/ep33/fcem2_33g_sc3_s52 --t-lock 3.2 --t-off 15.0 --ramp 0.2 --knots 10 --scoop-w 3 --samples 128 --iterations 50 --workers 48 --seed 52 --scene $FS --init /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT32_teacher/ep33/fcem2_33d_scoop3/progress.json" >> $T/ep33/fcem2_33g.list
echo "33 $B $NEW $T/ep33/fcem2_33g_sc2_off17 --t-lock 3.2 --t-off 17.0 --ramp 0.2 --knots 10 --scoop-w 2 --samples 128 --iterations 50 --workers 48 --seed 53 --scene $FS --init /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT32_teacher/ep33/fcem2_33d_scoop3/progress.json" >> $T/ep33/fcem2_33g.list
J=$(sbatch --parsable --export=ALL,FC=fcem2.py --array=0-2 --nodelist=node01 -c 48 --time=03:00:00 -J T33-fcem2g -o $T/logs/fcem2_33g_%a.log $T/code/fcem.sbatch $T/ep33/fcem2_33g.list); echo "fcem2_33 array $J"
