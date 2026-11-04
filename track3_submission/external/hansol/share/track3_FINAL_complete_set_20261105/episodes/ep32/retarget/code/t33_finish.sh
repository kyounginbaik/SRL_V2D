#!/bin/bash
# T32 finish chain: bake beam table into the actor (mresH save, official eval in the W_RP2 world) -> streamB stager (stage_b.py) -> run_one (canonical build-v2)
# -> vs.sh -> k32splice -> splice32 into FINAL -> dry-run -> render video. usage: t32_finish.sh NAME ACTOR TABLE TEACHER
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; K=$C22/streamK33_ep33; T=$C22/streamT32_teacher; V16=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; R=/data1/SRL_V2D/challenge_submissions_SRL
DX=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$DX/.venv/bin/python
N=$1; A=$2; TB=$3; TE=$4; M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/P_ep33_regrasp_cem3/episode_000033/meshes/episode_000033; EPD=$T/ep33/W_RP33/episode_corrected/episode_000033
export E31_TGT=$K/targets/K33_RP E31_EPD=$EPD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$DX/code:/data1/SRL_V2D/third_party/DexAgent TMPDIR=$K/tmp
SV=$K/runs/t33/save_$N; cd $TMPDIR
[ -f $SV/actor/evaluation/result.json ] || $PY -B -u $K/code/mresH.py save --ep 33 --actor $A --meshes $M --out $SV --table $TB 2>&1 | grep -v -E "Warn|warn\(" | tail -n 3 | cut -c1-300
python3 -I -c "import json;r=json.load(open('$SV/actor/evaluation/result.json'));m=r['metrics'];print('SAVE proxy',m.get('reference_proxy_mp_sr'),{k:(round(v['mean_position_error_cm'][0],2),round(v['mean_orientation_error_degrees'][0],1)) for k,v in m['official_style_reference_proxy']['objects'].items()})"
cd /tmp; $PY -B -u $T/code/stage_b.py --episode 33 --distilled $SV/actor --base $T/staging/$N --teacher $TE --teacher-note "T33 $N: beam-polished (wbeamH) actor on K33_RP world W_RP33 (GT-free)" \
  --lineage-file $T/code/build_world_b.py --lineage-file $T/code/blib.py --lineage-file $T/code/delta_port.py --lineage-file $T/code/prepare_pickup_b.py --lineage-file $T/code/teacher_ilc_b.py --lineage-file $TE/result.json 2>&1 | grep -v -E "Warn|warn\(" | tail -n 4 | cut -c1-200
TAG=T33_$N; bash $V16/common_v2_candidates/run_one.sbatch 33 $T/staging/$N/components/episode_000033 $TAG 2>&1 | tail -n 2 | cut -c1-200
python3 -I -c "import json;c=json.load(open('$V16/common_v2_candidates/$TAG/episode_000033/completed.json'));print('BUILD exact',c.get('exact_previous_physical_states_actions'),'eval_script',c.get('eval_script_sha256','')[:8],'eval_code',c.get('eval_code_sha256','')[:8])"
bash $K/code/vs.sh 33 $TAG K33_RP 2>&1 | grep -E "^VS" | head -3 | cut -c1-240
DON=$R/ep33_fix_20261104/$TAG/all_20_episodes; $PY -B $K/code/k33splice.py $TAG $DON $TAG 2>&1 | tail -1 | cut -c1-160
PKG=$R/track3_final_ep33_${N}_20261104/all_20_episodes; mkdir -p $PKG
python3 -I $R/ep33_fix_20261104/splice33.py $R/FINAL_20261104/all_20_episodes/track3_submission.parquet $DON/track3_submission.parquet $PKG/track3_submission.parquet 2>&1 | tail -2
cp -r $R/FINAL_20261104/all_20_episodes/submission_kit $PKG/; cd $PKG/submission_kit && $R/.venv/bin/python submit/v2d_submit.py ../track3_submission.parquet --track track_3 -m dry --dry-run 2>&1 | grep -E "looks valid|rror" | head -2
echo "PACKAGE $PKG sha $(sha256sum $PKG/track3_submission.parquet | cut -c1-16)"
D=$R/videos_ep32_RP2_20261104; W=$T/ep33/W_RP33/episode_corrected/episode_000033; CC=$V16/common_v2_candidates
printf '{\n "episode_dir": "%s",\n "baseline_report": "%s/K33_K33_H5r16_c3/episode_000033/backend/evaluation/execution/execution_report.json",\n "refined_report": "%s/%s/episode_000033/backend/evaluation/execution/execution_report.json",\n "out_dir": "%s/ep33_%s",\n "reference_scene": "%s/%s/episode_000033/backend/evaluation/execution/scene_physics.xml",\n "width": 640, "height": 480, "stride": 1, "include_source": true,\n "titles": ["Reference: K33_RP (our final recon, physics world)", "H5r16_c3 - in FINAL (recorded physics)", "%s (recorded physics)"],\n "video_name": "ep33_%s.mp4"\n}\n' "$W" "$CC" "$CC" "$TAG" "$D" "$N" "$CC" "$TAG" "$N" "$N" > $D/args_ep33_$N.json
sed -e "s#VID-ep32_RP2_c1#VID-ep32_$N#" -e "s#logs/ep32_RP2_c1#logs/ep32_$N#" -e "s#args_ep33_RP2_c1.json#args_ep33_$N.json#" $D/r_ep33_RP2_c1.sbatch > $D/r_ep33_$N.sbatch; sbatch --parsable $D/r_ep33_$N.sbatch
