#!/bin/bash
# Matched 3-arm x 3-seed pilot. Full study requires passing pilot first.
#SBATCH --job-name=fc-tissue-bilevel
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-8%2
#SBATCH --output=/data1/SRL_V2D/slurm_logs/fc_bilevel_%A_%a.out
#SBATCH --error=/data1/SRL_V2D/slurm_logs/fc_bilevel_%A_%a.err
set -euo pipefail
ROOT=/data1/SRL_V2D
FLASH="$ROOT/robotic_grounding/flash_chord"
MODES=(fixed reactor reactor_wrench)
SEEDS=(42 123 456)
MODE=${MODES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
if [[ ${EDITOR_STUDY:-0} == 1 ]]; then
    VARIANTS=(smooth per_frame global_local)
    VARIANT=${VARIANT_OVERRIDE:-${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}}
    MODE=reactor_wrench
    ENGINE=ttsa_flashsac
fi
WORLDS=${WORLDS:-32}
STEPS=${STEPS:-4096}
INTERVAL=${INTERVAL:-2048}
GROUP=${GROUP:-tissue_bilevel_pilot}
REPLAY=32768
MINIMUM=1024
BATCH=256
SAVE_INTERVAL=$STEPS
LOG_MODE=disabled
EXTRA=()
if [[ ${STUDY:-pilot} == full ]]; then
    WORLDS=2048
    STEPS=98304000
    INTERVAL=${FULL_INTERVAL:-9830400}
    GROUP=${FULL_GROUP:-tissue_bilevel_20261006}
    REPLAY=2000000
    MINIMUM=102400
    BATCH=2048
    SAVE_INTERVAL=9830400
    LOG_MODE=online
    # Abort early rather than failing authentication after simulator compilation.
    export WANDB_API_KEY="$(python -c 'import netrc; print(netrc.netrc().authenticators("api.wandb.ai")[2])')"
    test -n "$WANDB_API_KEY"
    EXTRA+=("curriculum.thresholds=[7372800,14745600,22118400,29491200,36864000,44236800,51609600,58982400,66355200,83558400,98304000]")
fi
if [[ ${EDITOR_STUDY:-0} == 1 ]]; then
    EXTRA+=("+bilevel.study_variant=$VARIANT")
    NAME=${VARIANT}_seed${SEED}
else
    NAME=${MODE}_seed${SEED}
fi
if [[ ${PENETRATION_AWARE:-0} == 1 ]]; then
    EXTRA+=("+bilevel.penetration_aware=true"
            "+bilevel.penetration_measure=visual_hull"
            "+bilevel.penetration_weight=${PENETRATION_WEIGHT:-1.0}"
            "+bilevel.penetration_reference_stride=${PENETRATION_STRIDE:-8}"
            "+bilevel.penetration_rollout_stride=${PENETRATION_STRIDE:-8}")
fi
if [[ ${SELECT_BEST_TRIAL:-0} == 1 ]]; then
    EXTRA+=("+bilevel.select_best_trial=true")
fi
if [[ ${GUIDED_PENETRATION_PROPOSAL:-0} == 1 ]]; then
    EXTRA+=("+bilevel.guided_penetration_proposal=true"
            "+bilevel.guided_search_stride=${GUIDED_SEARCH_STRIDE:-32}"
            "+bilevel.guided_action_amplitude=${GUIDED_ACTION_AMPLITUDE:-0.32}")
fi
OUT="$FLASH/outputs/$GROUP/$NAME"
mkdir -p "$OUT/cache"
CACHE_DIR=${CACHE_DIR:-$OUT/cache}
[[ ! -e "$OUT/upper_updates.jsonl" ]] || { echo "Refusing to overwrite prior upper-loop run: $OUT"; exit 2; }
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[[ ${#GPU_UUIDS[@]} == 1 ]] || { echo 'Expected one allocated GPU'; exit 2; }
CONTAINER=fc-bilevel-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
CPUSET=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)
trap 'docker stop -t 15 "$CONTAINER" >/dev/null 2>&1 || true' EXIT TERM INT
BASE=(docker run --rm --name "$CONTAINER" --pids-limit 4096 --runtime=nvidia
 --cpuset-cpus "$CPUSET"
 --gpus "device=${GPU_UUIDS[0]}" --user "$(id -u):$(id -g)" --ipc host --network host
 -e HOME=/tmp -e PYTHONUNBUFFERED=1 -e NVIDIA_DRIVER_CAPABILITIES=all
 -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4
 -e WANDB_API_KEY -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e PYGLET_HEADLESS=true
 -v "$FLASH:/workspace" -v "$ROOT/.e2e/tissue_box:/v2d/assets:ro" -v "$CACHE_DIR:/tmp/.cache"
 -w /workspace flash_chord:latest)
printf 'mode=%s seed=%s worlds=%s steps=%s gpu=%s\n' "$MODE" "$SEED" "$WORLDS" "$STEPS" "${GPU_UUIDS[0]}" > "$OUT/launch.txt"
sha256sum "$FLASH/src/flash_chord/retargeting/manipulation_bilevel.py" \
 "$FLASH/src/flash_chord/retargeting/ttsa_editor.py" \
 "$FLASH/src/flash_chord/retargeting/penetration.py" \
 "$FLASH/src/flash_chord/retargeting/reactor.py" \
 "$FLASH/src/flash_chord/training/flash_sac/runner.py" > "$OUT/source_sha256.txt"
/usr/bin/time -v -o "$OUT/runtime.txt" "${BASE[@]}" python scripts/train_flash_sac.py \
 experiment=dexmate_sharpa_flash_sac \
 "task.parquet='/v2d/assets/human_motion_data/ego_recon/processed_mink_1_3_tuned/sequence_id=tissue_box/robot_name=vega_sharpa'" \
 scene.world_count="$WORLDS" training.world_count="$WORLDS" training.seed="$SEED" reset.seed="$SEED" \
 training.total_environment_steps="$STEPS" training.checkpoint.save_interval_environment_steps="$SAVE_INTERVAL" \
 training.replay.capacity="$REPLAY" training.replay.minimum_size="$MINIMUM" training.replay.batch_size="$BATCH" \
 training.replay.n_step=3 training.discount=0.99 logging.mode="$LOG_MODE" \
 logging.project="${LOG_PROJECT:-v2d-tissue-bilevel}" logging.run_name="$NAME" logging.upload_checkpoints=false \
 output_dir="/workspace/outputs/$GROUP/$NAME" \
 "+bilevel={mode:$MODE,engine:${ENGINE:-reactor},warmup:$INTERVAL,interval:$INTERVAL,eval_worlds:4,upper_trials:${UPPER_TRIALS:-8}}" "${EXTRA[@]}"
"${BASE[@]}" python scripts/evaluate_policy.py \
 "evaluation.checkpoint=/workspace/outputs/$GROUP/$NAME/policy_${STEPS}.safetensors" \
 evaluation.world_count=64 evaluation.reset_mode=explicit \
 "evaluation.metrics_output=/workspace/outputs/$GROUP/$NAME/evaluation_unassisted.json" logging.mode=disabled
if [[ ${STUDY:-pilot} == full ]]; then
    for checkpoint_step in 9830400 49152000 98304000; do
        "${BASE[@]}" python scripts/view_policy.py \
          "evaluation.checkpoint=/workspace/outputs/$GROUP/$NAME/policy_${checkpoint_step}.safetensors" \
          scene.world_count=1 training.world_count=1 viewer.backend=mp4 \
          env.voc_scale=0 env.voc.max_force=0 env.voc.max_torque=0 reset.voc_decay_steps=0 \
          "viewer.output_path=/workspace/outputs/$GROUP/$NAME/rollout_${checkpoint_step}.mp4" \
          'viewer.camera_position=[1.05,-1.45,1.05]' 'viewer.camera_target=[0.0,0.0,0.85]' \
          viewer.num_frames=649 viewer.video_fps=20 viewer.headless=true viewer.realtime=false markers.enabled=false
    done
fi
