#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch ego.sbatch "N ..."   Simulation from the recording's camera, beside the source video if $SRC/episode_XXXXXX.mp4 exists.
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; M=/uniretarget_yam; B=/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103; SRC=${SRC:-/none}
for N in $1; do EP=$(printf recon_%06d $N); Z=$(printf episode_%06d $N); L=$(printf ep%02d $N); U=$R/videos/$EP/run_mU; [ -d $U ] || continue
apptainer exec --nv --containall --no-home --home /tmp --env MUJOCO_GL=egl,PYTHONDONTWRITEBYTECODE=1,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" -B "$H:$M" -B /scratch2 -B /lustre --pwd $M/code $R/flash_chord_yam.sif python $M/code/render_ego.py --episode $M/episodes/$EP --bundle $B/bundles/$Z --run $U --title "$L UniRetarget" --output $R/videos/side_by_side/${L}_Source_vs_Simulation.mp4 --source $SRC/$Z.mp4 2>&1 | grep -vE 'pix_fmt' | tail -3
done
echo EGO_DONE
