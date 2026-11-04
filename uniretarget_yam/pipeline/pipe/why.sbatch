#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/why-%j.log
for a in "recon_000020 left 1" "recon_000020 left 0"; do apptainer exec --containall --no-home --home /tmp --env OMP_NUM_THREADS=2 -B "/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam:/uniretarget_yam" -B "/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/flash_chord:/workspace:ro" -B /scratch2/exaFLOPs/srl_yam_run/pipe:/sp --pwd /workspace /scratch2/exaFLOPs/srl_yam_run/flash_chord_yam.sif python /sp/why2.py $a 2>&1 | grep -vE "Warn|warn"; done; echo WHY_DONE
