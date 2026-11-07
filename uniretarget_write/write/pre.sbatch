#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write/logs/%x-%j.log
set -euo pipefail
R=/lustre/exaflops/retarget_bench; H=/scratch2/exaFLOPs/srl_yam_run/ablate/oak/write; DD=/scratch2/exaFLOPs/spider_wide/ds; T=write_wb; P=$DD/processed/oakinkv2/sharpa/right/$T; A="--task=$T --dataset-name=oakinkv2 --data-id=0 --embodiment-type=right --dataset-dir=$DD"
unset DISPLAY; export MUJOCO_GL=egl PATH="$R/.venv_spider/bin:$PATH"
cd $R/repos/spider
python $H/proc.py
python spider/preprocess/decompose_fast.py $A
python spider/preprocess/detect_contact.py $A || echo contact_warn
python spider/preprocess/generate_xml.py $A --robot-type=sharpa || true
python $R/daid_mjwp/check_pairs.py $P/scene.xml
python spider/preprocess/ik_fast.py $A --robot-type=sharpa --no-show-viewer
python $H/add_board.py
python $R/daid_mjwp/check_pairs.py $P/scene.xml
echo PRE_DONE
