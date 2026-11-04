#!/bin/bash
#SBATCH --job-name=B-pandas
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/hansol/v2d/outputs/variants_20261030/B_work/logs/install-%j.log
set -x
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
LIB=/home/hansol/v2d/.venvs/pylib_score
mkdir -p $LIB
$PY -m pip install --no-deps --target $LIB "pandas<2.3" python-dateutil six pytz tzdata
export PYTHONPATH=$LIB
$PY -c 'import pandas; print("pandas", pandas.__version__)'
cd /data1/SRL_V2D
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
time $PY -B tools/reconstruction_audit/score_bundle.py /home/hansol/v2d/outputs/variants_20261030/B_work/canonical/public/episode_000011/exports/test_r8 --metrics
