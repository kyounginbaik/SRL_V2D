#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH -o /data1/hansol/spoon3537/logs/renderA-%j.log
export LD_LIBRARY_PATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.software_gl/usr/lib64:$LD_LIBRARY_PATH MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
cd /data1/hansol/spoon3537 && /home/hansol/v2d/.venvs/v2d-host/bin/python -B code/render_s5.py
