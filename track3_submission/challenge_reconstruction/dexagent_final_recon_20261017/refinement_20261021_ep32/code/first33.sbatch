#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --job-name=first33
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/logs/first33-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/spoon_in_cup.py 33 teacher:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/sc_3.5_5.8_w1.0/execution/trajectory.npz | grep -v radial
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/actor_grid_fine_legacy.py --episode 33 --source /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/sc_3.5_5.8_w1.0 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/sc_3.5_5.8_w1.0/actor_grid_fine --workers 8
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/actor_grid_legacy.py --episode 33 --source /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/sc_3.5_5.8_w1.0 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/sc_3.5_5.8_w1.0/actor_grid --workers 8
ls -d /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/search/sc_3.5_5.8_w1.0/actor_grid*/actor_* | while read a; do echo "$a /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep33/release_wrapped_first/$(basename $(dirname $a))__$(basename $a)"; done | xargs -P 8 -L 1 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/wrap_release33.py
