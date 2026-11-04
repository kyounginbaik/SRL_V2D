#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=A2-chain35
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA2_crotch/logs/chain35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA2_crotch; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter; PY=$R/.venv/bin/python
case $SLURM_ARRAY_TASK_ID in
 0) cd /tmp && $PY -B -u $W/code/rebuild_pickup.py $W/ep35/leftpinch/leftpinch_h12_sA $W/ep35/chain/h12_rel19.5 --release 19.5 && $PY -B -u $A/code/release_variant.py --teacher $W/ep35/chain/h12_rel19.5 --episode-dir $EPD --out $W/ep35/chain/h12_rel19.5_relR19.3 --side right --t-rel 19.3 --open-dur 0.3 --t-lift 19.9 && $PY -B -u $A/code/release_variant.py --teacher $W/ep35/chain/h12_rel19.5_relR19.3 --episode-dir $EPD --out $W/ep35/chain/h12_rel19.5_relR19.3_relL19.4 --side left --t-rel 19.4 --open-dur 0.3 --t-lift 20.0 ;;
 1) cd /tmp && $PY -B -u $W/code/rebuild_pickup.py $W/ep35/leftpinch/leftpinch_h9_sA $W/ep35/chain/h9_rel19.5 --release 19.5 && $PY -B -u $A/code/release_variant.py --teacher $W/ep35/chain/h9_rel19.5 --episode-dir $EPD --out $W/ep35/chain/h9_rel19.5_relR19.3 --side right --t-rel 19.3 --open-dur 0.3 --t-lift 19.9 && $PY -B -u $A/code/release_variant.py --teacher $W/ep35/chain/h9_rel19.5_relR19.3 --episode-dir $EPD --out $W/ep35/chain/h9_rel19.5_relR19.3_relL19.4 --side left --t-rel 19.4 --open-dur 0.3 --t-lift 20.0 ;;
 2) cd /tmp && $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $W/ep35/leftpinch/leftpinch_h12_sA --out $W/ep35/ilc_h12/left_gr0.3_gp0 --side left --object wooden_spoon --t0 1.5 --t1 18.5 --t-clear 18.8 --sigma-s 0.5 --episode-dir $EPD --iterations 3 --gain-rot 0.3 --gain-pos 0.0 ;;
 3) cd /tmp && $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $W/ep35/leftpinch/leftpinch_h12_sA --out $W/ep35/ilc_h12/right_gr0.3_gp0 --side right --object wooden_spoon --t0 1.5 --t1 18.5 --t-clear 18.8 --sigma-s 0.5 --episode-dir $EPD --iterations 3 --gain-rot 0.3 --gain-pos 0.0 ;;
 4) cd /tmp && $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $W/ep35/leftpinch/leftpinch_h12_sA --out $W/ep35/ilc_h12/left_gr0.5_gp0.3 --side left --object wooden_spoon --t0 1.5 --t1 18.5 --t-clear 18.8 --sigma-s 0.5 --episode-dir $EPD --iterations 3 --gain-rot 0.5 --gain-pos 0.3 ;;
 5) cd /tmp && $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $W/ep35/leftpinch/leftpinch_h12_sA --out $W/ep35/ilc_h12/right_gr0.5_gp0.3 --side right --object wooden_spoon --t0 1.5 --t1 18.5 --t-clear 18.8 --sigma-s 0.5 --episode-dir $EPD --iterations 3 --gain-rot 0.5 --gain-pos 0.3 ;;
esac
