R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
J=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233; A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code
FR=/data1/SRL_V2D/challenge_reconstruction/final_recon; ILC=$R/refinement_20261021_ep38/code/ilc_carry_generic.py; SILC=$J/code/spoon_ilc_j.py
E33=$R/episodes/episode_000033; E32=$R/refinement_20261018/expanded_batch/repaired_export_inputs/episode_000032; E32R=$R/episodes/episode_000032
CUP32=$R/refinement_20261018/expanded_batch/cup32_height_candidate/scale_0.900/assets/blue_cup_visual.obj
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$R/refinement_20261021_ep32/code:$R/code:/data1/SRL_V2D/third_party/DexAgent
rel() { $PY -B -u $A/release_variant.py --teacher $1 --episode-dir $2 --out $3 --side $4 --object $5 --t-rel $6 --open-dur 0.3 --t-lift $7 --blend-dur 0.6; }
seat() { $PY -B -u $A/seated_grasp.py --teacher $1 --episode-dir $2 --out $3 --side right --object wooden_spoon --t-seat $4 --t-on $5 --t-off $6 --ramp ${7:-0.5} ${8:-}; }
ilc() { $PY -B -u $ILC --source $1 --out $2 --side $3 --object $4 --t0 $5 --t1 $6 --t-clear $7 --sigma-s 0.5 --episode-dir $8 --iterations $9 --gain-rot ${10} --gain-pos ${10}; }
silc() { $PY -B -u $SILC $1 $2 --t0 $3 --t1 $4 --iterations $5 --gain-rot $6 --gain-pos $6 --sigma-s 0.5 --episode-dir $7 ${8:-}; }
