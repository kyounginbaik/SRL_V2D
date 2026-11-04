# Stream I common environment (sourced by the sbatch files). WORLD must be a build_world.py workspace.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
R7=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019; RS=$R7/rackscan_20261021
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamM_ep14
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:$R7/code
EPDIR=$WORLD/episodes/episode_000014; export EPDIR
