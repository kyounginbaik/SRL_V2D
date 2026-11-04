# Stream G shared paths
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; V22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; SD=$V22/streamG_ep10
L=$C/streamL_ep10; P10=left=blue_cup,right=beige_cup
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
