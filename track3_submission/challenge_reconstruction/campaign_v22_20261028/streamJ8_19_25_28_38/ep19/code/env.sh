R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023; V22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
F=$V22/streamF_margins/ep19; J=$V22/streamJ8_19_25_28_38/ep19
EPD=$F/world_st/episode_target/episode_000019
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
