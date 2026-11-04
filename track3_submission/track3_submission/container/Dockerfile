# SRL Track 3 submission container (DRAFT v1). Based on the kit's examples/container/Dockerfile.
#
# Build tree (assembled next to this file, see ../docs/BUILD_CONTAINER.md):
#   submission/
#     submission.json   (container/submission.json with TODOs filled and "_" notes removed)
#     run.sh            (container/run.sh)
#     code/             SRL_V2D checkout at the commit in submission.json, .git included, clean
#     kit/              the kit copy that packed and uploaded FINAL (FINAL_20261104/all_20_episodes/submission_kit)
#     verifier_kit/     kit >= 9cad120d (ships CONTAINER_SUBMISSION.md and tools/verify_container.py)
#     submitted/        track3_submission.parquet (9bc322af) + its receipts
#     weights/final/    20 evaluated components (collect_weights.sh; ep33 from teacher33, ep36 from spoon3537), policy.npz sha256 == checkpoint_sha256
#     meshes/final/     the reconstructed meshes packed into FINAL
#     outputs/final/    the build-v2 rollouts that FINAL rows were exported from
#
#   docker build -t srl-v2d-track3:final --build-arg BASE_IMAGE=... -f Dockerfile .
#
# BASE_IMAGE: the team did not evaluate inside a container. Use an image with the same Python and
# MuJoCo versions as challenge_reconstruction/dexagent_final_recon_20261017/.venv
# (requirements.lock.txt), plus git and the kit requirements (pandas, pyarrow).
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends git rsync libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY submission /submission
# Reproduce the evaluation venv from the lock file (no network at run time; install at build time).
RUN python -m venv /opt/srl-venv \
    && /opt/srl-venv/bin/pip install --no-cache-dir -r /submission/code/challenge_reconstruction/dexagent_final_recon_20261017/requirements.lock.txt \
    && python -m pip install --no-cache-dir pandas pyarrow
# The pipeline uses absolute /data1/SRL_V2D paths; point them at the shipped checkout.
RUN mkdir -p /data1 /home && ln -s /submission/code /data1/SRL_V2D \
    && ln -s /submission/code/external/hansol /data1/hansol \
    && ln -s /submission/code/external/hansol_home /home/hansol
RUN chmod 755 /submission/run.sh && chmod -R a+rX /submission
WORKDIR /submission
CMD ["/submission/run.sh", "verify"]
