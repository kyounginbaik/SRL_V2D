# Build from the FlashCHORD directory, not this recipe directory.
ARG BASE_IMAGE=flash_chord:uniretarget-base
FROM ${BASE_IMAGE}
# Checkpoint videos/overlays use cv2; keep the training lock's numpy unchanged.
RUN uv pip install --python /opt/venv/bin/python --no-deps opencv-python-headless==4.11.0.86 \
    && python -c "import cv2, jax, flax, wandb, pyarrow; print(cv2.__version__)"
