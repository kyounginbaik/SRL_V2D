# Build from the FlashCHORD directory, not this recipe directory.
ARG BASE_IMAGE=flash_chord:uniretarget-base
FROM ${BASE_IMAGE}
# Checkpoint videos/overlays use cv2; keep the training lock's numpy unchanged.
RUN uv pip install --python /opt/venv/bin/python --no-deps opencv-python-headless==4.11.0.86 \
        pandas==2.2.3 python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.2 six==1.17.0 \
    && python -c "import cv2, jax, flax, wandb, pyarrow, pandas; print(cv2.__version__, pandas.__version__)"
