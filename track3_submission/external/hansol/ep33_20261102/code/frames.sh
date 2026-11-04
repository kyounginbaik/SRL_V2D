#!/bin/bash
/home/hansol/v2d/.venvs/v2d-host/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 -y -loglevel error -i /home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA_sharpa/videos_sharpa/epi33.mp4 -vf "select='eq(n\,176)+eq(n\,248)',scale=1014:-2" -vsync 0 /data1/hansol/ep33_20261102/out/hga_ik_%d.jpg
