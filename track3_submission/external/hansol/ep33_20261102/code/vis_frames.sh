#!/bin/bash
/home/hansol/v2d/.venvs/v2d-host/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 -hide_banner -i /data1/hansol/ep33_20261102/out/pir_v1_vis.mp4 2>&1 | grep -E "Duration|Stream"
/home/hansol/v2d/.venvs/v2d-host/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 -y -loglevel error -i /data1/hansol/ep33_20261102/out/pir_v1_vis.mp4 -vf "fps=1/2,scale=480:-2,tile=4x3" -frames:v 1 /data1/hansol/ep33_20261102/out/pir_v1_vis_sheet.jpg
