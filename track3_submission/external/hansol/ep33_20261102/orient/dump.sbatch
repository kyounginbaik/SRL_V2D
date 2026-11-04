#!/bin/bash
#SBATCH -p debug
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orientdump_%j.log
FF=$(/home/hansol/v2d/.venvs/v2d-host/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
$FF -v error -y -i /home/hansol/v2d/outputs/variants_20261030/recon_E14_best_B2A_S1_HGA_sharpa/videos_sharpa/epi33.mp4 -vf "select='between(n\,190\,200)+between(n\,256\,264)',tile=4x5" -frames:v 1 -vsync vfr /data1/hansol/ep33_20261102/orient/mink_tiles.jpg
