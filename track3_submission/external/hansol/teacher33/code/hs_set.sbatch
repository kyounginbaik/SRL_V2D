#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH -o /data1/hansol/teacher33/logs/completeset-%j.log
/home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/teacher33/code/mk_complete_set.py
