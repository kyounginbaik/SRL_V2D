#!/bin/bash
cd /home/hansol/v2d/outputs/variants_20261030/A_work
until grep -q GENDONE logs/gen14.log; do sleep 15; done
code/score_exp.sh A14_g8 A14_s32 > logs/score14.log 2>&1
