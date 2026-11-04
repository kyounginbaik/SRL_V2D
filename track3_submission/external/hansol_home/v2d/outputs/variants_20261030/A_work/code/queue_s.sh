#!/bin/bash
cd /home/hansol/v2d/outputs/variants_20261030/A_work
while kill -0 3485852 2>/dev/null; do sleep 15; done
until grep -q GENDONE logs/gen_s.log; do sleep 15; done
code/score_tags.sh s32 s128 > logs/score_s.log 2>&1
code/qc_tag.sh s32 > logs/qc_s.log 2>&1; code/qc_tag.sh s128 >> logs/qc_s.log 2>&1
