#!/bin/bash
# usage: gen_all.sh PARAMS_JSON ; guarded smoothing of all 20 public/held-out + 19 eval clips (ep32 excluded), 4 at a time
W=/home/hansol/v2d/outputs/variants_20261030/A_work
B=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v8_g10b
P=$1
{ for e in $W/pub/*/episode_*; do echo "public $e"; done
  for e in $B/episode_*; do [ $(basename $e) = episode_000032 ] || echo "eval $e"; done; } |
  xargs -P 4 -L 1 sh -c '$0/code/genv.sh $0/code/guard_clip.py $1 $2 '$P' || echo "GENFAIL $2"' $W
echo GENDONE
