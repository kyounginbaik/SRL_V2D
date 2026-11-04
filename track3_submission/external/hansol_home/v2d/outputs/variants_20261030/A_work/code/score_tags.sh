#!/bin/bash
# usage: score_tags.sh TAG... ; scores A_<TAG> on the 20 public/held-out mirror clips with the unchanged scorer, 4 at a time
W=/home/hansol/v2d/outputs/variants_20261030/A_work
L=$W/logs/score_$(date +%H%M%S).list
for t in "$@"; do
  for f in $W/pub/*/episode_*/exports/A_$t; do [ -e $f/public_score.json ] || echo $f; done
done > $L
echo "$(wc -l < $L) folders -> $L"
xargs -a $L -P 4 -I{} sh -c '$0 {} && echo "DONE {}"' $W/code/score.sh
echo ALLDONE
