#!/bin/bash
# usage: verify.sh EP TAG : exactness of common_v2_candidates/TAG + score_any vs target_final_v5
CC=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates; NM=$(printf episode_%06d $1)
echo "$2 exact=$(grep -o '"exact_previous_physical_states_actions": [a-z]*' $CC/$2/$NM/completed.json | awk '{print $2}') $(/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/code/score_any.sh $1 $CC/$2/$NM $2 | tail -1)"
