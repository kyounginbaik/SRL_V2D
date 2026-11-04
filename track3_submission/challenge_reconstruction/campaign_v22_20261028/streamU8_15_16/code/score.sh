#!/bin/bash
# usage: score.sh EP TAG   -> scores the build-v2 output common_v2_candidates/TAG/episode_X with campaign_v22 score_any.sh
C16=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/code/score_any.sh $1 $C16/$2/$(printf episode_%06d $1) $2 && cat /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/scores/cand/$2/score.json
