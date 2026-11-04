#!/usr/bin/env bash
# Usage: ./pip.sh N [Fixed_Reference_Low_Level|UniRetarget_Physics_Result|UniRetarget_Reference_Refinement]
# Puts the original video small at the left middle of a rendered video. Renders play at half speed, so the source is slowed to match; it holds on its first frame through
# settling frames added in front, and holds for 1.5 s where the render pauses on its frontier frame.
N=$1; NAME=${2:-UniRetarget_Physics_Result}; L=$(printf ep%02d $N); EP=$(printf recon_%06d $N); Z=$(printf episode_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; V=$R/videos/$EP${VT:-}; IN=$V/${L}_$NAME.mp4; OUT=$V/${L}_${NAME}_With_Original.mp4; cd $R/pipe
PAD=$(python3 -c "import json;print(json.load(open('/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/episodes/$EP/recon_meta.json')).get('pad',0))")
LOG=$(grep -l "/vid/${L}_$NAME.mp4" rr$N-*.log rr${N}b-*.log v$N-*.log vv$N-*.log 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
FIRST=$(grep -h "/vid/${L}_$NAME.mp4" $LOG 2>/dev/null | tail -1 | python3 -c "import sys,json;l=sys.stdin.read().strip();print(json.loads(l)['first'] if l else -1)")
BASE="[1:v]setpts=2*PTS,tpad=start_duration=$(python3 -c "print($PAD/10)"):start_mode=clone:stop_duration=30:stop_mode=clone"
if [ "$FIRST" -ge 0 ]; then T=$(python3 -c "print($FIRST/10)"); SRC="$BASE,split[x][y];[x]trim=0:$T,setpts=PTS-STARTPTS,tpad=stop_duration=1.5:stop_mode=clone[p];[y]trim=start=$T,setpts=PTS-STARTPTS[q];[p][q]concat=n=2:v=1[c];[c]"; else SRC="$BASE,"; fi
ffmpeg -y -loglevel error -i $IN -i $R/source_videos/ego_cam_a/$Z.mp4 -filter_complex "${SRC}scale=480:-2,pad=iw+6:ih+6:3:3:white[s];[0:v][s]overlay=24:(H-h)/2-30:shortest=1,format=yuv420p[v]" -map '[v]' -c:v libx264 -crf 18 -preset medium $OUT && echo "$OUT first=$FIRST pad=$PAD"
