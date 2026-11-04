#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -J T33-report
#SBATCH -o /data1/hansol/teacher33/logs/%x-%j.log
# append one candidate's result to out/STATUS_20261105.md. usage: sbatch t33report.sh LABEL PKGNAME SAVE_DIR JSAVE JPERT JTAB JPKG
R=/data1/hansol/teacher33; S=$R/out/STATUS_20261105.md; L=$1; P=$2; SV=$3; PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
lg() { ls $R/logs/*-$1.log 2>/dev/null | head -1; }
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
{
echo; echo "### $L — reported $(TZ=Asia/Seoul date '+%F %H:%M KST')"
echo "- package: $R/packages/$P $( [ -d $R/packages/$P ] && echo '(exists)' || echo '(MISSING)')"
echo "- pkg: $(grep -h -E '^SHA|KIT validate|^VERIFY|Error|error' $(lg $7) 2>/dev/null | cut -c1-200 | tr '\n' ' ')"
echo "- kit score (official eval): $(grep -h '^FINAL' $(lg $4) 2>/dev/null | cut -c1-200)"
echo "- perturbation (16 draws): $(grep -h '^PERTURB' $(lg $5) 2>/dev/null | $PY -c "
import sys, json
for l in sys.stdin:
    n, s = l.split(' ', 2)[1:]; d = json.loads(s); print(n, 'pass', d.get('frac_pass'), 'spoon_lost', d.get('frac_spoon_lost'), 'median spoon', d.get('spoon_cm_median'), 'cm', d.get('spoon_deg_median'), 'deg')
")"
echo "- 0.5 s table: $(lg $6)"
echo "- end-of-clip (r33endjit, t / spoon cm/s / spoon z cm / R geoms on spoon / spoon on table):"
echo '```'
[ -f $SV/actor/evaluation/result.json ] && $PY -B $R/../retarget33/code/r33endjit.py $SV/actor/evaluation/result.json 2>/dev/null | grep -E '^ +1[4-6]\.' | awk -F'|' '{print $1 "|" $4 "|" $5 "|" $6 "|" $7}' | head -16
echo '```'
grep -h -E '^ +1[3-6]\.[05]-' $(lg $6) 2>/dev/null | cut -c1-90
} >> $S 2>&1
cat $S | tail -40
