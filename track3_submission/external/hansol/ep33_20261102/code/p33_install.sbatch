#!/bin/bash
#SBATCH -J p33install
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/p33_install_%j.log
# Install the ep33 spoon roll fix (HGAR8s15 / HGBR8s15) into S1_HGA / S1_HGB: data bundles and zips, then the sharpa bundles
# (videos_sharpa/epi33.mp4 and mink_ik/episode_000033/* from the Mink IK run already made on the fixed bundles, 11/02 14:43).
# Current versions -> *_prev_p33. Same pattern as media/track3_ep28/code/p28_install.sh.
set -e
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; V=/home/hansol/v2d/outputs/variants_20261030
R=/data1/hansol/mink_S1_20261031/out_contact; ST=$V/_build_p33; cd $V
declare -A FIX=([HGA]=HGAR8s15 [HGB]=HGBR8s15)
rm -rf $ST; mkdir -p $ST
for m in HGA HGB; do $PY $W/code/build_p33.py $V/recon_E14_best_B2A_S1_$m $W/roll/recon_E14_best_B2A_S1_${FIX[$m]}/episode_000033 $ST/recon_E14_best_B2A_S1_$m $m; done
for m in HGA HGB; do b=recon_E14_best_B2A_S1_$m; p=${b}_prev_p33
  mv $b $p; mv $b.zip $p.zip; mv $b.zip.sha256 $p.zip.sha256; sed -i "s/$b.zip/$p.zip/" $p.zip.sha256
  echo "Superseded 2026-11-03 by $b (ep33 spoon roll fix). Kept for reference." > $p.SUPERSEDED.txt
  mv $ST/$b $b; zip -q -r -X $b.zip $b; sha256sum $b.zip > $b.zip.sha256; unzip -tq $b.zip; cat $b.zip.sha256
done
rmdir $ST
for m in HGA HGB; do b=recon_E14_best_B2A_S1_$m; s=${b}_sharpa; p=${s}_prev_p33; E=$s/mink_ik/episode_000033; N=recon_E14_best_B2A_S1_${FIX[$m]}
  ls $R/$N/episode_000033/sharpa_ik_qpos.npz $R/$N/episode_000033/ik_report.json $R/$N/epi33.mp4 > /dev/null
  mv $s $p; mv $s.zip $p.zip; mv $s.zip.sha256 $p.zip.sha256; sed -i "s/$s.zip/$p.zip/" $p.zip.sha256
  echo "Superseded 2026-11-03 by $s (ep33 spoon roll fix; ep33 IK and video from the fixed bundle). Kept for reference." > $p.SUPERSEDED.txt
  mkdir $s; cp -a $b/. $s/; cp -a $p/videos_sharpa $p/mink_ik $p/README_sharpa.md $s/
  cp $R/$N/epi33.mp4 $s/videos_sharpa/epi33.mp4
  cp $R/$N/episode_000033/sharpa_ik_qpos.npz $R/$N/episode_000033/ik_report.json $R/$N/episode_000033/contact_sheet.jpg $E/
  printf '\nUpdate 2026-11-03 (ep33 spoon roll fix, see README.md): videos_sharpa/epi33.mp4 and mink_ik/episode_000033/* from the\nMink IK run on the fixed bundle (same tool, Slurm, 2026-11-02).\n' >> $s/README_sharpa.md
  (cd $s && awk '{print $2}' ../$p/SHA256SUMS_sharpa.txt | xargs sha256sum > SHA256SUMS_sharpa.txt)
  diff -rq $b $s | grep -v -E 'Only in .*: (videos_sharpa|mink_ik|README_sharpa.md|SHA256SUMS_sharpa.txt)' || true
  zip -q -r -X $s.zip $s; sha256sum $s.zip > $s.zip.sha256; unzip -tq $s.zip; cat $s.zip.sha256
done
ls -la $V/recon_E14_best_B2A_S1_HG?.zip $V/recon_E14_best_B2A_S1_HG?_sharpa.zip
echo P33-INSTALL-DONE
