#!/bin/bash
# Install the ep33 spoon flip at 9.2-9.5 s (HGAYc2 / HGBYc2, option 7b) into S1_HGA / S1_HGB. Run on node100 (it
# only submits Slurm jobs and waits): 1) build job: staged bundles; 2) Mink IK + render of ep33 on both staged bundles;
# 3) install job: *_prev_p33f, data zips, sharpa bundles (epi33 video + mink_ik/episode_000033) and zips.
set -e
W=/data1/hansol/ep33_20261102; V=/home/hansol/v2d/outputs/variants_20261030; M=/data1/hansol/mink_S1_20261031; ST=$V/_build_p33f
O=$W/orient/fit2; MK=$W/orient/ep33_marks_merged_20261103c2.json
# #SBATCH lines after the first command are ignored, so each job's -o goes before HEAD's commands
SB='#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:30:00'
HEAD='set -e
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"'
# 1) build
cat > $W/logs/p33f_build.sbatch <<EOF
$SB
#SBATCH -o $W/logs/p33f_build_%j.log
$HEAD
rm -rf $ST; mkdir -p $ST
for m in HGA HGB; do \$PY $W/code/build_p33f.py $V/recon_E14_best_B2A_S1_\$m $O/recon_E14_best_B2A_S1_\${m}Yc2/episode_000033 $MK $ST/recon_E14_best_B2A_S1_\$m; done
EOF
sbatch --wait $W/logs/p33f_build.sbatch > /dev/null; tail -4 $(ls -t $W/logs/p33f_build_*.log | head -1)
# 2) Mink IK on ep33 of the staged bundles
for m in HGA HGB; do N=recon_E14_best_B2A_S1_${m}_p33f; rm -rf $M/p33f/$N $M/out_contact/$N; mkdir -p $M/p33f; cp -a $ST/recon_E14_best_B2A_S1_$m $M/p33f/$N
  (cd $M && sbatch --array=33 code/run_episode_contact.sbatch $M/p33f/$N | awk '{print $4}') > $W/logs/p33f_ik_$m.jobid; done
for m in HGA HGB; do j=$(cat $W/logs/p33f_ik_$m.jobid); while squeue -h -j $j 2>/dev/null | grep -q .; do sleep 30; done; done
for m in HGA HGB; do ls -la $M/out_contact/recon_E14_best_B2A_S1_${m}_p33f/epi33.mp4 $M/out_contact/recon_E14_best_B2A_S1_${m}_p33f/episode_000033/sharpa_ik_qpos.npz; cat $M/out_contact/recon_E14_best_B2A_S1_${m}_p33f/episode_000033/ik_report.json | head -c 300; echo; done
# 3) install
cat > $W/logs/p33f_install.sbatch <<EOF
$SB
#SBATCH -o $W/logs/p33f_install_%j.log
$HEAD
V=$V; R=$M/out_contact; ST=$ST; cd \$V
for m in HGA HGB; do b=recon_E14_best_B2A_S1_\$m; p=\${b}_prev_p33f
  mv \$b \$p; mv \$b.zip \$p.zip; mv \$b.zip.sha256 \$p.zip.sha256; sed -i "s/\$b.zip/\$p.zip/" \$p.zip.sha256
  echo "Superseded 2026-11-03 by \$b (ep33 spoon flip at 9.2-9.5 s, option 7b). Kept for reference." > \$p.SUPERSEDED.txt
  mv \$ST/\$b \$b; zip -q -r -X \$b.zip \$b; sha256sum \$b.zip > \$b.zip.sha256; unzip -tq \$b.zip; cat \$b.zip.sha256
done
rmdir \$ST
for m in HGA HGB; do b=recon_E14_best_B2A_S1_\$m; s=\${b}_sharpa; p=\${s}_prev_p33f; E=\$s/mink_ik/episode_000033; N=\${b}_p33f
  mv \$s \$p; mv \$s.zip \$p.zip; mv \$s.zip.sha256 \$p.zip.sha256; sed -i "s/\$s.zip/\$p.zip/" \$p.zip.sha256
  echo "Superseded 2026-11-03 by \$s (ep33 spoon flip at 9.2-9.5 s, option 7b; ep33 IK and video re-rendered). Kept for reference." > \$p.SUPERSEDED.txt
  mkdir \$s; cp -a \$b/. \$s/; cp -a \$p/videos_sharpa \$p/mink_ik \$p/README_sharpa.md \$s/
  cp \$R/\$N/epi33.mp4 \$s/videos_sharpa/epi33.mp4
  cp \$R/\$N/episode_000033/sharpa_ik_qpos.npz \$R/\$N/episode_000033/ik_report.json \$R/\$N/episode_000033/contact_sheet.jpg \$E/
  [ -f \$R/\$N/episode_000033/hands_map_report.json ] && cp \$R/\$N/episode_000033/hands_map_report.json \$E/
  printf '\nUpdate 2026-11-03 (ep33 spoon flip at 9.2-9.5 s, see README.md): videos_sharpa/epi33.mp4 and\nmink_ik/episode_000033/* re-rendered from this bundle (same tool, Slurm).\n' >> \$s/README_sharpa.md
  (cd \$s && awk '{print \$2}' ../\$p/SHA256SUMS_sharpa.txt | xargs sha256sum > SHA256SUMS_sharpa.txt)
  diff -rq \$b \$s | grep -v -E 'Only in .*: (videos_sharpa|mink_ik|README_sharpa.md|SHA256SUMS_sharpa.txt)' || true
  zip -q -r -X \$s.zip \$s; sha256sum \$s.zip > \$s.zip.sha256; unzip -tq \$s.zip; cat \$s.zip.sha256
done
ls -la \$V/recon_E14_best_B2A_S1_HG?.zip \$V/recon_E14_best_B2A_S1_HG?_sharpa.zip
echo P33F-INSTALL-DONE
EOF
sbatch --wait $W/logs/p33f_install.sbatch > /dev/null; tail -16 $(ls -t $W/logs/p33f_install_*.log | head -1)
