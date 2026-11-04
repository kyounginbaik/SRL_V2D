#!/usr/bin/env bash
# Collect the 20 evaluated per-episode components of the final file 9bc322af (FINAL 0fca055b + ep32 T32c_b1 + ep33 R6M2 + ep36 s4)
# into a container build tree, check every policy.npz against the recorded checkpoint_sha256, and copy each episode's
# build-v2 candidate (fresh rollout, results, completed.json) to outputs/final/rollouts.
#
# Run inside Slurm (reads /data1/SRL_V2D and /data1/hansol (teacher33, spoon3537) read-only, writes only under DEST; about 1.7 GB + candidates):
#   sbatch -p debug -c 2 --mem=8G -t 01:00:00 -o collect_%j.log --wrap "bash collect_weights.sh /path/to/build/submission"
set -euo pipefail
DEST="${1:?usage: collect_weights.sh DEST_SUBMISSION_DIR}"
case "$(realpath -m "$DEST")" in /data1/SRL_V2D*) echo "refusing to write under /data1/SRL_V2D" >&2; exit 2;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TSV="${TSV:-${HERE}/../provenance/FINAL_9bc322af_episodes.tsv}"
mkdir -p "${DEST}/weights/final" "${DEST}/outputs/final/rollouts"
FAIL=0
while IFS=$'\t' read -r EP ROWS SRC CAND STREAM SHA COMP CANDDIR; do
    NAME=$(printf episode_%06d "$EP")
    OUT="${DEST}/weights/final/${NAME}"
    rsync -a --exclude '__pycache__' "${COMP}/" "${OUT}/"
    rsync -a --exclude '__pycache__' "${CANDDIR}/" "${DEST}/outputs/final/rollouts/${NAME}/"
    GOT=$(sha256sum "${OUT}/portable/original/policy.npz" | cut -d' ' -f1)
    if [[ "$GOT" == "$SHA" ]]; then echo "ok   ep${EP} ${CAND} ${SHA:0:8}"; else echo "FAIL ep${EP} ${CAND} want ${SHA:0:8} got ${GOT:0:8}"; FAIL=1; fi
done < <(tail -n +2 "$TSV")
exit $FAIL
