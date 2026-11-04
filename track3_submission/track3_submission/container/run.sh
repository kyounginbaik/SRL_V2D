#!/usr/bin/env bash
# SRL Track 3 submission container entry point (DRAFT v1, adapted from the kit's examples/container/run.sh).
#
# Usage:
#   run.sh verify [verify_container.py options]
#       kit check of every hash and the commit in /submission/submission.json
#   run.sh reproduce --label final --output DIR [--episodes 1,3,...]
#       re-execute each episode's evaluated policy component under evaluator build v2
#       (common_evaluator.py evaluate --backend legacy), one fresh physics rollout per episode
#
#   docker run --rm --network none -v DATASET:/data:ro -v OUT:/output IMAGE \
#       /submission/run.sh reproduce --label final --output /output
#
# The team pipeline was written against absolute paths under /data1/SRL_V2D. The image links
# /data1/SRL_V2D -> /submission/code so those paths resolve to the shipped checkout.
# CPU only (MuJoCo). NOTE: physics results were observed to depend on the CPU model (README).
set -euo pipefail

SUBMISSION="${SUBMISSION:-/submission}"
MANIFEST="${SUBMISSION}/submission.json"
KIT_PYTHON="${KIT_PYTHON:-python3}"
VERIFIER_KIT="${VERIFIER_KIT:-${SUBMISSION}/verifier_kit}"
# Python environment equivalent to challenge_reconstruction/dexagent_final_recon_20261017/.venv
# (requirements.lock.txt in that folder), with DexAgent and our floating-Sharpa port importable.
EVAL_PYTHON="${EVAL_PYTHON:-/opt/srl-venv/bin/python}"
CODE="${SUBMISSION}/code"
V2="${CODE}/challenge_reconstruction/evaluator_build_v2_20261023"
REFINE_KIT="${CODE}/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261017/submission_readiness/latest_official/v2d_submission_kit"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export PYTHONPATH="${CODE}/challenge_reconstruction/dexagent_final_recon_20261017/code:${CODE}/third_party/DexAgent${PYTHONPATH:+:${PYTHONPATH}}"
export HOME="${HOME:-/tmp}"; [[ -w "${HOME}" ]] || export HOME=/tmp

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

COMMAND="${1:-}"; shift || true
if [[ "${COMMAND}" == "verify" ]]; then
    exec "${KIT_PYTHON}" "${VERIFIER_KIT}/tools/verify_container.py" --manifest "${MANIFEST}" "$@"
fi
LABEL=""; OUTPUT=""; EPISODES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --dataset-root|--parallel) shift 2 ;;   # accepted for kit compatibility; not used by Track 3 here
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "${COMMAND}" == "reproduce" ]] || { usage >&2; exit 2; }
[[ -n "${LABEL}" && -n "${OUTPUT}" ]] || { echo "--label and --output are required" >&2; exit 2; }

OUT="${OUTPUT}/${LABEL}"; mkdir -p "${OUT}/rollouts" "${OUT}/logs"
WANTED="$("${KIT_PYTHON}" - "${MANIFEST}" "${LABEL}" "${EPISODES}" <<'EOF'
import json, sys
entry = next(s for s in json.load(open(sys.argv[1]))["submissions"] if s["label"] == sys.argv[2])
eps = sorted(int(k) for k in entry["checkpoints"])
want = eps if not sys.argv[3] else [int(e) for e in sys.argv[3].split(",")]
missing = [e for e in want if e not in eps]
if missing: sys.exit(f"no checkpoint for episode(s) {missing}")
print(" ".join(map(str, want)))
EOF
)"
STATUS=0
for EP in ${WANTED}; do
    NAME=$(printf episode_%06d "${EP}")
    COMP="${SUBMISSION}/weights/${LABEL}/${NAME}"
    echo "== ${NAME}: ${COMP}"
    (cd /tmp && "${EVAL_PYTHON}" -B -u "${V2}/package_code/common_evaluator.py" evaluate --backend legacy \
        --episode "${EP}" --legacy-component "${COMP}" --build-manifest "${V2}/build_manifest.json" \
        --kit "${REFINE_KIT}" --output "${OUT}/rollouts/${NAME}") > "${OUT}/logs/${NAME}.log" 2>&1 \
        || { echo "   FAILED (see ${OUT}/logs/${NAME}.log)"; STATUS=1; }
done
echo "rollouts in ${OUT}/rollouts"
# TODO(v2): assemble the 20 fresh rollouts into the submission parquet with the same packaging path as the final file
# (v32build.py for FINAL 0fca055b, mk_ep32ep33.py for ep32/ep33, spoon3537/rt/code/s3pkg.py for ep36; mesh payloads from
# meshes/final) and compare it with submitted/track3_submission.parquet (9bc322af). The ep33 component lives under
# /data1/hansol in the original tree: the image must also map /data1/hansol (see docs/BUILD_CONTAINER.md).
exit "${STATUS}"
