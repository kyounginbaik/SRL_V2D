#!/usr/bin/env bash
# Install only the CPU learned-policy/replay/packing dependencies.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd -- "$script_dir/.." && pwd)"
repo_root="$(cd -- "$experiment_root/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3.11}"
if [[ ! -x "$experiment_root/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$experiment_root/.venv"
fi
"$experiment_root/.venv/bin/python" -m pip install -r "$script_dir/requirements_policy.txt"
dexagent_checkout="$repo_root/third_party/DexAgent"
dexagent_commit=e29bdad8ed3e95b2c9ee2eea9044a6b4115ec633
if [[ ! -e "$dexagent_checkout" ]]; then
  mkdir -p "$repo_root/third_party"
  git clone https://github.com/dexagent1/DexAgent.git "$dexagent_checkout"
  git -C "$dexagent_checkout" checkout --detach "$dexagent_commit"
fi
if [[ "$(git -C "$dexagent_checkout" rev-parse HEAD)" != "$dexagent_commit" ]]; then
  echo 'Existing DexAgent checkout differs from the tested commit; it was left untouched.' >&2
  exit 2
fi
if [[ -n "$(git -C "$dexagent_checkout" status --porcelain --untracked-files=no)" ]]; then
  echo 'DexAgent has tracked local modifications; reproducibility requires the tested source.' >&2
  exit 2
fi
echo "CPU policy environment ready: $experiment_root/.venv/bin/python"
