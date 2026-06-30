#!/usr/bin/env bash
# Train PPO sequentially on the four styles of the four-style env.
#
#   styles: weapon, camouflage, daredevil, portal  (one run each)
#
# Unlike the multi-style runner there is no upper/lower corridor split, so the
# ONLY per-run change is MODE in ppo/util.py. CORRIDOR_DIR is set to None once
# (the four-style env has no corridor). Every other PPO arg keeps its default.
# ppo/util.py is backed up and restored on exit, so the repo copy is untouched.
#
# Run it yourself:
#   bash run_ppo_four_styles.sh
#
# NOTE: this only sets MODE/CORRIDOR_DIR and launches ppo/experiment.py. For it
# to actually train on MiniGridFourStyles, the PPO env builder (ppo/make_env +
# configs) must construct the four-style env for these modes — otherwise it will
# train on whatever env make_env currently builds.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL="$REPO/ppo/util.py"

STYLES=( camouflage daredevil portal)

# restore the original util.py when the script exits (normal end or Ctrl-C)
cp "$UTIL" "$UTIL.bak"
restore() { mv -f "$UTIL.bak" "$UTIL" 2>/dev/null || true; }
trap restore EXIT
trap 'exit 130' INT TERM

# four-style env has no corridor — set CORRIDOR_DIR=None once
sed -i -E "s|^CORRIDOR_DIR = .*|CORRIDOR_DIR = None|" "$UTIL"

for mode in "${STYLES[@]}"; do
  sed -i -E "s|^MODE = .*|MODE = \"$mode\"|" "$UTIL"
  rm -rf "$REPO/ppo/__pycache__"

  echo "=================================================================="
  echo ">>> PPO  mode=$mode"
  echo "=================================================================="

  ( cd "$REPO" && PYTHONPATH="$REPO:$REPO/ppo" python ppo/experiment.py )
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "!! FAILED  mode=$mode  (exit $status) — continuing"
  else
    echo "<< done    mode=$mode"
  fi
done

echo "All runs finished."