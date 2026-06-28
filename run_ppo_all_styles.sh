#!/usr/bin/env bash
# Train PPO sequentially on all four styles.
#
#   weapon / camouflage / bypass : two runs each — CORRIDOR_DIR "upper" then "lower"
#   daredevil                    : one run        — CORRIDOR_DIR None
#
#   ent_coef : camouflage 0.02, weapon 0.02, bypass 0.01, daredevil 0.018
#
# Every other PPO arg keeps its default from ppo/util.py. MODE and CORRIDOR_DIR
# are module-level globals that the argparse defaults AND the checkpoint filename
# read from (and --bypass_corridor's choices reject the string "None"), so they
# are set by editing ppo/util.py in place before each run. The file is backed up
# and restored on exit, so the repo copy is left untouched.
#
# Run it yourself from anywhere:
#   bash run_ppo_all_styles.sh
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL="$REPO/ppo/util.py"

# (mode | corridor | ent_coef), executed in this order
CONFIGS=(
  "weapon|upper|0.02"
  "weapon|lower|0.02"
  "camouflage|upper|0.02"
  "camouflage|lower|0.02"
  "bypass|upper|0.01"
  "bypass|lower|0.01"
  "daredevil|None|0.018"
)

# restore the original util.py when the script exits (normal end or Ctrl-C)
cp "$UTIL" "$UTIL.bak"
restore() { mv -f "$UTIL.bak" "$UTIL" 2>/dev/null || true; }
trap restore EXIT
trap 'exit 130' INT TERM

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r mode corridor ent <<< "$cfg"

  if [ "$corridor" = "None" ]; then
    corridor_repl='CORRIDOR_DIR = None'
  else
    corridor_repl="CORRIDOR_DIR = \"$corridor\""
  fi

  # set the globals the defaults + checkpoint names derive from
  sed -i -E "s|^MODE = .*|MODE = \"$mode\"|"        "$UTIL"
  sed -i -E "s|^CORRIDOR_DIR = .*|$corridor_repl|"  "$UTIL"

  # drop stale bytecode so the edited globals are picked up
  rm -rf "$REPO/ppo/__pycache__"

  echo "=================================================================="
  echo ">>> PPO  mode=$mode  corridor=$corridor  ent_coef=$ent"
  echo ">>> checkpoint -> multi_style_env_hard_${mode}_${corridor}_00_PPO.pt"
  echo "=================================================================="

  ( cd "$REPO" && PYTHONPATH="$REPO:$REPO/ppo" python ppo/experiment.py --ent_coef "$ent" )
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "!! FAILED  mode=$mode corridor=$corridor  (exit $status) — continuing"
  else
    echo "<< done    mode=$mode corridor=$corridor"
  fi
done

echo "All runs finished."