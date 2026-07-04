#!/bin/bash
# Super-quick end-to-end smoke test on synthetic data (no MNIST download, no real
# JetClass files, no W&B login needed).  Run from anywhere:
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

export WANDB_MODE=disabled          # never try to reach W&B during the test

echo "=== byte-compile all modules ==="
python -m py_compile cli.py models/*.py data/*.py data/jetclass/*.py utils/*.py experiments/*.py
echo "compile OK"

python scripts/smoke_test.py
