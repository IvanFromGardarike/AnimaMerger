#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -x venv/bin/python ] || "$PY" -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
  || venv/bin/python -m pip install torch
venv/bin/python -m pip install -r requirements.txt
echo "Done. Start with ./run.sh"
