#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x venv/bin/python ] || { echo "Run ./install.sh first"; exit 1; }
export ANIMA_RUNNER=1
while true; do
  venv/bin/python app.py
  code=$?
  [ "$code" -eq 3 ] || exit "$code"
  export ANIMA_NO_BROWSER=1   # language switch: restart without a new browser tab
done
