#!/usr/bin/env bash
# Launcher that runs `kesonleadas-claude.py` with the project's `.venv` Python.
VENV_PY="${PWD}/.venv/bin/python"
if [ -x "$VENV_PY" ]; then
  exec "$VENV_PY" "${PWD}/kesonleadas/kesonleadas-claude.py" "$@"
else
  echo "Error: virtualenv python not found at .venv/bin/python"
  echo "Create it and install deps:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
