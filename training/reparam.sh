#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/reparam.py" "$@"
