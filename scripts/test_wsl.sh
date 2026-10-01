#!/usr/bin/env bash
# Run the repo's unit tests inside the WSL venv (where vLLM lives).
set -uo pipefail
cd /mnt/e/inference-engineering/projects/ZeroPrefill
source /root/.venvs/zeroprefill/bin/activate
python -m pip install -q pytest 2>&1 | tail -2
python -m pytest tests/ -v 2>&1 | tail -30
