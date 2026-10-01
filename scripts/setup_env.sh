#!/usr/bin/env bash
# WSL2 provisioning for ZeroPrefill.
#
# Run as root, no sudo: `wsl -d Ubuntu-24.04 -u root -e bash scripts/setup_env.sh`
# Sudo is deliberately avoided because it blocks on a password prompt with no
# TTY, which silently hangs the whole provisioning run.
#
# The venv goes in /root/.venvs rather than the repo: the repo lives on /mnt/e,
# a 9p mount that is several times slower for the many small safetensors reads
# the restore path performs.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO_WSL=/mnt/e/inference-engineering/projects/ZeroPrefill
VENV=/root/.venvs/zeroprefill

echo "==> distro"
uname -r

echo "==> apt: python3.12-venv + pip"
apt-get update -qq 2>&1 | tail -3
apt-get install -y -qq python3.12-venv python3-pip curl 2>&1 | tail -5

echo "==> venv at $VENV"
if [ ! -d "$VENV" ]; then
  python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel 2>&1 | tail -2

echo "==> installing vllm (large download, several minutes)"
python -m pip install vllm safetensors transformers 2>&1 | tail -8

echo "==> smoke test"
python - <<'PY'
import torch, vllm
print("vllm :", vllm.__version__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("avail:", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print(f"gpu  : {p.name} sm_{p.major}{p.minor} {p.total_memory/1024**3:.2f}GiB")
cc = p.major * 10 + p.minor
assert cc >= 75, f"vLLM needs compute capability >= 7.5, got {cc}"
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
assert torch.isfinite((x @ x).float()).all()
print("fp16 matmul on device: OK")
PY

mkdir -p /root/zp-logs

cat <<'EOF'

Setup done.

  Logs and the connector summary live in /root/zp-logs (NOT /tmp, which is
  tmpfs in WSL2 and is wiped whenever the distro stops).

  Cache root: /root/zeroprefill-cache
  Repo:      /mnt/e/inference-engineering/projects/ZeroPrefill

Next:
  wsl -d Ubuntu-24.04 -u root -e bash <repo>/scripts/verify.sh
  wsl -d Ubuntu-24.04 -u root -e bash <repo>/scripts/offline_e2e.sh
EOF
