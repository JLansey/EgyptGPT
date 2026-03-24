#!/usr/bin/env bash
# CUDA smoke test as the researcher user.
# When run from the researcher shell directly, pass --direct to skip sudo.
set -euo pipefail

CMD='python -c "import torch; print(\"CUDA:\", torch.cuda.is_available()); print(\"devices:\", torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no gpu\")"'

ENV_VARS=(
  "PATH=/usr/local/nvidia/bin:$PATH"
  "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
)

if [[ "${1:-}" == "--direct" ]]; then
  env "${ENV_VARS[@]}" bash -c "$CMD"
else
  sudo -u researcher env \
    HOME=/home/researcher \
    "${ENV_VARS[@]}" \
    bash -c "$CMD"
fi
