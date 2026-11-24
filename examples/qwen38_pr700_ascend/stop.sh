#!/usr/bin/env bash
set -euo pipefail
if (( $# )); then
    echo "Usage: bash /workspace/stop.sh" >&2
    exit 2
fi
python - <<'PY'
import os
from pathlib import Path
import signal

work = Path("/workspace")
run = (work / "latest_service").resolve(strict=True)
if run.parent != work or not run.name.startswith("service_"):
    raise SystemExit(f"Unexpected service directory: {run}")
pid = int((run / "server.pid").read_text())
proc = Path(f"/proc/{pid}")
if not proc.exists():
    print("The recorded service has stopped.")
    raise SystemExit(0)
args = (proc / "cmdline").read_bytes().split(b"\0")
if ((proc / "cwd").resolve() != run or b"serve" not in args
        or b"/models/Qwen3.8-Flash-Next" not in args
        or not any(args[i:i + 2] == [b"--port", b"8098"] for i in range(len(args) - 1))):
    raise SystemExit("The recorded process does not match this Qwen service.")
os.kill(pid, signal.SIGTERM)
print(f"Sent SIGTERM to Qwen service {pid} on port 8098.")
PY
