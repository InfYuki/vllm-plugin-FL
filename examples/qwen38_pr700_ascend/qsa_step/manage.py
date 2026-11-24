# SPDX-License-Identifier: Apache-2.0
"""Install, inspect or undo the BF16 scheduling backport in the pinned checkout."""

import argparse
from pathlib import Path
import subprocess


PIN = "24487f4f48ee99cf45d86d1c504e645549b8c1ef"
ROOT = Path(__file__).resolve().parent
MODULE = "vllm_ascend/models/qwen4_exp/step_metadata.py"


def patch_bytes():
    source = (ROOT / "step_metadata.py").read_text(encoding="utf-8")
    lines = (source.rstrip("\n") + "\n").splitlines(keepends=True)
    header = (
        f"diff --git a/{MODULE} b/{MODULE}\nnew file mode 100644\n"
        f"--- /dev/null\n+++ b/{MODULE}\n@@ -0,0 +1,{len(lines)} @@\n"
    )
    existing = (ROOT / "runtime.patch").read_text(encoding="utf-8")
    return ((existing.rstrip("\n") + "\n") + header + "".join("+" + line for line in lines)).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "revert"))
    parser.add_argument("--repo", type=Path, default=Path("/vllm-workspace/vllm-ascend"))
    args = parser.parse_args()
    git = ["git", "-C", str(args.repo)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        parser.exit(1, f"Unexpected vllm-ascend revision {head}. Expected {PIN}.\n")
    patch = patch_bytes()
    apply = [*git, "apply", "--whitespace=nowarn"]
    forward = subprocess.run([*apply, "--check"], input=patch, capture_output=True)
    reverse = subprocess.run([*apply, "--check", "--reverse"], input=patch, capture_output=True)
    if (forward.returncode == 0) == (reverse.returncode == 0):
        parser.exit(1, "Could not establish patch state.\n" + forward.stderr.decode() + reverse.stderr.decode())
    installed = reverse.returncode == 0
    if args.action != "check" and installed != (args.action == "apply"):
        opts = ["--reverse"] if args.action == "revert" else []
        subprocess.run([*apply, *opts], input=patch, check=True)
        installed = not installed
        verify = ["--reverse"] if installed else []
        subprocess.run([*apply, "--check", *verify], input=patch, check=True)
    print(f"{'applied' if installed else 'original'} {head}")


if __name__ == "__main__":
    main()
