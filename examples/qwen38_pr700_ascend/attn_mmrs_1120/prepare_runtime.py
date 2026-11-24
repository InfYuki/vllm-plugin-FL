# SPDX-License-Identifier: Apache-2.0
"""Build an isolated import tree from the accepted MC2 runtime and its existing extensions."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
REPO = Path("/vllm-workspace/vllm-ascend")
PACKAGE = Path("vllm_ascend")
MODEL = PACKAGE / "models/qwen4_exp"
ORIGINAL = {
    "model.py": "3d70b03adbf7872e6bbf39cf4b9d56661de130625bbe2497ca8f2477dc1ce467",
    "model_state.py": "28ac100e31ad113cd43722cc11fbbd4b342810d02420e3a359a2f4ae18faa13e",
}
PATCH = HERE / "runtime.patch"


def verify(runtime):
    subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=runtime, check=True)
    assert (runtime / MODEL / "attention_mmrs.py").read_bytes() == (HERE / "attention_mmrs.py").read_bytes()
    for name in (*ORIGINAL, "attention_mmrs.py"):
        module = runtime / MODEL / name
        compile(module.read_bytes(), str(module), "exec")


def main():
    for name, digest in ORIGINAL.items():
        if hashlib.sha256((REPO / MODEL / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Accepted MC2 source differs at {REPO / MODEL / name}")
    final = HERE / "runtime"
    if final.exists():
        verify(final)
        return
    with tempfile.TemporaryDirectory(dir=HERE, prefix=".prepare_") as work:
        runtime = Path(work) / "runtime"
        for path, skip in ((PACKAGE, {"models"}), (PACKAGE / "models", {"qwen4_exp"}),
                           (MODEL, {*ORIGINAL, "attention_mmrs.py"})):
            dest = runtime / path
            dest.mkdir(parents=True)
            for item in (REPO / path).iterdir():
                if item.name not in skip and item.name != "__pycache__":
                    (dest / item.name).symlink_to(item)
        for name in ORIGINAL:
            shutil.copyfile(REPO / MODEL / name, runtime / MODEL / name)
        shutil.copyfile(HERE / "attention_mmrs.py", runtime / MODEL / "attention_mmrs.py")
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=runtime, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=runtime, check=True)
        verify(runtime)
        runtime.rename(final)


if __name__ == "__main__":
    main()
