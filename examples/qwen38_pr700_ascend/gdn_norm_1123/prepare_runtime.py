# SPDX-License-Identifier: Apache-2.0
"""Apply vllm-ascend PR 21678 to the accepted attention MMRS runtime."""

import ast
from hashlib import sha256
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
PACKAGE = Path("vllm_ascend")
BASE = Path("/workspace/attn_mmrs_1120/runtime") / PACKAGE
SOURCE = Path("/vllm-workspace/vllm-ascend") / PACKAGE
MODULE = Path("device/device_op.py")
ORIGINAL = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
METHOD = '''    def fla_gdn_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        scale: float,
        prebuilt_meta,
        fused_fwd,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the A2/A3 Phase6 FLA NPU GDN prefill kernel."""
        # l2norm is done inside the kernel (use_qk_l2norm_in_kernel=True),
        # so the external l2norm_fwd calls are removed.
        head_first_qkv = tuple(x.transpose(1, 2) for x in (q, k, v))
        if all(x.is_contiguous() for x in head_first_qkv):
            q, k, v = head_first_qkv
            layout = "BNSD"
        else:
            q, k, v = (x.contiguous() for x in (q, k, v))
            layout = "BSND"
        g = g.to(torch.float32).contiguous()
        # fused_gdn_gating already emits sigmoid(beta), so beta is forwarded
        # as-is and the kernel must not apply the sigmoid a second time.
        beta = beta.to(v.dtype).contiguous()
        initial_state = initial_state.contiguous()

        cu_seqlens = prebuilt_meta.cu_seqlens_host
        chunk_indices = prebuilt_meta.chunk_indices_chunk64_host
        keep_meta = prebuilt_meta.keep_meta
        initial_state_kern = initial_state
        if keep_meta is not None:
            cu_seqlens = prebuilt_meta.cu_seqlens_kern
            initial_state_kern = initial_state[keep_meta]

        output, final_state, *_ = fused_fwd(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state_kern,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            layout=layout,
            use_exp2=False,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            allow_neg_eigval=False,
            disable_recompute=False,
            state_v_first=True,
        )
        if keep_meta is not None:
            full_final_state = initial_state.clone()
            full_final_state[keep_meta] = final_state
            final_state = full_final_state
        return output, final_state'''


def candidate():
    path = SOURCE / MODULE
    data = path.read_bytes()
    if sha256(data).hexdigest() != ORIGINAL or not path.samefile(BASE / MODULE):
        raise RuntimeError(f"Unexpected MMRS source at {path}")
    source = data.decode("utf-8")
    owner, = (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
              and n.name == "BaseDeviceAdaptor")
    method, = (n for n in owner.body if isinstance(n, ast.FunctionDef)
               and n.name == "fla_gdn_prefill")
    lines = source.splitlines(keepends=True)
    lines[method.lineno - 1:method.end_lineno] = [METHOD + "\n"]
    result = "".join(lines)
    compile(result, str(path), "exec")
    return result.encode("utf-8")


def inherited(folder, private):
    for source in (BASE / folder).iterdir():
        if source.name not in (private, "__pycache__"):
            yield source


def verify(runtime, expected):
    pkg = runtime / PACKAGE
    target = pkg / MODULE
    if target.resolve() != target or target.read_bytes() != expected:
        raise RuntimeError(f"Unexpected private module at {target}")
    for folder, private in (("", "device"), ("device", "device_op.py")):
        for source in inherited(folder, private):
            if not (pkg / folder / source.name).samefile(source):
                raise RuntimeError(f"Unexpected inherited module for {source}")


def main():
    expected = candidate()
    final = HERE / "runtime"
    if final.exists():
        verify(final, expected)
        return
    with tempfile.TemporaryDirectory(dir=HERE, prefix=".prepare_") as work:
        runtime = Path(work) / "runtime"
        pkg = runtime / PACKAGE
        for folder, private in (("", "device"), ("device", "device_op.py")):
            dest = pkg / folder
            dest.mkdir(parents=True)
            for source in inherited(folder, private):
                (dest / source.name).symlink_to(source)
        (pkg / MODULE).write_bytes(expected)
        verify(runtime, expected)
        runtime.rename(final)


if __name__ == "__main__":
    main()
