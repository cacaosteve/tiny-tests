#!/usr/bin/env python3
"""Time half GEMM on AMD:AMD — explicit hand WMMA vs codegen Tensor.matmul.

Hand WMMA is not wired into Tensor.matmul; call rdna3_wmma_gemm explicitly.

Run from ~/tinygrad with venv active:

  python ~/github/tiny-tests/amd_wmma_dispatch.py
  python ~/github/tiny-tests/amd_wmma_dispatch.py --sizes 1024,2048,4096
  python ~/github/tiny-tests/amd_wmma_dispatch.py --codegen   # Tensor.matmul only
  python ~/github/tiny-tests/amd_wmma_dispatch.py --check
"""
from __future__ import annotations
import argparse, os, sys, time

def main() -> int:
  p = argparse.ArgumentParser(description="AMD:AMD hand WMMA vs codegen matmul bench")
  p.add_argument("--sizes", default="1024,2048,4096")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--codegen", action="store_true", help="use Tensor.matmul (codegen WMMA), not hand kernel")
  p.add_argument("--no-dispatch", action="store_true", help=argparse.SUPPRESS)  # old flag alias
  p.add_argument("--check", action="store_true", help="correctness vs numpy at 512")
  args = p.parse_args()
  if args.no_dispatch: args.codegen = True

  root = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))
  if root not in sys.path: sys.path.insert(0, root)
  os.environ["DEV"] = "AMD:AMD"
  os.environ["DEBUG"] = "0"

  from tinygrad import Tensor, dtypes, Device
  from tinygrad.engine.realize import run_linear
  from extra.gemm.rdna3_asm_wmma_gemm import can_use_rdna3_wmma_gemm, rdna3_wmma_gemm

  ren = type(Device["AMD"].renderer).__name__
  mode = "codegen" if args.codegen else "hand"
  print(f"DEV=AMD:AMD [{ren}] mode={mode}", flush=True)

  if args.check:
    import numpy as np
    n = 512
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)).astype(np.float16)
    B = rng.standard_normal((n, n)).astype(np.float16)
    ref = A.astype(np.float32) @ B.astype(np.float32)
    a, b = Tensor(A), Tensor(B)
    got = (a.matmul(b, dtype=dtypes.float) if args.codegen else rdna3_wmma_gemm(a, b, out_dtype=dtypes.float)).realize().numpy()
    err = float(np.max(np.abs(got - ref)))
    print(f"check {n}: max abs err={err:.4e}  can_use={can_use_rdna3_wmma_gemm(a,b)}", flush=True)
    if err > 0.05: raise SystemExit(f"FAIL err={err}")
    print("[ok]", flush=True)

  for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
    a = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    b = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    print(f"  n={n} can_use={can_use_rdna3_wmma_gemm(a,b)}", flush=True)
    c = a.matmul(b, dtype=dtypes.float) if args.codegen else rdna3_wmma_gemm(a, b, out_dtype=dtypes.float)
    linear = c.schedule_linear()
    for _ in range(args.warmup): run_linear(linear)
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters): run_linear(linear)
    Device["AMD"].synchronize()
    sec = (time.perf_counter() - t0) / args.iters
    print(f"  {n:>5}  {2*n**3/sec/1e9:8.0f} GFLOPS  ({sec*1e3:7.2f} ms/iter)", flush=True)
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
