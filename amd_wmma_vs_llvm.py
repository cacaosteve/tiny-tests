#!/usr/bin/env python3
"""Half->float GEMM: ASM (AMD:AMD) vs LLVM (AMD), clean single-process timing.

Run from ~/tinygrad with venv active. Sets DEV itself per backend, so DON'T prefix DEV=:

  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py
  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --sizes 1024,2048,4096
  DEV=AMD:AMD DEBUG=0 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --worker AMD:AMD --sizes 4096

Each backend runs in its own subprocess because Device.DEFAULT is fixed after first import.
Worker exits with os._exit(0) after sync to avoid HCQ teardown timeouts on gfx11.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

SCRIPT = os.path.abspath(__file__)

def _worker(dev: str, sizes: list[int], warmup: int, iters: int) -> None:
  from tinygrad import Tensor, dtypes, Device
  ren = type(Device["AMD"].renderer).__name__
  for n in sizes:
    a = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    b = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    for _ in range(warmup): a.matmul(b, dtype=dtypes.float).realize()
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): a.matmul(b, dtype=dtypes.float).realize()
    Device["AMD"].synchronize()
    sec = (time.perf_counter() - t0) / iters
    print(f"  DEV={dev:<8s} [{ren:<12s}]  {n:>5}  {2*n**3/sec/1e9:8.0f} GFLOPS  ({sec*1e3:7.2f} ms/iter)", flush=True)
  Device["AMD"].synchronize()
  os._exit(0)

def main() -> int:
  p = argparse.ArgumentParser(description="half->float GEMM ASM vs LLVM")
  p.add_argument("--sizes", default="1024,2048,4096")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--worker", help=argparse.SUPPRESS)
  args = p.parse_args()
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

  if args.worker:
    _worker(args.worker, sizes, args.warmup, args.iters)
    return 0

  rc = 0
  for dev in ("AMD:AMD", "AMD"):
    print(f"\n=== {dev} ===", flush=True)
    env = os.environ.copy()
    env["DEV"] = dev
    env["DEBUG"] = "0"
    cmd = [sys.executable, SCRIPT, "--worker", dev,
           "--sizes", args.sizes, "--warmup", str(args.warmup), "--iters", str(args.iters)]
    rc = subprocess.call(cmd, env=env) or rc
  return rc

if __name__ == "__main__":
  raise SystemExit(main())
