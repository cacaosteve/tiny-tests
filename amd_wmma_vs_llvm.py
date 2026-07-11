#!/usr/bin/env python3
"""Half GEMM: auto ASM vs LLVM vs hand WMMA. Clean single-process timing.

Run from ~/tinygrad with venv active. Sets DEV itself per backend, so DON'T prefix DEV=:

  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py
  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --sizes 1024,2048,4096
  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --hand --sizes 4096

Each backend runs in its own subprocess because Device.DEFAULT is fixed after first import.
Worker exits with os._exit(0) after sync to avoid HCQ teardown timeouts on gfx11.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

SCRIPT = os.path.abspath(__file__)

def _worker(dev: str, sizes: list[int], warmup: int, iters: int, hand: bool) -> None:
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.engine.realize import run_linear
  ren = type(Device["AMD"].renderer).__name__
  label = "HAND" if hand else ren
  for n in sizes:
    a = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    b = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    if hand:
      from extra.gemm.rdna3_asm_wmma_gemm import can_use_rdna3_wmma_gemm, rdna3_wmma_gemm
      if not can_use_rdna3_wmma_gemm(a, b):
        print(f"  DEV={dev:<8s} [{label:<12s}]  {n:>5}  SKIP (can_use=False)", flush=True)
        continue
      # Schedule once; time only run_linear (same as amd_hand_wmma / run_matmul).
      c = rdna3_wmma_gemm(a, b, out_dtype=dtypes.float)
      linear = c.schedule_linear()
      def run():
        run_linear(linear)
    else:
      def run(): a.matmul(b, dtype=dtypes.float).realize()
    for _ in range(warmup): run()
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): run()
    Device["AMD"].synchronize()
    sec = (time.perf_counter() - t0) / iters
    print(f"  DEV={dev:<8s} [{label:<12s}]  {n:>5}  {2*n**3/sec/1e9:8.0f} GFLOPS  ({sec*1e3:7.2f} ms/iter)", flush=True)
  Device["AMD"].synchronize()
  os._exit(0)

def main() -> int:
  p = argparse.ArgumentParser(description="half GEMM ASM vs LLVM vs hand WMMA")
  p.add_argument("--sizes", default="1024,2048,4096")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--hand", action="store_true", help="also time hand WMMA custom_kernel path")
  p.add_argument("--worker", help=argparse.SUPPRESS)
  p.add_argument("--worker-hand", action="store_true", help=argparse.SUPPRESS)
  args = p.parse_args()
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

  if args.worker:
    root = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))
    if root not in sys.path: sys.path.insert(0, root)
    _worker(args.worker, sizes, args.warmup, args.iters, args.worker_hand)
    return 0

  rc = 0
  # AMD:AMD codegen path disables matmul hand dispatch so we still see the auto gap.
  # HAND column uses the custom_kernel directly (float out, matches matmul dtype=float).
  jobs = [("AMD:AMD", False, {"RDNA3_WMMA_GEMM": "0"}), ("AMD", False, {})]
  if args.hand: jobs.append(("AMD:AMD", True, {}))
  for dev, hand, extra_env in jobs:
    tag = "HAND" if hand else dev
    print(f"\n=== {tag} ===", flush=True)
    env = os.environ.copy()
    env["DEV"] = dev
    env["DEBUG"] = "0"
    env.setdefault("TINYGRAD", os.path.expanduser("~/tinygrad"))
    env.update(extra_env)
    cmd = [sys.executable, SCRIPT, "--worker", dev,
           "--sizes", args.sizes, "--warmup", str(args.warmup), "--iters", str(args.iters)]
    if hand: cmd.append("--worker-hand")
    rc = subprocess.call(cmd, env=env) or rc
  return rc

if __name__ == "__main__":
  raise SystemExit(main())
