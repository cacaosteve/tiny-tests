#!/usr/bin/env python3
"""Matmul GFLOPS sweep. Run from ~/tinygrad with venv active.

  DEV=AMD:AMD python ~/github/tiny-tests/amd_gemm_bench.py
  DEV=AMD     python ~/github/tiny-tests/amd_gemm_bench.py
  python ~/github/tiny-tests/amd_gemm_bench.py --both
  python ~/github/tiny-tests/amd_gemm_bench.py --sizes 512,1024 --warmup 5
"""
from __future__ import annotations
import argparse, os, time

def bench(dev: str, sizes: list[int], warmup: int, iters: int) -> None:
  os.environ["DEV"] = dev
  from tinygrad import Tensor, Device

  print(f"\n=== DEV={dev} ({Device.DEFAULT}) ===")
  for n in sizes:
    a = Tensor.randn(n, n, device="AMD").realize()
    b = Tensor.randn(n, n, device="AMD").realize()
    for _ in range(warmup):
      (a @ b).realize()
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
      (a @ b).realize()
    Device["AMD"].synchronize()
    sec = (time.perf_counter() - t0) / iters
    gflops = 2 * n**3 / sec / 1e9
    print(f"  {n:>5}  {gflops:8.0f} GFLOPS  ({sec*1000:.2f} ms/iter)")

def main() -> None:
  p = argparse.ArgumentParser(description="AMD matmul GFLOPS benchmark")
  p.add_argument("--sizes", default="512,1024,2048", help="comma-separated N for NxN matmul")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--both", action="store_true", help="run DEV=AMD:AMD then DEV=AMD (LLVM)")
  args = p.parse_args()
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

  if args.both:
    bench("AMD:AMD", sizes, args.warmup, args.iters)
    bench("AMD", sizes, args.warmup, args.iters)
  else:
    dev = os.environ.get("DEV", "AMD:AMD")
    bench(dev, sizes, args.warmup, args.iters)

if __name__ == "__main__":
  main()
