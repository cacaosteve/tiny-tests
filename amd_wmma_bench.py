#!/usr/bin/env python3
"""Half-precision matmul GFLOPS (WMMA / TC path). Run from ~/tinygrad with venv active.

  DEV=AMD:AMD python ~/github/tiny-tests/amd_wmma_bench.py
  DEV=AMD:AMD python ~/github/tiny-tests/amd_wmma_bench.py --compare-tc
  python ~/github/tiny-tests/amd_wmma_bench.py --both --compare-tc
  python ~/github/tiny-tests/amd_wmma_bench.py --check-wmma
"""
from __future__ import annotations
import argparse, os, time

def _wmma_triggered(n: int) -> bool:
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.uop.ops import Ops
  from test.helpers import replace_opts
  from test.backend.test_linearizer import helper_realized_ast
  from test.opt.test_tensor_cores import _wmma_uops_in_program

  a = Tensor.rand(n, n, dtype=dtypes.half)
  b = Tensor.rand(n, n, dtype=dtypes.half)
  r = a.matmul(b, dtype=dtypes.float)
  ast, _ = helper_realized_ast(r)
  ast = replace_opts(ast, [Opt(OptOps.TC, 0, (0, 0, 1))])
  pu = to_program(ast, Device[Device.DEFAULT].renderer)
  return len(_wmma_uops_in_program(pu)) > 0

def bench(dev: str, sizes: list[int], warmup: int, iters: int, tc: int) -> None:
  os.environ["DEV"] = dev
  os.environ["TC"] = str(tc)
  from tinygrad import Tensor, dtypes, Device

  print(f"\n=== DEV={dev} TC={tc} ({Device.DEFAULT}) ===")
  for n in sizes:
    a = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    b = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    for _ in range(warmup):
      (a @ b).realize()
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
      (a @ b).realize()
    Device["AMD"].synchronize()
    sec = (time.perf_counter() - t0) / iters
    gflops = 2 * n**3 / sec / 1e9
    print(f"  {n:>5}  {gflops:8.0f} GFLOPS  ({sec*1000:.2f} ms/iter)  half→float")

def main() -> None:
  p = argparse.ArgumentParser(description="AMD half matmul GFLOPS (WMMA / TC)")
  p.add_argument("--sizes", default="1024,2048,4096", help="comma-separated N for NxN matmul")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--both", action="store_true", help="run DEV=AMD:AMD then DEV=AMD (LLVM)")
  p.add_argument("--compare-tc", action="store_true", help="bench TC=1 then TC=0 on same dev")
  p.add_argument("--check-wmma", action="store_true", help="print whether forced TC uses WMMA on 512² tile")
  args = p.parse_args()
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
  devs = ["AMD:AMD", "AMD"] if args.both else [os.environ.get("DEV", "AMD:AMD")]

  if args.check_wmma:
    for dev in devs:
      os.environ["DEV"] = dev
      from tinygrad import Device  # noqa: F401 — sets backend from DEV
      print(f"DEV={dev} forced TC WMMA on 512²: {_wmma_triggered(512)}")

  for dev in devs:
    if args.compare_tc:
      bench(dev, sizes, args.warmup, args.iters, tc=1)
      bench(dev, sizes, args.warmup, args.iters, tc=0)
    else:
      bench(dev, sizes, args.warmup, args.iters, tc=int(os.environ.get("TC", "1")))

if __name__ == "__main__":
  main()
