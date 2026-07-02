#!/usr/bin/env python3
"""Half-precision matmul GFLOPS (WMMA / TC path). Run from ~/tinygrad with venv active.

  python ~/github/tiny-tests/amd_wmma_bench.py
  python ~/github/tiny-tests/amd_wmma_bench.py --compare-tc
  python ~/github/tiny-tests/amd_wmma_bench.py --both --compare-tc
  python ~/github/tiny-tests/amd_wmma_bench.py --check-wmma --both

Multi-backend / multi-TC runs use subprocess (Device is fixed after first import).
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from dataclasses import replace

TINYGRAD = os.environ.get("TINYGRAD", os.getcwd())
PY = sys.executable
SCRIPT = os.path.abspath(__file__)


def _setup_path() -> None:
  root = os.environ.get("TINYGRAD", os.getcwd())
  if root not in sys.path:
    sys.path.insert(0, root)


def _realized_ast(r):
  from tinygrad import Tensor
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.device import Buffer
  from tinygrad.engine.realize import run_linear

  if not isinstance(r, list):
    r = [r]
  linear, var_vals = Tensor.linear_with_vars(*r)
  run_linear(UOp(Ops.LINEAR, src=linear.src[:-1]), var_vals)
  last_call = linear.src[-1]
  ast = last_call.src[0]
  last_bufs = [s.buffer for s in last_call.src[1:] if s.op is not Ops.BIND]
  bufs = [Buffer(x.device, x.size, x.dtype).allocate() if i < len(ast.src) else x for i, x in enumerate(last_bufs)]
  for b in bufs:
    b.ensure_allocated()
  return ast, bufs


def _has_wmma(pu) -> bool:
  from tinygrad.renderer.isa.amd import AMDOps
  from tinygrad.uop.ops import Ops

  lin = pu.src[1].src
  if any(u.op is Ops.INS and u.arg is AMDOps.WMMA for u in lin):
    return True
  return any(u.op is Ops.WMMA for u in lin)


def _wmma_triggered(n: int) -> bool:
  _setup_path()
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt import Opt, OptOps

  a = Tensor.rand(n, n, dtype=dtypes.half)
  b = Tensor.rand(n, n, dtype=dtypes.half)
  r = a.matmul(b, dtype=dtypes.float)
  ast, _ = _realized_ast(r)
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (0, 0, 1)),)))
  pu = to_program(ast, Device[Device.DEFAULT].renderer)
  return _has_wmma(pu)


def _autogen_uses_tc(n: int) -> bool:
  _setup_path()
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt import OptOps

  a = Tensor.randn(n, n, dtype=dtypes.half).realize()
  b = Tensor.randn(n, n, dtype=dtypes.half).realize()
  ast, _ = _realized_ast(a @ b)
  pu = to_program(ast)
  return any(o.op is OptOps.TC for o in pu.src[0].arg.applied_opts)


def bench(dev: str, sizes: list[int], warmup: int, iters: int, tc: int) -> None:
  _setup_path()
  os.environ["DEV"] = dev
  os.environ["TC"] = str(tc)
  from tinygrad import Device, Tensor, dtypes

  ren = type(Device[Device.DEFAULT].renderer).__name__
  print(f"\n=== DEV={dev} TC={tc} renderer={ren} ===")
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
    tc_note = "autogen TC" if _autogen_uses_tc(n) else "no TC opt"
    print(f"  {n:>5}  {gflops:8.0f} GFLOPS  ({sec*1000:.2f} ms/iter)  half→float  [{tc_note}]")


def _env(dev: str, tc: int | None = None) -> dict[str, str]:
  e = os.environ.copy()
  e["DEV"] = dev
  e["TINYGRAD"] = os.environ.get("TINYGRAD", os.getcwd())
  if tc is not None:
    e["TC"] = str(tc)
  return e


def _spawn(extra: list[str], *, dev: str, tc: int | None = None) -> int:
  cmd = [PY, SCRIPT, "--worker", "--dev", dev, *extra]
  if tc is not None:
    cmd += ["--tc", str(tc)]
  return subprocess.call(cmd, cwd=os.environ.get("TINYGRAD", os.getcwd()), env=_env(dev, tc))


def _worker_main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--worker", action="store_true")
  p.add_argument("--dev", required=True)
  p.add_argument("--tc", type=int, default=1)
  p.add_argument("--sizes", default="1024,2048,4096")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--check-wmma", action="store_true")
  args = p.parse_args(argv)
  if args.check_wmma:
    print(f"DEV={args.dev} forced TC WMMA on 512²: {_wmma_triggered(512)}")
    return
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
  bench(args.dev, sizes, args.warmup, args.iters, args.tc)


def main() -> int:
  if "--worker" in sys.argv:
    _worker_main()
    return 0

  p = argparse.ArgumentParser(description="AMD half matmul GFLOPS (WMMA / TC)")
  p.add_argument("--sizes", default="1024,2048,4096")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--both", action="store_true", help="run DEV=AMD:AMD then DEV=AMD (LLVM)")
  p.add_argument("--compare-tc", action="store_true", help="bench TC=1 then TC=0")
  p.add_argument("--check-wmma", action="store_true", help="forced TC emits WMMA on 512² tile")
  args = p.parse_args()

  sizes_arg = ["--sizes", args.sizes, "--warmup", str(args.warmup), "--iters", str(args.iters)]
  devs = ["AMD:AMD", "AMD"] if args.both else [os.environ.get("DEV", "AMD:AMD")]
  tcs = [1, 0] if args.compare_tc else [int(os.environ.get("TC", "1"))]
  rc = 0

  if args.check_wmma:
    for dev in devs:
      rc = _spawn(["--check-wmma", *sizes_arg], dev=dev) or rc
    return rc

  jobs = [(dev, tc) for dev in devs for tc in tcs]
  if len(jobs) == 1:
    dev, tc = jobs[0]
    bench(dev, [int(x) for x in args.sizes.split(",") if x.strip()], args.warmup, args.iters, tc)
    return 0

  for dev, tc in jobs:
    rc = _spawn(sizes_arg, dev=dev, tc=tc) or rc
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
