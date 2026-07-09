#!/usr/bin/env python3
"""Half-precision matmul GFLOPS (WMMA / TC path). Run from ~/tinygrad with venv active.

  python ~/github/tiny-tests/amd_wmma_bench.py --inspect --sizes 4096
  python ~/github/tiny-tests/amd_wmma_vs_llvm.py          # ASM vs LLVM timing (preferred)
  python ~/github/tiny-tests/amd_wmma_bench.py --check --sizes 512,1024,4096
  python ~/github/tiny-tests/amd_wmma_bench.py --both --compare-tc

Do not prefix DEV= when using --both / --check (each subprocess sets DEV itself).

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


def _schedule_ast(r):
  from tinygrad import Tensor
  if not isinstance(r, list):
    r = [r]
  return r[0].schedule_linear().src[-1].src[0]


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
  ast = _schedule_ast(a.matmul(b, dtype=dtypes.float))
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (0, 0, 1)),)))
  pu = to_program(ast, Device[Device.DEFAULT].renderer)
  return _has_wmma(pu)


def _matmul_expr(a, b):
  from tinygrad import dtypes
  return a.matmul(b, dtype=dtypes.float)


def _program_stats(n: int, tc: int | None = None):
  _setup_path()
  from tinygrad import Tensor, dtypes, Device, Context
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt import OptOps
  from tinygrad.device import CompileError
  from tinygrad.renderer.isa.amd import AMDOps
  from tinygrad.uop.ops import Ops

  ctx = {"TC": tc} if tc is not None else {}
  with Context(**ctx):
    a = Tensor.empty(n, n, dtype=dtypes.half, device="AMD")
    b = Tensor.empty(n, n, dtype=dtypes.half, device="AMD")
    ast = _schedule_ast(_matmul_expr(a, b))
    ren = Device[Device.DEFAULT].renderer
    try:
      prg = to_program(ast, ren)
    except CompileError as e:
      return None, False, 0, 0, str(e)
  lin = list(prg.src[1].src)
  wmma = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.WMMA)
  mulacc = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.MULACC)
  uses_tc = any(o.op is OptOps.TC for o in prg.src[0].arg.applied_opts)
  return prg, uses_tc, wmma, mulacc, None


def _autogen_uses_tc(n: int) -> bool:
  return _program_stats(n)[1]


def _tc_tag(n: int, tc: int) -> str:
  prg, uses_tc, wmma, _, err = _program_stats(n, tc)
  if err:
    return f"compile failed: {err}"
  note = "autogen TC" if uses_tc else "no TC opt"
  ren = type(prg.arg).__name__ if prg else ""
  if wmma:
    return f"{note}, WMMA={wmma}"
  _setup_path()
  from tinygrad import Device
  if type(Device[Device.DEFAULT].renderer).__name__ == "HIPRenderer":
    return f"{note}, WMMA=llvm"
  return f"{note}, WMMA=0"


def inspect(dev: str, sizes: list[int], tc: int) -> None:
  _setup_path()
  os.environ["DEV"] = dev
  os.environ["TC"] = str(tc)
  from tinygrad import Device

  ren = type(Device[Device.DEFAULT].renderer).__name__
  print(f"\n=== inspect DEV={dev} TC={tc} renderer={ren} ===")
  for n in sizes:
    prg, uses_tc, wmma, mulacc, err = _program_stats(n, tc)
    if err:
      print(f"  {n:>5}  COMPILE ERROR: {err}")
      continue
    opts = prg.src[0].arg.applied_opts
    loc, glob = prg.arg.local_size, prg.arg.global_size
    est = prg.src[0].arg.estimates
    from tinygrad.renderer.isa.amd import AMDOps
    from tinygrad.uop.ops import Ops
    lin = list(prg.src[1].src)
    spill = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.SPILL)
    fill = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.FILL)
    lload = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.LLOAD)
    lstore = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.LSTORE)
    barrier = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.BARRIER)
    print(f"  {n:>5}  TC={uses_tc}  WMMA_ins={wmma}  MULACC={mulacc}  SPILL={spill}  FILL={fill}")
    print(f"        LLOAD={lload}  LSTORE={lstore}  BARRIER={barrier}")
    print(f"        opts={opts}")
    print(f"        local={loc}  global={glob}  est_ops={est.ops}  est_mem={est.mem}")


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
      _matmul_expr(a, b).realize()
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
      _matmul_expr(a, b).realize()
    Device["AMD"].synchronize()
    sec = (time.perf_counter() - t0) / iters
    gflops = 2 * n**3 / sec / 1e9
    print(f"  {n:>5}  {gflops:8.0f} GFLOPS  ({sec*1000:.2f} ms/iter)  half→float")


def check(dev: str, sizes: list[int], tc: int) -> None:
  _setup_path()
  os.environ["DEV"] = dev
  os.environ["TC"] = str(tc)
  from tinygrad import Device, Tensor, dtypes

  ren = type(Device[Device.DEFAULT].renderer).__name__
  print(f"\n=== check DEV={dev} TC={tc} renderer={ren} ===")
  ok = True
  for n in sizes:
    a = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    b = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    out = _matmul_expr(a, b).realize()
    ref = (a.float() @ b.float()).realize()
    max_err = (out - ref).abs().max().item()
    mse = (out - ref).square().mean().item()
    good = max_err < 1.0
    ok = ok and good
    mark = "ok" if good else "FAIL"
    print(f"  {n:>5}  max_abs_err={max_err:.4e}  mse={mse:.4e}  [{mark}]")
  if not ok:
    raise SystemExit(1)


def _env(dev: str, tc: int | None = None) -> dict[str, str]:
  e = os.environ.copy()
  e["DEV"] = dev
  e["TINYGRAD"] = os.environ.get("TINYGRAD", os.getcwd())
  e.setdefault("DEBUG", "0")
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
  p.add_argument("--check", action="store_true", help="half→float matmul vs float reference")
  p.add_argument("--inspect", action="store_true")
  args = p.parse_args(argv)
  if args.check:
    check(args.dev, [int(x) for x in args.sizes.split(",") if x.strip()], args.tc)
    return
  if args.check_wmma:
    print(f"DEV={args.dev} forced TC WMMA on 512²: {_wmma_triggered(512)}")
    return
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
  if args.inspect:
    inspect(args.dev, sizes, args.tc)
    return
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
  p.add_argument("--check", action="store_true", help="half→float matmul vs float reference")
  p.add_argument("--inspect", action="store_true", help="dump schedule/WMMA stats (no timing)")
  args = p.parse_args()

  sizes_arg = ["--sizes", args.sizes, "--warmup", str(args.warmup), "--iters", str(args.iters)]
  devs = ["AMD:AMD", "AMD"] if args.both else [os.environ.get("DEV", "AMD:AMD")]
  tcs = [1, 0] if args.compare_tc else [int(os.environ.get("TC", "1"))]
  rc = 0

  if args.check:
    for dev in devs:
      for tc in tcs:
        rc = _spawn(["--check", *sizes_arg], dev=dev, tc=tc) or rc
    return rc

  if args.check_wmma:
    for dev in devs:
      rc = _spawn(["--check-wmma", *sizes_arg], dev=dev) or rc
    return rc

  if args.inspect:
    for dev in devs:
      for tc in tcs:
        rc = _spawn(["--inspect", *sizes_arg], dev=dev, tc=tc) or rc
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
