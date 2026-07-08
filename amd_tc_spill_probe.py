#!/usr/bin/env python3
"""Probe WMMA kernel register pressure vs post-TC tile size.

Replicates heuristic.py's post-TC path (TC -> UPCAST M -> UPCAST N -> LOCAL N)
but with configurable sizes, then counts SPILL/FILL/WMMA in the compiled kernel.

  DEV=MOCKKFD+AMD:AMD PYTHONPATH=. python ~/github/tiny-tests/amd_tc_spill_probe.py --n 512
"""
from __future__ import annotations
import argparse, os, sys

def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--n", type=int, default=512)
  p.add_argument("--configs", default="4,4,4;2,2,4;2,2,2;2,2,1;1,1,4;1,1,1",
                 help="semicolon list of up_m,up_n,local_n")
  args = p.parse_args()

  root = os.environ.get("TINYGRAD", os.getcwd())
  if root not in sys.path: sys.path.insert(0, root)
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt import Opt, OptOps
  from tinygrad.codegen.opt.postrange import Scheduler
  from tinygrad.renderer.isa.amd import AMDOps
  from tinygrad.uop.ops import Ops
  from tinygrad.device import CompileError

  n = args.n
  ren = Device["AMD"].renderer
  a = Tensor.rand(n, n, dtype=dtypes.half)
  b = Tensor.rand(n, n, dtype=dtypes.half)
  base_ast = a.matmul(b, dtype=dtypes.float).schedule_linear().src[-1].src[0]

  print(f"n={n} renderer={type(ren).__name__}")
  print(f"{'up_m,up_n,loc':>14} | {'WMMA':>5} {'SPILL':>6} {'FILL':>5} | opts")
  for cfg in args.configs.split(";"):
    up_m, up_n, loc_n = (int(x) for x in cfg.split(","))
    k = Scheduler(base_ast, ren)
    k.convert_loop_to_global()
    try:
      rngs = k.apply_opt(Opt(OptOps.TC, 0, (-1, 0, 1)))
      if rngs is not None:
        for tc_dim, up in [(1, up_m), (0, up_n)]:
          if up > 1 and rngs[tc_dim].src[0].divides(up) is not None:
            rngs[tc_dim] = k.apply_opt(Opt(OptOps.UPCAST, k.rngs.index(rngs[tc_dim]), up))[0]
        if loc_n > 1 and rngs[0].src[0].divides(loc_n) is not None:
          k.apply_opt(Opt(OptOps.LOCAL, k.rngs.index(rngs[0]), loc_n))
      opt_ast = k.get_optimized_ast()
      pu = to_program(opt_ast, ren)
      lin = list(pu.src[1].src)
      wmma = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.WMMA)
      spill = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.SPILL)
      fill = sum(1 for u in lin if u.op is Ops.INS and u.arg is AMDOps.FILL)
      opts = [o.op.name for o in pu.src[0].arg.applied_opts]
      print(f"{cfg:>14} | {wmma:>5} {spill:>6} {fill:>5} | {opts}")
    except (CompileError, Exception) as e:
      print(f"{cfg:>14} | FAIL: {type(e).__name__}: {str(e)[:60]}")
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
