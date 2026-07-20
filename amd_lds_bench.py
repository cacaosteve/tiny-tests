#!/usr/bin/env python3
"""Safe TC_LDS_AB half-GEMM check + bench for gfx1100 gaming PC.

Run from ~/tinygrad with venv active (do NOT prefix DEV= for --vs-llvm):

  # recommended after pull — mse@256 then default vs LDS @4096
  python ~/github/tiny-tests/amd_lds_bench.py

  # spill/coop gate only (no GPU launch beyond compile)
  python ~/github/tiny-tests/amd_lds_bench.py --counts-only

  # also compare LLVM / hand
  python ~/github/tiny-tests/amd_lds_bench.py --vs-llvm

Stops before big benches if mse is bad or SPILL!=0 (address spills hang the display GPU).
"""
from __future__ import annotations
import argparse, os, subprocess, sys

TINYGRAD = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def _setup() -> None:
  if TINYGRAD not in sys.path:
    sys.path.insert(0, TINYGRAD)


def _clear_tc() -> None:
  from tinygrad.helpers import getenv
  from tinygrad.codegen import to_program_cache
  for k in list(os.environ):
    if k.startswith("TC_"): os.environ.pop(k)
  getenv.cache_clear()
  to_program_cache.clear()


def _set_lds(on: bool) -> None:
  from tinygrad.helpers import getenv
  from tinygrad.codegen import to_program_cache
  if on: os.environ["TC_LDS_AB"] = "1"
  else: os.environ.pop("TC_LDS_AB", None)
  getenv.cache_clear()
  to_program_cache.clear()


def _counts(n: int, label: str) -> dict[str, int]:
  _setup()
  from tinygrad import Tensor, Context, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.device import Device
  from tinygrad.renderer.isa.amd import AMDRenderer, AMDOps
  from tinygrad.uop import Ops
  ren = Device["AMD"].renderer
  if not isinstance(ren, AMDRenderer):
    raise SystemExit(f"need AMDRenderer, got {type(ren).__name__}")
  with Context(BEAM=0):
    ast = (Tensor.empty(n, n, dtype=dtypes.half, device="AMD") @
           Tensor.empty(n, n, dtype=dtypes.half, device="AMD")).cast(dtypes.float)
    prg = to_program(ast.schedule_linear().src[-1].src[0], ren)
  los = [u.arg for u in prg.src[1].src if u.op is Ops.INS]
  keys = ("LOAD", "LLOAD", "LSTORE", "STORE", "WMMA", "SPILL", "FILL", "BARRIER")
  cnt = {k: los.count(getattr(AMDOps, k)) for k in keys}
  print(f"{label:24s} N={n}  {cnt}", flush=True)
  coop = cnt["LLOAD"] > cnt["LSTORE"] and cnt["LOAD"] < cnt["LLOAD"]
  print(f"{'':24s}      coop={'OK' if coop else 'FAIL'}", flush=True)
  try:
    from test.amd.test_amd_renderer import _amd_inst_names
    names = _amd_inst_names(prg)
    mem = {k: sum(1 for x in names if x == k) for k in
           ("GLOBAL_LOAD_B32", "GLOBAL_LOAD_B128", "DS_LOAD_B32", "DS_LOAD_B64", "DS_LOAD_B128")}
    print(f"{'':24s}      mem={mem}", flush=True)
  except Exception as e:
    print(f"{'':24s}      mem=(skip: {e})", flush=True)
  return cnt


def _mse(n: int = 256) -> float:
  _setup()
  from tinygrad import Tensor, Context, dtypes
  _set_lds(True)
  with Context(BEAM=0):
    a = Tensor.randn(n, n, dtype=dtypes.half).realize()
    b = Tensor.randn(n, n, dtype=dtypes.half).realize()
    c = (a @ b).cast(dtypes.float).realize()
    ref = (a.float() @ b.float()).realize()
    return float((c - ref).square().mean().item())


def _bench(n: int, cnt: int, label: str) -> float:
  _setup()
  from tinygrad import Tensor, Context, dtypes, GlobalCounters
  a = Tensor.randn(n, n, dtype=dtypes.half).realize()
  b = Tensor.randn(n, n, dtype=dtypes.half).realize()
  with Context(BEAM=0, DEBUG=0): (a @ b).realize()
  ets = []
  with Context(BEAM=0, DEBUG=2):
    for _ in range(cnt):
      GlobalCounters.reset()
      (a @ b).realize()
      ets.append(GlobalCounters.time_sum_s)
  gflops = (2.0 * n * n * n) / min(ets) * 1e-9
  print(f"{label:24s} N={n}  min={min(ets)*1e6:8.2f} us  {gflops:9.1f} GFLOPS", flush=True)
  return gflops


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--n", type=int, default=int(os.environ.get("N", "4096")))
  p.add_argument("--cnt", type=int, default=int(os.environ.get("CNT", "3")))
  p.add_argument("--counts-only", action="store_true", help="compile/spill/coop only @256 + @N")
  p.add_argument("--skip-mse", action="store_true")
  p.add_argument("--vs-llvm", action="store_true", help="also run amd_wmma_vs_llvm default,lds,hip")
  args = p.parse_args()

  os.environ.setdefault("DEV", "AMD:AMD")
  print(f"DEV={os.environ.get('DEV')}  N={args.n}  CNT={args.cnt}", flush=True)

  # 1) spill gate @256 (cheap enough; catches hang-class address spills)
  _clear_tc()
  _set_lds(False)
  d = _counts(256, "default@256")
  if d["SPILL"] or d["FILL"]:
    print("FAIL: default has spills — do not bench", file=sys.stderr)
    return 1
  _set_lds(True)
  l = _counts(256, "TC_LDS_AB@256")
  if l["SPILL"] or l["FILL"]:
    print("FAIL: TC_LDS_AB has address spills (hang class) — abort", file=sys.stderr)
    return 1
  if not (l["LLOAD"] > l["LSTORE"] and l["LOAD"] < l["LLOAD"]):
    print("FAIL: TC_LDS_AB coop gate", file=sys.stderr)
    return 1

  if args.n != 256:
    _set_lds(False)
    _counts(args.n, "default")
    _set_lds(True)
    _counts(args.n, "TC_LDS_AB=1")

  if args.counts_only:
    print("counts-only OK", flush=True)
    return 0

  # 2) correctness @256 before any big launch
  if not args.skip_mse:
    mse = _mse(256)
    print(f"TC_LDS_AB=1 mse@256 = {mse}", flush=True)
    if mse > 1e-2:
      print(f"FAIL: mse too high ({mse})", file=sys.stderr)
      return 1

  # 3) benches
  _set_lds(False)
  _bench(args.n, args.cnt, "default")
  _set_lds(True)
  _bench(args.n, args.cnt, "TC_LDS_AB=1")

  if args.vs_llvm:
    cmd = [PY, os.path.join(HERE, "amd_wmma_vs_llvm.py"),
           "--configs", "default,lds,hip", "--sizes", str(args.n),
           "--warmup", "2", "--iters", str(args.cnt)]
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=TINYGRAD)

  print("done", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
