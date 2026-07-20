#!/usr/bin/env python3
"""Safe UPCAST16 (4×4 / 16 WMMA) half-GEMM check + bench for gfx1100.

Best next lever after ~28k default (was hang-class; remat made it spill-free offline).

  cd ~/tinygrad && source venv/bin/activate
  python ~/github/tiny-tests/amd_upcast_bench.py

Aborts if SPILL/FILL nonzero or mse bad. Stop if the desktop disconnects.
"""
from __future__ import annotations
import argparse, os, sys

TINYGRAD = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))


def _setup() -> None:
  if TINYGRAD not in sys.path:
    sys.path.insert(0, TINYGRAD)


def _clear_caches() -> None:
  from tinygrad.helpers import getenv
  from tinygrad.codegen import to_program_cache
  getenv.cache_clear()
  to_program_cache.clear()


def _set_upcast16(on: bool) -> None:
  if on:
    os.environ["TC_UPCAST"] = "4"
    os.environ["TC_UPCAST_TILES"] = "16"
  else:
    os.environ.pop("TC_UPCAST", None)
    os.environ.pop("TC_UPCAST_TILES", None)
  os.environ.pop("TC_LDS_AB", None)
  _clear_caches()


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
  print(f"{'':24s}      opts={tuple(prg.src[0].arg.applied_opts)}", flush=True)
  try:
    from test.amd.test_amd_renderer import _amd_inst_names
    names = _amd_inst_names(prg)
    mem = {k: sum(1 for x in names if x == k) for k in
           ("GLOBAL_LOAD_B32", "GLOBAL_LOAD_B128", "DS_LOAD_B128")}
    print(f"{'':24s}      mem={mem}", flush=True)
  except Exception as e:
    print(f"{'':24s}      mem=(skip: {e})", flush=True)
  return cnt


def _mse(n: int = 256) -> float:
  _setup()
  from tinygrad import Tensor, Context, dtypes
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
  p.add_argument("--counts-only", action="store_true")
  p.add_argument("--skip-mse", action="store_true")
  args = p.parse_args()

  os.environ.setdefault("DEV", "AMD:AMD")
  print(f"DEV={os.environ.get('DEV')}  N={args.n}  CNT={args.cnt}", flush=True)
  print("mode=UPCAST16 (TC_UPCAST=4 TC_UPCAST_TILES=16)", flush=True)

  _set_upcast16(False)
  d = _counts(256, "default@256")
  if d["SPILL"] or d["FILL"]:
    print("FAIL: default has spills", file=sys.stderr)
    return 1

  _set_upcast16(True)
  u = _counts(256, "UPCAST16@256")
  if u["SPILL"] or u["FILL"]:
    print("FAIL: UPCAST16 has address spills (hang class) — abort", file=sys.stderr)
    return 1
  if u["WMMA"] < 16:
    print(f"FAIL: expected >=16 WMMA, got {u['WMMA']}", file=sys.stderr)
    return 1

  if args.counts_only:
    print("counts-only OK", flush=True)
    return 0

  if not args.skip_mse:
    mse = _mse(256)
    print(f"UPCAST16 mse@256 = {mse}", flush=True)
    if mse > 1e-2:
      print(f"FAIL: mse too high ({mse})", file=sys.stderr)
      return 1

  _set_upcast16(False)
  _bench(args.n, args.cnt, "default")
  _set_upcast16(True)
  _bench(args.n, args.cnt, "UPCAST16")

  print("done", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
