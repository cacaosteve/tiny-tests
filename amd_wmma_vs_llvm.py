#!/usr/bin/env python3
"""Half GEMM: auto ASM vs LLVM vs hand WMMA. Clean single-process timing.

Run from ~/tinygrad with venv active. Sets DEV itself per backend, so DON'T prefix DEV=:

  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py
  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --sizes 1024,2048,4096
  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --hand --sizes 4096
  # schedule matrix (what actually closes the gap):
  AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --configs default,lds,lds_up4,hip,hand --sizes 2048,4096

Each backend runs in its own subprocess because Device.DEFAULT is fixed after first import.
Worker exits with os._exit(0) after sync to avoid HCQ teardown timeouts on gfx11.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

SCRIPT = os.path.abspath(__file__)

# Named ASM schedule knobs. HIP/HAND ignore TC_* .
CONFIGS = {
  "default": ("AMD:AMD", False, {}),
  "up4": ("AMD:AMD", False, {"TC_UPCAST": "4"}),
  "up16": ("AMD:AMD", False, {"TC_UPCAST": "4", "TC_UPCAST_TILES": "16"}),
  "lds": ("AMD:AMD", False, {"TC_LDS_AB": "1"}),
  "lds_up4": ("AMD:AMD", False, {"TC_LDS_AB": "1", "TC_UPCAST": "4"}),
  "hip": ("AMD", False, {}),
  "hand": ("AMD:AMD", True, {}),
}

def _worker(dev: str, sizes: list[int], warmup: int, iters: int, hand: bool) -> None:
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.engine.realize import run_linear
  ren = type(Device["AMD"].renderer).__name__
  label = "HAND" if hand else ren
  tc = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("TC_")) or "TC=default"
  for n in sizes:
    a = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    b = Tensor.randn(n, n, dtype=dtypes.half, device="AMD").realize()
    if hand:
      from extra.gemm.rdna3_asm_wmma_gemm import can_use_rdna3_wmma_gemm, rdna3_wmma_gemm
      if not can_use_rdna3_wmma_gemm(a, b):
        print(f"  [{label:<12s}] {tc:<28s} {n:>5}  SKIP (can_use=False)", flush=True)
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
    print(f"  [{label:<12s}] {tc:<28s} {n:>5}  {2*n**3/sec/1e9:8.0f} GFLOPS  ({sec*1e3:7.2f} ms/iter)", flush=True)
  Device["AMD"].synchronize()
  os._exit(0)

def main() -> int:
  p = argparse.ArgumentParser(description="half GEMM ASM vs LLVM vs hand WMMA")
  p.add_argument("--sizes", default="1024,2048,4096")
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--iters", type=int, default=10)
  p.add_argument("--hand", action="store_true", help="also time hand WMMA custom_kernel path")
  p.add_argument("--configs", default="",
                 help=f"comma list of {','.join(CONFIGS)} (default: default,hip; --hand adds hand)")
  p.add_argument("--worker", help=argparse.SUPPRESS)
  p.add_argument("--worker-hand", action="store_true", help=argparse.SUPPRESS)
  args = p.parse_args()
  sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

  if args.worker:
    root = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))
    if root not in sys.path: sys.path.insert(0, root)
    _worker(args.worker, sizes, args.warmup, args.iters, args.worker_hand)
    return 0

  if args.configs.strip():
    names = [x.strip() for x in args.configs.split(",") if x.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
      print(f"error: unknown configs {unknown}; choose from {list(CONFIGS)}", file=sys.stderr)
      return 2
    jobs = [(n, *CONFIGS[n]) for n in names]
  else:
    jobs = [("default", *CONFIGS["default"]), ("hip", *CONFIGS["hip"])]
    if args.hand: jobs.append(("hand", *CONFIGS["hand"]))

  rc = 0
  for name, dev, hand, extra_env in jobs:
    print(f"\n=== {name} (DEV={dev}) ===", flush=True)
    env = os.environ.copy()
    env["DEV"] = dev
    env["DEBUG"] = "0"
    env.setdefault("TINYGRAD", os.path.expanduser("~/tinygrad"))
    # drop leftover TC_* from parent so configs are clean
    for k in list(env):
      if k.startswith("TC_"): env.pop(k)
    env.update(extra_env)
    cmd = [sys.executable, SCRIPT, "--worker", dev,
           "--sizes", args.sizes, "--warmup", str(args.warmup), "--iters", str(args.iters)]
    if hand: cmd.append("--worker-hand")
    rc = subprocess.call(cmd, env=env) or rc
  return rc

if __name__ == "__main__":
  raise SystemExit(main())
