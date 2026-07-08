#!/usr/bin/env python3
"""Quick f16→f32 GEMM correctness check (compiler WMMA path). Run from ~/tinygrad with venv active.

  DEV=AMD:AMD python ~/github/tiny-tests/amd_compiler_gemm.py
  DEV=AMD:AMD python ~/github/tiny-tests/amd_compiler_gemm.py --size 1024
"""
from __future__ import annotations
import argparse, os, sys

def main() -> int:
  p = argparse.ArgumentParser(description="compiler WMMA GEMM sanity (half in, float out)")
  p.add_argument("--size", type=int, default=512, help="M=N=K (default 512)")
  p.add_argument("--atol", type=float, default=0.05, help="max abs err threshold")
  args = p.parse_args()
  if "DEV" not in os.environ:
    os.environ["DEV"] = "AMD:AMD"
    print(f"DEV not set; using {os.environ['DEV']}")

  import numpy as np
  from tinygrad import Tensor, dtypes

  n = args.size
  a = Tensor.rand(n, n, dtype=dtypes.half)
  b = Tensor.rand(n, n, dtype=dtypes.half)
  c = a.matmul(b, dtype=dtypes.float).numpy()
  ref = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
  err = float(np.abs(c - ref).max())
  ok = err < args.atol
  print(f"shape {n}x{n}x{n}  max_abs_err={err:.6f}  {'OK' if ok else 'BAD'}")
  return 0 if ok else 1

if __name__ == "__main__":
  raise SystemExit(main())
