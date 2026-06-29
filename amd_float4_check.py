#!/usr/bin/env python3
"""Run after slice 3 (supports_float4). Run from ~/tinygrad with venv active.

  DEV=AMD:AMD python ~/github/tiny-tests/amd_float4_check.py
"""
from __future__ import annotations
import os, subprocess, sys

def main() -> int:
  dev = os.environ.get("DEV", "AMD:AMD")
  if "DEV" not in os.environ:
    os.environ["DEV"] = dev
  py = sys.executable
  suites = [
    [py, "-m", "unittest", "test.amd.test_amd_renderer", "-q"],
    [py, "-m", "unittest", "test.opt.test_gen_float4", "-q"],
    [py, "-m", "unittest", "test.backend.test_ops.TestOps.test_matmul", "-q"],
  ]
  rc = 0
  for cmd in suites:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    rc = subprocess.call(cmd) or rc
  if rc == 0:
    print("\nFloat4 suites OK — run amd_gemm_bench.py --both for perf.\n")
  return rc

if __name__ == "__main__":
  raise SystemExit(main())
