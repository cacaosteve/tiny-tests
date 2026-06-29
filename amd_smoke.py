#!/usr/bin/env python3
"""Quick AMD asm correctness smoke. Run from ~/tinygrad with venv active.

  DEV=AMD:AMD python ~/github/tiny-tests/amd_smoke.py
  DEV=AMD:AMD python ~/github/tiny-tests/amd_smoke.py --ops   # + full test_ops (~2–3 min)
"""
from __future__ import annotations
import argparse, os, subprocess, sys

def run(cmd: list[str]) -> int:
  print(f"\n$ {' '.join(cmd)}\n", flush=True)
  return subprocess.call(cmd)

def main() -> int:
  p = argparse.ArgumentParser(description="AMD asm smoke tests")
  p.add_argument("--ops", action="store_true", help="also run full test.backend.test_ops")
  args = p.parse_args()
  py = sys.executable
  dev = os.environ.get("DEV", "AMD:AMD")
  if "DEV" not in os.environ:
    os.environ["DEV"] = dev
    print(f"DEV not set; using {dev}")

  tests = [
    [py, "-m", "unittest", "test.amd.test_amd_renderer", "-q"],
    [py, "-m", "unittest", "test.backend.test_ops.TestOps.test_matmul", "-q"],
  ]
  if args.ops:
    tests.append([py, "-m", "unittest", "test.backend.test_ops", "-q"])

  rc = 0
  for cmd in tests:
    rc = run(cmd) or rc
  return rc

if __name__ == "__main__":
  raise SystemExit(main())
