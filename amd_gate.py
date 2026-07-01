#!/usr/bin/env python3
"""gfx1100 validation gate — everything except full test_ops (~417 tests).

Run from ~/tinygrad with venv active:

  DEV=AMD:AMD python ~/github/tiny-tests/amd_gate.py
  python ~/github/tiny-tests/amd_gate.py --no-bench     # skip GFLOPS sweep
  python ~/github/tiny-tests/amd_gate.py --full-ops     # add full test_ops (slow)

Requires: pytest (pip install pytest). torch comes with test_ops.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

TINYGRAD = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DEV_ASM = "AMD:AMD"
DEV_LLVM = "AMD"

PYTEST_SLICE = "max_unpool2d or simple_conv2d_nhwc or asymmetric_padding_conv2d or matmul or conv2d or add or mul or reduce"


def _env(dev: str) -> dict[str, str]:
  e = os.environ.copy()
  e["DEV"] = dev
  return e


def run_step(name: str, cmd: list[str], *, dev: str | None = None, cwd: str | None = None) -> int:
  root = cwd or TINYGRAD
  tag = f" [DEV={dev}]" if dev else ""
  print(f"\n{'='*72}\n{name}{tag}\n$ {' '.join(cmd)}\n{'='*72}\n", flush=True)
  t0 = time.perf_counter()
  rc = subprocess.call(cmd, cwd=root, env=_env(dev) if dev else os.environ.copy())
  print(f"\n→ {name}: {'OK' if rc == 0 else f'FAIL (exit {rc})'} in {time.perf_counter()-t0:.1f}s\n", flush=True)
  return rc


def main() -> int:
  p = argparse.ArgumentParser(description="AMD asm backend gfx1100 gate (no full test_ops)")
  p.add_argument("--no-bench", action="store_true", help="skip amd_gemm_bench.py")
  p.add_argument("--full-ops", action="store_true", help="also run full test.backend.test_ops (~1–2h)")
  p.add_argument("--tinygrad", default=TINYGRAD, help=f"tinygrad root (default {TINYGRAD})")
  args = p.parse_args()
  root = args.tinygrad

  if not os.path.isdir(os.path.join(root, "test")):
    print(f"error: tinygrad not found at {root}", file=sys.stderr)
    return 2

  dev = os.environ.get("DEV", DEV_ASM)
  if dev != DEV_ASM:
    print(f"warning: DEV={dev!r}; gate expects {DEV_ASM} for asm tests", flush=True)

  steps: list[tuple[str, list[str], str | None]] = [
    ("test_amd_renderer (112)", [PY, "-m", "unittest", "test.amd.test_amd_renderer", "-q"], DEV_ASM),
    ("test_tiny", [PY, "-m", "unittest", "test.test_tiny", "-q"], DEV_ASM),
    ("test_ops slice (pytest)", [PY, "-m", "pytest", "test/backend/test_ops.py", "-q", "-k", PYTEST_SLICE], DEV_ASM),
    ("test_linearizer", [PY, "-m", "unittest", "test.backend.test_linearizer", "-q"], DEV_ASM),
  ]

  if args.full_ops:
    steps.append(("test_ops FULL", [PY, "-m", "pytest", "test/backend/test_ops.py", "-q"], DEV_ASM))

  results: list[tuple[str, int]] = []
  for name, cmd, step_dev in steps:
    results.append((name, run_step(name, cmd, dev=step_dev, cwd=root)))

  if not args.no_bench:
    bench_cmd = [PY, os.path.join(HERE, "amd_gemm_bench.py"), "--both"]
    results.append(("f32 GEMM benchmark (ASM + LLVM)", run_step("f32 GEMM benchmark", bench_cmd)))

  print("\n" + "=" * 72)
  print("SUMMARY")
  print("=" * 72)
  failed = [(n, rc) for n, rc in results if rc != 0]
  for name, rc in results:
    print(f"  {'OK' if rc == 0 else 'FAIL':4s}  {name}")
  print("=" * 72)
  if failed:
    print(f"\n{len(failed)} step(s) failed — fix before reopen / WMMA.\n")
    return 1
  print("\nGate green. Full test_ops still optional: amd_gate.py --full-ops\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
