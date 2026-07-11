#!/usr/bin/env python3
"""Safe progressive test for gfx1100 hand-tuned WMMA GEMM.

Wraps rdna3_asm_wmma_matmul.py (this repo) with compile-first checks and per-step
subprocess timeouts. If the 7900 drives the display, a GPU hang can still drop
Parsec/RDP — wait 2–3 min, reconnect (SSH is more reliable than web UI), then:

  dmesg --ctime | tail -40 | grep -iE 'amdgpu|gpu|reset|fault|hang'
  python ~/github/tiny-tests/amd_hand_wmma.py --from <failed-step>

Run from ~/tinygrad with venv active:

  DEV=AMD:AMD python ~/github/tiny-tests/amd_hand_wmma.py
  python ~/github/tiny-tests/amd_hand_wmma.py --compile-only
  python ~/github/tiny-tests/amd_hand_wmma.py --from run-256
  python ~/github/tiny-tests/amd_hand_wmma.py --bench
  python ~/github/tiny-tests/amd_hand_wmma.py --list
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from dataclasses import dataclass
from datetime import datetime

TINYGRAD = os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad"))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
KERNEL = os.path.join(TINYGRAD, "extra/gemm/rdna3_asm_wmma_gemm.py")
LOG_PATH = os.path.join(HERE, "amd_hand_wmma.last.log")
DEV_ASM = "AMD:AMD"
TIMEOUT_RC = 124


@dataclass(frozen=True)
class Step:
  name: str
  gpu: bool
  timeout_s: int
  env: dict[str, str]


STEPS: list[Step] = [
  Step("compile-256", False, 30, {"COMPILE_ONLY": "1", "N": "256"}),
  Step("compile-4096", False, 60, {"COMPILE_ONLY": "1", "N": "4096"}),
  Step("isolate-no-alu", True, 90, {"N": "256", "CNT": "1", "VERIFY": "0", "NO_ALU": "1", "DEBUG": "0", "HCQDEV_WAIT_TIMEOUT_MS": "20000"}),
  Step("run-256", True, 90, {"N": "256", "CNT": "1", "VERIFY": "1", "DEBUG": "0", "HCQDEV_WAIT_TIMEOUT_MS": "20000"}),
  Step("run-512", True, 120, {"N": "512", "CNT": "1", "VERIFY": "1", "DEBUG": "0", "HCQDEV_WAIT_TIMEOUT_MS": "25000"}),
  Step("run-1024", True, 180, {"N": "1024", "CNT": "1", "VERIFY": "1", "DEBUG": "0", "HCQDEV_WAIT_TIMEOUT_MS": "30000"}),
  Step("run-4096", True, 300, {"N": "4096", "CNT": "1", "VERIFY": "1", "DEBUG": "0", "HCQDEV_WAIT_TIMEOUT_MS": "45000"}),
]

BENCH_STEP = Step("bench-4096", True, 600, {"N": "4096", "CNT": "5", "VERIFY": "1", "DEBUG": "2", "HCQDEV_WAIT_TIMEOUT_MS": "60000"})


def _log(msg: str, log_fp) -> None:
  line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
  print(line, flush=True)
  log_fp.write(line + "\n")
  log_fp.flush()


def _dmesg_tail() -> str:
  try:
    p = subprocess.run(["dmesg", "--ctime"], capture_output=True, text=True, timeout=10)
    if p.returncode != 0:
      return ""
    lines = [ln for ln in p.stdout.splitlines() if any(k in ln.lower() for k in ("amdgpu", "kfd", "gpu reset", "mmu", "hang", "fault"))]
    return "\n".join(lines[-25:])
  except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    return ""


def _run_step(step: Step, root: str, kernel: str, log_fp) -> int:
  env = os.environ.copy()
  env["DEV"] = DEV_ASM
  env["TINYGRAD"] = root
  env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  env.update(step.env)
  cmd = [PY, kernel]
  if step.gpu:
    cmd = ["timeout", "-k", "5", str(step.timeout_s), *cmd]
  tag = " ".join(f"{k}={v}" for k, v in sorted(step.env.items()))
  _log(f"START {step.name} ({tag}) timeout={step.timeout_s}s", log_fp)
  _log(f"$ {' '.join(cmd)}", log_fp)
  t0 = time.perf_counter()
  rc = subprocess.call(cmd, cwd=root, env=env)
  dt = time.perf_counter() - t0
  if rc == TIMEOUT_RC:
    _log(f"TIMEOUT {step.name} after {dt:.1f}s (GPU hang likely — wait before reconnect)", log_fp)
    tail = _dmesg_tail()
    if tail:
      _log("dmesg (amdgpu/kfd tail):\n" + tail, log_fp)
    return rc
  status = "OK" if rc == 0 else f"FAIL exit {rc}"
  _log(f"END {step.name}: {status} in {dt:.1f}s", log_fp)
  if rc != 0:
    tail = _dmesg_tail()
    if tail:
      _log("dmesg (amdgpu/kfd tail):\n" + tail, log_fp)
  return rc


def _check_paths(root: str, kernel: str) -> str | None:
  if not os.path.isdir(os.path.join(root, "tinygrad")):
    return f"tinygrad not found at {root}"
  if not os.path.isfile(kernel):
    return f"kernel not found: {kernel}"
  return None


def main() -> int:
  p = argparse.ArgumentParser(description="Safe progressive gfx1100 hand WMMA GEMM test")
  p.add_argument("--tinygrad", default=TINYGRAD, help=f"tinygrad root (default {TINYGRAD})")
  p.add_argument("--from", dest="from_step", metavar="STEP", help="resume at this step name (see --list)")
  p.add_argument("--only", metavar="STEP", help="run a single step")
  p.add_argument("--bench", action="store_true", help="append full 4096²×5 bench after safe steps")
  p.add_argument("--compile-only", action="store_true", help="only run compile-* steps (no GPU)")
  p.add_argument("--list", action="store_true", help="list step names and exit")
  p.add_argument("--diagnose", action="store_true", help="run DEBUG_VERIFY bad-cell localization --runs times")
  p.add_argument("--n", type=int, default=256, help="matrix size for --diagnose (default 256)")
  p.add_argument("--runs", type=int, default=2, help="repeat count for --diagnose (default 2, to catch races)")
  args = p.parse_args()

  root = args.tinygrad
  kernel = os.path.join(HERE, "rdna3_asm_wmma_matmul.py")

  if args.diagnose:
    if err := _check_paths(root, kernel):
      print(f"error: {err}", file=sys.stderr)
      return 2
    step = Step(f"diagnose-{args.n}", True, 120,
                {"N": str(args.n), "CNT": "1", "VERIFY": "1", "DEBUG_VERIFY": "1", "HCQDEV_WAIT_TIMEOUT_MS": "30000"})
    with open(LOG_PATH, "a", encoding="utf-8") as log_fp:
      for i in range(args.runs):
        _log(f"--- diagnose run {i+1}/{args.runs} (N={args.n}) ---", log_fp)
        _run_step(step, root, kernel, log_fp)  # continue through failures so we see every run
    return 0

  steps = list(STEPS)
  if args.bench:
    steps.append(BENCH_STEP)

  if args.list:
    for s in steps:
      kind = "gpu" if s.gpu else "compile"
      print(f"  {s.name:16s}  {kind:7s}  timeout={s.timeout_s}s")
    return 0

  if err := _check_paths(root, kernel):
    print(f"error: {err}", file=sys.stderr)
    return 2

  if args.only:
    steps = [s for s in steps if s.name == args.only]
    if not steps:
      print(f"error: unknown step {args.only!r} (try --list)", file=sys.stderr)
      return 2
  elif args.compile_only:
    steps = [s for s in steps if not s.gpu]
  elif args.from_step:
    names = [s.name for s in steps]
    if args.from_step not in names:
      print(f"error: unknown step {args.from_step!r} (try --list)", file=sys.stderr)
      return 2
    steps = steps[names.index(args.from_step):]

  print(f"log: {LOG_PATH}")
  print("tip: use SSH for GPU tests; web UI may stay down until amdgpu finishes reset (~2–3 min)\n", flush=True)

  with open(LOG_PATH, "a", encoding="utf-8") as log_fp:
    _log("=" * 72, log_fp)
    _log(f"amd_hand_wmma run ({len(steps)} step(s))", log_fp)
    for step in steps:
      rc = _run_step(step, root, kernel, log_fp)
      if rc != 0:
        _log(f"stopped at {step.name} — resume: python {os.path.relpath(__file__, root)} --from {step.name}", log_fp)
        return 1
    _log("all steps passed", log_fp)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
