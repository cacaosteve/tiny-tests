# Thin wrapper — kernel lives in tinygrad extra/gemm/rdna3_asm_wmma_gemm.py
#   DEV=AMD:AMD python ~/github/tiny-tests/rdna3_asm_wmma_matmul.py
#   python ~/github/tiny-tests/amd_hand_wmma.py --bench
import os, sys
sys.path.insert(0, os.environ.get("TINYGRAD", os.path.expanduser("~/tinygrad")))
from extra.gemm.rdna3_asm_wmma_gemm import *  # noqa: F401,F403
from extra.gemm.rdna3_asm_wmma_gemm import run_matmul, assemble_kernel
from tinygrad.helpers import getenv

if __name__ == "__main__":
  if getenv("COMPILE_ONLY", 0):
    n = int(getenv("N", 4096))
    ni, nb = assemble_kernel(n)
    print(f"compile OK: {ni} insts, {nb} byte elf")
  else:
    run_matmul()
