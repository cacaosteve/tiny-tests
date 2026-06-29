# tiny-tests

Helper scripts for tinygrad on the remote gfx1100 box. **Run from `~/tinygrad` with venv active.**

| Script | When | Command |
|--------|------|---------|
| `amd_smoke.py` | After slice 2+ commits | `DEV=AMD:AMD python ~/github/tiny-tests/amd_smoke.py` |
| `amd_smoke.py --ops` | Pre-PR full ops (~2.5 min) | `DEV=AMD:AMD python ~/github/tiny-tests/amd_smoke.py --ops` |
| `amd_gemm_bench.py` | Perf check (slice 3+) | `python ~/github/tiny-tests/amd_gemm_bench.py --both` |
| `amd_float4_check.py` | After slice 3 lands | `DEV=AMD:AMD python ~/github/tiny-tests/amd_float4_check.py` |

`--both` on the bench script runs ASM (`DEV=AMD:AMD`) then LLVM (`DEV=AMD`) back-to-back.
