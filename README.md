# tiny-tests

Helper scripts for tinygrad on the remote gfx1100 box. **Run from `~/tinygrad` with venv active.**

## One command (recommended)

```bash
cd ~/tinygrad && source venv/bin/activate
python ~/github/tiny-tests/amd_gate.py
```

Runs (in order):

1. `test_amd_renderer` — 112 tests, real hardware
2. `test_tiny`
3. `test_ops` pytest slice (matmul, conv, add, mul, reduce, padding conv) — **not** full 417
4. `test_linearizer`
5. f32 GEMM GFLOPS — `DEV=AMD:AMD` then `DEV=AMD` (LLVM), via `amd_gemm_bench.py --both`

~15–45 min depending on box. Skip GEMM: `--no-bench`. Add full ops overnight: `--full-ops`.

## Individual scripts

| Script | When | Command |
|--------|------|---------|
| **`amd_gate.py`** | Pre-PR / post-pull validation | `python ~/github/tiny-tests/amd_gate.py` |
| `amd_smoke.py` | Quick 2-test check | `DEV=AMD:AMD python ~/github/tiny-tests/amd_smoke.py` |
| `amd_gemm_bench.py` | Perf only | `python ~/github/tiny-tests/amd_gemm_bench.py --both` |
| `amd_wmma_bench.py` | Half matmul / WMMA (TC) perf + correctness | `python ~/github/tiny-tests/amd_wmma_bench.py --check --sizes 512,1024` |
| `amd_wmma_vs_llvm.py` | Clean ASM vs LLVM half→float timing | `AMD=1 python ~/github/tiny-tests/amd_wmma_vs_llvm.py --sizes 1024,2048,4096` |
| **`amd_hand_wmma.py`** | Hand-tuned WMMA GEMM (safe ramp) | `python ~/github/tiny-tests/amd_hand_wmma.py` |
| `amd_float4_check.py` | After float4 work | `DEV=AMD:AMD python ~/github/tiny-tests/amd_float4_check.py` |

Always use **`DEV=AMD:AMD`** for asm (not `DEV=AMD` alone). The gate script sets it per step.

Until LDS blocking feeds WMMA, prefer `TC_LDS_BLOCK=999999` for asm WMMA benches (avoids useless GROUP staging).

## Mac (mock only)

```bash
DEV=MOCKKFD+AMD:AMD python ~/github/tiny-tests/amd_smoke.py
# or 91 non-hardware renderer tests — see tinygrad notes
```

Mock `test_ops` slice is unreliable; trust gfx1100 for pytest slice.

## Hand-tuned WMMA GEMM (safe ramp)

The kernel is `rdna3_asm_wmma_matmul.py` in this repo (not tinygrad). A GPU hang can drop Parsec/RDP if the 7900 drives the display — wait 2–3 min, reconnect via SSH, check `dmesg`.

```bash
cd ~/tinygrad && source venv/bin/activate
python ~/github/tiny-tests/amd_hand_wmma.py              # compile → 256 → 512 → 1024 → 4096 verify
python ~/github/tiny-tests/amd_hand_wmma.py --compile-only
python ~/github/tiny-tests/amd_hand_wmma.py --from run-256   # resume after disconnect
python ~/github/tiny-tests/amd_hand_wmma.py --bench          # + 4096²×5 timing run
python ~/github/tiny-tests/amd_hand_wmma.py --list
```

Log: `~/github/tiny-tests/amd_hand_wmma.last.log`
