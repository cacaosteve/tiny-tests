# RDNA3 128x128 half→float GEMM using WMMA (gfx1100 / 7900 XTX)
# Run from ~/tinygrad with venv active (needs tinygrad on PYTHONPATH):
#   DEV=AMD:AMD python ~/github/tiny-tests/rdna3_asm_wmma_matmul.py
#   VERIFY=1 CNT=10 DEV=AMD:AMD python ~/github/tiny-tests/rdna3_asm_wmma_matmul.py
# Or use the safe harness: python ~/github/tiny-tests/amd_hand_wmma.py
import time
import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv, colored
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.engine.realize import Estimates, run_linear
from tinygrad.renderer.amd.dsl import s, v, VCC_LO, NULL
from tinygrad.runtime.autogen.amd.rdna3.ins import *

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 16
TILES_M, TILES_N = 4, 4
THREADS, ELEM = 128, 2
LDS_A_ROW = BLOCK_K*ELEM   # 32
LDS_B_ROW = BLOCK_N*ELEM  # 256
LDS_A_SIZE = BLOCK_M * LDS_A_ROW  # 4096
LDS_B_SIZE = BLOCK_K * LDS_B_ROW  # 4096
LDS_SIZE = LDS_A_SIZE + LDS_B_SIZE  # 8192
LDS_B_OFF = LDS_A_SIZE
ACC, DA, DB, FA, FB, ET = 60, 188, 196, 204, 44, 10

def build_kernel(N, arch='gfx1100'):
  assert N % BLOCK_M == 0 and N >= 256
  NO_ALU, NO_DS, NO_GLOBAL = getenv("NO_ALU", 0), getenv("NO_DS", 0), getenv("NO_GLOBAL", 0)
  I, L, B = [], {}, []
  def e(i): I.append(i); return i
  def label(n): L[n] = sum(i.size() for i in I)
  def br(i, t): B.append((len(I)-1, t))

  e(s_load_b128(sdata=s[4:7], sbase=s[0:1], offset=0, soffset=NULL))
  e(s_load_b64(sdata=s[8:9], sbase=s[0:1], offset=0x10, soffset=NULL))
  e(s_waitcnt_lgkmcnt(simm16=0))
  # RDNA3: wg ids in s[2:4] (see amd_lib WGID). RDNA4 uses ttmp[9]/ttmp[7] instead.
  e(s_lshl_b32(s[10], s[2], 7)); e(s_lshl_b32(s[11], s[3], 7))
  e(s_mov_b32(s[12], N)); e(s_lshl_b32(s[13], s[12], 1))
  e(s_mul_i32(s[14], s[12], BLOCK_K*ELEM))
  e(s_add_i32(s[17], s[12], -2*BLOCK_K))  # loop bound

  e(v_and_b32_e32(v[1], 31, v[0])); e(v_lshrrev_b32_e32(v[2], 5, v[0]))
  e(v_and_b32_e32(v[3], 1, v[2])); e(v_lshrrev_b32_e32(v[2], 1, v[2]))

  # A LDS store: swizzled layout must match load_a (LLA) read addresses
  e(v_and_b32_e32(v[48], 63, v[0]))
  e(v_lshrrev_b32_e32(v[49], 4, v[48]))
  e(v_lshlrev_b32_e32(v[4], 9, v[49]))
  e(v_and_b32_e32(v[49], 15, v[1]))
  e(v_lshlrev_b32_e32(v[49], 5, v[49]))
  e(v_add_nc_u32_e32(v[4], v[4], v[49]))
  e(v_lshrrev_b32_e32(v[49], 4, v[1]))
  e(v_lshlrev_b32_e32(v[49], 4, v[49]))
  e(v_add_nc_u32_e32(v[4], v[4], v[49]))
  e(v_lshlrev_b32_e32(v[49], 11, v[2]))
  e(v_add_nc_u32_e32(v[4], v[4], v[49]))
  e(v_and_b32_e32(v[48], 7, v[0])); e(v_lshlrev_b32_e32(v[5], 9, v[48]))
  e(v_lshrrev_b32_e32(v[48], 3, v[0])); e(v_lshlrev_b32_e32(v[48], 5, v[48]))
  e(v_add_nc_u32_e32(v[5], v[5], v[48])); e(v_add_nc_u32_e32(v[5], LDS_B_OFF, v[5]))

  e(v_add_nc_u32_e32(v[48], s[11], v[0]))
  e(v_mul_lo_u32(v[6], v[48], N*ELEM)); e(v_mov_b32_e32(v[7], 0))
  e(v_lshrrev_b32_e32(v[48], 3, v[0])); e(v_mul_lo_u32(v[8], v[48], N*ELEM))
  e(v_and_b32_e32(v[48], 7, v[0])); e(v_lshlrev_b32_e32(v[48], 5, v[48]))
  e(v_add_nc_u32_e32(v[8], v[8], v[48]))
  e(s_mul_i32(s[15], s[10], ELEM)); e(v_add_nc_u32_e32(v[8], s[15], v[8]))
  e(v_mov_b32_e32(v[9], 0))

  LLA, LLB = 40, 43
  e(v_and_b32_e32(v[50], 15, v[1])); e(v_lshrrev_b32_e32(v[51], 4, v[1]))
  e(v_lshlrev_b32_e32(v[LLA], 5, v[50]))
  e(v_lshlrev_b32_e32(v[51], 4, v[51]))
  e(v_add_nc_u32_e32(v[LLA], v[LLA], v[51]))
  e(v_lshlrev_b32_e32(v[52], 11, v[2]))
  e(v_add_nc_u32_e32(v[LLA], v[LLA], v[52]))
  e(v_lshlrev_b32_e32(v[LLB], 5, v[50]))
  e(v_add_nc_u32_e32(v[LLB], v[LLB], v[51]))
  e(v_lshlrev_b32_e32(v[52], 11, v[3]))
  e(v_add_nc_u32_e32(v[LLB], v[LLB], v[52]))
  e(v_add_nc_u32_e32(v[LLB], LDS_B_OFF, v[LLB]))

  for i in range(0, 128, 2):
    e(VOPD(VOPDOp.V_DUAL_MOV_B32, VOPDOp.V_DUAL_MOV_B32, vdstx=v[ACC+i], vdsty=v[ACC+i+1], srcx0=0, srcy0=0))
  e(s_mov_b32(s[16], 0))

  if not NO_GLOBAL:
    for i in range(2): e(global_load_b128(vdst=v[DA+i*4:DA+i*4+3], addr=v[6], saddr=s[4:5], offset=i*16))
    for i in range(2): e(global_load_b128(vdst=v[DB+i*4:DB+i*4+3], addr=v[8], saddr=s[6:7], offset=i*16))
    e(s_waitcnt_vmcnt(simm16=0))
  if not NO_DS:
    for i in range(2): e(ds_store_b128(addr=v[4], data0=v[DA+i*4:DA+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
    for i in range(2): e(ds_store_b128(addr=v[5], data0=v[DB+i*4:DB+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
  if not NO_GLOBAL:
    e(v_add_nc_u32_e32(v[6], BLOCK_K*ELEM, v[6]))
    e(v_add_nc_u32_e32(v[8], s[14], v[8]))

  def load_a(tm):
    aoff = tm * 16 * LDS_A_ROW
    b = FA + tm * 8
    e(ds_load_b128(vdst=v[b:b+3], addr=v[LLA], offset0=aoff&0xFF, offset1=aoff>>8))
    e(ds_load_b128(vdst=v[b+4:b+7], addr=v[LLA], offset0=(aoff+16)&0xFF, offset1=(aoff+16)>>8))

  def load_b(bi, off0, off1):
    b = FB + bi * 8
    e(ds_load_b128(vdst=v[b:b+3], addr=v[LLB], offset0=off0, offset1=off1))
    e(ds_load_b128(vdst=v[b+4:b+7], addr=v[LLB], offset0=off0, offset1=off1+2))

  def emit_iter_body(load_set='AB'):
    if not NO_DS:
      e(s_waitcnt_lgkmcnt(simm16=0))
      e(s_barrier())
    if not NO_GLOBAL:
      if 'A' in load_set:
        for i in range(2): e(global_load_b128(vdst=v[DA+i*4:DA+i*4+3], addr=v[6], saddr=s[4:5], offset=i*16))
        e(v_add_nc_u32_e32(v[6], BLOCK_K*ELEM, v[6]))
      if 'B' in load_set:
        for i in range(2): e(global_load_b128(vdst=v[DB+i*4:DB+i*4+3], addr=v[8], saddr=s[6:7], offset=i*16))
        e(v_add_nc_u32_e32(v[8], s[14], v[8]))
    if not NO_DS:
      for tm in range(TILES_M): load_a(tm)
      load_b(0, 0, 0); load_b(1, 0, 0)
      e(s_waitcnt_lgkmcnt(simm16=0))
    if not NO_ALU:
      if not NO_DS: load_b(2, 0, 4)
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+0)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB:FB+7], src2=v[ac:ac+7]))
      if not NO_DS: load_b(3, 0, 6)
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+1)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+8:FB+15], src2=v[ac:ac+7]))
      if not NO_DS: e(s_waitcnt_lgkmcnt(simm16=1))
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+2)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+16:FB+23], src2=v[ac:ac+7]))
      if not NO_DS: e(s_waitcnt_lgkmcnt(simm16=0))
      for tm in range(TILES_M):
        ac = ACC + (tm*TILES_N+3)*8
        e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+24:FB+31], src2=v[ac:ac+7]))
    if not NO_GLOBAL and not NO_DS: e(s_waitcnt_vmcnt(simm16=0))
    if not NO_DS:
      for i in range(2): e(ds_store_b128(addr=v[4], data0=v[DA+i*4:DA+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
      for i in range(2): e(ds_store_b128(addr=v[5], data0=v[DB+i*4:DB+i*4+3], offset0=(i*16)&0xFF, offset1=(i*16)>>8))
    e(s_add_i32(s[16], s[16], BLOCK_K))

  label('LOOP')
  emit_iter_body(load_set='A')
  emit_iter_body(load_set='B')
  e(s_cmp_lt_i32(s[16], s[17])); e(s_cbranch_scc1(simm16=0)); br(I[-1], 'LOOP')

  emit_iter_body(load_set='AB')

  if not NO_DS:
    e(s_waitcnt_lgkmcnt(simm16=0))
    e(s_barrier())
  if not NO_DS:
    for tm in range(TILES_M): load_a(tm)
    load_b(0, 0, 0); load_b(1, 0, 0)
    e(s_waitcnt_lgkmcnt(simm16=0))
  if not NO_ALU:
    if not NO_DS: load_b(2, 0, 4)
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+0)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB:FB+7], src2=v[ac:ac+7]))
    if not NO_DS: load_b(3, 0, 6)
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+1)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+8:FB+15], src2=v[ac:ac+7]))
    if not NO_DS: e(s_waitcnt_lgkmcnt(simm16=1))
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+2)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+16:FB+23], src2=v[ac:ac+7]))
    if not NO_DS: e(s_waitcnt_lgkmcnt(simm16=0))
    for tm in range(TILES_M):
      ac = ACC + (tm*TILES_N+3)*8
      e(v_wmma_f32_16x16x16_f16(vdst=v[ac:ac+7], src0=v[FA+tm*8:FA+tm*8+7], src1=v[FB+24:FB+31], src2=v[ac:ac+7]))

  label('EPILOGUE')
  e(v_and_b32_e32(v[ET], 15, v[1]))
  # RDNA3 WMMA: lanes 0-15 -> even rows, 16-31 -> odd rows (not RDNA4's +8 row band).
  e(v_lshrrev_b32_e32(v[ET+1], 4, v[1]))
  e(v_lshlrev_b32_e32(v[ET+2], 6, v[2])); e(v_add_nc_u32_e32(v[ET+2], s[11], v[ET+2]))
  e(v_lshlrev_b32_e32(v[ET+3], 6, v[3])); e(v_add_nc_u32_e32(v[ET+3], s[10], v[ET+3]))
  e(v_add_nc_u32_e32(v[ET+3], v[ET+3], v[ET]))

  for tm in range(TILES_M):
    for tn in range(TILES_N):
      ac = ACC + (tm*TILES_N+tn)*8; r_off, c_off = tm*16, tn*16
      e(v_add_nc_u32_e32(v[ET+6], r_off, v[ET+2])); e(v_add_nc_u32_e32(v[ET+6], v[ET+1], v[ET+6]))
      e(v_mul_lo_u32(v[ET+4], v[ET+6], s[12])); e(v_add_nc_u32_e32(v[ET+4], v[ET+4], v[ET+3]))
      if c_off: e(v_add_nc_u32_e32(v[ET+4], c_off, v[ET+4]))
      e(v_lshlrev_b32_e32(v[ET+4], 1, v[ET+4]))
      for elem in range(8):
        e(v_cvt_f16_f32_e32(v[ET+7], v[ac+elem]))
        e(global_store_b16(addr=v[ET+4], data=v[ET+7], saddr=s[8:9]))
        if elem < 7:
          e(v_add_nc_u32_e32(v[ET+4], s[13], v[ET+4]))
          e(v_add_nc_u32_e32(v[ET+4], s[13], v[ET+4]))  # WMMA acc elems are 2 rows apart

  e(s_waitcnt_vscnt(simm16=0)); e(s_sendmsg(simm16=3)); e(s_endpgm())

  for idx, target in B:
    off = (L[target] - sum(i.size() for i in I[:idx+1])) // 4
    assert -32768 <= off <= 32767; I[idx].simm16 = off
  return I

def assemble_kernel(n: int, arch: str = 'gfx1100') -> tuple[int, int]:
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDRenderer
  insts = build_kernel(n, arch)
  r = AMDRenderer(Target(device='AMD', arch=arch, renderer='AMD'))
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))
  blob = r.asm(UOp(Ops.PROGRAM, src=(UOp.sink(), lin)), lin)
  return len(insts), len(blob)

def run_matmul(n: int | None = None):
  n = int(n if n is not None else getenv("N", 4096))
  dev = Device[Device.DEFAULT]
  arch = getattr(dev.renderer, 'arch', 'gfx1100')
  print(f"N={n}  Device arch: {arch}")
  insts = build_kernel(n, arch)

  rng = np.random.default_rng(42)
  a = Tensor(rng.random((n, n), dtype=np.float32).astype(np.float16))
  b = Tensor(rng.random((n, n), dtype=np.float32).astype(np.float16))
  c = Tensor.zeros(n, n, dtype=dtypes.half)
  Tensor.realize(a, b, c)

  grid, local = (n//BLOCK_N, n//BLOCK_M, 1), (THREADS, 1, 1)
  print(f"Grid: {grid}, Local: {local}")

  def asm_kernel(A, B, C):
    gidxs = [UOp.special(sz, f"gidx{i}") for i, sz in enumerate(grid)]
    lidxs = [UOp.special(THREADS, "lidx0")]
    lds_size = max(LDS_SIZE, 65536//getenv("LIMIT_OCC",2))
    lds = UOp.placeholder((lds_size,), dtypes.uint8, 0, AddrSpace.LOCAL)
    sink = UOp.sink(A.base, B.base, C.base, lds, *gidxs, *lidxs,
                    arg=KernelInfo(name=colored("kernel","cyan"), estimates=Estimates(ops=n*n*n*2, mem=n*n*2*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in insts]))))

  c = Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2]
  linear = c.schedule_linear()

  dev = Device[Device.DEFAULT]
  ets = []
  with Context(DEBUG=int(getenv("DEBUG", 0))):
    for _ in range(int(getenv("CNT", 5))):
      t0 = time.perf_counter()
      run_linear(linear)
      dev.synchronize()
      ets.append(time.perf_counter() - t0)
  best = min(ets)
  if best > 0:
    print(f"REAL TFLOPS {n*n*n*2 / best * 1e-12:.2f}  ({best*1000:.2f} ms best of {len(ets)})")
  else:
    print(f"timing unavailable (ets={ets})")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    dev.synchronize()
    c_np = c.float().numpy()
    a_np, b_np = a.float().numpy(), b.float().numpy()
    ref = a_np @ b_np
    err = np.sqrt(np.mean((c_np - ref)**2)) / np.sqrt(np.mean(ref**2))
    nan_cnt = int(np.isnan(c_np).sum())
    zero_cnt = int((c_np == 0).sum())
    print(f"relative RMSE {err:.6f}  (c nan={nan_cnt}/{c_np.size} zero={zero_cnt} sample={c_np[0,0]:.4g})")
    if getenv("DEBUG_VERIFY", 0):
      bs = BLOCK_M
      for gy in range(n // bs):
        for gx in range(n // bs):
          sl, r = c_np[gy*bs:(gy+1)*bs, gx*bs:(gx+1)*bs], ref[gy*bs:(gy+1)*bs, gx*bs:(gx+1)*bs]
          m = ~np.isnan(sl)
          brmse = float(np.sqrt(np.mean((sl[m]-r[m])**2))) if m.any() else float("nan")
          print(f"  tile wg=({gx},{gy}) nan={int(np.isnan(sl).sum())}/{sl.size} valid_rmse={brmse:.4g} valid={m.sum()}")
      print(f"  nan col parity even={np.isnan(c_np[:,0::2]).mean():.3f} odd={np.isnan(c_np[:,1::2]).mean():.3f}")
    if err != err or err > 0.05: raise RuntimeError(f"matmul is wrong! RMSE={err}")

if __name__ == "__main__":
  if getenv("COMPILE_ONLY", 0):
    n = int(getenv("N", 4096))
    ni, nb = assemble_kernel(n)
    print(f"compile OK: {ni} insts, {nb} byte elf")
  else:
    run_matmul()
