// FlashVSR's locality-constrained sparse attention: the ROUTING half.
//
// The attention itself is attn_steel's own has_block_mask path -- this
// file only decides which key blocks each query block may read, and
// writes that decision in the per-(query tile, key) byte form that
// kernel already accepts. Sol-Attn's routing lives next door for the
// same reason: the decision is the model, the skipping is a kernel.
//
// WHY IT MOVED HERE FROM THE HOST. The scores are over WINDOW BLOCKS,
// so they are thousands of numbers rather than millions and the host was
// never the bottleneck in arithmetic. What it cost was a STALL: the
// block means had to come back, per block, thirty times a chunk, which
// serialises the GPU behind a round trip that does almost no work.
//
// Four passes, because the top-k needs the softmax finished and the
// softmax needs the scores finished:
//
//   scores    q-block mean . k-block mean, with the local band as -inf
//   softmax   per query block, in place
//   threshold per GROUP, the (k+1)-th largest, by bisection
//   flags     p > threshold, expanded to the flash kernel's tiles
//
// THE GROUP IS NOT THE ROW. The reference takes its top-k over all the
// query blocks of one temporal window at once, not per query block --
// `rearrange('h (it s1) s2 -> (h it) s1 s2')` -- so a window spends its
// budget where it is needed rather than each row spending an equal
// share. Getting that wrong keeps a plausible number of blocks and the
// wrong ones.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// p[h][i][j] = <qm[h][i], km[h][j]> / sqrt(HD), or -inf where the
// always-exact spatial band excludes the pair.
//
// The band is indexed MODULO the spatial block count: block i of any
// temporal window sits at spatial position i % S, which is what makes
// one [S, S] mask serve every window pair.
//
//   0:qm [H,Nq,HD]  1:km [H,Nk,HD]  2:band [S,S] uchar (1 = allowed)
//   3:p [H,Nq,Nk] f32  4:HD 5:Nq 6:Nk 7:S 8:scale
//   grid {Nk, Nq, H}.
kernel void flashvsr_route_scores(
    const device VPIPE_ELT* qm    [[buffer(0)]],
    const device VPIPE_ELT* km    [[buffer(1)]],
    const device uchar*     band  [[buffer(2)]],
    device float*           p     [[buffer(3)]],
    constant int&           HD    [[buffer(4)]],
    constant int&           Nq    [[buffer(5)]],
    constant int&           Nk    [[buffer(6)]],
    constant int&           S     [[buffer(7)]],
    constant float&         scale [[buffer(8)]],
    uint3 tpig [[thread_position_in_grid]])
{
  const uint j = tpig.x, i = tpig.y, h = tpig.z;
  if (j >= (uint)Nk || i >= (uint)Nq) { return; }
  const ulong o = ((ulong)h * (uint)Nq + i) * (uint)Nk + j;
  if (band[(ulong)(i % (uint)S) * (uint)S + (j % (uint)S)] == 0u) {
    p[o] = -INFINITY;
    return;
  }
  const device VPIPE_ELT* a = qm + ((ulong)h * (uint)Nq + i) * (uint)HD;
  const device VPIPE_ELT* b = km + ((ulong)h * (uint)Nk + j) * (uint)HD;
  float acc = 0.0f;
  for (int d = 0; d < HD; ++d) { acc += (float)a[d] * (float)b[d]; }
  p[o] = acc * scale;
}

#define RT_TG 256

// Softmax over one query block's row, IN PLACE.
//
// A band-excluded pair becomes exactly zero here, and that is what lets
// the threshold below drop it: the two mechanisms are not independent,
// and at a grid the band covers entirely the top-k drops only the single
// lowest block.
//
//   0:p [H,Nq,Nk] f32  1:Nk.  grid {RT_TG, Nq*H}, threadgroup {RT_TG}.
kernel void flashvsr_route_softmax(
    device float* p  [[buffer(0)]],
    constant int& Nk [[buffer(1)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
  device float* row = p + (ulong)tid.y * (uint)Nk;
  const uint lid = ltid.x;
  threadgroup float part[RT_TG / 32];

  float m = -INFINITY;
  for (int k = (int)lid; k < Nk; k += RT_TG) { m = max(m, row[k]); }
  m = simd_max(m);
  if ((lid & 31u) == 0u) { part[lid / 32u] = m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < RT_TG / 32) {
    float v = part[lid];
    v = simd_max(v);
    if (lid == 0u) { part[0] = v; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mx = part[0];

  float s = 0.0f;
  for (int k = (int)lid; k < Nk; k += RT_TG) {
    // A row the band excluded everywhere would be all -inf; exp of that
    // is zero and the sum below stays zero, which the normalise guards.
    const float e = isinf(row[k]) && row[k] < 0.0f ? 0.0f
                                                   : exp(row[k] - mx);
    row[k] = e;
    s += e;
  }
  s = simd_sum(s);
  if ((lid & 31u) == 0u) { part[lid / 32u] = s; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < RT_TG / 32) {
    float v = part[lid];
    v = simd_sum(v);
    if (lid == 0u) { part[0] = v; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float den = part[0];
  if (den <= 0.0f) { return; }
  for (int k = (int)lid; k < Nk; k += RT_TG) { row[k] /= den; }
}

// The (apply+1)-th largest value of one GROUP, by bisection on the float
// BIT PATTERN.
//
// Exact, and that matters: the reference keeps `p > v[apply+1]`, so
// ties at the threshold are dropped, and a threshold that is merely
// close keeps or drops a different set. For non-negative floats the IEEE
// bits order the same way the values do, so a 31-step binary search over
// the bits lands on an actual element value rather than near one --
// counts only change where an element sits.
//
// A full sort would also be exact, but this is 31 counting passes over a
// group of at most a few tens of thousands, which is nothing, and it
// needs no scratch.
//
//   0:p [H,Nq,Nk] f32  1:thr [H*groups] f32
//   2:Nk 3:s1 (query blocks per group) 4:apply
//   grid {RT_TG, H*groups}, threadgroup {RT_TG}.
kernel void flashvsr_route_threshold(
    const device float* p     [[buffer(0)]],
    device float*       thr   [[buffer(1)]],
    constant int&       Nk    [[buffer(2)]],
    constant int&       s1    [[buffer(3)]],
    constant int&       apply [[buffer(4)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
  const uint g = tid.y;
  const uint lid = ltid.x;
  const int n = s1 * Nk;
  const device float* base = p + (ulong)g * (uint)n;
  threadgroup uint part[RT_TG / 32];
  threadgroup uint total;

  // Invariant: count(bits >= lo) >= apply+1 and count(bits >= hi) < it.
  // 0x3F800000 is 1.0f, the largest a probability can be.
  uint lo = 0u, hi = 0x3F800001u;
  const uint want = (uint)apply + 1u;
  for (int it = 0; it < 31; ++it) {
    if (hi - lo <= 1u) { break; }
    const uint mid = lo + (hi - lo) / 2u;
    uint c = 0u;
    for (int k = (int)lid; k < n; k += RT_TG) {
      if (as_type<uint>(base[k]) >= mid) { c += 1u; }
    }
    c = simd_sum(c);
    if ((lid & 31u) == 0u) { part[lid / 32u] = c; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < RT_TG / 32) {
      uint v = part[lid];
      v = simd_sum(v);
      if (lid == 0u) { total = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c_all = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c_all >= want) { lo = mid; } else { hi = mid; }
  }
  if (lid == 0u) { thr[g] = as_type<float>(lo); }
}

// The decision, in the byte form attn_steel's has_block_mask reads:
// [H, NQ, kL], NONZERO EXCLUDES, per query TILE and per KEY.
//
// One routing block spans 128/bq of the flash kernel's query tiles and
// 128 of its keys, so this is a broadcast rather than a computation.
//
//   0:p [H,Nq,Nk] f32  1:thr [H*groups] f32  2:flags [H,NQ,kL] uchar
//   3:Nq 4:Nk 5:NQ 6:kL 7:bq 8:s1
//   grid {kL, NQ, H}.
kernel void flashvsr_route_flags(
    const device float* p     [[buffer(0)]],
    const device float* thr   [[buffer(1)]],
    device uchar*       flags [[buffer(2)]],
    constant int&       Nq    [[buffer(3)]],
    constant int&       Nk    [[buffer(4)]],
    constant int&       NQ    [[buffer(5)]],
    constant int&       kL    [[buffer(6)]],
    constant int&       bq    [[buffer(7)]],
    constant int&       s1    [[buffer(8)]],
    uint3 tpig [[thread_position_in_grid]])
{
  const uint k = tpig.x, qb = tpig.y, h = tpig.z;
  if (k >= (uint)kL || qb >= (uint)NQ) { return; }
  const int i = min(Nq - 1, (int)(qb * (uint)bq) / 128);
  const int j = min(Nk - 1, (int)k / 128);
  const int groups = (s1 > 0) ? (Nq / s1) : 1;
  const int g = (s1 > 0) ? min(groups - 1, i / s1) : 0;
  const float t = thr[(ulong)h * (uint)groups + (uint)g];
  const float v = p[((ulong)h * (uint)Nq + (uint)i) * (uint)Nk + (uint)j];
  flags[((ulong)h * (uint)NQ + qb) * (uint)kL + k] = (v > t) ? 0u : 1u;
}

// ---------------------------------------------------------------------
// THE SPAN PATH.
//
// attn_steel has two ways to be told what to skip, and only one of them
// actually skips. `has_block_mask` is a per-key flag: the kernel still
// walks every key block and writes -inf into the ones it must not read,
// so a 39%-kept routing costs exactly what dense costs. MEASURED: with
// the mask forced to exclude nothing, the attention took the same time
// to the noise floor. `has_spans` instead hands each query block its own
// ASCENDING LIST of key blocks and loops over that, which is where the
// saving lives -- the same reason Sol-Attn runs its exact half there.
//
// THE CONVERSION IS EXACT HERE, with nothing rounded. The kernel's key
// block is BK = 16 and its query tile is BQ = 32, while the routing
// block is 128 of each; 128 divides by both, and every sequence length
// this model produces is a multiple of 128 (f*h*w with h and w multiples
// of 8). So one routing block is exactly 8 whole key blocks and 4 whole
// query tiles, there is no partial block at any edge, and the mask is
// not needed alongside the spans at all.
//
// The window mask attn_steel applies under has_spans is switched off by
// tokens_per_frame = 0, which makes its `q_video` test false for every
// row; span_bounds is then never dereferenced.

// keep[h][i][j] = the BLOCK-level decision, which the span builders
// consume. Same rule as flashvsr_route_flags, one entry per routing
// block pair rather than per key.
//
//   0:p [H,Nq,Nk] f32  1:thr [H*groups] f32  2:keep [H,Nq,Nk] uchar
//   3:Nq 4:Nk 5:s1.  grid {Nk, Nq, H}.
kernel void flashvsr_route_keep(
    const device float* p    [[buffer(0)]],
    const device float* thr  [[buffer(1)]],
    device uchar*       keep [[buffer(2)]],
    constant int&       Nq   [[buffer(3)]],
    constant int&       Nk   [[buffer(4)]],
    constant int&       s1   [[buffer(5)]],
    uint3 tpig [[thread_position_in_grid]])
{
  const uint j = tpig.x, i = tpig.y, h = tpig.z;
  if (j >= (uint)Nk || i >= (uint)Nq) { return; }
  const int groups = (s1 > 0) ? (Nq / s1) : 1;
  const int g = (s1 > 0) ? min(groups - 1, (int)i / s1) : 0;
  const ulong o = ((ulong)h * (uint)Nq + i) * (uint)Nk + j;
  keep[o] = (p[o] > thr[(ulong)h * (uint)groups + (uint)g]) ? 1u : 0u;
}

// FlashVSR+'s answer to a query block the threshold left with NO key
// block: keep its first candidate, as its reference does. A strict
// `p > threshold` can do that when the group's scores tie, and the span
// kernel's softmax over nothing is 0/0 -- a NaN row that every later
// block's attention spreads.
//
// "First" is in the REFERENCE's key order, which is oldest window first.
// The ring holds the same windows in slot order, so the oldest is at
// slot `first_win`; within a window the band is the only candidate rule
// and it is spatial, so the first candidate is that window's lowest
// spatial block inside the row's band.
//
//   0:keep [H,Nq,Nk] uchar  1:band [S,S] uchar
//   2:Nq 3:Nk 4:S 5:first_win.  grid {Nq, H}.
kernel void flashvsr_route_fallback(
    device uchar*       keep      [[buffer(0)]],
    const device uchar* band      [[buffer(1)]],
    constant int&       Nq        [[buffer(2)]],
    constant int&       Nk        [[buffer(3)]],
    constant int&       S         [[buffer(4)]],
    constant int&       first_win [[buffer(5)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const uint i = tpig.x, h = tpig.y;
  if (i >= (uint)Nq) { return; }
  device uchar* row = keep + ((ulong)h * (uint)Nq + i) * (uint)Nk;
  for (int j = 0; j < Nk; ++j) {
    if (row[j] != 0u) { return; }
  }
  const uint si = i % (uint)S;
  for (int j = 0; j < S; ++j) {
    if (band[(ulong)si * (uint)S + (uint)j] != 0u) {
      const int k = first_win * S + j;
      if (k < Nk) { row[k] = 1u; }
      return;
    }
  }
}

// qb_off[H][NQ + 1], the CSR offsets into qb_blocks.
//
// Serial in ONE thread on purpose: the array is H*(NQ+1) entries -- a
// couple of thousand at any geometry this model runs -- so a parallel
// scan would cost more in launch and barriers than the whole loop, and
// the offsets have to be globally monotonic across heads because
// qb_blocks is one flat array.
//
//   0:keep [H,Nq,Nk] uchar  1:qb_off [H*(NQ+1)] int
//   2:Nq 3:Nk 4:NQ 5:H 6:per_k (key blocks per routing block)
//   7:per_q (query tiles per routing block).  grid {1,1,1}.
kernel void flashvsr_span_offsets(
    const device uchar* keep   [[buffer(0)]],
    device int*         qb_off [[buffer(1)]],
    constant int&       Nq     [[buffer(2)]],
    constant int&       Nk     [[buffer(3)]],
    constant int&       NQ     [[buffer(4)]],
    constant int&       H      [[buffer(5)]],
    constant int&       per_k  [[buffer(6)]],
    constant int&       per_q  [[buffer(7)]])
{
  int total = 0;
  for (int h = 0; h < H; ++h) {
    for (int qb = 0; qb < NQ; ++qb) {
      qb_off[h * (NQ + 1) + qb] = total;
      const int i = min(Nq - 1, qb / per_q);
      int n = 0;
      for (int j = 0; j < Nk; ++j) {
        if (keep[((ulong)h * (uint)Nq + (uint)i) * (uint)Nk + (uint)j] != 0u) {
          ++n;
        }
      }
      total += n * per_k;
    }
    qb_off[h * (NQ + 1) + NQ] = total;
  }
}

// qb_blocks: for each query tile, the ascending key-BLOCK indices it
// visits. ASCENDING IS LOAD-BEARING -- the kernel's loaders only move
// forward and step by the gap since the last visited block, so an
// out-of-order list reads the wrong keys rather than failing.
//
//   0:keep  1:qb_off  2:qb_blocks int  3:Nq 4:Nk 5:NQ 6:per_k 7:per_q
//   grid {NQ, H}.
kernel void flashvsr_span_emit(
    const device uchar* keep      [[buffer(0)]],
    const device int*   qb_off    [[buffer(1)]],
    device int*         qb_blocks [[buffer(2)]],
    constant int&       Nq        [[buffer(3)]],
    constant int&       Nk        [[buffer(4)]],
    constant int&       NQ        [[buffer(5)]],
    constant int&       per_k     [[buffer(6)]],
    constant int&       per_q     [[buffer(7)]],
    uint2 tpig [[thread_position_in_grid]])
{
  const uint qb = tpig.x, h = tpig.y;
  if (qb >= (uint)NQ) { return; }
  int w = qb_off[(int)h * (NQ + 1) + (int)qb];
  const int i = min(Nq - 1, (int)qb / per_q);
  for (int j = 0; j < Nk; ++j) {
    if (keep[((ulong)h * (uint)Nq + (uint)i) * (uint)Nk + (uint)j] == 0u) {
      continue;
    }
    for (int t = 0; t < per_k; ++t) { qb_blocks[w++] = j * per_k + t; }
  }
}
