// fish_dac.metal -- the ops of Fish Audio's modded DAC codec (the S2 Pro /
// Drama 3 codec.pth) that the shared conv / norm / attention kernels do not
// already cover: the residual VQ's decode and encode, the windowed
// transformers' q/k/v split with their interleaved rotary, the heads back
// to token-major, and the layer-scaled residual add.
//
// Everything is TIME-MAJOR [T, C], as in audio_vae_1d.metal. Element
// (storage) type half; every sum and product f32.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define FD_NCB 10

// The quantized latent of `T` frames: z[t] = sum over the ten codebooks of
// table_q[code_q] -- each table is its codebook already through its
// out_proj (weight-normed 1x1 conv + bias), so a lookup is the quantizer's
// whole decode. Codes are clamped as the reference clamps them (semantic
// < sizes[0], residual < sizes[1]).
//   0:codes[T,10] 1:tables(f32) 2:z[T,C] 3:C 4:sem_size 5:res_size
//   grid (C, T).
kernel void fish_vq_decode(
    const device int*   codes  [[buffer(0)]],
    const device float* tables [[buffer(1)]],
    device VPIPE_ELT*   z      [[buffer(2)]],
    constant int&       C      [[buffer(3)]],
    constant int&       sem_size [[buffer(4)]],
    constant int&       res_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, t = (int)gid.y;
  if (c >= C) { return; }
  const device int* cd = codes + (int64_t)t * FD_NCB;
  float acc = 0.0f;
  int64_t base = 0;
  for (int q = 0; q < FD_NCB; ++q) {
    const int n = q == 0 ? sem_size : res_size;
    const int code = clamp(cd[q], 0, n - 1);
    acc += tables[(base + code) * C + c];
    base += n;
  }
  z[(int64_t)t * C + c] = (VPIPE_ELT)acc;
}

// The residual VQ's encode, one threadgroup a frame: for each codebook in
// turn, e = in_proj(residual) (8 wide), the code whose L2-NORMALISED entry
// is nearest the normalised e (ties to the lower index), then residual -=
// table[code]. `win` [q][8][C] and `bin` [q][8] are the in_projs (weight
// norm folded, f32), `cbn` the codebooks normalised [sum sizes][8] f32.
//   0:z[T,C] 1:win 2:bin 3:cbn 4:tables 5:codes[T,10] 6:C 7:sem_size
//   8:res_size.   grid (256, T); threadgroup (256).
kernel void fish_vq_encode(
    const device VPIPE_ELT* z      [[buffer(0)]],
    const device float*     win    [[buffer(1)]],
    const device float*     bin    [[buffer(2)]],
    const device float*     cbn    [[buffer(3)]],
    const device float*     tables [[buffer(4)]],
    device int*             codes  [[buffer(5)]],
    constant int&           C      [[buffer(6)]],
    constant int&           sem_size [[buffer(7)]],
    constant int&           res_size [[buffer(8)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]])
{
  constexpr int TG = 256, D = 8, CMAX = 1024;
  threadgroup float r[CMAX];
  threadgroup float part[TG / 32][D];
  threadgroup float e[D];
  threadgroup float bestv[TG / 32];
  threadgroup int besti[TG / 32];
  threadgroup int chosen;
  const int t = (int)tg.y;
  const uint lid = ltid.x;
  for (int c = (int)lid; c < C; c += TG) {
    r[c] = (float)z[(int64_t)t * C + c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int64_t base = 0;
  for (int q = 0; q < FD_NCB; ++q) {
    const int n = q == 0 ? sem_size : res_size;
    // e = W_in r + b.
    float acc[D];
    for (int j = 0; j < D; ++j) { acc[j] = 0.0f; }
    for (int c = (int)lid; c < C; c += TG) {
      const float x = r[c];
      for (int j = 0; j < D; ++j) {
        acc[j] += win[((int64_t)q * D + j) * C + c] * x;
      }
    }
    for (int j = 0; j < D; ++j) {
      const float s = simd_sum(acc[j]);
      if ((lid & 31) == 0) { part[lid >> 5][j] = s; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < (uint)D) {
      float s = bin[q * D + (int)lid];
      for (int g = 0; g < TG / 32; ++g) { s += part[g][lid]; }
      e[lid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // F.normalize: e / max(|e|, 1e-12).
    float nn = 0.0f;
    for (int j = 0; j < D; ++j) { nn += e[j] * e[j]; }
    const float inv = 1.0f / max(sqrt(nn), 1e-12f);
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = (int)lid; i < n; i += TG) {
      // -dist up to |e|^2, a constant of the row: 2 e.c - |c|^2, as the
      // reference ranks it (|c| is 1 only to rounding).
      const device float* cv = cbn + (base + i) * D;
      float dot = 0.0f, cc = 0.0f;
      for (int j = 0; j < D; ++j) {
        dot += e[j] * inv * cv[j];
        cc += cv[j] * cv[j];
      }
      const float d = 2.0f * dot - cc;
      if (d > bv || (d == bv && i < bi)) { bv = d; bi = i; }
    }
    for (uint o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(bv, o);
      const int oi = simd_shuffle_down(bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((lid & 31) == 0) { bestv[lid >> 5] = bv; besti[lid >> 5] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
      float v = bestv[0];
      int i0 = besti[0];
      for (int g = 1; g < TG / 32; ++g) {
        if (bestv[g] > v || (bestv[g] == v && besti[g] < i0)) {
          v = bestv[g]; i0 = besti[g];
        }
      }
      chosen = i0;
      codes[(int64_t)t * FD_NCB + q] = i0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int code = chosen;
    for (int c = (int)lid; c < C; c += TG) {
      r[c] -= tables[(base + code) * C + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    base += n;
  }
}

// A transformer's fused q|k|v [T, 3*H*D] split HEAD-MAJOR: q into
// [H, qstride, D], k and v into [H, kvstride, D], q and k rotated: GPT-J
// pairs (2i, 2i+1) over the whole head at the host's cos / sin table
// [pos][D] (cos at 2i, sin at 2i+1), which holds the reference's
// bf16-rounded values.
// Row t of the call is position pos0 + t (the table's row) and K/V row
// kvrow0 + t: a streamed chunk appends to a history.
//   0:qkv 1:q 2:k 3:v 4:cs 5:H 6:D 7:qstride 8:kvstride 9:pos0 10:kvrow0.
//   grid (D/2, H, T).
kernel void fish_qkv_rope(
    const device VPIPE_ELT* qkv [[buffer(0)]],
    device VPIPE_ELT*       q   [[buffer(1)]],
    device VPIPE_ELT*       k   [[buffer(2)]],
    device VPIPE_ELT*       v   [[buffer(3)]],
    const device float*     cs  [[buffer(4)]],
    constant int&           H   [[buffer(5)]],
    constant int&           D   [[buffer(6)]],
    constant int&           qstride  [[buffer(7)]],
    constant int&           kvstride [[buffer(8)]],
    constant int&           pos0     [[buffer(9)]],
    constant int&           kvrow0   [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x, h = (int)gid.y, t = (int)gid.z;
  if (2 * i >= D) { return; }
  const int W = H * D;
  const device VPIPE_ELT* row = qkv + (int64_t)t * 3 * W + h * D + 2 * i;
  const float c = cs[(int64_t)(pos0 + t) * D + 2 * i];
  const float s = cs[(int64_t)(pos0 + t) * D + 2 * i + 1];
  const int64_t oq = ((int64_t)h * qstride + t) * D + 2 * i;
  const int64_t ok = ((int64_t)h * kvstride + kvrow0 + t) * D + 2 * i;
  {
    const float x0 = (float)row[0], x1 = (float)row[1];
    q[oq] = (VPIPE_ELT)(x0 * c - x1 * s);
    q[oq + 1] = (VPIPE_ELT)(x1 * c + x0 * s);
  }
  {
    const float x0 = (float)row[W], x1 = (float)row[W + 1];
    k[ok] = (VPIPE_ELT)(x0 * c - x1 * s);
    k[ok + 1] = (VPIPE_ELT)(x1 * c + x0 * s);
  }
  v[ok] = row[2 * W];
  v[ok + 1] = row[2 * W + 1];
}

// Attention output [H, stride, D] back to token-major [T, H*D].
//   0:att 1:out 2:H 3:D 4:stride.   grid (H*D, T).
kernel void fish_heads_tok(
    const device VPIPE_ELT* att [[buffer(0)]],
    device VPIPE_ELT*       out [[buffer(1)]],
    constant int&           H   [[buffer(2)]],
    constant int&           D   [[buffer(3)]],
    constant int&           stride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x, t = (int)gid.y;
  if (i >= H * D) { return; }
  const int h = i / D, d = i % D;
  out[(int64_t)t * H * D + i] = att[((int64_t)h * stride + t) * D + d];
}

// x[t, c] += g[c] * y[t, c] -- a LayerScale'd residual (g f32).
//   0:x (inout) 1:y 2:g 3:C 4:n.   grid (n).
kernel void fish_scale_add(
    device VPIPE_ELT*       x [[buffer(0)]],
    const device VPIPE_ELT* y [[buffer(1)]],
    const device float*     g [[buffer(2)]],
    constant int&           C [[buffer(3)]],
    constant int&           n [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= (uint)n) { return; }
  const int c = (int)(gid % (uint)C);
  x[gid] = (VPIPE_ELT)((float)x[gid] + g[c] * (float)y[gid]);
}

// Causal depthwise conv with bias: out[t, c] = b[c] + sum_k w[c, k] *
// in[t + k - (K - 1), c], zero before the start.
//   0:in[T,C] 1:w[C,K](f32) 2:b[C](f32) 3:out 4:T 5:C 6:K.  grid (C, T).
kernel void fish_dwconv_causal(
    const device VPIPE_ELT* in  [[buffer(0)]],
    const device float*     w   [[buffer(1)]],
    const device float*     b   [[buffer(2)]],
    device VPIPE_ELT*       out [[buffer(3)]],
    constant int&           T   [[buffer(4)]],
    constant int&           C   [[buffer(5)]],
    constant int&           K   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int c = (int)gid.x, t = (int)gid.y;
  if (c >= C || t >= T) { return; }
  float acc = b[c];
  for (int k = 0; k < K; ++k) {
    const int ti = t + k - (K - 1);
    if (ti < 0) { continue; }
    acc += w[c * K + k] * (float)in[(int64_t)ti * C + c];
  }
  out[(int64_t)t * C + c] = (VPIPE_ELT)acc;
}
