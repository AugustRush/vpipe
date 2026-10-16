// band_split_roformer.metal -- the ops of a BS-RoFormer source separator
// (generative-models/bs-roformer/) that no other model here has. The
// transformer itself runs on the shared kernels (rms_norm_f16, the dense
// GEMMs, attn_steel / attn_steel_nax, gelu_erf_f16, residual_add_f16,
// transpose_abd_f16); what is here is the BAND structure around it and
// the two things its attention does differently:
//
//   * the band split: per band, an RMSNorm over that band's slice of the
//     frame's spectrum and a Linear of its own into the model width;
//   * rotary positions on rows that stay TOKEN-MAJOR -- the time and the
//     frequency attention read the same qkv buffer through strides, so
//     the position of a token is its frame or its band, not its row;
//   * the per-head sigmoid gates on the attention output;
//   * the mask head's tanh and its GLU, written straight into the
//     frame's spectrum layout.
//
// f16 storage, f32 math. The spectrum and the mask stay f32: they meet
// the STFT on the host, which is f32.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

// Rows of frames per threadgroup in the band split, and the widest band
// it takes (a band's features: bins x channels x 2).
#define BSR_ROWS 4
#define BSR_MAXK 1024

// The band split. For band b of frame t, over its features
// x = spec[t, off_b : off_b + len_b]:
//
//   y = F.normalize(x) * sqrt(len_b) * gamma_b
//                       (F.normalize: x / max(|x|, 1e-12))
//   out[t, b, :] = y @ W_b^T + bias_b
//
//   0:spec f32 [T, Fin]   1:band int [nb, 2] (off, len)
//   2:gamma f32 [Fin]     3:wt f32 [Fin, D] (W_b^T stacked in band order)
//   4:bias f16 [nb, D]    5:out f16 [T, nb, D]
//   6:T 7:Fin 8:nb 9:D
// One threadgroup of D threads (D % 32 == 0, D <= 1024) per BSR_ROWS
// frames of one band: grid (D, ceil(T / BSR_ROWS), nb) threadgroups.
kernel void bsr_band_split_f16(
    const device float*     spec  [[buffer(0)]],
    const device int*       band  [[buffer(1)]],
    const device float*     gamma [[buffer(2)]],
    const device float*     wt    [[buffer(3)]],
    const device VPIPE_ELT* bias  [[buffer(4)]],
    device VPIPE_ELT*       out   [[buffer(5)]],
    constant int& T   [[buffer(6)]],
    constant int& Fin [[buffer(7)]],
    constant int& nb  [[buffer(8)]],
    constant int& D   [[buffer(9)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]],
    uint3 ntg3 [[threads_per_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]],
    uint  sl  [[thread_index_in_simdgroup]])
{
  threadgroup float xs[BSR_ROWS * BSR_MAXK];
  threadgroup float red[BSR_ROWS * 32];
  const uint ntg = ntg3.x;
  const int b = (int)tg.z;
  const int t0 = (int)tg.y * BSR_ROWS;
  const int off = band[2 * b], len = band[2 * b + 1];

  float ss[BSR_ROWS];
  for (int r = 0; r < BSR_ROWS; ++r) {
    ss[r] = 0.0f;
    const int t = t0 + r;
    for (int k = (int)lid; k < len; k += (int)ntg) {
      const float v = t < T ? spec[(long)t * Fin + off + k] : 0.0f;
      xs[r * BSR_MAXK + k] = v;
      ss[r] += v * v;
    }
  }
  for (int r = 0; r < BSR_ROWS; ++r) {
    const float s = simd_sum(ss[r]);
    if (sl == 0) { red[r * 32 + sg] = s; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int nsg = ((int)ntg + 31) / 32;
  float scale[BSR_ROWS];
  for (int r = 0; r < BSR_ROWS; ++r) {
    float tot = 0.0f;
    for (int i = 0; i < nsg; ++i) { tot += red[r * 32 + i]; }
    scale[r] = sqrt((float)len) / max(sqrt(tot), 1e-12f);
  }
  for (int r = 0; r < BSR_ROWS; ++r) {
    for (int k = (int)lid; k < len; k += (int)ntg) {
      xs[r * BSR_MAXK + k] *= scale[r] * gamma[off + k];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int o = (int)lid;
  if (o >= D) { return; }
  float acc[BSR_ROWS];
  for (int r = 0; r < BSR_ROWS; ++r) { acc[r] = 0.0f; }
  const device float* w = wt + (long)off * D + o;
  for (int k = 0; k < len; ++k) {
    const float wk = w[(long)k * D];
    for (int r = 0; r < BSR_ROWS; ++r) {
      acc[r] = fma(xs[r * BSR_MAXK + k], wk, acc[r]);
    }
  }
  const float bo = float(bias[b * D + o]);
  for (int r = 0; r < BSR_ROWS; ++r) {
    const int t = t0 + r;
    if (t < T) {
      out[((long)t * nb + b) * D + o] = VPIPE_ELT(acc[r] + bo);
    }
  }
}

// Rotary positions in place on token-major q and k, the pairs ADJACENT
// (rotary-embedding-torch's convention: x[2i], x[2i+1] rotate together
// by pos * freqs[i]).
//
// A token's position is (token / div) % mod: its frame (div = bands)
// for the time attention, its band (div = 1, mod = bands) for the
// frequency one -- the buffer is the same and only the reading changes.
// The angles come from a host table of (cos, sin) per (position, pair),
// computed from the f32 angle exactly as the reference forms it.
//
//   0:qkv [N, row]: q at column 0, k at column heads * Dh
//   1:cs float2 [positions, Dh / 2]
//   2:row 3:heads 4:Dh 5:div 6:mod 7:N
// One thread per (token, head of q or k), two pairs at a time:
// grid (2 * heads, N). Dh % 4 == 0.
kernel void bsr_rope_rows_f16(
    device VPIPE_ELT*    qkv [[buffer(0)]],
    const device float2* cs  [[buffer(1)]],
    constant int& row   [[buffer(2)]],
    constant int& heads [[buffer(3)]],
    constant int& Dh    [[buffer(4)]],
    constant int& div   [[buffer(5)]],
    constant int& mod   [[buffer(6)]],
    constant int& N     [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int hh = (int)gid.x;
  const int tok = (int)gid.y;
  if (hh >= 2 * heads || tok >= N) { return; }
  const int half_d = Dh / 2;
  const int pos = (tok / div) % mod;
  const int which = hh / heads, h = hh % heads;
  device VPIPE_ELT* x = qkv + (long)tok * row + (long)which * heads * Dh +
                        (long)h * Dh;
  const device float2* c = cs + (long)pos * half_d;
  for (int i = 0; i < half_d; i += 2) {
    device vec<VPIPE_ELT, 4>* p = (device vec<VPIPE_ELT, 4>*)(x + 2 * i);
    const float4 v = float4(*p);
    const float2 a = c[i], b = c[i + 1];
    *p = vec<VPIPE_ELT, 4>(float4(v.x * a.x - v.y * a.y,
                                  v.y * a.x + v.x * a.y,
                                  v.z * b.x - v.w * b.y,
                                  v.w * b.x + v.z * b.y));
  }
}

#define BSR_MAXH 16

// The attention's per-head gates, in place:
//   att[tok, h, :] *= sigmoid(xn[tok, :] . wg[h, :] + bg[h])
// where xn is the normed input the attention projected q, k, v from.
//
//   0:att [N, H * Dh]  1:xn [N, D]  2:wg [H, D]  3:bg [H]
//   4:N 5:D 6:H 7:Dh
// One simdgroup per token (H <= 16): grid (32 * N) threads.
kernel void bsr_head_gate_f16(
    device VPIPE_ELT*       att [[buffer(0)]],
    const device VPIPE_ELT* xn  [[buffer(1)]],
    const device VPIPE_ELT* wg  [[buffer(2)]],
    const device VPIPE_ELT* bg  [[buffer(3)]],
    constant int& N  [[buffer(4)]],
    constant int& D  [[buffer(5)]],
    constant int& H  [[buffer(6)]],
    constant int& Dh [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint sl  [[thread_index_in_simdgroup]])
{
  const int tok = (int)(tid / 32);
  if (tok >= N) { return; }
  float g[BSR_MAXH];
  const device VPIPE_ELT* x = xn + (long)tok * D;
  for (int h = 0; h < H; ++h) {
    float p = 0.0f;
    for (int k = (int)sl; k < D; k += 32) {
      p = fma(float(x[k]), float(wg[h * D + k]), p);
    }
    const float z = simd_sum(p) + float(bg[h]);
    g[h] = 1.0f / (1.0f + exp(-z));
  }
  device VPIPE_ELT* a = att + (long)tok * H * Dh;
  for (int e = (int)sl; e < H * Dh; e += 32) {
    a[e] = VPIPE_ELT(float(a[e]) * g[e / Dh]);
  }
}

// y = tanh(y + bias[col]) in place over [rows, N]: a mask-head hidden
// layer after a biasless GEMM.
//   0:y 1:bias [N] 2:N 3:total (rows * N)
kernel void bsr_bias_tanh_f16(
    device VPIPE_ELT*       y    [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    constant int&  N     [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= total) { return; }
  const float v = float(y[gid]) + float(bias[gid % (uint)N]);
  y[gid] = VPIPE_ELT(precise::tanh(v));
}

// One band's mask, from its last Linear (biasless GEMM output m) through
// the GLU, into the frame's spectrum layout:
//   mask[t, off + j] = (m[t, j] + b[j]) * sigmoid(m[t, w + j] + b[w + j])
//
//   0:m f16 [T, 2w]  1:bias f16 [2w]  2:mask f32 [T, Fin]
//   3:T 4:w 5:Fin 6:off
// grid (w, T).
kernel void bsr_band_glu_f16(
    const device VPIPE_ELT* m    [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    device float*           mask [[buffer(2)]],
    constant int& T   [[buffer(3)]],
    constant int& w   [[buffer(4)]],
    constant int& Fin [[buffer(5)]],
    constant int& off [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int j = (int)gid.x, t = (int)gid.y;
  if (j >= w || t >= T) { return; }
  const device VPIPE_ELT* r = m + (long)t * 2 * w;
  const float a = float(r[j]) + float(bias[j]);
  const float g = float(r[w + j]) + float(bias[w + j]);
  mask[(long)t * Fin + off + j] = a / (1.0f + exp(-g));
}

// erf by Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7) -- the same
// polynomial llm_elementwise's gelu_erf_f16 uses, so the fused form
// below rounds as the unfused pair did.
inline float
bsr_erf_(float x)
{
  const float p = 0.3275911f;
  const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f,
              a4 = -1.453152027f, a5 = 1.061405429f;
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  const float ax = fabs(x);
  const float t = 1.0f / (1.0f + p * ax);
  const float y = 1.0f -
      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
  return s * y;
}

// y = gelu(y + bias[col]) in place over [rows, N], exact (erf) GELU:
// the feed-forward's hidden layer after a GEMM that took no bias
// (matmul2d's), in ONE pass over the block's widest activation instead
// of a bias pass and a GELU pass.
//   0:y 1:bias [N] 2:N 3:total (rows * N)
kernel void bsr_bias_gelu_f16(
    device VPIPE_ELT*       y    [[buffer(0)]],
    const device VPIPE_ELT* bias [[buffer(1)]],
    constant int&  N     [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= total) { return; }
  // The bias add rounds to storage first, as the unfused pair did.
  const float v = float(VPIPE_ELT(float(y[gid]) + float(bias[gid % (uint)N])));
  y[gid] = VPIPE_ELT(0.5f * v * (1.0f + bsr_erf_(v * 0.7071067811865476f)));
}

// x += y + bias[col] over [rows, N]: a biased projection's output into
// the residual stream, after a GEMM that took no bias.
//   0:x 1:y 2:bias [N] 3:N 4:total
kernel void bsr_bias_residual_f16(
    device VPIPE_ELT*       x    [[buffer(0)]],
    const device VPIPE_ELT* y    [[buffer(1)]],
    const device VPIPE_ELT* bias [[buffer(2)]],
    constant int&  N     [[buffer(3)]],
    constant uint& total [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= total) { return; }
  const float b = float(VPIPE_ELT(float(y[gid]) + float(bias[gid % (uint)N])));
  x[gid] = VPIPE_ELT(float(x[gid]) + b);
}
