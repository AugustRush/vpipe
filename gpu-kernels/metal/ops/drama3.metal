// drama3.metal -- the kernels of Fish Audio Drama 3's speech LM that are
// its own (generative-models/drama3/): the backbone's frame embedding, the
// codebook-MTP tower's input mix, its q/k norm + interleaved partial RoPE
// into the per-layer K/V caches, its gated attention output, the fused
// residual + norm, and the frame sampler.
//
// A FRAME is one int row [main, c0 .. c9]: `main` is the token the
// backbone reads -- a text id, or semantic_begin + c0 for an audio frame --
// and c0..c9 are the codec's ten codebooks (zero on a text row).
//
// Element (storage) type: half by default; -DVPIPE_ELT=bfloat builds the
// bf16 variant (drama3_bf16), which is the one the model runs. Every
// reduction and every product is f32.

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT half
#endif

#define D3_FRAME 11

inline bool d3_is_sem(int t, int sem0, int sem1)
{
  return t >= sem0 && t <= sem1;
}

// Backbone input rows: x[r] = E_lm[main] on a text row; on an audio row
// (E_lm[main] + E_cb[cbsize + c1]) / sqrt(2) -- code 1 is the one codebook
// besides the semantic code the backbone sees (vq_backbone_codebooks 1).
// E_lm[main] comes from `elm`: rows the host gathered in row order
// (gathered 1, a prompt), or a band of the table starting at token `base`
// (gathered 0, the per-frame step: the semantic band).
//   0:frames[M,11] 1:elm 2:E_cb[10*cbsize,H] 3:x[M,H] 4:H 5:sem0 6:sem1
//   7:cbsize 8:gathered 9:base 10:band_rows.   grid (H, M).
kernel void drama3_embed_rows(
    const device int*       frames [[buffer(0)]],
    const device VPIPE_ELT* elm    [[buffer(1)]],
    const device VPIPE_ELT* ecb    [[buffer(2)]],
    device VPIPE_ELT*       x      [[buffer(3)]],
    constant int&           H      [[buffer(4)]],
    constant int&           sem0   [[buffer(5)]],
    constant int&           sem1   [[buffer(6)]],
    constant int&           cbsize [[buffer(7)]],
    constant int&           gathered [[buffer(8)]],
    constant int&           base   [[buffer(9)]],
    constant int&           band_rows [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int d = (int)gid.x, r = (int)gid.y;
  if (d >= H) { return; }
  const device int* f = frames + (int64_t)r * D3_FRAME;
  const int tok = f[0];
  const int64_t er = gathered != 0 ? (int64_t)r
                                   : (int64_t)clamp(tok - base, 0,
                                                    band_rows - 1);
  float v = (float)elm[er * H + d];
  if (d3_is_sem(tok, sem0, sem1)) {
    const int c1 = clamp(f[2], 0, cbsize - 1);
    v = (v + (float)ecb[(int64_t)(cbsize + c1) * H + d]) * 0.70710678118654752f;
  }
  x[(int64_t)r * H + d] = (VPIPE_ELT)v;
}

// One threadgroup reduction (256 threads) of a per-thread partial sum.
inline float d3_tg_sum(float v, threadgroup float* sh, uint lid, uint nthg)
{
  v = simd_sum(v);
  if ((lid & 31) == 0) { sh[lid >> 5] = v; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0.0f;
  for (uint i = 0; i < (nthg >> 5); ++i) { t += sh[i]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// The tower layer's input, ahead of eh_proj: cat(enorm(e), hnorm(h)) [M, 2H]
// where e = the layer's code embedding table[code] on a row whose LABEL
// (the next frame) is audio, else 0, plus -- on the first layer only -- the
// INPUT frame's codes 2..9 through the shared codebook table, on a row
// whose input is audio. Both norms are the zero-centred RMSNorm; every
// norm weight here is f32 with its +1 already added (exact -- in the
// storage type the +1 would round a weight of 0.01 by up to 0.4%).
//   0:labels[M,11] 1:table[cbsize,H] 2:h[M,H] 3:enorm[H] 4:hnorm[H]
//   5:out[M,2H] 6:H 7:eps 8:code_col (1..10, the label column embedded)
//   9:sem0 10:sem1 11:cbsize 12:has_extra 13:frames[M,11] (inputs)
//   14:E_cb[10*cbsize,H] 15:extra_first 16:extra_end.
//   grid (256, M); threadgroup (256).
kernel void drama3_mix_norm(
    const device int*       labels    [[buffer(0)]],
    const device VPIPE_ELT* table     [[buffer(1)]],
    const device VPIPE_ELT* h         [[buffer(2)]],
    const device float*     enorm     [[buffer(3)]],
    const device float*     hnorm     [[buffer(4)]],
    device VPIPE_ELT*       out       [[buffer(5)]],
    constant int&           H         [[buffer(6)]],
    constant float&         eps       [[buffer(7)]],
    constant int&           code_col  [[buffer(8)]],
    constant int&           sem0      [[buffer(9)]],
    constant int&           sem1      [[buffer(10)]],
    constant int&           cbsize    [[buffer(11)]],
    constant int&           has_extra [[buffer(12)]],
    const device int*       frames    [[buffer(13)]],
    const device VPIPE_ELT* ecb       [[buffer(14)]],
    constant int&           x_first   [[buffer(15)]],
    constant int&           x_end     [[buffer(16)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]])
{
  threadgroup float sh[32];
  const uint lid = ltid.x, nthg = tptg.x;
  const int r = (int)tg.y;
  const device int* lab = labels + (int64_t)r * D3_FRAME;
  const bool lsem = d3_is_sem(lab[0], sem0, sem1);
  const int code = clamp(lab[code_col], 0, cbsize - 1);
  const device int* inp = frames + (int64_t)r * D3_FRAME;
  const bool isem = has_extra != 0 && d3_is_sem(inp[0], sem0, sem1);
  constexpr int PER = 8;              // H <= 256 * 8
  float ev[PER], hv[PER];
  float se = 0.0f, shh = 0.0f;
  for (int j = 0; j < PER; ++j) {
    const int d = (int)lid + j * (int)nthg;
    float e = 0.0f, x = 0.0f;
    if (d < H) {
      if (lsem) { e = (float)table[(int64_t)code * H + d]; }
      if (isem) {
        for (int k = x_first; k < x_end; ++k) {
          const int c = clamp(inp[1 + k], 0, cbsize - 1);
          e += (float)ecb[(int64_t)(k * cbsize + c) * H + d];
        }
      }
      x = (float)h[(int64_t)r * H + d];
    }
    ev[j] = e; hv[j] = x;
    se += e * e; shh += x * x;
  }
  se = d3_tg_sum(se, sh, lid, nthg);
  shh = d3_tg_sum(shh, sh, lid, nthg);
  const float ie = rsqrt(se / (float)H + eps);
  const float ih = rsqrt(shh / (float)H + eps);
  device VPIPE_ELT* o = out + (int64_t)r * 2 * H;
  for (int j = 0; j < PER; ++j) {
    const int d = (int)lid + j * (int)nthg;
    if (d < H) {
      o[d] = (VPIPE_ELT)(ev[j] * ie * (float)enorm[d]);
      o[H + d] = (VPIPE_ELT)(hv[j] * ih * (float)hnorm[d]);
    }
  }
}

// r = a + b (stored to `res`), out = rmsnorm(r) * w -- the tower's two
// pre-norm residual joins (f32 weight with its +1). has_b 0: a plain norm
// of a (b and res unread / unwritten).
//   0:a[M,H] 1:b[M,H] 2:res[M,H] 3:w[H] 4:out[M,H] 5:H 6:eps 7:has_b
//   grid (256, M); threadgroup (256).
kernel void drama3_add_norm(
    const device VPIPE_ELT* a   [[buffer(0)]],
    const device VPIPE_ELT* b   [[buffer(1)]],
    device VPIPE_ELT*       res [[buffer(2)]],
    const device float*     w   [[buffer(3)]],
    device VPIPE_ELT*       out [[buffer(4)]],
    constant int&           H   [[buffer(5)]],
    constant float&         eps [[buffer(6)]],
    constant int&           has_b [[buffer(7)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint3 ltid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]])
{
  threadgroup float sh[32];
  const uint lid = ltid.x, nthg = tptg.x;
  const int r = (int)tg.y;
  constexpr int PER = 8;
  float v[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; ++j) {
    const int d = (int)lid + j * (int)nthg;
    float x = 0.0f;
    if (d < H) {
      const int64_t i = (int64_t)r * H + d;
      if (has_b != 0) {
        // Round the sum to the storage type first: the residual stream is
        // what the next join reads, so the norm sees the stored value.
        x = (float)(VPIPE_ELT)((float)a[i] + (float)b[i]);
        res[i] = (VPIPE_ELT)x;
      } else {
        x = (float)a[i];
      }
    }
    v[j] = x;
    ss += x * x;
  }
  ss = d3_tg_sum(ss, sh, lid, nthg);
  const float inv = rsqrt(ss / (float)H + eps);
  for (int j = 0; j < PER; ++j) {
    const int d = (int)lid + j * (int)nthg;
    if (d < H) {
      out[(int64_t)r * H + d] = (VPIPE_ELT)(v[j] * inv * (float)w[d]);
    }
  }
}

// q/k of one tower layer: per-head zero-centred RMSNorm (f32 weight, +1 in),
// then GPT-J (INTERLEAVED) rotary on the first `rot` channels -- pairs
// (2i, 2i+1) at angle pos * inv_freq[i], from the host's f32 cos/sin table
// -- and the rest passed through. q goes HEAD-MAJOR into qh[Hq, qstride, D]
// (the attention's layout), k and v into the layer's caches [Hkv, cap, D]
// at their positions. The gate stays where the projection left it.
// The projection's row is [q (Hq*D) | gate (Hq*D) | k (Hkv*D) | v (Hkv*D)].
//   0:qkv[M, 2*Hq*D + 2*Hkv*D] 1:qn[D] 2:kn[D] 3:qh 4:kc 5:vc 6:cs[pos, rot]
//   (cos at [p][2i], sin at [p][2i+1]) 7:D 8:Hq 9:Hkv 10:rot 11:pos0
//   12:qstride 13:cap 14:eps.
//   grid (32, Hq + 2*Hkv, M); threadgroup (32): one simdgroup a head-row.
kernel void drama3_qk_rope(
    const device VPIPE_ELT* qkv  [[buffer(0)]],
    const device float*     qn   [[buffer(1)]],
    const device float*     kn   [[buffer(2)]],
    device VPIPE_ELT*       qh   [[buffer(3)]],
    device VPIPE_ELT*       kc   [[buffer(4)]],
    device VPIPE_ELT*       vc   [[buffer(5)]],
    const device float*     cs   [[buffer(6)]],
    constant int&           D    [[buffer(7)]],
    constant int&           Hq   [[buffer(8)]],
    constant int&           Hkv  [[buffer(9)]],
    constant int&           rot  [[buffer(10)]],
    constant int&           pos0 [[buffer(11)]],
    constant int&           qstride [[buffer(12)]],
    constant int&           cap  [[buffer(13)]],
    constant float&         eps  [[buffer(14)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
  const int head = (int)tg.y, r = (int)tg.z;
  const int qd = Hq * D, kd = Hkv * D;
  const int64_t row = (int64_t)r * (2 * qd + 2 * kd);
  const int pos = pos0 + r;
  constexpr int PER = 8;              // D <= 256: 8 contiguous a lane
  const int base = (int)lane * PER;
  if (head >= Hq + Hkv) {             // v: a plain copy into the cache
    const int hv = head - Hq - Hkv;
    const device VPIPE_ELT* src = qkv + row + 2 * qd + kd + hv * D;
    device VPIPE_ELT* dst = vc + ((int64_t)hv * cap + pos) * D;
    for (int j = 0; j < PER; ++j) {
      if (base + j < D) { dst[base + j] = src[base + j]; }
    }
    return;
  }
  const bool isq = head < Hq;
  const device VPIPE_ELT* src =
      isq ? qkv + row + head * D : qkv + row + 2 * qd + (head - Hq) * D;
  const device float* w = isq ? qn : kn;
  float v[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; ++j) {
    const int d = base + j;
    v[j] = d < D ? (float)src[d] : 0.0f;
    ss += v[j] * v[j];
  }
  ss = simd_sum(ss);
  const float inv = rsqrt(ss / (float)D + eps);
  for (int j = 0; j < PER; ++j) {
    const int d = base + j;
    // Round the norm's output to storage, as the reference's norm returns
    // its activation dtype before the rotation reads it.
    v[j] = d < D ? (float)(VPIPE_ELT)(v[j] * inv * (float)w[d]) : 0.0f;
  }
  // PER is even and base is a multiple of PER, so a lane holds whole pairs.
  for (int j = 0; j < PER; j += 2) {
    const int d = base + j;
    if (d + 1 < rot) {
      const float c = cs[(int64_t)pos * rot + d];
      const float s = cs[(int64_t)pos * rot + d + 1];
      const float x0 = v[j], x1 = v[j + 1];
      v[j] = x0 * c - x1 * s;
      v[j + 1] = x1 * c + x0 * s;
    }
  }
  device VPIPE_ELT* dst =
      isq ? qh + ((int64_t)head * qstride + r) * D
          : kc + ((int64_t)(head - Hq) * cap + pos) * D;
  for (int j = 0; j < PER; ++j) {
    if (base + j < D) { dst[base + j] = (VPIPE_ELT)v[j]; }
  }
}

// The attention output, gated and back to TOKEN-major for o_proj:
// o[r, h*D + d] = att[h][r][d] * sigmoid(gate[r][h*D + d]).
//   0:att[Hq, astride, D] 1:qkv[M, 2*Hq*D + 2*Hkv*D] 2:o[M, Hq*D] 3:D 4:Hq
//   5:Hkv 6:astride.   grid (Hq*D, M).
kernel void drama3_gate_out(
    const device VPIPE_ELT* att  [[buffer(0)]],
    const device VPIPE_ELT* qkv  [[buffer(1)]],
    device VPIPE_ELT*       o    [[buffer(2)]],
    constant int&           D    [[buffer(3)]],
    constant int&           Hq   [[buffer(4)]],
    constant int&           Hkv  [[buffer(5)]],
    constant int&           astride [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
  const int i = (int)gid.x, r = (int)gid.y;
  const int qd = Hq * D;
  if (i >= qd) { return; }
  const int h = i / D, d = i % D;
  const float a = (float)att[((int64_t)h * astride + r) * D + d];
  const float g = (float)qkv[(int64_t)r * (2 * qd + 2 * Hkv * D) + qd + i];
  o[(int64_t)r * qd + i] = (VPIPE_ELT)(a / (1.0f + exp(-g)));
}

// ---- The frame sampler ----------------------------------------------------
//
// One draw is the reference's (vllm-omni's draw_top_k_top_p, sglang-omni's
// twin): logits / max(T, 1e-5); rank the candidates by logit; take each
// rank's probability over the WHOLE row; keep rank r when its inclusive
// cumulative probability is <= top_p AND r < top_k, rank 0 always; draw
// argmax_r p_r / Exp_r over the kept ranks with Exp_r = -log(u_r), u_r
// uniform -- which samples the kept set in proportion to p. The ranks are
// extracted one at a time, stopping at the first rank no draw keeps (the
// cumulative probability only grows), so a draw costs its kept set, not a
// sort of the row. Ties rank the lower index first.
//
// The noise is a hash of (seed, frame, draw, rank): a run is a function of
// its seed, and the draws of one frame are independent of each other.

inline uint d3_hash(uint x)
{
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

inline float d3_u01(uint seed, uint frame, uint draw, uint rank)
{
  uint h = d3_hash(seed ^ d3_hash(frame * 0x9E3779B9U + 0x632BE5ABU));
  h = d3_hash(h ^ (draw * 0x85EBCA77U + rank * 0xC2B2AE3DU + 1U));
  // 24 high bits -> (0, 1).
  return ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

#define D3_TG   256
#define D3_PER  16                     // V <= D3_TG * D3_PER = 4096
#define D3_NDRAW 2

struct D3Draw {
  float inv_t;      // 1 / max(T, 1e-5)
  int   top_k;
  float top_p;
  uint  draw;       // noise stream
  float zmax;       // row max of logits * inv_t
  float zsum;       // sum exp(l * inv_t - zmax)
  float cum;
  bool  open;
  float best;
  int   best_i;
};

// Rank the row and run up to D3_NDRAW draws over the same ranking (the main
// token's first draw and its repetition-aware re-draw share their
// candidates, as the reference's do). out_idx[j] = draw j's column.
inline void d3_draws(const device VPIPE_ELT* logits, int V, thread D3Draw* dr,
                     int ndraw, uint seed, uint frame, int max_rank,
                     threadgroup float* shv, threadgroup int* shi,
                     uint lid, thread int* out_idx)
{
  float lv[D3_PER];
  float lmax = -INFINITY;
  for (int j = 0; j < D3_PER; ++j) {
    const int i = (int)lid + j * D3_TG;
    lv[j] = i < V ? (float)logits[i] : -INFINITY;
    lmax = max(lmax, lv[j]);
  }
  // Row max (raw logits; every draw's T is positive, so its scaled max is
  // this times its inv_t).
  float m = simd_max(lmax);
  if ((lid & 31) == 0) { shv[lid >> 5] = m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = -INFINITY;
  for (int i = 0; i < D3_TG / 32; ++i) { m = max(m, shv[i]); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k = 0; k < ndraw; ++k) {
    dr[k].zmax = m * dr[k].inv_t;
    float s = 0.0f;
    for (int j = 0; j < D3_PER; ++j) {
      if (lv[j] > -INFINITY) { s += exp(lv[j] * dr[k].inv_t - dr[k].zmax); }
    }
    s = simd_sum(s);
    if ((lid & 31) == 0) { shv[lid >> 5] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = 0.0f;
    for (int i = 0; i < D3_TG / 32; ++i) { s += shv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dr[k].zsum = s;
    dr[k].cum = 0.0f;
    dr[k].open = true;
    dr[k].best = -1.0f;
    dr[k].best_i = -1;
  }
  // This thread's best remaining (value, index).
  float tv = -INFINITY; int ti = 0x7fffffff;
  for (int j = 0; j < D3_PER; ++j) {
    const int i = (int)lid + j * D3_TG;
    if (lv[j] > tv || (lv[j] == tv && i < ti)) { tv = lv[j]; ti = i; }
  }
  for (int r = 0; r < max_rank; ++r) {
    bool any = false;
    for (int k = 0; k < ndraw; ++k) { any = any || dr[k].open; }
    if (!any) { break; }
    // Threadgroup argmax of (tv, lowest ti).
    float bv = tv; int bi = ti;
    for (uint o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(bv, o);
      const int oi = simd_shuffle_down(bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if ((lid & 31) == 0) { shv[lid >> 5] = bv; shi[lid >> 5] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = shv[0]; bi = shi[0];
    for (int i = 1; i < D3_TG / 32; ++i) {
      if (shv[i] > bv || (shv[i] == bv && shi[i] < bi)) {
        bv = shv[i]; bi = shi[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!(bv > -INFINITY)) { break; }
    for (int k = 0; k < ndraw; ++k) {
      if (!dr[k].open) { continue; }
      const float p = exp(bv * dr[k].inv_t - dr[k].zmax) / dr[k].zsum;
      dr[k].cum += p;
      const bool keep = r == 0 || (dr[k].cum <= dr[k].top_p &&
                                   r < dr[k].top_k);
      if (!keep) { dr[k].open = false; continue; }
      const float u = d3_u01(seed, frame, dr[k].draw, (uint)r);
      const float sc = p / max(-log(u), 1e-30f);
      if (sc > dr[k].best) { dr[k].best = sc; dr[k].best_i = bi; }
    }
    // The owner of the winner drops it and finds its next best.
    if (bi == ti) {
      for (int j = 0; j < D3_PER; ++j) {
        if ((int)lid + j * D3_TG == bi) { lv[j] = -INFINITY; }
      }
      tv = -INFINITY; ti = 0x7fffffff;
      for (int j = 0; j < D3_PER; ++j) {
        const int i = (int)lid + j * D3_TG;
        if (lv[j] > tv || (lv[j] == tv && i < ti)) { tv = lv[j]; ti = i; }
      }
    }
  }
  for (int k = 0; k < ndraw; ++k) { out_idx[k] = max(dr[k].best_i, 0); }
}

// The main token. logits [V] over the candidates the reference draws from:
// column 0 = end of speech, column j = semantic code j (1..V-1; code 0 is
// never a candidate). The first draw takes the request's T / top-k / top-p;
// when it lands on a semantic code that one of the last `window` main
// tokens already was, the frame takes a second draw at T 1.0, top-k 30,
// top-p 0.95 instead (repetition-aware sampling, the reference's fixed
// RAS settings). Writes out[0] = the token id, out[1] = code 0 (0 for end
// of speech), and slides the token into the window.
//   0:logits[V] 1:out(frame, 11 ints) 2:ras[window] 3:V 4:temp 5:top_k
//   6:top_p 7:seed 8:frame 9:eos 10:sem0 11:window 12:max_rank.
//   grid (256); threadgroup (256).
kernel void drama3_sample_main(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device int*             out    [[buffer(1)]],
    device int*             ras    [[buffer(2)]],
    constant int&           V      [[buffer(3)]],
    constant float&         temp   [[buffer(4)]],
    constant int&           top_k  [[buffer(5)]],
    constant float&         top_p  [[buffer(6)]],
    constant uint&          seed   [[buffer(7)]],
    constant uint&          frame  [[buffer(8)]],
    constant int&           eos    [[buffer(9)]],
    constant int&           sem0   [[buffer(10)]],
    constant int&           window [[buffer(11)]],
    constant int&           max_rank [[buffer(12)]],
    uint lid [[thread_index_in_threadgroup]])
{
  threadgroup float shv[D3_TG / 32];
  threadgroup int shi[D3_TG / 32];
  D3Draw dr[D3_NDRAW];
  dr[0].inv_t = 1.0f / max(temp, 1e-5f);
  dr[0].top_k = top_k; dr[0].top_p = top_p; dr[0].draw = 0;
  dr[1].inv_t = 1.0f; dr[1].top_k = 30; dr[1].top_p = 0.95f; dr[1].draw = 1;
  int idx[D3_NDRAW];
  d3_draws(logits, V, dr, 2, seed, frame, max_rank, shv, shi, lid, idx);
  if (lid == 0) {
    const int first = idx[0] == 0 ? eos : sem0 + idx[0];
    bool repeat = false;
    for (int i = 0; i < window; ++i) { repeat = repeat || ras[i] == first; }
    const bool use_ras = repeat && idx[0] != 0;
    const int col = use_ras ? idx[1] : idx[0];
    const int tok = col == 0 ? eos : sem0 + col;
    out[0] = tok;
    out[1] = col;                    // code 0; 0 on end of speech
    for (int i = 0; i + 1 < window; ++i) { ras[i] = ras[i + 1]; }
    if (window > 0) { ras[window - 1] = tok; }
  }
}

// Code k (1..9) of the frame from codebook head k-1's logits [V], at the
// request's T / top-k / top-p; 0 when the frame's main token is not audio
// (end of speech carries no codes). out = the frame.
//   0:logits[V] 1:out(frame) 2:V 3:temp 4:top_k 5:top_p 6:seed 7:frame
//   8:k 9:sem0 10:sem1 11:max_rank.   grid (256); threadgroup (256).
kernel void drama3_sample_code(
    const device VPIPE_ELT* logits [[buffer(0)]],
    device int*             out    [[buffer(1)]],
    constant int&           V      [[buffer(2)]],
    constant float&         temp   [[buffer(3)]],
    constant int&           top_k  [[buffer(4)]],
    constant float&         top_p  [[buffer(5)]],
    constant uint&          seed   [[buffer(6)]],
    constant uint&          frame  [[buffer(7)]],
    constant int&           k      [[buffer(8)]],
    constant int&           sem0   [[buffer(9)]],
    constant int&           sem1   [[buffer(10)]],
    constant int&           max_rank [[buffer(11)]],
    uint lid [[thread_index_in_threadgroup]])
{
  threadgroup float shv[D3_TG / 32];
  threadgroup int shi[D3_TG / 32];
  D3Draw dr[1];
  dr[0].inv_t = 1.0f / max(temp, 1e-5f);
  dr[0].top_k = top_k; dr[0].top_p = top_p; dr[0].draw = 1 + (uint)k;
  int idx[1];
  d3_draws(logits, V, dr, 1, seed, frame, max_rank, shv, shi, lid, idx);
  if (lid == 0) {
    out[1 + k] = d3_is_sem(out[0], sem0, sem1) ? idx[0] : 0;
  }
}

// Teacher forcing: copy the label frame's column `col` into `out`'s (so a
// tower run over known frames reads the codes the sampler would write).
//   0:src(frame) 1:dst(frame) 2:col.   grid (1).
kernel void drama3_copy_code(
    const device int* src [[buffer(0)]],
    device int*       dst [[buffer(1)]],
    constant int&     col [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid == 0) { dst[col] = src[col]; }
}
