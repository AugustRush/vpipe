// vsa_gate.metal -- VSA compression gate for MiniMax-H3 (FastVideo VSA-H3).
//
// The reference's arithmetic (video_sparse_attn_h3.py, `logical_gate is not
// None`):
//
//   v_pooled[g]   = mean over tile g's rows of V          [tiles, H, D]
//   scores[g0,g1] = (q_pooled[g0] . k_pooled[g1]) / sqrt(D)
//   out_c[g]      = sum_g1 softmax_g1(scores[g, g1]) * v_pooled[g1]
//   out[r]        = out[r] + gate[r] * out_c[tile(r)]
//
// THE TILE IS A 3-D CUBE, NOT A RUN OF ROWS. FastVideo's
// get_tile_partition_indices cuts the (T, H, W) latent grid into
// ceil(T/ts) x ceil(H/hs) x ceil(W/ws) boxes and flattens each in row-major
// order, so tile g's members are the rows of one box -- consecutive in W,
// strided in H, and further apart in T. Pooling a run of 64 consecutive rows
// instead mixes unrelated positions (it spans several frames of one row) and
// the branch turns into a low-frequency blur; MEASURED, that was visible in
// the picture.
//
// A box is identified by (bt, bh, bw); its rows are
//   video_start + ((bt*ts + t) * GH + (bh*hs + h)) * GW + (bw*ws + w)
// for t in [0, ts), h in [0, hs), w in [0, ws) clipped to the grid. The count
// of LIVE rows is what the mean divides by -- the reference's
// variable_block_sizes.
//
// fp32 throughout, written back as the attention kernel's dtype.
//
// Grid: one threadgroup per TILE, 64 threads; heads looped inside.
//
//   0 q,1 k,2 v (device const T*), 3 gate (const T*), 4 out (T*),
//   5 qp,6 kp,7 vp (device float*), 8 scores (device float*),
//   9 rows,10 tiles,11 heads,12 dim (uint&), 13 scale (float&),
//  14 video_start,15 grid_h,16 grid_w,17 frames (uint&),
//  18 ts,19 hs,20 ws (uint&)   -- the cube's shape in (T, H, W)

#include <metal_stdlib>
using namespace metal;

#ifndef VPIPE_ELT
#define VPIPE_ELT bfloat
#endif
typedef VPIPE_ELT T;

kernel void vsa_pool(device const T* q      [[buffer(0)]],
                     device const T* k      [[buffer(1)]],
                     device const T* v      [[buffer(2)]],
                     device float*   qp     [[buffer(5)]],
                     device float*   kp     [[buffer(6)]],
                     device float*   vp     [[buffer(7)]],
                     constant uint&  rows   [[buffer(9)]],
                     constant uint&  tiles  [[buffer(10)]],
                     constant uint&  heads  [[buffer(11)]],
                     constant uint&  dim    [[buffer(12)]],
                     constant uint&  vstart [[buffer(14)]],
                     constant uint&  GH     [[buffer(15)]],
                     constant uint&  GW     [[buffer(16)]],
                     constant uint&  frames [[buffer(17)]],
                     constant uint&  ts     [[buffer(18)]],
                     constant uint&  hs     [[buffer(19)]],
                     constant uint&  ws     [[buffer(20)]],
                     uint            gx     [[thread_position_in_grid]],
                     uint            tid    [[thread_position_in_threadgroup]])
{
  if (gx >= tiles) { return; }
  const uint nbh = (GH + hs - 1) / hs;
  const uint nbw = (GW + ws - 1) / ws;
  const uint bt = gx / (nbh * nbw), r = gx % (nbh * nbw);
  const uint bh = r / nbw, bw = r % nbw;
  // Live extents: a cube at the edge covers fewer rows.
  const uint t_hi = min(ts, frames - bt * ts);
  const uint h_hi = min(hs, GH - bh * hs);
  const uint w_hi = min(ws, GW - bw * ws);
  const uint live = t_hi * h_hi * w_hi;
  if (live == 0u) { return; }
  const float inv = 1.0f / float(live);

  for (uint h = 0; h < heads; ++h) {
    for (uint d = tid; d < dim; d += 64u) {
      float sq = 0.0f, sk = 0.0f, sv = 0.0f;
      for (uint t = 0; t < t_hi; ++t) {
        for (uint yy = 0; yy < h_hi; ++yy) {
          for (uint xx = 0; xx < w_hi; ++xx) {
            const uint row = vstart + ((bt * ts + t) * GH + (bh * hs + yy)) * GW
                                    + (bw * ws + xx);
            if (row >= rows) { continue; }
            const uint o = (row * heads + h) * dim + d;
            sq += float(q[o]); sk += float(k[o]); sv += float(v[o]);
          }
        }
      }
      const uint po = (gx * heads + h) * dim + d;
      qp[po] = sq * inv; kp[po] = sk * inv; vp[po] = sv * inv;
    }
  }
}

kernel void vsa_scores(device const float* qp     [[buffer(5)]],
                       device const float* kp     [[buffer(6)]],
                       device float*       scores [[buffer(8)]],
                       constant uint&      tiles  [[buffer(10)]],
                       constant uint&      heads  [[buffer(11)]],
                       constant uint&      dim    [[buffer(12)]],
                       constant float&     scale  [[buffer(13)]],
                       uint                gx     [[thread_position_in_grid]],
                       uint                tid    [[thread_position_in_threadgroup]])
{
  if (gx >= tiles) { return; }
  for (uint h = 0; h < heads; ++h) {
    const uint qo = (gx * heads + h) * dim;
    device float* row = scores + (gx * heads + h) * tiles;
    for (uint g = tid; g < tiles; g += 64u) {
      const uint ko = (g * heads + h) * dim;
      float acc = 0.0f;
      for (uint d = 0; d < dim; ++d) { acc += qp[qo + d] * kp[ko + d]; }
      row[g] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      float m = -INFINITY;
      for (uint g = 0; g < tiles; ++g) { m = max(m, row[g]); }
      float s = 0.0f;
      for (uint g = 0; g < tiles; ++g) { row[g] = exp(row[g] - m); s += row[g]; }
      const float invs = s > 0.0f ? 1.0f / s : 0.0f;
      for (uint g = 0; g < tiles; ++g) { row[g] *= invs; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void vsa_merge(device const float* scores [[buffer(8)]],
                      device const float* vp     [[buffer(7)]],
                      device const T*     gate   [[buffer(3)]],
                      device T*           out    [[buffer(4)]],
                      constant uint&      rows   [[buffer(9)]],
                      constant uint&      tiles  [[buffer(10)]],
                      constant uint&      heads  [[buffer(11)]],
                      constant uint&      dim    [[buffer(12)]],
                      constant uint&      vstart [[buffer(14)]],
                      constant uint&      GH     [[buffer(15)]],
                      constant uint&      GW     [[buffer(16)]],
                      constant uint&      frames [[buffer(17)]],
                      constant uint&      ts     [[buffer(18)]],
                      constant uint&      hs     [[buffer(19)]],
                      constant uint&      ws     [[buffer(20)]],
                      uint                gx     [[thread_position_in_grid]],
                      uint                tid    [[thread_position_in_threadgroup]])
{
  if (gx >= tiles) { return; }
  const uint nbh = (GH + hs - 1) / hs;
  const uint nbw = (GW + ws - 1) / ws;
  const uint bt = gx / (nbh * nbw), r = gx % (nbh * nbw);
  const uint bh = r / nbw, bw = r % nbw;
  const uint t_hi = min(ts, frames - bt * ts);
  const uint h_hi = min(hs, GH - bh * hs);
  const uint w_hi = min(ws, GW - bw * ws);

  for (uint h = 0; h < heads; ++h) {
    const device float* w = scores + (gx * heads + h) * tiles;
    for (uint d = tid; d < dim; d += 64u) {
      float acc = 0.0f;
      for (uint g = 0; g < tiles; ++g) {
        acc += w[g] * vp[(g * heads + h) * dim + d];
      }
      for (uint t = 0; t < t_hi; ++t) {
        for (uint yy = 0; yy < h_hi; ++yy) {
          for (uint xx = 0; xx < w_hi; ++xx) {
            const uint row = vstart + ((bt * ts + t) * GH + (bh * hs + yy)) * GW
                                    + (bw * ws + xx);
            if (row >= rows) { continue; }
            const uint o = (row * heads + h) * dim + d;
            out[o] = T(float(out[o]) + float(gate[o]) * acc);
          }
        }
      }
    }
  }
}
