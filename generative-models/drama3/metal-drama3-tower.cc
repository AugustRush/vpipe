#include "generative-models/drama3/metal-drama3-tower.h"

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_f32(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// The attention's two kernels: the multi-simdgroup scan below this many
// query rows, the simdgroup-matrix flash kernel from it on.
constexpr int kMmaAttnRows = 32;
// matmul2d's 128-row tile only pays once M amortizes it.
constexpr int kMmaMinRows = 64;
// SDPA_MBC_ELT / head_dim simdgroups split the keys (sdpa.metal).
constexpr int kMbcElt = 4096;
// sdpa_causal_mma_f16's query tile and its padding precondition.
constexpr int kMmaBq = 32;
// The sampler's candidate cap (the reference's max_top_k).
constexpr int kMaxRank = 64;

}  // namespace

MetalDrama3Tower::~MetalDrama3Tower() = default;

std::unique_ptr<MetalDrama3Tower>
MetalDrama3Tower::load(const std::shared_ptr<WeightSet>& ws, MetalCompute* mc,
                       const Config& cfg, std::string* err)
{
  if (ws == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no checkpoint or no metal device"; }
    return nullptr;
  }
  auto t = std::unique_ptr<MetalDrama3Tower>(new MetalDrama3Tower());
  t->_mc = mc;
  t->_cfg = cfg;
  if (!t->init_(ws, err)) { return nullptr; }
  return t;
}

bool
MetalDrama3Tower::init_(const std::shared_ptr<WeightSet>& ws,
                        std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[drama3 tower] " + std::move(m); }
    return false;
  };
  const Config& c = _cfg;
  if (c.head_dim != 256 || c.dim % 256 != 0 || c.dim > 2048 ||
      c.rotary_dim % 8 != 0 || c.rotary_dim > c.head_dim ||
      c.n_head % c.n_kv != 0) {
    return fail(std::format("unsupported geometry: dim {} head_dim {} "
                            "rotary {}", c.dim, c.head_dim, c.rotary_dim));
  }
  _ws = ws;
  WeightSet& wts = *ws;
  MetalCompute* mc = _mc;

  _lib_d3    = mc->load_library("drama3_bf16");
  _lib_dense = mc->load_library("dense_gemm_bf16");
  _lib_sdpa  = mc->load_library("sdpa_bf16");
  _lib_elt   = mc->load_library("llm_elementwise_bf16");
  _fn_mix      = _lib_d3.function("drama3_mix_norm");
  _fn_add_norm = _lib_d3.function("drama3_add_norm");
  _fn_qk_rope  = _lib_d3.function("drama3_qk_rope");
  _fn_gate_out = _lib_d3.function("drama3_gate_out");
  _fn_sample_code = _lib_d3.function("drama3_sample_code");
  _fn_gemv     = _lib_dense.function("dense_gemv_t_f16");
  _fn_gemm     = _lib_dense.function("dense_gemm_t_f16");
  _fn_swiglu1  = _lib_dense.function("dense_moe_gather_swiglu_f16");
  _fn_sdpa_mb  = _lib_sdpa.function("sdpa_causal_mb_f16");
  _fn_sdpa_mma = _lib_sdpa.function("sdpa_causal_mma_f16");
  _fn_swiglu_il = _lib_elt.function("swiglu_interleaved_f16");
  for (const auto* f : {&_fn_mix, &_fn_add_norm, &_fn_qk_rope, &_fn_gate_out,
                        &_fn_sample_code, &_fn_gemv, &_fn_gemm, &_fn_swiglu1,
                        &_fn_sdpa_mb, &_fn_sdpa_mma, &_fn_swiglu_il}) {
    if (!f->valid()) { return fail("a kernel did not load"); }
  }
  if (mc->supports_matrix_cores()) {
    _lib_mma = mc->load_library("dense_gemm_mma_bf16");
    _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _use_mma = _fn_mma.valid() && _fn_mma_deep.valid();
  }
  _zero_eid = mc->make_shared_buffer(sizeof(std::int32_t));
  if (_zero_eid.empty()) { return fail("out of memory"); }
  *static_cast<std::int32_t*>(_zero_eid.contents()) = 0;

  const auto& src = wts.src();
  // A bf16 tensor the tower keeps, checked for its shape.
  auto keep = [&](const std::string& nm, std::int64_t rows,
                  std::int64_t cols) -> SharedBuffer {
    const auto* info = src.info(nm);
    if (info == nullptr || info->dtype != "BF16") { return {}; }
    std::int64_t r = info->shape.empty() ? 0 : info->shape[0];
    std::int64_t k = 1;
    for (std::size_t i = 1; i < info->shape.size(); ++i) {
      k *= info->shape[i];
    }
    if (info->shape.size() == 1) { k = r; r = 1; }
    if (r != rows || k != cols) { return {}; }
    return wts.read(nm, mc, WeightSet::Residency::Copied);
  };
  // A zero-centred norm weight as f32 with its +1 added (exact).
  auto norm = [&](const std::string& nm, int n) -> SharedBuffer {
    SharedBuffer raw = keep(nm, 1, n);
    if (raw.empty()) { return {}; }
    SharedBuffer out = mc->make_shared_buffer((std::size_t)n * 4);
    if (out.empty()) { return {}; }
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    auto* d = static_cast<float*>(out.contents());
    for (int i = 0; i < n; ++i) { d[i] = bf16_f32(s[i]) + 1.0f; }
    return out;
  };
  const int H = c.dim, D = c.head_dim, F = c.ffn, V = c.codebook_size;
  const int QKV = c.qkv_width(), QD = c.n_head * D;
  _layers.resize((std::size_t)c.layers());
  for (int L = 0; L < c.layers(); ++L) {
    Layer& ly = _layers[(std::size_t)L];
    const bool shared = L < c.shared_layers;
    const std::string p =
        shared ? std::format("audio_decoder_mtp_shared.layers.{}.", L)
               : std::format("audio_decoder_mtp.layers.{}.",
                             L - c.shared_layers);
    const std::string t = p + "transformer_layer.";
    ly.emb = keep(p + (shared ? "semantic_embedding.weight"
                              : "codebook_embedding.weight"), V, H);
    ly.enorm = norm(p + "enorm.weight", H);
    ly.hnorm = norm(p + "hnorm.weight", H);
    ly.fnorm = norm(p + "final_layernorm.weight", H);
    ly.eh = keep(p + "eh_proj.weight", H, 2 * H);
    ly.attn_norm = norm(t + "attention_norm.weight", H);
    ly.ffn_norm = norm(t + "ffn_norm.weight", H);
    ly.wqkv = keep(t + "attention.wqkv.weight", QKV, H);
    ly.qn = norm(t + "attention.q_norm.weight", D);
    ly.kn = norm(t + "attention.k_norm.weight", D);
    ly.wo = keep(t + "attention.wo.weight", H, QD);
    ly.w2 = keep(t + "feed_forward.w2.weight", H, F);
    {
      // gate (w1) / up (w3) rows interleaved -- row 2g gate g, 2g+1 up g --
      // the layout the single-slab SwiGLU gather and swiglu_interleaved
      // read. Consumed into the product, so read uncached.
      SharedBuffer w1 = keep(t + "feed_forward.w1.weight", F, H);
      SharedBuffer w3 = keep(t + "feed_forward.w3.weight", F, H);
      if (!w1.empty() && !w3.empty()) {
        ly.w13 = mc->make_shared_buffer((std::size_t)2 * F * H * 2);
        if (!ly.w13.empty()) {
          auto* o = static_cast<std::uint8_t*>(ly.w13.contents());
          const auto* g = static_cast<const std::uint8_t*>(w1.contents());
          const auto* u = static_cast<const std::uint8_t*>(w3.contents());
          const std::size_t row = (std::size_t)H * 2;
          for (int r = 0; r < F; ++r) {
            std::memcpy(o + (2 * (std::size_t)r) * row, g + r * row, row);
            std::memcpy(o + (2 * (std::size_t)r + 1) * row, u + r * row, row);
          }
        }
      }
    }
    if (!shared) {
      ly.head = keep(std::format("audio_decoder_mtp.heads.{}.weight",
                                 L - c.shared_layers), V, H);
    }
    if (ly.emb.empty() || ly.enorm.empty() || ly.hnorm.empty() ||
        ly.fnorm.empty() || ly.eh.empty() || ly.attn_norm.empty() ||
        ly.ffn_norm.empty() || ly.wqkv.empty() || ly.qn.empty() ||
        ly.kn.empty() || ly.wo.empty() || ly.w2.empty() || ly.w13.empty() ||
        (!shared && ly.head.empty())) {
      return fail(std::format("layer {} ({}) is missing or misshapen", L, p));
    }
  }
  _ecb = keep("audio_decoder.codebook_embeddings.weight",
              (std::int64_t)c.num_codebooks * V, H);
  if (_ecb.empty()) {
    return fail("audio_decoder.codebook_embeddings is missing or "
                "misshapen");
  }
  return true;
}

std::size_t
MetalDrama3Tower::kv_bytes_for(const Config& c, int rows)
{
  return (std::size_t)c.layers() * 2 * c.n_kv * c.head_dim * 2 *
         (std::size_t)std::max(rows, 0);
}

std::size_t
MetalDrama3Tower::kv_bytes() const
{
  return kv_bytes_for(_cfg, _cap);
}

std::size_t
MetalDrama3Tower::resident_bytes() const
{
  std::size_t b = _ecb.byte_size();
  for (const Layer& ly : _layers) {
    for (const SharedBuffer* s :
         {&ly.emb, &ly.enorm, &ly.hnorm, &ly.attn_norm, &ly.ffn_norm,
          &ly.fnorm, &ly.qn, &ly.kn, &ly.eh, &ly.wqkv, &ly.wo, &ly.w13,
          &ly.w2, &ly.head}) {
      b += s->byte_size();
    }
  }
  return b;
}

bool
MetalDrama3Tower::reserve(int rows)
{
  if (rows <= _cap) { return true; }
  const Config& c = _cfg;
  const int cap = ((rows + 255) / 256) * 256;
  const std::size_t D = (std::size_t)c.head_dim;
  for (Layer& ly : _layers) {
    for (SharedBuffer* kv : {&ly.kc, &ly.vc}) {
      SharedBuffer nb =
          _mc->make_shared_buffer((std::size_t)c.n_kv * cap * D * 2);
      if (nb.empty()) { return false; }
      if (!kv->empty()) {
        // Per head: the old rows, at the new stride.
        for (int h = 0; h < c.n_kv; ++h) {
          std::memcpy(static_cast<std::uint8_t*>(nb.contents()) +
                          (std::size_t)h * cap * D * 2,
                      static_cast<const std::uint8_t*>(kv->contents()) +
                          (std::size_t)h * _cap * D * 2,
                      (std::size_t)_cap * D * 2);
        }
      }
      *kv = std::move(nb);
    }
  }
  // The rotary table, cos | sin per pair, as the reference forms it:
  // inv_freq in f32, angle = f32(pos) * inv_freq in f32, cos / sin of it.
  const int rot = c.rotary_dim;
  SharedBuffer tab = _mc->make_shared_buffer((std::size_t)cap * rot * 4);
  if (tab.empty()) { return false; }
  std::vector<float> inv((std::size_t)rot / 2);
  for (int i = 0; i < rot / 2; ++i) {
    inv[(std::size_t)i] =
        1.0f / std::pow(c.rope_base, (float)(2 * i) / (float)rot);
  }
  auto* t = static_cast<float*>(tab.contents());
  for (int p = 0; p < cap; ++p) {
    for (int i = 0; i < rot / 2; ++i) {
      const float a = (float)p * inv[(std::size_t)i];
      t[(std::size_t)p * rot + 2 * i] = (float)std::cos((double)a);
      t[(std::size_t)p * rot + 2 * i + 1] = (float)std::sin((double)a);
    }
  }
  _rope = std::move(tab);
  _cap = cap;
  return true;
}

bool
MetalDrama3Tower::ensure_scratch_(int n)
{
  if (n <= _scratch_rows) { return true; }
  const Config& c = _cfg;
  const std::size_t N = (std::size_t)n, H = (std::size_t)c.dim;
  const std::size_t D = (std::size_t)c.head_dim;
  const std::size_t QD = (std::size_t)c.n_head * D;
  const std::size_t F = (std::size_t)c.ffn;
  auto b = [&](std::size_t elems) {
    return _mc->make_shared_buffer(elems * 2);
  };
  _cat = b(N * 2 * H);
  _u = b(N * H);
  _xn = b(N * H);
  _qkv = b(N * (std::size_t)c.qkv_width());
  // The flash kernel over-reads up to a query tile past the last head.
  _qh = b((N + kMmaBq) * QD);
  _att = b(N * QD);
  _o = b(N * QD);
  _ao = b(N * H);
  _r = b(N * H);
  _fn = b(N * H);
  _gu = b(N * 2 * F);
  _act = b(N * F);
  _dn = b(N * H);
  _h0 = b(N * H);
  _h1 = b(N * H);
  if (_logits.empty()) { _logits = b((std::size_t)c.codebook_size); }
  for (const SharedBuffer* s : {&_cat, &_u, &_xn, &_qkv, &_qh, &_att, &_o,
                                &_ao, &_r, &_fn, &_gu, &_act, &_dn, &_h0,
                                &_h1, &_logits}) {
    if (s->empty()) { _scratch_rows = 0; return false; }
  }
  _scratch_rows = n;
  return true;
}

void
MetalDrama3Tower::encode_gemm(ComputeEncoder& enc, const SharedBuffer& w,
                              const SharedBuffer& x, std::size_t xoff,
                              const SharedBuffer& y, std::size_t yoff, int K,
                              int N, int M)
{
  // y[M, N] = x[M, K] @ w[N, K]^T, bf16 storage, f32 accumulation.
  if (M == 1) {
    enc.set_function(_fn_gemv);
    enc.set_buffer(0, x, xoff);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, K);
    enc.set_constant(4, N);
    enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    return;
  }
  const bool mma = _use_mma && M >= kMmaMinRows;
  const bool deep = mma && mma_use_wide_tile(N, K);
  enc.set_function(mma ? (deep ? _fn_mma_deep : _fn_mma) : _fn_gemm);
  enc.set_buffer(0, x, xoff);
  enc.set_buffer(1, w);
  enc.set_buffer(2, w);             // bias slot unused (has_bias 0)
  enc.set_buffer(3, y, yoff);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  if (mma) {
    const int BN = deep ? 256 : 128;
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
}

bool
MetalDrama3Tower::encode_rows(ComputeEncoder& enc, const SharedBuffer& h,
                              const SharedBuffer& frames,
                              const SharedBuffer& labels, int n, int pos0,
                              const SharedBuffer* head_logits)
{
  return encode_(enc, h, frames, labels, n, pos0, head_logits, nullptr);
}

bool
MetalDrama3Tower::encode_step(ComputeEncoder& enc, const SharedBuffer& h,
                              const SharedBuffer& in_frame,
                              const SharedBuffer& out_frame, int pos,
                              const Draw& draw)
{
  return encode_(enc, h, in_frame, out_frame, 1, pos, nullptr, &draw);
}

bool
MetalDrama3Tower::encode_(ComputeEncoder& enc, const SharedBuffer& h,
                          const SharedBuffer& frames,
                          const SharedBuffer& labels, int n, int pos0,
                          const SharedBuffer* head_logits, const Draw* draw)
{
  const Config& c = _cfg;
  if (n <= 0 || pos0 < 0 || pos0 + n > _cap || (draw != nullptr && n != 1)) {
    return false;
  }
  if (!ensure_scratch_(n)) { return false; }
  const int H = c.dim, D = c.head_dim, Hq = c.n_head, Hkv = c.n_kv;
  const int QD = Hq * D, QKV = c.qkv_width(), F = c.ffn;
  const int V = c.codebook_size, rot = c.rotary_dim;
  const float eps = c.eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const int x_first = 1 + c.backbone_codebooks, x_end = c.num_codebooks;
  const SharedBuffer* hin = &h;
  for (int L = 0; L < c.layers(); ++L) {
    const Layer& ly = _layers[(std::size_t)L];
    const bool shared = L < c.shared_layers;
    // ---- mix: cat(enorm(e), hnorm(h)) -> eh_proj ----------------------
    {
      const int code_col = 1 + (shared ? 0 : L - c.shared_layers);
      const int extra = L == 0 ? 1 : 0;
      enc.set_function(_fn_mix);
      enc.set_buffer(0, labels);
      enc.set_buffer(1, ly.emb);
      enc.set_buffer(2, *hin);
      enc.set_buffer(3, ly.enorm);
      enc.set_buffer(4, ly.hnorm);
      enc.set_buffer(5, _cat);
      enc.set_constant(6, H);
      enc.set_constant(7, eps);
      enc.set_constant(8, code_col);
      enc.set_constant(9, c.sem0);
      enc.set_constant(10, c.sem1);
      enc.set_constant(11, V);
      enc.set_constant(12, extra);
      enc.set_buffer(13, frames);
      enc.set_buffer(14, _ecb);
      enc.set_constant(15, x_first);
      enc.set_constant(16, x_end);
      enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    }
    encode_gemm(enc, ly.eh, _cat, 0, _u, 0, 2 * H, H, n);
    // ---- attention ----------------------------------------------------
    auto add_norm = [&](const SharedBuffer& a, const SharedBuffer* b,
                        const SharedBuffer& res, const SharedBuffer& w,
                        const SharedBuffer& out) {
      enc.set_function(_fn_add_norm);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b != nullptr ? *b : a);
      enc.set_buffer(2, res);
      enc.set_buffer(3, w);
      enc.set_buffer(4, out);
      enc.set_constant(5, H);
      enc.set_constant(6, eps);
      enc.set_constant(7, b != nullptr ? 1 : 0);
      enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    };
    add_norm(_u, nullptr, _u, ly.attn_norm, _xn);
    encode_gemm(enc, ly.wqkv, _xn, 0, _qkv, 0, H, QKV, n);
    enc.set_function(_fn_qk_rope);
    enc.set_buffer(0, _qkv);
    enc.set_buffer(1, ly.qn);
    enc.set_buffer(2, ly.kn);
    enc.set_buffer(3, _qh);
    enc.set_buffer(4, ly.kc);
    enc.set_buffer(5, ly.vc);
    enc.set_buffer(6, _rope);
    enc.set_constant(7, D);
    enc.set_constant(8, Hq);
    enc.set_constant(9, Hkv);
    enc.set_constant(10, rot);
    enc.set_constant(11, pos0);
    enc.set_constant(12, n);
    enc.set_constant(13, _cap);
    enc.set_constant(14, eps);
    enc.dispatch({32, (unsigned)(Hq + 2 * Hkv), (unsigned)n}, {32, 1, 1});
    {
      const bool flash = n >= kMmaAttnRows;
      enc.set_function(flash ? _fn_sdpa_mma : _fn_sdpa_mb);
      enc.set_buffer(0, _qh);
      enc.set_buffer(1, ly.kc);
      enc.set_buffer(2, ly.vc);
      enc.set_buffer(3, _att);
      enc.set_constant(4, scale);
      enc.set_constant(5, pos0 + n);
      enc.set_constant(6, D);
      enc.set_constant(7, Hq);
      enc.set_constant(8, Hkv);
      enc.set_constant(9, n);
      enc.set_constant(10, pos0);
      enc.set_constant(11, _cap);
      enc.set_constant(12, 0);          // no window
      enc.set_constant(13, 0);          // no ring
      if (flash) {
        const unsigned thr = (unsigned)(4 * (D / 64) * 32);
        enc.dispatch({thr, (unsigned)Hq, (unsigned)((n + kMmaBq - 1) /
                                                    kMmaBq)},
                     {thr, 1, 1});
      } else {
        const unsigned thr = (unsigned)((kMbcElt / D) * 32);
        enc.dispatch({thr, (unsigned)Hq, (unsigned)n}, {thr, 1, 1});
      }
    }
    enc.set_function(_fn_gate_out);
    enc.set_buffer(0, _att);
    enc.set_buffer(1, _qkv);
    enc.set_buffer(2, _o);
    enc.set_constant(3, D);
    enc.set_constant(4, Hq);
    enc.set_constant(5, Hkv);
    enc.set_constant(6, n);
    enc.dispatch({(unsigned)QD, (unsigned)n, 1}, {256, 1, 1});
    encode_gemm(enc, ly.wo, _o, 0, _ao, 0, QD, H, n);
    add_norm(_ao, &_u, _r, ly.ffn_norm, _fn);
    // ---- SwiGLU ---------------------------------------------------------
    if (n == 1) {
      enc.set_function(_fn_swiglu1);
      enc.set_buffer(0, ly.w13);
      enc.set_buffer(1, _fn);
      enc.set_buffer(2, _act);
      enc.set_constant(3, H);
      enc.set_constant(4, 2 * F);
      enc.set_buffer(5, _zero_eid);
      enc.set_constant(6, 1);
      enc.dispatch({32, (unsigned)(F / 2), 1}, {32, 2, 1});
    } else {
      encode_gemm(enc, ly.w13, _fn, 0, _gu, 0, H, 2 * F, n);
      enc.set_function(_fn_swiglu_il);
      enc.set_buffer(0, _gu);
      enc.set_buffer(1, _act);
      enc.set_constant(2, n);
      enc.set_constant(3, F);
      enc.dispatch({(unsigned)(n * F), 1, 1}, {256, 1, 1});
    }
    encode_gemm(enc, ly.w2, _act, 0, _dn, 0, F, H, n);
    const SharedBuffer& hout = (L % 2 == 0) ? _h0 : _h1;
    add_norm(_dn, &_r, _dn, ly.fnorm, hout);
    // ---- codebook head ------------------------------------------------
    if (!shared) {
      const int j = L - c.shared_layers;          // head j -> code j + 1
      if (draw != nullptr) {
        encode_gemm(enc, ly.head, hout, 0, _logits, 0, H, V, 1);
        enc.set_function(_fn_sample_code);
        enc.set_buffer(0, _logits);
        enc.set_buffer(1, labels);
        enc.set_constant(2, V);
        enc.set_constant(3, draw->temperature);
        enc.set_constant(4, std::clamp(draw->top_k, 1, kMaxRank));
        enc.set_constant(5, draw->top_p);
        enc.set_constant(6, draw->seed);
        enc.set_constant(7, draw->frame);
        enc.set_constant(8, j + 1);
        enc.set_constant(9, c.sem0);
        enc.set_constant(10, c.sem1);
        enc.set_constant(11, kMaxRank);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
      } else if (head_logits != nullptr) {
        encode_gemm(enc, ly.head, hout, 0, *head_logits,
                    (std::size_t)j * n * V * 2, H, V, n);
      }
    }
    hin = &hout;
  }
  return true;
}

}  // namespace vpipe::genai
