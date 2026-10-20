#include "generative-models/drama3/metal-fish-codec.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <format>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "fish-dac/f16|";

constexpr int kDilations[3] = {1, 3, 9};
// The decoder's / encoder's strides, input side first.
constexpr int kDecRates[4] = {8, 8, 4, 2};
constexpr int kEncRates[4] = {2, 4, 8, 8};
// Samples an 86 Hz row of the decoder spans (the four upsample strides).
constexpr int kRowSamples = 8 * 8 * 4 * 2;
// The conv stacks run in tiles of rows with a causal LEFT halo at least
// their receptive field (decoder ~20 rows, encoder ~11), which makes a
// tile's kept rows exactly what one pass over everything gives.
constexpr int kTileRows = 256;
constexpr int kHaloRows = 32;

// VPIPE_FISH_CODEC_TILE_ROWS: the conv stacks' tile, the A/B of the halo.
int
tile_rows()
{
  static const int v = []() {
    const char* e = std::getenv("VPIPE_FISH_CODEC_TILE_ROWS");
    const int n = e != nullptr ? std::atoi(e) : 0;
    return n > 0 ? n : kTileRows;
  }();
  return v;
}
// The banded im2col's column budget, in elements.
constexpr std::size_t kColCap = (std::size_t)16 << 20;
// The flash attention over-reads up to one 64-key block past the end.
constexpr int kKvPad = 64;
constexpr int kHeadDim = 64;

std::uint16_t
f32_bf16_rne(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = ((u >> 16) & 1u) + 0x7FFFu;
  return (std::uint16_t)((u + r) >> 16);
}

float
bf16_f32(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

bool
read_f32(WeightSet& ws, MetalCompute* mc, const std::string& nm,
         std::vector<float>* out, std::vector<std::int64_t>* shape = nullptr)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->dtype != "F32") { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.byte_size() < n * 4) { return false; }
  out->assign(static_cast<const float*>(raw.contents()),
              static_cast<const float*>(raw.contents()) + n);
  if (shape != nullptr) { *shape = info->shape; }
  return true;
}

}  // namespace

MetalFishCodec::~MetalFishCodec() = default;

std::unique_ptr<MetalFishCodec>
MetalFishCodec::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                     bool with_encoder, std::string* err)
{
  if (ws == nullptr || mc == nullptr) {
    if (err != nullptr) { *err = "no codec.pth or no metal device"; }
    return nullptr;
  }
  std::unique_ptr<MetalFishCodec> m(new MetalFishCodec());
  m->_mc = mc;
  if (!m->init_(ws, with_encoder, err)) { return nullptr; }
  return m;
}

SharedBuffer
MetalFishCodec::f16_(const std::string& nm)
{
  WeightSet& ws = *_ws;
  if (ws.src().info(nm) == nullptr) { return {}; }
  return ws.derived(std::string(kKey) + nm, [&]() -> SharedBuffer {
    std::vector<float> v;
    if (!read_f32(ws, _mc, nm, &v)) { return {}; }
    SharedBuffer out = _mc->make_shared_buffer(v.size() * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<_Float16*>(out.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
    return out;
  });
}

SharedBuffer
MetalFishCodec::f32_(const std::string& nm)
{
  WeightSet& ws = *_ws;
  if (ws.src().info(nm) == nullptr) { return {}; }
  return ws.tensor(nm, _mc, WeightSet::Residency::Copied);
}

// Fold the weight norm (g * v / |v|, the norm over every dimension but the
// first -- for a ConvTranspose1d that is the INPUT width), transpose into
// the GEMM layout and narrow to f16 in one pass. Three spellings: the new
// parametrization (original0 = g, original1 = v), the old (weight_g /
// weight_v), and a plain weight.
MetalFishCodec::Conv
MetalFishCodec::conv_(const std::string& nm, bool transposed, int dilation,
                      int stride)
{
  Conv c;
  WeightSet& ws = *_ws;
  const auto& src = ws.src();
  std::string vname, gname;
  if (src.info(nm + ".parametrizations.weight.original1") != nullptr) {
    vname = nm + ".parametrizations.weight.original1";
    gname = nm + ".parametrizations.weight.original0";
  } else if (src.info(nm + ".weight_v") != nullptr) {
    vname = nm + ".weight_v";
    gname = nm + ".weight_g";
  } else {
    vname = nm + ".weight";
  }
  const auto* info = src.info(vname);
  if (info == nullptr || info->shape.size() != 3) { return c; }
  const int d0 = (int)info->shape[0];
  const int d1 = (int)info->shape[1];
  const int k  = (int)info->shape[2];
  c.cin = transposed ? d0 : d1;
  c.cout = transposed ? d1 : d0;
  c.k = k;
  c.dilation = dilation;
  c.stride = stride;
  c.transposed = transposed;
  c.b = f16_(nm + ".bias");
  const std::string key =
      std::string(kKey) + (transposed ? "ct|" : "c|") + nm;
  c.w = ws.derived(key, [&]() -> SharedBuffer {
    std::vector<float> v, g;
    if (!read_f32(ws, _mc, vname, &v)) { return {}; }
    if (!gname.empty() && !read_f32(ws, _mc, gname, &g)) { return {}; }
    const std::size_t n = (std::size_t)d0 * d1 * k;
    if (v.size() < n) { return {}; }
    std::vector<float> scale((std::size_t)d0, 1.0f);
    if (!gname.empty()) {
      if (g.size() < (std::size_t)d0) { return {}; }
      for (int i = 0; i < d0; ++i) {
        double sq = 0.0;
        const float* row = v.data() + (std::size_t)i * d1 * k;
        for (std::size_t j = 0; j < (std::size_t)d1 * k; ++j) {
          sq += (double)row[j] * (double)row[j];
        }
        const double nrm = std::sqrt(sq);
        scale[(std::size_t)i] =
            nrm > 0.0 ? (float)((double)g[(std::size_t)i] / nrm) : 0.0f;
      }
    }
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<_Float16*>(out.contents());
    if (!transposed) {
      // [cout, cin, k] -> [cout, k * cin], (k, cin) within a row.
      for (int co = 0; co < d0; ++co) {
        for (int ci = 0; ci < d1; ++ci) {
          for (int kk = 0; kk < k; ++kk) {
            d[((std::size_t)co * k + kk) * d1 + ci] =
                (_Float16)(v[((std::size_t)co * d1 + ci) * k + kk] *
                           scale[(std::size_t)co]);
          }
        }
      }
    } else {
      // [cin, cout, k] -> [k * cout, cin]: every tap side by side, folded
      // by col2im.
      for (int ci = 0; ci < d0; ++ci) {
        for (int co = 0; co < d1; ++co) {
          for (int kk = 0; kk < k; ++kk) {
            d[((std::size_t)kk * d1 + co) * d0 + ci] =
                (_Float16)(v[((std::size_t)ci * d1 + co) * k + kk] *
                           scale[(std::size_t)ci]);
          }
        }
      }
    }
    return out;
  });
  if (c.w.empty()) { return Conv{}; }
  return c;
}

// DAC's Snake: x + sin(alpha x)^2 / (alpha + 1e-9), alpha stored as is.
MetalFishCodec::Snake
MetalFishCodec::snake_(const std::string& nm)
{
  Snake s;
  WeightSet& ws = *_ws;
  std::vector<float> a;
  if (!read_f32(ws, _mc, nm + ".alpha", &a) || a.empty()) { return s; }
  s.c = (int)a.size();
  s.ealpha = ws.derived(std::string(kKey) + "a|" + nm, [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(a.size() * 4);
    if (!o.empty()) { std::memcpy(o.contents(), a.data(), a.size() * 4); }
    return o;
  });
  s.rbeta = ws.derived(std::string(kKey) + "ra|" + nm, [&]() -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(a.size() * 4);
    if (o.empty()) { return o; }
    auto* d = static_cast<float*>(o.contents());
    for (std::size_t i = 0; i < a.size(); ++i) {
      d[i] = 1.0f / (a[i] + 1e-9f);
    }
    return o;
  });
  if (s.ealpha.empty() || s.rbeta.empty()) { return Snake{}; }
  return s;
}

MetalFishCodec::Transformer
MetalFishCodec::transformer_(const std::string& nm, int n_layer, int window)
{
  Transformer tf;
  tf.window = window;
  for (int L = 0; L < n_layer; ++L) {
    const std::string p = nm + ".layers." + std::to_string(L) + ".";
    TLayer ly;
    ly.wqkv = f16_(p + "attention.wqkv.weight");
    ly.wo = f16_(p + "attention.wo.weight");
    ly.w1 = f16_(p + "feed_forward.w1.weight");
    ly.w3 = f16_(p + "feed_forward.w3.weight");
    ly.w2 = f16_(p + "feed_forward.w2.weight");
    ly.anorm = f16_(p + "attention_norm.weight");
    ly.fnorm = f16_(p + "ffn_norm.weight");
    ly.ascale = f32_(p + "attention_layer_scale.gamma");
    ly.fscale = f32_(p + "ffn_layer_scale.gamma");
    if (ly.wqkv.empty() || ly.wo.empty() || ly.w1.empty() || ly.w3.empty() ||
        ly.w2.empty() || ly.anorm.empty() || ly.fnorm.empty() ||
        ly.ascale.empty() || ly.fscale.empty()) {
      return Transformer{};
    }
    tf.layers.push_back(std::move(ly));
  }
  tf.norm = f16_(nm + ".norm.weight");
  if (tf.norm.empty()) { return Transformer{}; }
  return tf;
}

MetalFishCodec::ConvNeXt
MetalFishCodec::convnext_(const std::string& nm)
{
  ConvNeXt n;
  n.dw_w = f32_(nm + ".dwconv.conv.weight");      // [C, 1, 7]
  n.dw_b = f32_(nm + ".dwconv.conv.bias");
  n.ln_w = f16_(nm + ".norm.weight");
  n.ln_b = f16_(nm + ".norm.bias");
  n.pw1 = f16_(nm + ".pwconv1.weight");
  n.pw1_b = f16_(nm + ".pwconv1.bias");
  n.pw2 = f16_(nm + ".pwconv2.weight");
  n.pw2_b = f16_(nm + ".pwconv2.bias");
  n.gamma = f32_(nm + ".gamma");
  return n;
}

bool
MetalFishCodec::init_(const std::shared_ptr<WeightSet>& ws,
                      bool with_encoder, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[fish codec] " + std::move(m); }
    return false;
  };
  _ws = ws;
  MetalCompute* mc = _mc;
  _lib_gemm = mc->load_library("dense_gemm");
  _lib_elt  = mc->load_library("llm_elementwise");
  _lib_avae = mc->load_library("audio_vae_1d");
  _lib_fd   = mc->load_library("fish_dac");
  _lib_rms  = mc->load_library("rms_norm");
  _lib_sdpa = mc->load_library("sdpa");
  _fn_gemm   = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_im2col = _lib_avae.function("im2col_1d_tc_f16");
  _fn_col2im = _lib_avae.function("col2im_1d_tc_f16");
  _fn_snake  = _lib_avae.function("snake_beta_f16");
  _fn_add    = _lib_elt.function("residual_add_f16");
  _fn_ln     = _lib_elt.function("layer_norm_affine_f16");
  _fn_gelu   = _lib_elt.function("gelu_erf_f16");
  _fn_swiglu = _lib_elt.function("swiglu_f16");
  _fn_bias_rows = _lib_elt.function("bias_add_rows_f16");
  _fn_copy   = _lib_elt.function("copy_f16");
  _fn_rms    = _lib_rms.function("rms_norm_f16");
  _fn_flash  = _lib_sdpa.function("sdpa_causal_flash_f16");
  _fn_vq_dec = _lib_fd.function("fish_vq_decode");
  _fn_vq_enc = _lib_fd.function("fish_vq_encode");
  _fn_qkv    = _lib_fd.function("fish_qkv_rope");
  _fn_heads  = _lib_fd.function("fish_heads_tok");
  _fn_scale_add = _lib_fd.function("fish_scale_add");
  _fn_dw     = _lib_fd.function("fish_dwconv_causal");
  for (const auto* f : {&_fn_gemm, &_fn_im2col, &_fn_col2im, &_fn_snake,
                        &_fn_add, &_fn_ln, &_fn_gelu, &_fn_swiglu,
                        &_fn_bias_rows, &_fn_rms, &_fn_flash, &_fn_vq_dec,
                        &_fn_vq_enc, &_fn_qkv, &_fn_heads, &_fn_scale_add,
                        &_fn_dw, &_fn_copy}) {
    if (!f->valid()) { return fail("a codec kernel did not resolve"); }
  }
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_FISH_CODEC_NO_MMA2") == nullptr) {
    _lib_mma = mc->load_library("dense_gemm_mma");
    _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _use_mma = _fn_mma.valid() && _fn_mma_deep.valid();
  }

  // ---- the quantizer's decode tables: codebook -> out_proj, f32 ---------
  {
    WeightSet& w = *ws;
    std::vector<std::string> qs;
    qs.push_back("quantizer.semantic_quantizer.quantizers.0");
    for (int i = 0; i < kCodebooks - 1; ++i) {
      qs.push_back("quantizer.quantizer.quantizers." + std::to_string(i));
    }
    std::vector<float> all;
    for (std::size_t qi = 0; qi < qs.size(); ++qi) {
      std::vector<float> cb, v, g, b;
      std::vector<std::int64_t> cbs, vs;
      if (!read_f32(w, mc, qs[qi] + ".codebook.weight", &cb, &cbs) ||
          !read_f32(w, mc, qs[qi] + ".out_proj.weight_v", &v, &vs) ||
          !read_f32(w, mc, qs[qi] + ".out_proj.weight_g", &g) ||
          !read_f32(w, mc, qs[qi] + ".out_proj.bias", &b) ||
          cbs.size() != 2 || vs.size() != 3) {
        return fail("quantizer " + qs[qi] + " is incomplete");
      }
      const int n = (int)cbs[0], cd = (int)cbs[1];
      const int C = (int)vs[0];
      if ((int)vs[1] != cd || (int)b.size() != C) {
        return fail("quantizer " + qs[qi] + " is misshapen");
      }
      if (qi == 0) { _sem = n; _dim = C; } else { _res = n; }
      // W = g * v / |v| per output channel ([C, cd, 1]).
      std::vector<float> W((std::size_t)C * cd);
      for (int o = 0; o < C; ++o) {
        double sq = 0.0;
        for (int j = 0; j < cd; ++j) {
          const double x = v[(std::size_t)o * cd + j];
          sq += x * x;
        }
        const double s = sq > 0.0 ? g[(std::size_t)o] / std::sqrt(sq) : 0.0;
        for (int j = 0; j < cd; ++j) {
          W[(std::size_t)o * cd + j] =
              (float)(v[(std::size_t)o * cd + j] * s);
        }
      }
      const std::size_t off = all.size();
      all.resize(off + (std::size_t)n * C);
      for (int e = 0; e < n; ++e) {
        for (int o = 0; o < C; ++o) {
          float acc = b[(std::size_t)o];
          for (int j = 0; j < cd; ++j) {
            acc += W[(std::size_t)o * cd + j] * cb[(std::size_t)e * cd + j];
          }
          all[off + (std::size_t)e * C + o] = acc;
        }
      }
    }
    _tables = mc->make_shared_buffer(all.size() * 4);
    if (_tables.empty()) { return fail("out of memory"); }
    std::memcpy(_tables.contents(), all.data(), all.size() * 4);
  }
  if (_dim != 1024) { return fail("an unexpected latent width"); }

  // ---- decode -----------------------------------------------------------
  _post = transformer_("quantizer.post_module", 8, 128);
  if (_post.empty()) { return fail("quantizer.post_module is incomplete"); }
  for (int i = 0; i < 2; ++i) {
    const std::string p = "quantizer.upsample." + std::to_string(i);
    _up[i].conv = conv_(p + ".0.conv", true, 1, 2);
    _up[i].nx = convnext_(p + ".1");
    if (_up[i].conv.empty() || _up[i].nx.pw2.empty() ||
        _up[i].nx.gamma.empty() || _up[i].nx.dw_w.empty()) {
      return fail(p + " is incomplete");
    }
  }
  _dec_in = conv_("decoder.model.0.conv", false, 1, 1);
  if (_dec_in.empty()) { return fail("decoder.model.0 is missing"); }
  for (int bi = 0; bi < 4; ++bi) {
    const std::string p =
        "decoder.model." + std::to_string(bi + 1) + ".block.";
    DecBlock& b = _dec[bi];
    b.act = snake_(p + "0");
    b.up = conv_(p + "1.conv", true, 1, kDecRates[bi]);
    for (int j = 0; j < 3; ++j) {
      const std::string r = p + std::to_string(2 + j) + ".block.";
      ResUnit& u = b.res[j];
      u.a1 = snake_(r + "0");
      u.c1 = conv_(r + "1.conv", false, kDilations[j], 1);
      u.a2 = snake_(r + "2");
      u.c2 = conv_(r + "3.conv", false, 1, 1);
      if (u.a1.empty() || u.c1.empty() || u.a2.empty() || u.c2.empty()) {
        return fail("decoder residual unit " + r + " is incomplete");
      }
    }
    if (b.act.empty() || b.up.empty() || b.up.k != 2 * kDecRates[bi]) {
      return fail("decoder block " + std::to_string(bi + 1) +
                  " is incomplete");
    }
  }
  _dec_act = snake_("decoder.model.5");
  _dec_out = conv_("decoder.model.6.conv", false, 1, 1);
  if (_dec_act.empty() || _dec_out.empty() || _dec_out.cout != 1) {
    return fail("the decoder's output stage is incomplete");
  }
  if (with_encoder && !init_encoder_(err)) { return false; }
  return true;
}

bool
MetalFishCodec::init_encoder_(std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[fish codec] " + std::move(m); }
    return false;
  };
  WeightSet& w = *_ws;
  _enc_in = conv_("encoder.block.0.conv", false, 1, 1);
  if (_enc_in.empty()) { return fail("encoder.block.0 is missing"); }
  for (int bi = 0; bi < 4; ++bi) {
    const std::string p =
        "encoder.block." + std::to_string(bi + 1) + ".block.";
    EncBlock& b = _enc[bi];
    for (int j = 0; j < 3; ++j) {
      const std::string r = p + std::to_string(j) + ".block.";
      ResUnit& u = b.res[j];
      u.a1 = snake_(r + "0");
      u.c1 = conv_(r + "1.conv", false, kDilations[j], 1);
      u.a2 = snake_(r + "2");
      u.c2 = conv_(r + "3.conv", false, 1, 1);
      if (u.a1.empty() || u.c1.empty() || u.a2.empty() || u.c2.empty()) {
        return fail("encoder residual unit " + r + " is incomplete");
      }
    }
    b.act = snake_(p + "3");
    b.down = conv_(p + "4.conv", false, 1, kEncRates[bi]);
    if (b.act.empty() || b.down.empty() || b.down.k != 2 * kEncRates[bi]) {
      return fail("encoder block " + std::to_string(bi + 1) +
                  " is incomplete");
    }
  }
  // The last encoder block's transformer: its config passes no window,
  // so the default 512 holds.
  _enc_tf = transformer_("encoder.block.4.block.5", 4, 512);
  _enc_act = snake_("encoder.block.5");
  _enc_out = conv_("encoder.block.6.conv", false, 1, 1);
  if (_enc_tf.empty() || _enc_act.empty() || _enc_out.empty()) {
    return fail("the encoder's last stage is incomplete");
  }
  for (int i = 0; i < 2; ++i) {
    const std::string p = "quantizer.downsample." + std::to_string(i);
    _down[i].conv = conv_(p + ".0.conv", false, 1, 2);
    _down[i].nx = convnext_(p + ".1");
    if (_down[i].conv.empty() || _down[i].nx.pw2.empty()) {
      return fail(p + " is incomplete");
    }
  }
  _pre = transformer_("quantizer.pre_module", 8, 128);
  if (_pre.empty()) { return fail("quantizer.pre_module is incomplete"); }
  // The VQ's in_projs (weight norm folded) and normalised codebooks, f32.
  std::vector<std::string> qs;
  qs.push_back("quantizer.semantic_quantizer.quantizers.0");
  for (int i = 0; i < kCodebooks - 1; ++i) {
    qs.push_back("quantizer.quantizer.quantizers." + std::to_string(i));
  }
  std::vector<float> win, bin, cbn;
  for (const std::string& q : qs) {
    std::vector<float> v, g, b, cb;
    std::vector<std::int64_t> vs, cbs;
    if (!read_f32(w, _mc, q + ".in_proj.weight_v", &v, &vs) ||
        !read_f32(w, _mc, q + ".in_proj.weight_g", &g) ||
        !read_f32(w, _mc, q + ".in_proj.bias", &b) ||
        !read_f32(w, _mc, q + ".codebook.weight", &cb, &cbs) ||
        vs.size() != 3 || vs[0] != 8 || vs[1] != _dim) {
      return fail("quantizer " + q + "'s in_proj is incomplete");
    }
    for (int o = 0; o < 8; ++o) {
      double sq = 0.0;
      for (int j = 0; j < _dim; ++j) {
        const double x = v[(std::size_t)o * _dim + j];
        sq += x * x;
      }
      const double s = sq > 0.0 ? g[(std::size_t)o] / std::sqrt(sq) : 0.0;
      for (int j = 0; j < _dim; ++j) {
        win.push_back((float)(v[(std::size_t)o * _dim + j] * s));
      }
      bin.push_back(b[(std::size_t)o]);
    }
    const int n = (int)cbs[0];
    for (int e = 0; e < n; ++e) {
      double sq = 0.0;
      for (int j = 0; j < 8; ++j) {
        const double x = cb[(std::size_t)e * 8 + j];
        sq += x * x;
      }
      const float inv = (float)(1.0 / std::max(std::sqrt(sq), 1e-12));
      for (int j = 0; j < 8; ++j) {
        cbn.push_back(cb[(std::size_t)e * 8 + j] * inv);
      }
    }
  }
  auto up = [&](const std::vector<float>& h) {
    SharedBuffer b = _mc->make_shared_buffer(h.size() * 4);
    if (!b.empty()) { std::memcpy(b.contents(), h.data(), h.size() * 4); }
    return b;
  };
  _win = up(win);
  _bin = up(bin);
  _cbn = up(cbn);
  if (_win.empty() || _bin.empty() || _cbn.empty()) {
    return fail("out of memory");
  }
  _enc_ready = true;
  return true;
}

std::size_t
MetalFishCodec::resident_bytes() const
{
  std::size_t b = _tables.byte_size() + _win.byte_size() + _bin.byte_size() +
                  _cbn.byte_size();
  auto conv = [&](const Conv& c) { b += c.w.byte_size() + c.b.byte_size(); };
  auto snake = [&](const Snake& s) {
    b += s.ealpha.byte_size() + s.rbeta.byte_size();
  };
  auto ru = [&](const ResUnit& r) {
    snake(r.a1); snake(r.a2); conv(r.c1); conv(r.c2);
  };
  auto tf = [&](const Transformer& t) {
    for (const TLayer& l : t.layers) {
      for (const SharedBuffer* s : {&l.wqkv, &l.wo, &l.w1, &l.w3, &l.w2,
                                    &l.anorm, &l.fnorm, &l.ascale,
                                    &l.fscale}) {
        b += s->byte_size();
      }
    }
    b += t.norm.byte_size();
  };
  auto nx = [&](const ConvNeXt& n) {
    for (const SharedBuffer* s : {&n.dw_w, &n.dw_b, &n.ln_w, &n.ln_b,
                                  &n.pw1, &n.pw1_b, &n.pw2, &n.pw2_b,
                                  &n.gamma}) {
      b += s->byte_size();
    }
  };
  tf(_post);
  for (const Resample& r : _up) { conv(r.conv); nx(r.nx); }
  conv(_dec_in); conv(_dec_out); snake(_dec_act);
  for (const DecBlock& d : _dec) {
    snake(d.act); conv(d.up);
    for (const ResUnit& r : d.res) { ru(r); }
  }
  if (_enc_ready) {
    conv(_enc_in); conv(_enc_out); snake(_enc_act); tf(_enc_tf); tf(_pre);
    for (const EncBlock& e : _enc) {
      for (const ResUnit& r : e.res) { ru(r); }
      snake(e.act); conv(e.down);
    }
    for (const Resample& r : _down) { conv(r.conv); nx(r.nx); }
  }
  return b;
}

std::size_t
MetalFishCodec::plan_bytes(const std::string& path, bool with_encoder)
{
  auto w = MetalLlamaWeights::open_model(path);
  if (!w.has_value()) { return 0; }
  auto starts = [](const std::string& s, std::string_view p) {
    return s.compare(0, p.size(), p) == 0;
  };
  std::size_t n = 0;
  for (const std::string& nm : w->tensor_names()) {
    const auto* ti = w->info(nm);
    if (ti == nullptr || ti->dtype != "F32") { continue; }
    const bool dec = starts(nm, "decoder.") ||
                     starts(nm, "quantizer.post_module.") ||
                     starts(nm, "quantizer.upsample.");
    const bool enc = starts(nm, "encoder.") ||
                     starts(nm, "quantizer.pre_module.") ||
                     starts(nm, "quantizer.downsample.");
    if (dec || (with_encoder && enc)) { n += (std::size_t)ti->nbytes / 2; }
  }
  // The ten codebooks through their out_projs, f32 [4096 + 9 * 1024, 1024]
  // (+ the encoder's in_projs and normalised codebooks, small).
  n += (std::size_t)(4096 + 9 * 1024) * 1024 * 4;
  if (with_encoder) {
    n += (std::size_t)kCodebooks * 8 * 1024 * 4 +
         (std::size_t)(4096 + 9 * 1024) * 8 * 4;
  }
  return n;
}

// ---- kernels ------------------------------------------------------------

void
MetalFishCodec::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                      const SharedBuffer& w, const SharedBuffer* bias,
                      const SharedBuffer& y, std::size_t ye, int M, int N,
                      int K, std::size_t xe)
{
  // matmul2d where it fills its 128-row tile and stays under the 2 GB
  // operand line; the steel tile otherwise.
  if (_use_mma && M >= 256 && N >= 64 && M <= mma_row_band(N, K)) {
    const bool wide = mma_use_wide_tile(N, K);
    const int RN = wide ? 256 : 128;
    enc.set_function(wide ? _fn_mma_deep : _fn_mma);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    if (bias != nullptr) {
      const std::uint32_t total = (std::uint32_t)((std::size_t)M * N);
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, *bias);
      enc.set_constant(2, N);
      enc.set_constant(3, total);
      enc.dispatch({total, 1, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalFishCodec::conv_run_(ComputeEncoder& enc, const Conv& c,
                          const SharedBuffer& in, const SharedBuffer& out,
                          int T)
{
  const SharedBuffer* bias = c.b.empty() ? nullptr : &c.b;
  const int To = T / c.stride;
  if (c.k == 1 && c.stride == 1) {
    gemm_(enc, in, c.w, bias, out, 0, T, c.cout, c.cin);
    return;
  }
  // CAUSAL: left-pad (k - 1) * dilation + 1 - stride; rows past the end
  // read as zero (T is a whole number of strides here).
  const int pad = (c.k - 1) * c.dilation + 1 - c.stride;
  const std::size_t per_row = (std::size_t)c.k * c.cin;
  std::size_t band = std::max<std::size_t>(_col_cap / per_row, 1);
  band = std::min(band, (std::size_t)To);
  for (std::size_t r0 = 0; r0 < (std::size_t)To; r0 += band) {
    const int n = (int)std::min(band, (std::size_t)To - r0);
    enc.set_function(_fn_im2col);
    enc.set_buffer(0, in);
    enc.set_buffer(1, _col);
    enc.set_constant(2, T);
    enc.set_constant(3, c.cin);
    enc.set_constant(4, c.k);
    enc.set_constant(5, c.dilation);
    enc.set_constant(6, pad);
    enc.set_constant(7, c.stride);
    enc.set_constant(8, To);
    enc.set_constant(9, (int)r0);
    enc.set_constant(10, n);
    enc.dispatch({(unsigned)per_row, (unsigned)n, 1}, {64, 1, 1});
    gemm_(enc, _col, c.w, bias, out, r0 * (std::size_t)c.cout, n, c.cout,
          (int)per_row);
  }
}

void
MetalFishCodec::convt_run_(ComputeEncoder& enc, const Conv& c,
                           const SharedBuffer& in, const SharedBuffer& out,
                           int T, int in0, int out0)
{
  // Every input row's k taps side by side, then folded; the causal crop
  // keeps the first T * stride outputs.
  // _fold was sized for this before encoding began (ensure_fold_): a
  // buffer replaced mid-encode would be freed under the encoder.
  gemm_(enc, in, c.w, nullptr, _fold, 0, T, c.k * c.cout, c.cin,
        (std::size_t)in0 * c.cin);
  const int To = T * c.stride;
  const unsigned tx = (unsigned)std::min(c.cout, 32);
  const unsigned ty = std::max(1u, 256u / tx);
  enc.set_function(_fn_col2im);
  enc.set_buffer(0, _fold);
  enc.set_buffer(1, c.b.empty() ? _fold : c.b);
  enc.set_buffer(2, out, (std::size_t)out0 * c.cout * 2);
  enc.set_constant(3, T);
  enc.set_constant(4, c.cout);
  enc.set_constant(5, c.k);
  enc.set_constant(6, c.stride);
  enc.set_constant(7, 0);
  enc.set_constant(8, To);
  enc.set_constant(9, c.b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)c.cout, (unsigned)To, 1}, {tx, ty, 1});
}

void
MetalFishCodec::snake_run_(ComputeEncoder& enc, const Snake& s,
                           const SharedBuffer& in, const SharedBuffer& out,
                           int T)
{
  const int C = s.c;
  const unsigned tx = (unsigned)std::min(C, 32);
  const unsigned ty = std::max(1u, 256u / tx);
  enc.set_function(_fn_snake);
  enc.set_buffer(0, in);
  enc.set_buffer(1, s.ealpha);
  enc.set_buffer(2, s.rbeta);
  enc.set_buffer(3, out);
  enc.set_constant(4, C);
  enc.set_constant(5, T * C);
  enc.dispatch({(unsigned)C, (unsigned)T, 1}, {tx, ty, 1});
}

void
MetalFishCodec::resunit_run_(ComputeEncoder& enc, const ResUnit& r,
                             const SharedBuffer& x, const SharedBuffer& a,
                             const SharedBuffer& b, int T)
{
  snake_run_(enc, r.a1, x, a, T);
  conv_run_(enc, r.c1, a, b, T);
  snake_run_(enc, r.a2, b, a, T);
  conv_run_(enc, r.c2, a, b, T);
  const int n = T * r.c2.cout;
  enc.set_function(_fn_add);
  enc.set_buffer(0, x);
  enc.set_buffer(1, b);
  enc.set_buffer(2, x);
  enc.set_constant(3, n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

bool
MetalFishCodec::ensure_rope_(int rows)
{
  if (rows <= _rope_rows) { return true; }
  const int cap = ((rows + 1023) / 1024) * 1024;
  constexpr int D = kHeadDim;
  SharedBuffer t = _mc->make_shared_buffer((std::size_t)cap * D * 4);
  if (t.empty()) { return false; }
  // freqs_cis as the reference builds it: f32 frequencies, f32 angles
  // t * f, cos / sin, then ROUNDED TO BF16 (its default dtype).
  float freq[D / 2];
  for (int i = 0; i < D / 2; ++i) {
    freq[i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / (float)D);
  }
  auto* d = static_cast<float*>(t.contents());
  for (int p = 0; p < cap; ++p) {
    for (int i = 0; i < D / 2; ++i) {
      const float a = (float)p * freq[i];
      d[(std::size_t)p * D + 2 * i] =
          bf16_f32(f32_bf16_rne((float)std::cos((double)a)));
      d[(std::size_t)p * D + 2 * i + 1] =
          bf16_f32(f32_bf16_rne((float)std::sin((double)a)));
    }
  }
  _rope = std::move(t);
  _rope_rows = cap;
  return true;
}

bool
MetalFishCodec::ensure_scratch_(std::size_t act, std::size_t tf_rows)
{
  auto grow = [&](SharedBuffer& b, std::size_t elems) {
    if (b.byte_size() >= elems * 2) { return true; }
    b = _mc->make_shared_buffer(elems * 2);
    if (b.empty()) { return false; }
    std::memset(b.contents(), 0, elems * 2);
    return true;
  };
  bool ok = true;
  if (act > _act_cap) {
    ok = ok && grow(_a, act) && grow(_b, act) && grow(_c, act) &&
         grow(_d, act);
    if (ok) { _act_cap = act; }
  }
  if (_col_cap == 0) {
    ok = ok && grow(_col, kColCap);
    if (ok) { _col_cap = kColCap; }
  }
  if (tf_rows > _tf_cap) {
    const std::size_t W = (std::size_t)_dim;
    const std::size_t kv = (tf_rows + kKvPad) * W;
    // _tf1 also holds the ConvNeXt's 4C-wide hidden.
    ok = ok && grow(_tq, tf_rows * W) && grow(_tk, kv) && grow(_tv, kv) &&
         grow(_tatt, tf_rows * W) && grow(_tqkv, tf_rows * 3 * W) &&
         grow(_th, tf_rows * W) && grow(_tf1, tf_rows * 4 * W) &&
         grow(_tf3, tf_rows * 3 * W);
    if (ok) { _tf_cap = tf_rows; }
  }
  return ok;
}

bool
MetalFishCodec::ensure_fold_(std::size_t elems)
{
  if (_fold.byte_size() >= elems * 2) { return true; }
  _fold = _mc->make_shared_buffer(elems * 2);
  return !_fold.empty();
}

bool
MetalFishCodec::transformer_run_(ComputeEncoder& enc, const Transformer& tf,
                                 const SharedBuffer& x, int T,
                                 DecodeStream* st)
{
  // The table is grown before encoding starts (ensure_rope_), never here.
  const int pos0 = st != nullptr ? st->pos : 0;
  if (pos0 + T > _rope_rows) { return false; }
  if (st != nullptr && (st->k.size() != tf.layers.size() ||
                        st->pos - st->base + T + kKvPad > st->cap)) {
    return false;
  }
  const int W = _dim, H = W / kHeadDim, D = kHeadDim;
  // Whole: K/V are this call's rows. Streamed: the history's rows, this
  // call's appended at row0.
  const int kvs = st != nullptr ? st->cap : T + kKvPad;
  const int row0 = st != nullptr ? st->pos - st->base : 0;
  const float scale = 1.0f / std::sqrt((float)D);
  const float eps = 1e-5f;
  auto rms = [&](const SharedBuffer& in, const SharedBuffer& w,
                 const SharedBuffer& out) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, in);
    enc.set_buffer(1, w);
    enc.set_buffer(2, out);
    enc.set_constant(3, W);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)T, 1}, {256, 1, 1});
  };
  auto scale_add = [&](const SharedBuffer& y, const SharedBuffer& g) {
    const int n = T * W;
    enc.set_function(_fn_scale_add);
    enc.set_buffer(0, x);
    enc.set_buffer(1, y);
    enc.set_buffer(2, g);
    enc.set_constant(3, W);
    enc.set_constant(4, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  for (std::size_t li = 0; li < tf.layers.size(); ++li) {
    const TLayer& ly = tf.layers[li];
    const SharedBuffer& kb = st != nullptr ? st->k[li] : _tk;
    const SharedBuffer& vb = st != nullptr ? st->v[li] : _tv;
    rms(x, ly.anorm, _th);
    gemm_(enc, _th, ly.wqkv, nullptr, _tqkv, 0, T, 3 * W, W);
    enc.set_function(_fn_qkv);
    enc.set_buffer(0, _tqkv);
    enc.set_buffer(1, _tq);
    enc.set_buffer(2, kb);
    enc.set_buffer(3, vb);
    enc.set_buffer(4, _rope);
    enc.set_constant(5, H);
    enc.set_constant(6, D);
    enc.set_constant(7, T);
    enc.set_constant(8, kvs);
    enc.set_constant(9, pos0);
    enc.set_constant(10, row0);
    enc.dispatch({(unsigned)(D / 2), (unsigned)H, (unsigned)T}, {32, 1, 1});
    enc.set_function(_fn_flash);
    enc.set_buffer(0, _tq);
    enc.set_buffer(1, kb);
    enc.set_buffer(2, vb);
    enc.set_buffer(3, _tatt);
    enc.set_constant(4, scale);
    enc.set_constant(5, row0 + T);
    enc.set_constant(6, D);
    enc.set_constant(7, H);
    enc.set_constant(8, H);
    enc.set_constant(9, T);
    enc.set_constant(10, row0);
    enc.set_constant(11, kvs);
    enc.set_constant(12, tf.window);
    enc.set_constant(13, 0);
    enc.dispatch({8 * 32, (unsigned)H, (unsigned)((T + 7) / 8)},
                 {8 * 32, 1, 1});
    enc.set_function(_fn_heads);
    enc.set_buffer(0, _tatt);
    enc.set_buffer(1, _th);
    enc.set_constant(2, H);
    enc.set_constant(3, D);
    enc.set_constant(4, T);
    enc.dispatch({(unsigned)W, (unsigned)T, 1}, {256, 1, 1});
    gemm_(enc, _th, ly.wo, nullptr, _tf1, 0, T, W, W);
    scale_add(_tf1, ly.ascale);
    rms(x, ly.fnorm, _th);
    gemm_(enc, _th, ly.w1, nullptr, _tf1, 0, T, 3 * W, W);
    gemm_(enc, _th, ly.w3, nullptr, _tf3, 0, T, 3 * W, W);
    enc.set_function(_fn_swiglu);
    enc.set_buffer(0, _tf1);
    enc.set_buffer(1, _tf3);
    enc.set_buffer(2, _tf1);
    enc.set_constant(3, T * 3 * W);
    enc.dispatch({(unsigned)(T * 3 * W), 1, 1}, {256, 1, 1});
    gemm_(enc, _tf1, ly.w2, nullptr, _th, 0, T, W, 3 * W);
    scale_add(_th, ly.fscale);
  }
  // The final norm, through _th and back.
  rms(x, tf.norm, _th);
  enc.set_function(_fn_copy);
  enc.set_buffer(0, _th);
  enc.set_buffer(1, x);
  enc.set_constant(2, 0);
  enc.set_constant(3, T * W);
  enc.dispatch({(unsigned)(T * W), 1, 1}, {256, 1, 1});
  return true;
}

void
MetalFishCodec::convnext_run_(ComputeEncoder& enc, const ConvNeXt& n,
                              const SharedBuffer& x, int T)
{
  const int C = _dim;
  enc.set_function(_fn_dw);
  enc.set_buffer(0, x);
  enc.set_buffer(1, n.dw_w);
  enc.set_buffer(2, n.dw_b);
  enc.set_buffer(3, _th);
  enc.set_constant(4, T);
  enc.set_constant(5, C);
  enc.set_constant(6, 7);
  enc.dispatch({(unsigned)C, (unsigned)T, 1}, {256, 1, 1});
  enc.set_function(_fn_ln);
  enc.set_buffer(0, _th);
  enc.set_buffer(1, n.ln_w);
  enc.set_buffer(2, n.ln_b);
  enc.set_buffer(3, _tatt);
  enc.set_constant(4, C);
  enc.set_constant(5, 1e-6f);
  enc.dispatch({256, (unsigned)T, 1}, {256, 1, 1});
  // pwconv1 (C -> 4C) + GELU into the transformer's wide scratch.
  gemm_(enc, _tatt, n.pw1, &n.pw1_b, _tf1, 0, T, 4 * C, C);
  enc.set_function(_fn_gelu);
  enc.set_buffer(0, _tf1);
  enc.set_buffer(1, _tf1);
  enc.set_constant(2, T * 4 * C);
  enc.dispatch({(unsigned)(T * 4 * C), 1, 1}, {256, 1, 1});
  gemm_(enc, _tf1, n.pw2, &n.pw2_b, _th, 0, T, C, 4 * C);
  const int nn = T * C;
  enc.set_function(_fn_scale_add);
  enc.set_buffer(0, x);
  enc.set_buffer(1, _th);
  enc.set_buffer(2, n.gamma);
  enc.set_constant(3, C);
  enc.set_constant(4, nn);
  enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
}

// ---- decode ---------------------------------------------------------------

std::size_t
MetalFishCodec::decode_scratch_bytes(int frames)
{
  // A tile's widest activation (the last stage, 96 channels at the full
  // rate) x4, the im2col band, the transformer's buffers over the whole
  // utterance at 86 Hz rows (the ConvNeXt reuses them at 4C).
  const std::size_t rows = (std::size_t)kTileRows + kHaloRows;
  const std::size_t act = rows * kRowSamples * 96;
  const std::size_t tf = (std::size_t)std::max(frames, 1) * 4 + kKvPad;
  return (4 * act + kColCap) * 2 + tf * 1024 * 2 * 14 +
         (std::size_t)std::max(frames, 1) * 4 * 1024 * 2;
}

std::size_t
MetalFishCodec::encode_scratch_bytes(int frames)
{
  const std::size_t rows = (std::size_t)kTileRows + kHaloRows;
  const std::size_t act = rows * kRowSamples * 64 * 2;
  const std::size_t tf = (std::size_t)std::max(frames, 1) * 4 + kKvPad;
  return (4 * act + kColCap) * 2 + tf * 1024 * 2 * 14 +
         (std::size_t)std::max(frames, 1) * 4 * 1024 * 2;
}

bool
MetalFishCodec::decode_stages(const std::int32_t* codes, int frames,
                              std::vector<float>* vq,
                              std::vector<float>* post,
                              std::vector<float>* up, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[fish codec] " + std::move(m); }
    return false;
  };
  if (codes == nullptr || frames <= 0) { return fail("no frames"); }
  const int T = frames, W = _dim;
  const std::size_t T4 = (std::size_t)T * 4;
  // The stem runs whole: the windowed transformer's reach is long.
  if (!ensure_scratch_(T4 * W, T4) || !ensure_fold_(T4 * W) ||
      !ensure_rope_(T)) {
    return fail("out of memory");
  }
  SharedBuffer cb = _mc->make_shared_buffer((std::size_t)T * kCodebooks * 4);
  if (cb.empty()) { return fail("out of memory"); }
  std::memcpy(cb.contents(), codes, (std::size_t)T * kCodebooks * 4);
  auto read = [&](const SharedBuffer& b, std::size_t rows,
                  std::vector<float>* out) {
    if (out == nullptr) { return; }
    out->resize(rows * W);
    const auto* s = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < out->size(); ++i) { (*out)[i] = (float)s[i]; }
  };
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    enc.set_function(_fn_vq_dec);
    enc.set_buffer(0, cb);
    enc.set_buffer(1, _tables);
    enc.set_buffer(2, _a);
    enc.set_constant(3, W);
    enc.set_constant(4, _sem);
    enc.set_constant(5, _res);
    enc.dispatch({(unsigned)W, (unsigned)T, 1}, {256, 1, 1});
  }
  if (!s.commit().wait_ok()) { return fail("VQ decode failed"); }
  read(_a, (std::size_t)T, vq);
  {
    ComputeEncoder enc = s.begin_compute();
    if (!transformer_run_(enc, _post, _a, T)) {
      return fail("post transformer failed");
    }
  }
  if (!s.commit().wait_ok()) { return fail("post transformer failed"); }
  read(_a, (std::size_t)T, post);
  {
    ComputeEncoder enc = s.begin_compute();
    convt_run_(enc, _up[0].conv, _a, _b, T);
    convnext_run_(enc, _up[0].nx, _b, 2 * T);
    convt_run_(enc, _up[1].conv, _b, _a, 2 * T);
    convnext_run_(enc, _up[1].nx, _a, 4 * T);
  }
  if (!s.commit().wait_ok()) { return fail("upsample failed"); }
  read(_a, T4, up);
  return true;
}

void
MetalFishCodec::copy_rows_(ComputeEncoder& enc, const SharedBuffer& src,
                           int s0, const SharedBuffer& dst, int d0, int rows)
{
  if (rows <= 0) { return; }
  const int n = rows * _dim;
  enc.set_function(_fn_copy);
  enc.set_buffer(0, src, (std::size_t)s0 * _dim * 2);
  enc.set_buffer(1, dst);
  enc.set_constant(2, d0 * _dim);
  enc.set_constant(3, n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

void
MetalFishCodec::decoder_run_(ComputeEncoder& enc, int rows)
{
  int L = rows;
  conv_run_(enc, _dec_in, _d, _a, L);
  for (const DecBlock& b : _dec) {
    snake_run_(enc, b.act, _a, _b, L);
    convt_run_(enc, b.up, _b, _a, L);
    L *= b.up.stride;
    for (const ResUnit& r : b.res) { resunit_run_(enc, r, _a, _b, _c, L); }
  }
  snake_run_(enc, _dec_act, _a, _b, L);
  conv_run_(enc, _dec_out, _b, _c, L);
}

std::unique_ptr<MetalFishCodec::DecodeStream>
MetalFishCodec::decode_stream_begin(int max_chunk_frames)
{
  auto st = std::make_unique<DecodeStream>();
  st->max_chunk = std::max(max_chunk_frames, 1);
  // The window's history (127 frames, up to 63 more -- see the shift in
  // decode_stream_chunk) and a chunk, with the flash attention's
  // over-read past the end.
  st->cap = 191 + st->max_chunk + kKvPad;
  const std::size_t W = (std::size_t)_dim;
  const std::size_t kv = (std::size_t)st->cap * W * 2;
  for (std::size_t i = 0; i < _post.layers.size(); ++i) {
    st->k.push_back(_mc->make_shared_buffer(kv));
    st->v.push_back(_mc->make_shared_buffer(kv));
    if (st->k.back().empty() || st->v.back().empty()) { return nullptr; }
    std::memset(st->k.back().contents(), 0, kv);
    std::memset(st->v.back().contents(), 0, kv);
  }
  st->nx_hist[0] = _mc->make_shared_buffer(6 * W * 2);
  st->nx_hist[1] = _mc->make_shared_buffer(6 * W * 2);
  st->nx_buf[0] = _mc->make_shared_buffer((6 + 2 * st->max_chunk) * W * 2);
  st->nx_buf[1] = _mc->make_shared_buffer((6 + 4 * st->max_chunk) * W * 2);
  st->z_hist = _mc->make_shared_buffer((std::size_t)kHaloRows * W * 2);
  for (const SharedBuffer* b : {&st->nx_hist[0], &st->nx_hist[1],
                                &st->nx_buf[0], &st->nx_buf[1],
                                &st->z_hist}) {
    if (b->empty()) { return nullptr; }
  }
  return st;
}

bool
MetalFishCodec::decode_stream_chunk(DecodeStream& st,
                                    const std::int32_t* codes, int frames,
                                    std::vector<float>* pcm,
                                    std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[fish codec] " + std::move(m); }
    return false;
  };
  if (codes == nullptr || pcm == nullptr || frames <= 0 ||
      frames > st.max_chunk) {
    return fail("a stream chunk of 1..max_chunk frames");
  }
  const int n = frames, W = _dim;
  const int h0 = st.nx_rows[0], h1 = st.nx_rows[1], hz = st.z_rows;
  const int rows = hz + 4 * n;                // the decoder's input
  const std::size_t tile_act = (std::size_t)rows * kRowSamples * 96;
  if (!ensure_scratch_(std::max(tile_act, (std::size_t)rows * 1536),
                       (std::size_t)(6 + 4 * st.max_chunk)) ||
      !ensure_rope_(st.pos + n)) {
    return fail("out of memory");
  }
  {
    std::size_t f = (std::size_t)n * 4 * W, r = (std::size_t)rows;
    for (const DecBlock& b : _dec) {
      f = std::max(f, r * b.up.k * b.up.cout);
      r *= (std::size_t)b.up.stride;
    }
    if (!ensure_fold_(f)) { return fail("out of memory"); }
  }
  // Room for this chunk's keys: keep the window's 127 frames -- from a
  // position that is a multiple of 64, so the flash attention's 64-key
  // blocks fall where they fall in a whole decode (same blocks, same sums,
  // same bits).
  if (st.pos - st.base + n + kKvPad > st.cap) {
    const int nb = std::max(0, (st.pos - 127) / 64 * 64);
    const int keep = st.pos - nb;
    const int from = nb - st.base;
    const std::size_t H = (std::size_t)(W / kHeadDim);
    const std::size_t D = kHeadDim;
    for (std::size_t li = 0; li < st.k.size(); ++li) {
      for (SharedBuffer* b : {&st.k[li], &st.v[li]}) {
        auto* p = static_cast<std::uint8_t*>(b->contents());
        for (std::size_t h = 0; h < H; ++h) {
          std::memmove(p + (h * st.cap) * D * 2,
                       p + (h * st.cap + from) * D * 2,
                       (std::size_t)keep * D * 2);
        }
      }
    }
    st.base = nb;
  }
  SharedBuffer cb = _mc->make_shared_buffer((std::size_t)n * kCodebooks * 4);
  if (cb.empty()) { return fail("out of memory"); }
  std::memcpy(cb.contents(), codes, (std::size_t)n * kCodebooks * 4);
  const int n0 = h0 + 2 * n, n1 = h1 + 4 * n;
  const int k0 = std::min(6, n0), k1 = std::min(6, n1);
  const int kz = std::min(kHaloRows, rows);
  CommandStream s = _mc->make_command_stream();
  {
    ComputeEncoder enc = s.begin_compute();
    copy_rows_(enc, st.nx_hist[0], 0, st.nx_buf[0], 0, h0);
    copy_rows_(enc, st.nx_hist[1], 0, st.nx_buf[1], 0, h1);
    copy_rows_(enc, st.z_hist, 0, _d, 0, hz);
    enc.set_function(_fn_vq_dec);
    enc.set_buffer(0, cb);
    enc.set_buffer(1, _tables);
    enc.set_buffer(2, _a);
    enc.set_constant(3, W);
    enc.set_constant(4, _sem);
    enc.set_constant(5, _res);
    enc.dispatch({(unsigned)W, (unsigned)n, 1}, {256, 1, 1});
    if (!transformer_run_(enc, _post, _a, n, &st)) {
      return fail("post transformer failed");
    }
    // Upsample 1: its ConvNeXt's history saved before it runs in place.
    convt_run_(enc, _up[0].conv, _a, st.nx_buf[0], n, 0, h0);
    copy_rows_(enc, st.nx_buf[0], n0 - k0, st.nx_hist[0], 0, k0);
    convnext_run_(enc, _up[0].nx, st.nx_buf[0], n0);
    // Upsample 2, from upsample 1's new rows.
    convt_run_(enc, _up[1].conv, st.nx_buf[0], st.nx_buf[1], 2 * n, h0, h1);
    copy_rows_(enc, st.nx_buf[1], n1 - k1, st.nx_hist[1], 0, k1);
    convnext_run_(enc, _up[1].nx, st.nx_buf[1], n1);
    // The decoder's input: its halo, then the new rows.
    copy_rows_(enc, st.nx_buf[1], h1, _d, hz, 4 * n);
    copy_rows_(enc, _d, rows - kz, st.z_hist, 0, kz);
    decoder_run_(enc, rows);
  }
  std::string gerr;
  if (!s.commit().wait_ok(&gerr)) {
    return fail(gerr.empty() ? "stream decode failed" : gerr);
  }
  const auto* o = static_cast<const _Float16*>(_c.contents());
  const std::size_t skip = (std::size_t)hz * kRowSamples;
  const std::size_t got = (std::size_t)4 * n * kRowSamples;
  pcm->resize(got);
  for (std::size_t i = 0; i < got; ++i) {
    (*pcm)[i] = std::tanh((float)o[skip + i]);
  }
  st.pos += n;
  st.nx_rows[0] = k0;
  st.nx_rows[1] = k1;
  st.z_rows = kz;
  return true;
}

bool
MetalFishCodec::decode(const std::int32_t* codes, int frames,
                       std::vector<float>* pcm, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr && err->empty()) { *err = "[fish codec] " + m; }
    return false;
  };
  if (pcm == nullptr) { return fail("no output"); }
  if (!decode_stages(codes, frames, nullptr, nullptr, nullptr, err)) {
    return false;
  }
  const int W = _dim;
  const int R = frames * 4;                      // 86 Hz rows in _a
  // Keep the stem's output: the tiles reuse the activation scratch.
  SharedBuffer z = _mc->make_shared_buffer((std::size_t)R * W * 2);
  if (z.empty()) { return fail("out of memory"); }
  std::memcpy(z.contents(), _a.contents(), (std::size_t)R * W * 2);
  const std::size_t tile_act =
      (std::size_t)(tile_rows() + kHaloRows) * kRowSamples * 96;
  if (!ensure_scratch_(std::max(tile_act,
                                (std::size_t)(tile_rows() + kHaloRows) * 1536),
                       0)) {
    return fail("out of memory");
  }
  {
    // The widest transposed conv's pre-fold over a tile.
    std::size_t f = 0, rows = (std::size_t)(tile_rows() + kHaloRows);
    for (const DecBlock& b : _dec) {
      f = std::max(f, rows * b.up.k * b.up.cout);
      rows *= (std::size_t)b.up.stride;
    }
    if (!ensure_fold_(f)) { return fail("out of memory"); }
  }
  pcm->assign((std::size_t)R * kRowSamples, 0.0f);
  CommandStream s = _mc->make_command_stream();
  for (int r0 = 0; r0 < R; r0 += tile_rows()) {
    const int r1 = std::min(R, r0 + tile_rows());
    const int t0 = std::max(0, r0 - kHaloRows);
    const int L0 = r1 - t0;
    std::memcpy(_d.contents(),
                static_cast<const std::uint8_t*>(z.contents()) +
                    (std::size_t)t0 * W * 2,
                (std::size_t)L0 * W * 2);
    const int L = L0 * kRowSamples;
    {
      ComputeEncoder enc = s.begin_compute();
      decoder_run_(enc, L0);
    }
    std::string gerr;
    if (!s.commit().wait_ok(&gerr)) {
      return fail(gerr.empty() ? "decode failed" : gerr);
    }
    // Keep rows [r0, r1): samples [(r0 - t0) * 512, L) of the tile; tanh.
    const auto* o = static_cast<const _Float16*>(_c.contents());
    const std::size_t skip = (std::size_t)(r0 - t0) * kRowSamples;
    for (std::size_t i = skip; i < (std::size_t)L; ++i) {
      (*pcm)[(std::size_t)r0 * kRowSamples + (i - skip)] =
          std::tanh((float)o[i]);
    }
  }
  return true;
}

// ---- encode ---------------------------------------------------------------

bool
MetalFishCodec::encode(const float* pcm, int n,
                       std::vector<std::int32_t>* codes, std::string* err,
                       std::vector<float>* pre, std::vector<float>* latent,
                       std::vector<float>* down)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "[fish codec] " + std::move(m); }
    return false;
  };
  if (!_enc_ready) { return fail("loaded without the encoder"); }
  if (pcm == nullptr || n <= 0 || codes == nullptr) {
    return fail("no audio");
  }
  const int W = _dim;
  const int frames = (n + kHop - 1) / kHop;
  const int R = frames * 4;                   // 86 Hz rows
  // The conv stack, tiled over 86 Hz output rows with a causal halo.
  const std::size_t tile_act =
      (std::size_t)(tile_rows() + kHaloRows) * kRowSamples * 64 * 2;
  if (!ensure_scratch_(std::max(tile_act, (std::size_t)R * W),
                       (std::size_t)R) ||
      !ensure_rope_(R)) {
    return fail("out of memory");
  }
  SharedBuffer lat = _mc->make_shared_buffer((std::size_t)R * W * 2);
  if (lat.empty()) { return fail("out of memory"); }
  CommandStream s = _mc->make_command_stream();
  for (int r0 = 0; r0 < R; r0 += tile_rows()) {
    const int r1 = std::min(R, r0 + tile_rows());
    const int t0 = std::max(0, r0 - kHaloRows);
    const int L0 = (r1 - t0) * kRowSamples;
    auto* d = static_cast<_Float16*>(_d.contents());
    for (int i = 0; i < L0; ++i) {
      const long src = (long)t0 * kRowSamples + i;
      d[i] = (_Float16)(src < n ? pcm[src] : 0.0f);
    }
    int L = L0;
    {
      ComputeEncoder enc = s.begin_compute();
      conv_run_(enc, _enc_in, _d, _a, L);
      for (const EncBlock& b : _enc) {
        for (const ResUnit& r : b.res) { resunit_run_(enc, r, _a, _b, _c, L); }
        snake_run_(enc, b.act, _a, _b, L);
        conv_run_(enc, b.down, _b, _a, L);
        L /= b.down.stride;
      }
    }
    std::string gerr;
    if (!s.commit().wait_ok(&gerr)) {
      return fail(gerr.empty() ? "encode failed" : gerr);
    }
    std::memcpy(static_cast<std::uint8_t*>(lat.contents()) +
                    (std::size_t)r0 * W * 2,
                static_cast<const std::uint8_t*>(_a.contents()) +
                    (std::size_t)(r0 - t0) * W * 2,
                (std::size_t)(r1 - r0) * W * 2);
  }
  // Whole from here on: the transformers' reach is long, and 86 Hz rows of
  // 1024 channels are cheap.
  SharedBuffer cb = _mc->make_shared_buffer((std::size_t)frames * kCodebooks *
                                            4);
  if (cb.empty()) { return fail("out of memory"); }
  auto grab = [&](const SharedBuffer& b, std::size_t rows,
                  std::vector<float>* out) {
    if (out == nullptr) { return; }
    out->resize(rows * W);
    const auto* z = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < out->size(); ++i) { (*out)[i] = (float)z[i]; }
  };
  {
    ComputeEncoder enc = s.begin_compute();
    if (!transformer_run_(enc, _enc_tf, lat, R)) {
      return fail("encoder transformer failed");
    }
    snake_run_(enc, _enc_act, lat, _b, R);
    conv_run_(enc, _enc_out, _b, _a, R);
  }
  if (latent != nullptr || down != nullptr) {
    if (!s.commit().wait_ok()) { return fail("encode failed"); }
    grab(_a, (std::size_t)R, latent);
  }
  {
    ComputeEncoder enc = s.begin_compute();
    // Downsample 86 -> 43 -> 21.5 Hz, each a strided conv + ConvNeXt.
    conv_run_(enc, _down[0].conv, _a, _b, R);
    convnext_run_(enc, _down[0].nx, _b, R / 2);
    conv_run_(enc, _down[1].conv, _b, _a, R / 2);
    convnext_run_(enc, _down[1].nx, _a, R / 4);
  }
  if (down != nullptr) {
    if (!s.commit().wait_ok()) { return fail("encode failed"); }
    grab(_a, (std::size_t)frames, down);
  }
  {
    ComputeEncoder enc = s.begin_compute();
    if (!transformer_run_(enc, _pre, _a, frames)) {
      return fail("pre transformer failed");
    }
    enc.set_function(_fn_vq_enc);
    enc.set_buffer(0, _a);
    enc.set_buffer(1, _win);
    enc.set_buffer(2, _bin);
    enc.set_buffer(3, _cbn);
    enc.set_buffer(4, _tables);
    enc.set_buffer(5, cb);
    enc.set_constant(6, W);
    enc.set_constant(7, _sem);
    enc.set_constant(8, _res);
    enc.dispatch({256, (unsigned)frames, 1}, {256, 1, 1});
  }
  std::string gerr;
  if (!s.commit().wait_ok(&gerr)) {
    return fail(gerr.empty() ? "encode failed" : gerr);
  }
  codes->assign(static_cast<const std::int32_t*>(cb.contents()),
                static_cast<const std::int32_t*>(cb.contents()) +
                    (std::size_t)frames * kCodebooks);
  if (pre != nullptr) {
    pre->resize((std::size_t)frames * W);
    const auto* z = static_cast<const _Float16*>(_a.contents());
    for (std::size_t i = 0; i < pre->size(); ++i) { (*pre)[i] = (float)z[i]; }
  }
  return true;
}

}  // namespace vpipe::genai
