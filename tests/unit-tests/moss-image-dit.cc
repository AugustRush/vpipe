// MOSS-Image's DiT, schedule and rotary table against the reference.
//
// MOSS-Image (OpenMOSS) is a single-stream flow-matching DiT whose
// reference is the pipeline_moss_image.py the checkpoint ships. What is
// silent when wrong, and pinned here:
//
//   * ONE shared modulation for all blocks, with no SiLU in front of it;
//   * plain gates (x += o * gate) and the SITU activation over a fused
//     [hidden | gate] projection;
//   * the four-axis half-split rotary with the text length as an image
//     coordinate and centred h / w;
//   * the sequence order (image first) and the text blocks' own rows;
//   * the schedule's snap to the training grid;
//   * the negative branch's skipped block.
//
// Env: VPIPE_MOSS_IMAGE_GOLDEN (tools/dump_moss_image_golden.py: the
// `dit` and `schedule` dumps) for these; each skips vacuously without
// it, so check the tests actually RAN.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/moss-image/metal-moss-image-transformer.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/weight-set.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

#define MI_REQUIRE(cond)                                                   \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
golden_dir_()
{
  const char* g = std::getenv("VPIPE_MOSS_IMAGE_GOLDEN");
  return (g == nullptr || *g == '\0') ? std::string() : std::string(g);
}

std::vector<float>
read_f32_(const std::filesystem::path& p)
{
  std::vector<float> v;
  std::ifstream in(p, std::ios::binary);
  if (!in) { return v; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  v.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

float
bf16_(std::uint16_t h)
{
  std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& want)
{
  double num = 0.0, den = 0.0;
  const std::size_t n = std::min(got.size(), want.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

FlexData
manifest_(const std::string& tag)
{
  const std::string g = golden_dir_();
  if (g.empty()) { return FlexData(); }
  std::ifstream mf(std::filesystem::path(g) / (tag + "_manifest.json"));
  if (!mf) { return FlexData(); }
  try {
    return FlexData::from_json(mf);
  } catch (...) {
    return FlexData();
  }
}

struct Golden {
  bool ok = false;
  std::filesystem::path dir;
  std::string tag;
  int gh = 0, gw = 0, text_len = 0, text_dim = 0, layers = 0, heads = 0;
  int dim = 0, ffn = 0, skip = -1, lora_modules = 0;
  float timestep = 0.0f;
  double bf16_rel = 0.0;
  std::vector<float> latent, text, out, out_skip, rope, lora_out;
};

Golden
load_golden_(const char* tag = "dit")
{
  Golden G;
  G.tag = tag;
  const FlexData man = manifest_(tag);
  if (!man.is_object()) { return G; }
  G.dir = std::filesystem::path(golden_dir_());
  auto o = man.as_object();
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  G.gh = geti("grid_h", 0);
  G.gw = geti("grid_w", 0);
  G.text_len = geti("text_len", 0);
  G.text_dim = geti("text_dim", 0);
  G.layers = geti("num_layers", 0);
  G.heads = geti("num_heads", 0);
  G.dim = geti("dim", 0);
  G.ffn = geti("ffn_dim", 0);
  G.skip = geti("skip", -1);
  G.lora_modules = geti("lora_modules", 0);
  G.timestep =
      (float)(o.contains("timestep") ? o.at("timestep").as_real(0.0) : 0.0);
  G.bf16_rel =
      o.contains("bf16_rel") ? o.at("bf16_rel").as_real(0.0) : 0.0;
  auto f = [&](const char* nm) {
    return read_f32_(G.dir / (G.tag + "_" + nm + ".f32"));
  };
  G.latent = f("latent");
  G.text = f("text");
  G.out = f("out");
  G.out_skip = f("out_skip");
  G.rope = f("rope");
  G.lora_out = f("lora_out");
  G.ok = !G.latent.empty() && !G.text.empty() && !G.out.empty() &&
         G.gh > 0 && G.gw > 0;
  return G;
}

metal_compute::SharedBuffer
to_bf16_(MetalCompute* mc, const std::vector<float>& v)
{
  metal_compute::SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
  if (b.empty()) { return b; }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::uint32_t u;
    std::memcpy(&u, &v[i], 4);
    const std::uint32_t r = (u >> 16) & 1u;
    d[i] = (std::uint16_t)((u + 0x7fffu + r) >> 16);
  }
  return b;
}

// [C, H, W] -> [H*W, C] and back: the reference's
// flatten(2).transpose(1, 2) and its inverse.
std::vector<float>
to_tokens_(const std::vector<float>& chw, int C, int H, int W)
{
  std::vector<float> t((std::size_t)C * H * W);
  for (int c = 0; c < C; ++c) {
    for (int p = 0; p < H * W; ++p) {
      t[(std::size_t)p * C + c] = chw[(std::size_t)c * H * W + p];
    }
  }
  return t;
}

std::vector<float>
from_tokens_(const std::vector<float>& tok, int C, int H, int W)
{
  std::vector<float> o((std::size_t)C * H * W);
  for (int c = 0; c < C; ++c) {
    for (int p = 0; p < H * W; ++p) {
      o[(std::size_t)c * H * W + p] = tok[(std::size_t)p * C + c];
    }
  }
  return o;
}

std::vector<float>
run_(MetalMossImageTransformer* m, MetalCompute* mc, const Golden& G,
     const std::vector<int>& skip, std::string* err)
{
  std::vector<float> out;
  const int C = m->config().latent_channels;
  metal_compute::SharedBuffer lat =
      to_bf16_(mc, to_tokens_(G.latent, C, G.gh, G.gw));
  metal_compute::SharedBuffer txt = to_bf16_(mc, G.text);
  if (lat.empty() || txt.empty()) { return out; }
  MetalMossImageTransformer::Request req;
  req.latents = &lat;
  req.grid_h = G.gh;
  req.grid_w = G.gw;
  req.text = &txt;
  req.text_rows = G.text_len;
  req.timestep = G.timestep;
  req.skip_layers = skip;
  metal_compute::SharedBuffer v = m->forward(req, err);
  if (v.empty()) { return out; }
  const std::size_t n = (std::size_t)G.gh * G.gw * C;
  if (v.byte_size() < n * 2) { return out; }
  std::vector<float> flat(n);
  const auto* d = static_cast<const std::uint16_t*>(v.contents());
  for (std::size_t i = 0; i < n; ++i) { flat[i] = bf16_(d[i]); }
  return from_tokens_(flat, C, G.gh, G.gw);
}

double
worst_abs_(const std::vector<float>& a, const std::vector<float>& b)
{
  double w = 0.0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    w = std::max(w, (double)std::fabs(a[i] - b[i]));
  }
  return a.size() == b.size() ? w : 1e30;
}

}  // namespace

// ---- the schedule, which needs no model -------------------------------

// BIT-EXACT against the reference's own MOSSImageScheduler +
// FlowMatchEulerDiscreteScheduler, for step counts on both sides of
// where the snap starts merging nodes.
TEST(moss_image_sched, sigmas_match_the_reference)
{
  const std::string g = golden_dir_();
  if (g.empty()) { return; }
  std::ifstream in(std::filesystem::path(g) / "sched_schedule.json");
  if (!in) { return; }
  FlexData fd = FlexData::from_json(in);
  MI_REQUIRE(fd.is_object());
  auto root = fd.as_object();
  const double shift = root.at("shift").as_real(5.0);
  const FlexData sch = root.at("schedules");
  MI_REQUIRE(sch.is_object());
  auto so = sch.as_object();
  int checked = 0;
  for (const char* k : {"1", "2", "3", "4", "8", "16", "20", "28", "30",
                        "50", "64", "100", "200"}) {
    if (!so.contains(k)) { continue; }
    const FlexData e = so.at(k);
    const FlexData ws = e.as_object().at("sigmas");
    const FlexData wt = e.as_object().at("timesteps");
    auto wsa = ws.as_array();
    auto wta = wt.as_array();
    const std::vector<float> got = moss_image::sigmas(std::atoi(k), shift);
    bool same = got.size() == wsa.size();
    for (std::size_t i = 0; same && i < got.size(); ++i) {
      same = got[i] == (float)wsa.at(i).as_real(0.0);
    }
    // ... and the timestep the DiT is handed for each.
    for (std::size_t i = 0; same && i < wta.size(); ++i) {
      same = moss_image::timestep(got[i]) == (float)wta.at(i).as_real(0.0);
    }
    if (!same) {
      std::printf("[moss_image_sched] %s steps: %zu nodes against %zu\n", k,
                  got.size(), wsa.size());
    }
    EXPECT_TRUE(same);
    ++checked;
  }
  EXPECT_TRUE(checked >= 10);
}

// The rotary table against the reference's own _RoPE.forward, which
// writes [sin(4 axes x 16) | cos(4 x 16)] per row. The f32 table must
// agree to f32 rounding; the bf16-rounded one (what the forward uses,
// as the reference does) to bf16's half-ulp.
TEST(moss_image_dit, rope_table_matches_the_reference)
{
  const Golden G = load_golden_();
  if (!G.ok || G.rope.empty()) { return; }
  const int D = 128, half = D / 2;
  const int rows = G.gh * G.gw + G.text_len;
  MI_REQUIRE(G.rope.size() == (std::size_t)rows * D);
  for (bool round : {false, true}) {
    std::vector<float> c, s;
    moss_image::rope_tables(G.gh, G.gw, G.text_len, D, 10000.0f, round, &c,
                            &s);
    MI_REQUIRE(c.size() == (std::size_t)rows * half);
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
      for (int p = 0; p < half; ++p) {
        const float ws = G.rope[(std::size_t)r * D + p];
        const float wc = G.rope[(std::size_t)r * D + half + p];
        worst = std::max(worst, (double)std::fabs(
                                    s[(std::size_t)r * half + p] - ws));
        worst = std::max(worst, (double)std::fabs(
                                    c[(std::size_t)r * half + p] - wc));
      }
    }
    std::printf("[moss_image_dit] rope table (%s) worst |diff| %g\n",
                round ? "bf16" : "f32", worst);
    EXPECT_TRUE(worst < (round ? 4e-3 : 2e-6));
  }
}

TEST(moss_image_dit, read_dims_reads_the_generated_config)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  EXPECT_TRUE(cfg.n_layers == G.layers);
  EXPECT_TRUE(cfg.n_heads == G.heads);
  EXPECT_TRUE(cfg.hidden == G.dim);
  EXPECT_TRUE(cfg.ffn_inner == G.ffn);
  EXPECT_TRUE(cfg.text_dim == G.text_dim);
  EXPECT_TRUE(cfg.head_dim == 128);
  EXPECT_TRUE(cfg.latent_channels == 128);
  // A different class, and a head width the flash entries do not take.
  namespace fs = std::filesystem;
  const fs::path d = fs::temp_directory_path() / "vpipe_moss_image_cfg";
  std::error_code ec;
  fs::create_directories(d, ec);
  auto put = [&](const char* json) {
    std::ofstream o(d / "config.json");
    o << json;
  };
  MetalMossImageTransformer::Config c2;
  put(R"({"_class_name":"ZImageTransformer2DModel","dim":3072})");
  EXPECT_FALSE(MetalMossImageTransformer::Config::read_dims(d.string(), &c2));
  put(R"({"_class_name":"MOSSImageTransformer","dim":3072,"num_heads":48})");
  EXPECT_FALSE(MetalMossImageTransformer::Config::read_dims(d.string(), &c2));
  fs::remove_all(d, ec);
}

// The whole forward against the fp32 reference, and again with a block
// skipped (the negative branch's recipe). The bar is the reference's
// OWN bf16 run's distance from its fp32 one, with headroom: this forward
// rounds less than the reference does, so it should sit at or under it.
TEST(moss_image_dit, matches_reference_on_a_small_model)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(m != nullptr);
  // ENGAGEMENT BEFORE ACCURACY: gemm_mma_ declines under 64 rows.
  if (mc->supports_matrix_cores()) {
    EXPECT_TRUE(m->uses_mma2());
    EXPECT_TRUE(G.gh * G.gw + G.text_len >= 64);
  }
  std::string ferr;
  const std::vector<float> got = run_(m.get(), mc, G, {}, &ferr);
  if (got.empty()) {
    std::printf("[moss_image_dit] forward failed: %s\n", ferr.c_str());
  }
  MI_REQUIRE(!got.empty());
  const double rel = rel_l2_(got, G.out);
  const double bar = std::max(2.0 * G.bf16_rel, 0.02);
  std::printf("[moss_image_dit] velocity rel-L2 %.6f (reference bf16 "
              "%.6f, bar %.4f) over %zu, %d blocks, joint %d\n", rel,
              G.bf16_rel, bar, G.out.size(), G.layers,
              G.gh * G.gw + G.text_len);
  EXPECT_TRUE(rel < bar);
  if (G.skip >= 0 && !G.out_skip.empty()) {
    const std::vector<float> gs = run_(m.get(), mc, G, {G.skip}, &ferr);
    MI_REQUIRE(!gs.empty());
    const double rs = rel_l2_(gs, G.out_skip);
    // The skip must MOVE the output, or "matches" is vacuous.
    const double moved = rel_l2_(G.out_skip, G.out);
    std::printf("[moss_image_dit] skip %d: rel-L2 %.6f (skipping moves "
                "the reference %.4f)\n", G.skip, rs, moved);
    EXPECT_TRUE(rs < bar);
    EXPECT_TRUE(moved > 10.0 * bar);
  }
}

// STREAMED == PRELOADED, BYTE FOR BYTE, with a skipped block in the
// middle of the stream (whose read the prefetch must jump).
TEST(moss_image_dit, streamed_matches_preloaded)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  const std::vector<int> skip =
      G.skip >= 0 ? std::vector<int>{G.skip} : std::vector<int>{};
  auto pre = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(pre != nullptr);
  std::string e;
  const std::vector<float> a = run_(pre.get(), mc, G, {}, &e);
  const std::vector<float> as = run_(pre.get(), mc, G, skip, &e);
  pre.reset();
  for (int no_slots = 0; no_slots < 2; ++no_slots) {
    if (no_slots) { setenv("VPIPE_MOSS_IMAGE_NO_SLOTS", "1", 1); }
    auto str = MetalMossImageTransformer::load(ckpt, mc, cfg, true);
    std::vector<float> b, bs;
    if (str != nullptr) {
      EXPECT_TRUE(str->streaming());
      b = run_(str.get(), mc, G, {}, &e);
      bs = run_(str.get(), mc, G, skip, &e);
      // Again, now that residency may have promoted blocks.
      const std::vector<float> b2 = run_(str.get(), mc, G, {}, &e);
      EXPECT_TRUE(worst_abs_(b2, a) == 0.0);
    }
    if (no_slots) { unsetenv("VPIPE_MOSS_IMAGE_NO_SLOTS"); }
    MI_REQUIRE(!a.empty() && !b.empty() && !bs.empty());
    std::printf("[moss_image_dit] streamed%s vs preloaded: %g / skip %g\n",
                no_slots ? " (no slots)" : "", worst_abs_(a, b),
                worst_abs_(as, bs));
    EXPECT_TRUE(worst_abs_(a, b) == 0.0);
    EXPECT_TRUE(worst_abs_(as, bs) == 0.0);
  }
}

// RUNTIME LoRA against the reference's own dense run of the same
// factors FUSED into the base weights: it bound every module, it moves
// the output well past the bf16 floor, and scale 0 is the base EXACTLY.
TEST(moss_image_dit, runtime_lora_matches_a_fused_reference)
{
  const Golden G = load_golden_();
  if (!G.ok || G.lora_out.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  auto base = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(base != nullptr);
  std::string e;
  const std::vector<float> b0 = run_(base.get(), mc, G, {}, &e);
  base.reset();
  MetalMossImageTransformer::LoraSpec spec;
  spec.path = (G.dir / (G.tag + "_lora.safetensors")).string();
  spec.scale = 1.0f;
  auto m = MetalMossImageTransformer::load(ckpt, mc, cfg, false, {spec});
  MI_REQUIRE(m != nullptr);
  EXPECT_TRUE(m->lora_modules(0) == G.lora_modules);
  const std::vector<float> got = run_(m.get(), mc, G, {}, &e);
  MI_REQUIRE(!got.empty() && !b0.empty());
  const double rel = rel_l2_(got, G.lora_out);
  const double moved = rel_l2_(G.lora_out, G.out);
  std::printf("[moss_image_dit] lora rel-L2 %.6f against the fused "
              "reference (the adapter moves it %.4f)\n", rel, moved);
  EXPECT_TRUE(rel < std::max(2.0 * G.bf16_rel, 0.02));
  EXPECT_TRUE(moved > 5.0 * G.bf16_rel);
  m->set_lora_scale(0, 0.0f);
  const std::vector<float> off = run_(m.get(), mc, G, {}, &e);
  EXPECT_TRUE(worst_abs_(off, b0) == 0.0);
}

// The WIDE golden: 1088 image rows + 24 text, 8 heads (hidden 1024),
// six blocks -- the geometry that clears every fast path's floor
// (matmul2d's 64 rows, Sol's 1024 rows and 16 key blocks, i8's 1024
// rows AND K >= 1024), which the small golden is too small to reach.
TEST(moss_image_dit, matches_reference_at_the_wide_geometry)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  EXPECT_TRUE(G.gh * G.gw >= 1024 && cfg.hidden >= 1024);
  auto m = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(m != nullptr);
  std::string e;
  const std::vector<float> got = run_(m.get(), mc, G, {}, &e);
  const std::vector<float> gs = run_(m.get(), mc, G, {G.skip}, &e);
  MI_REQUIRE(!got.empty() && !gs.empty());
  const double bar = std::max(2.0 * G.bf16_rel, 0.02);
  const double r0 = rel_l2_(got, G.out), r1 = rel_l2_(gs, G.out_skip);
  std::printf("[moss_image_dit] wide: rel-L2 %.6f, skip %d %.6f (reference "
              "bf16 %.6f)\n", r0, G.skip, r1, G.bf16_rel);
  EXPECT_TRUE(r0 < bar);
  EXPECT_TRUE(r1 < bar);
}

// perf-flops: what the forward COUNTS equals what the shapes DERIVE,
// and the meter is CREDITED all of it -- with a skipped block counted as
// the nothing it is.
TEST(moss_image_dit, forward_credits_its_flops)
{
  namespace pf = vpipe::perf_flops;
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  auto m = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(m != nullptr);
  const long long H = cfg.hidden, F = cfg.ffn_inner;
  const long long IMG = (long long)G.gh * G.gw, T = G.text_len;
  const long long JT = IMG + T;
  auto derived = [&](int skipped) {
    std::uint64_t n = gemm_flops(1, H, cfg.freq_dim) + gemm_flops(1, H, H) +
                      gemm_flops(1, 6 * H, H) + gemm_flops(1, 2 * H, H) +
                      gemm_flops(IMG, H, cfg.latent_channels) +
                      gemm_flops(T, H, cfg.text_dim) +
                      gemm_flops(IMG, cfg.latent_channels, H);
    const std::uint64_t text =
        gemm_flops(T, 3 * H, H) + gemm_flops(T, H, H) +
        2 * gemm_flops(T, F, H) + gemm_flops(T, H, F) +
        attention_flops(cfg.n_heads, T * T, cfg.head_dim);
    const std::uint64_t main =
        4 * gemm_flops(JT, H, H) + gemm_flops(JT, 2 * F, H) +
        gemm_flops(JT, H, F) +
        attention_flops(cfg.n_heads, JT * JT, cfg.head_dim);
    n += (std::uint64_t)cfg.n_text_blocks * text;
    n += (std::uint64_t)(cfg.n_layers - skipped) * main;
    return n;
  };
  for (int skipped = 0; skipped < 2; ++skipped) {
    const pf::Totals g0 = pf::totals(pf::kGpu);
    const std::uint64_t f0 = m->gpu_flops().flushed();
    std::string e;
    const std::vector<float> out =
        run_(m.get(), mc, G,
             skipped ? std::vector<int>{G.skip} : std::vector<int>{}, &e);
    MI_REQUIRE(!out.empty());
    const std::uint64_t counted = m->gpu_flops().flushed() - f0;
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0.flops <
                                   counted; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const std::uint64_t credited = pf::totals(pf::kGpu).flops - g0.flops;
    std::printf("[moss_image_dit] flops (%d skipped): counted %llu, derived "
                "%llu, credited %llu\n", skipped,
                (unsigned long long)counted,
                (unsigned long long)derived(skipped),
                (unsigned long long)credited);
    EXPECT_TRUE(counted == derived(skipped));
    EXPECT_TRUE(credited == counted);
  }
}

// SOL, at the geometry that clears its floors: it engages, the picture
// moves (approximate by construction), and not far.
TEST(moss_image_dit, sol_attention_engages_and_stays_close)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  auto base = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(base != nullptr);
  std::string e;
  const std::vector<float> off = run_(base.get(), mc, G, {}, &e);
  base.reset();
  cfg.sol.enabled = true;
  auto m = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(m != nullptr);
  if (!m->uses_sol_attn()) {
    std::printf("[moss_image_dit] sol_attn unavailable here; skipped\n");
    return;
  }
  const std::vector<float> on = run_(m.get(), mc, G, {}, &e);
  MI_REQUIRE(!off.empty() && !on.empty());
  // Sol routed: fewer attention FLOPs credited than dense would be.
  const double d = rel_l2_(on, off);
  const double r = rel_l2_(on, G.out);
  std::printf("[moss_image_dit] sol (tau %.2f): rel-L2 vs dense %.6f, vs "
              "the reference %.6f\n", (double)cfg.sol.tau, d, r);
  EXPECT_TRUE(std::isfinite(d) && d > 0.0 && d < 0.6);
}

// A TIMED stop -- a thread sets the flag a third of the way into a
// forward, which is what a stop actually is (a flag tripped on the Nth
// CHECK is already set before any check runs). It must come back EMPTY
// with NO reason, well before the forward would have ended -- preloaded
// (the held stack's drains are what make it observable) and streamed.
TEST(moss_image_dit, a_timed_stop_lands_inside_a_forward)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  for (int arm = 0; arm < 2; ++arm) {
    const bool stream = arm == 1;
    // A streamed arm DECLARES NO RESERVE, so residency admits nothing
    // and every block really streams: a promoted block is on the held
    // path and would not measure this arm.
    auto m = MetalMossImageTransformer::load(ckpt, mc, cfg, stream);
    MI_REQUIRE(m != nullptr);
    std::string e;
    // Warm, then the whole forward's time with a stop installed that
    // never fires (so the held arm pays its drains in both runs).
    std::atomic<bool> flag{false};
    m->set_stream_stop([&]() { return flag.load(); });
    (void)run_(m.get(), mc, G, {}, &e);
    const auto w0 = std::chrono::steady_clock::now();
    const std::vector<float> whole = run_(m.get(), mc, G, {}, &e);
    const double whole_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - w0).count();
    MI_REQUIRE(!whole.empty());
    std::chrono::steady_clock::time_point fired;
    std::thread t([&]() {
      std::this_thread::sleep_for(std::chrono::microseconds(
          (long long)(whole_ms * 1000.0 / 3.0)));
      fired = std::chrono::steady_clock::now();
      flag.store(true);
    });
    std::string why;
    const std::vector<float> got = run_(m.get(), mc, G, {}, &why);
    const auto done = std::chrono::steady_clock::now();
    t.join();
    const double lat = std::chrono::duration<double, std::milli>(
        done - fired).count();
    std::printf("[moss_image_dit] timed stop (%s): returned %.1f ms after "
                "the flag; a whole forward is %.1f ms\n",
                stream ? "streamed" : "preloaded", lat, whole_ms);
    EXPECT_TRUE(got.empty());
    EXPECT_TRUE(why.empty());
    EXPECT_TRUE(lat < 0.6 * whole_ms);
    m->set_stream_stop({});
  }
}

// THE INT8 TIERS on a matrix-core box: each engages at the wide
// geometry (both floors cleared) and stays within a gross bar of the
// dense forward; preloaded == streamed BYTE-IDENTICAL on each.
TEST(moss_image_dit, i8_tiers_engage_and_stay_close)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  auto dense = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(dense != nullptr);
  std::string e;
  const std::vector<float> d0 = run_(dense.get(), mc, G, {}, &e);
  dense.reset();
  MI_REQUIRE(!d0.empty());
  for (int tier = 0; tier < 4; ++tier) {
    MetalMossImageTransformer::Config c = cfg;
    c.i8_gemm = true;
    c.i8_tier = tier;
    auto pre = MetalMossImageTransformer::load(ckpt, mc, c, false);
    MI_REQUIRE(pre != nullptr);
    EXPECT_TRUE(pre->uses_i8_gemm());
    if (tier > 0) { EXPECT_TRUE(pre->fast_tier()); }
    const std::vector<float> a = run_(pre.get(), mc, G, {}, &e);
    pre.reset();
    auto str = MetalMossImageTransformer::load(ckpt, mc, c, true);
    MI_REQUIRE(str != nullptr);
    const std::vector<float> b = run_(str.get(), mc, G, {}, &e);
    str.reset();
    MI_REQUIRE(!a.empty() && !b.empty());
    const double r = rel_l2_(a, d0);
    std::printf("[moss_image_dit] i8 tier %d: rel-L2 vs dense %.5f, vs the "
                "reference %.5f; streamed %s\n", tier, r,
                rel_l2_(a, G.out),
                worst_abs_(a, b) == 0.0 ? "identical" : "DIFFERS");
    EXPECT_TRUE(r < 0.25);
    // A streamed dense block stays on the accurate tier (its slot is the
    // checkpoint's bytes), so the holds agree only on the accurate tier.
    if (tier == 0) { EXPECT_TRUE(worst_abs_(a, b) == 0.0); }
  }
}

// On a box WITHOUT matrix cores every GEMM takes the dense tiled path,
// and the widest tile it loaded (bm64bn64). Emulated by turning BOTH
// matrix-core knobs off.
TEST(moss_image_dit, no_matrix_cores_stays_on_wide_tiles)
{
  const Golden G = load_golden_();
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  setenv("VPIPE_MOSS_IMAGE_NO_MMA2", "1", 1);
  setenv("VPIPE_MOSS_IMAGE_NO_NAX_ATTN", "1", 1);
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  std::unique_ptr<MetalMossImageTransformer> m;
  if (MetalMossImageTransformer::Config::read_dims(ckpt, &cfg)) {
    m = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  }
  std::vector<float> got;
  if (m != nullptr) {
    std::string e;
    got = run_(m.get(), mc, G, {}, &e);
  }
  unsetenv("VPIPE_MOSS_IMAGE_NO_MMA2");
  unsetenv("VPIPE_MOSS_IMAGE_NO_NAX_ATTN");
  MI_REQUIRE(m != nullptr && !got.empty());
  EXPECT_FALSE(m->uses_mma2());
  EXPECT_FALSE(m->uses_attn_nax());
  EXPECT_TRUE(m->gemm_tile() == 2);
  const double rel = rel_l2_(got, G.out);
  std::printf("[moss_image_dit] wide tiles (bm64bn64): rel-L2 %.6f\n", rel);
  EXPECT_TRUE(rel < std::max(2.0 * G.bf16_rel, 0.02));
}

// THE ROW BANDS the matmul2d routes run past the 2 GB line, exercised far
// below it by capping the band (VPIPE_MOSS_IMAGE_BAND_ROWS): rows are
// independent in the tiles, so a banded forward is BYTE-IDENTICAL to an
// unbanded one. Matrix-core boxes only -- elsewhere the route is not taken.
TEST(moss_image_dit, matmul2d_row_bands_are_exact)
{
  const Golden G = load_golden_("wide");
  if (!G.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalMossImageTransformer::Config cfg;
  const std::string ckpt = (G.dir / G.tag).string();
  MI_REQUIRE(MetalMossImageTransformer::Config::read_dims(ckpt, &cfg));
  std::string e;
  auto a = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  MI_REQUIRE(a != nullptr && a->uses_mma2());
  const std::vector<float> whole = run_(a.get(), mc, G, {}, &e);
  a.reset();
  setenv("VPIPE_MOSS_IMAGE_BAND_ROWS", "256", 1);
  auto b = MetalMossImageTransformer::load(ckpt, mc, cfg, false);
  std::vector<float> banded;
  if (b != nullptr) { banded = run_(b.get(), mc, G, {}, &e); }
  unsetenv("VPIPE_MOSS_IMAGE_BAND_ROWS");
  MI_REQUIRE(!whole.empty() && !banded.empty());
  std::printf("[moss_image_dit] 256-row bands over %d rows: worst |diff| "
              "%g\n", G.gh * G.gw + G.text_len, worst_abs_(whole, banded));
  EXPECT_TRUE(worst_abs_(whole, banded) == 0.0);
}
