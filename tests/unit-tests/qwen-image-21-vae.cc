// Qwen-Image-2.1's AutoencoderKLQwenImage21, which runs on the same
// MetalKrea2Vae as Krea-2 / Qwen-Image-Edit after the generalization that
// gave it a variable level count, asymmetric encoder/decoder widths, RGBA
// channels and the `is_residual` shortcuts.
//
// Three tiers, and they check different things on purpose:
//
//  * config   the real vae/config.json parses into the Config the rest of
//             this file assumes. Cheap, no GPU, no weights -- and it is
//             the check that matters most, because a half-read config
//             (four levels instead of five, base_dim standing in for
//             decoder_base_dim) asks for tensors that EXIST at widths
//             that do not, and the failure is a plausible picture.
//  * kernel   the two parameter-free shortcut kernels against a CPU
//             oracle, at small shapes, with no model on disk. NOTE WHAT
//             THIS DOES AND DOES NOT PROVE: the oracle re-states the same
//             closed-form index mapping the kernel implements, so it
//             catches kernel bugs -- bounds, the ulong index discipline,
//             the accumulate-into-out contract, a wrong grid -- and it
//             CANNOT catch a closed form that disagrees with diffusers.
//             That is the golden's job below.
//  * model    the real checkpoint loads, and a round trip through
//             encode+decode comes back as the same picture.
//
// Env: VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH = model root (uses <root>/vae),
// VPIPE_QWEN_IMAGE21_GOLDEN = golden dir. Each tier skips its own vars.

#include "minitest.h"

#include <mach/mach.h>

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// minitest's ASSERT_TRUE records a failure and CONTINUES, so anything
// guarding a buffer read needs its own return.
// `cond` is evaluated ONCE. An ASSERT_TRUE(cond) followed by
// if (!(cond)) evaluates it twice, which is invisible for a
// comparison and doubles the work for a call -- a model loaded, or a
// forward run, twice per use.
#define Q21_REQUIRE(cond)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      report_expect_true_failure(#cond, __FILE__, __LINE__);               \
      return;                                                              \
    }                                                                      \
  } while (0)

std::string
model_root_()
{
  const char* r = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH");
  return (r != nullptr && *r != '\0') ? std::string(r) : std::string();
}

std::string
golden_dir_()
{
  const char* g = std::getenv("VPIPE_QWEN_IMAGE21_GOLDEN");
  return (g != nullptr && *g != '\0') ? std::string(g) : std::string();
}

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Write vpipe's own result beside the golden when VPIPE_QWEN_IMAGE21_DUMP
// names a directory. This exists for BISECTING a numeric gap: with the
// output on disk it can be compared against reference variants (one
// component disabled at a time) without rebuilding anything.
void
dump_(const char* name, const std::vector<float>& v)
{
  const char* d = std::getenv("VPIPE_QWEN_IMAGE21_DUMP");
  if (d == nullptr || *d == '\0') { return; }
  std::error_code ec;
  std::filesystem::create_directories(d, ec);
  std::ofstream out(std::filesystem::path(d) / (std::string(name) + ".f32"),
                    std::ios::binary);
  if (out) {
    out.write(reinterpret_cast<const char*>(v.data()),
              (std::streamsize)(v.size() * sizeof(float)));
  }
}

// The checkpoint's own config, or a Null FlexData when it is not here.
FlexData
vae_config_json_()
{
  const std::string root = model_root_();
  if (root.empty()) { return {}; }
  std::ifstream in(std::filesystem::path(root) / "vae" / "config.json");
  if (!in) { return {}; }
  try {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) { return fd; }
  } catch (...) {
  }
  return {};
}

}  // namespace

// ---- config ---------------------------------------------------------

TEST(qwen_image_21_vae, config_from_json_reads_every_new_key)
{
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }          // model not on this box

  MetalKrea2Vae::Config cfg;                // Krea-2 defaults
  MetalKrea2Vae::config_from_json(fd, &cfg);

  // FIVE levels, and the spatial factor follows from the LENGTH rather
  // than from a separate key -- the default {1,2,4,4} must have been
  // replaced wholesale, not merged into.
  EXPECT_TRUE(cfg.levels() == 5);
  EXPECT_TRUE(cfg.spatial_factor() == 16);
  EXPECT_TRUE(cfg.dim_mult.size() == 5);
  if (cfg.dim_mult.size() == 5) {
    EXPECT_TRUE(cfg.dim_mult[0] == 1 && cfg.dim_mult[1] == 2 &&
                cfg.dim_mult[2] == 4 && cfg.dim_mult[3] == 8 &&
                cfg.dim_mult[4] == 8);
  }
  EXPECT_TRUE(cfg.z_dim == 64);
  // ASYMMETRIC: the encoder is 96 wide and the decoder 144. Reading the
  // decoder at base_dim is the bug that under-books every byte estimate
  // by a third.
  EXPECT_TRUE(cfg.base_dim == 96);
  EXPECT_TRUE(cfg.decoder_base_dim == 144);
  EXPECT_TRUE(cfg.dec_base() == 144);
  // RGBA on both sides.
  EXPECT_TRUE(cfg.in_channels == 4);
  EXPECT_TRUE(cfg.out_channels == 4);
  EXPECT_TRUE(cfg.is_residual);
  EXPECT_TRUE(cfg.num_res_blocks == 2);
  // The temporal flags survive on a still because the shortcut's channel
  // mapping needs them (see the kernels).
  EXPECT_TRUE(cfg.temperal_downsample.size() == 4);
  if (cfg.temperal_downsample.size() == 4) {
    EXPECT_TRUE(cfg.temperal_downsample[0] == 0);
    EXPECT_TRUE(cfg.temperal_downsample[1] == 1);
    EXPECT_TRUE(cfg.temperal_downsample[2] == 1);
    EXPECT_TRUE(cfg.temperal_downsample[3] == 1);
  }
  EXPECT_TRUE(cfg.latents_mean.size() == 64);
  EXPECT_TRUE(cfg.latents_std.size() == 64);
}

// A Krea-2 config must still read as Krea-2 through the same function --
// the generalization's whole risk is that it moved four shipping models.
TEST(qwen_image_21_vae, config_defaults_are_unchanged_for_four_levels)
{
  MetalKrea2Vae::Config cfg;
  EXPECT_TRUE(cfg.levels() == 4);
  EXPECT_TRUE(cfg.spatial_factor() == 8);
  EXPECT_TRUE(cfg.dec_base() == 96);        // sentinel resolves to base_dim
  EXPECT_TRUE(cfg.in_channels == 3 && cfg.out_channels == 3);
  EXPECT_FALSE(cfg.is_residual);

  // An object carrying only the keys the old reader knew leaves the rest
  // alone, so an existing checkpoint cannot acquire five levels or RGBA
  // by accident.
  FlexData fd = FlexData::make_object();
  {
    auto o = fd.as_object();
    o.insert_or_assign("z_dim", FlexData::make_int(16));
    o.insert_or_assign("base_dim", FlexData::make_int(96));
    o.insert_or_assign("num_res_blocks", FlexData::make_int(2));
  }
  MetalKrea2Vae::Config got;
  MetalKrea2Vae::config_from_json(fd, &got);
  EXPECT_TRUE(got.levels() == 4);
  EXPECT_TRUE(got.spatial_factor() == 8);
  EXPECT_TRUE(got.dec_base() == 96);
  EXPECT_FALSE(got.is_residual);
  EXPECT_TRUE(got.out_channels == 3);
}

// diffusers' AutoencoderKLQwenImage as an RGBA checkpoint publishes it: ONE
// channel key (`input_channels`) for both ends, and a scalar
// `scaling_factor` with null per-channel statistics. Both were read as
// absent, which loaded the RGBA weights (the conv shapes come from the
// tensors) and then decoded a 4-wide output at a stride of 3, and left
// the stages inert for want of latents_mean/std.
TEST(qwen_image_21_vae, config_reads_input_channels_and_scaling_factor)
{
  FlexData fd = FlexData::from_json(R"({
      "_class_name": "AutoencoderKLQwenImage", "base_dim": 96,
      "dim_mult": [1, 2, 4, 4], "z_dim": 16, "num_res_blocks": 2,
      "input_channels": 4, "latents_mean": null, "latents_std": null,
      "scaling_factor": 8.0064, "shift_factor": 0.0})");
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  EXPECT_TRUE(cfg.in_channels == 4 && cfg.out_channels == 4);
  ASSERT_TRUE(cfg.latents_mean.size() == 16 &&
              cfg.latents_std.size() == 16);
  if (cfg.latents_std.size() == 16 && cfg.latents_mean.size() == 16) {
    // z * std + mean == z / 8.0064 + 0, the reference's un-scale.
    EXPECT_TRUE(cfg.latents_mean[0] == 0.0f && cfg.latents_mean[15] == 0.0f);
    EXPECT_TRUE(std::fabs(cfg.latents_std[7] - 1.0f / 8.0064f) < 1e-9f);
  }

  // An explicit pair still wins over the shared key, and published
  // statistics are never replaced by the scalar.
  FlexData both = FlexData::from_json(R"({
      "input_channels": 3, "in_channels": 4, "out_channels": 4,
      "z_dim": 2, "latents_mean": [0.5, -0.5], "latents_std": [2.0, 3.0],
      "scaling_factor": 8.0})");
  MetalKrea2Vae::Config c2;
  MetalKrea2Vae::config_from_json(both, &c2);
  EXPECT_TRUE(c2.in_channels == 4 && c2.out_channels == 4);
  EXPECT_TRUE(c2.latents_std.size() == 2 && c2.latents_std[1] == 3.0f);

  // Krea-2's own config spells `input_channels: 3` beside real stats:
  // unchanged.
  FlexData krea = FlexData::from_json(R"({"input_channels": 3,
      "z_dim": 1, "latents_mean": [0.1], "latents_std": [1.5]})");
  MetalKrea2Vae::Config c3;
  MetalKrea2Vae::config_from_json(krea, &c3);
  EXPECT_TRUE(c3.in_channels == 3 && c3.out_channels == 3);
  EXPECT_TRUE(c3.latents_std.size() == 1 && c3.latents_std[0] == 1.5f);
}

// ---- the shortcut kernels, against a CPU oracle ---------------------

namespace {

// Dispatch one of the two shortcut kernels over host vectors and return
// `out` (which the kernel ACCUMULATES into, so it is seeded non-zero
// here -- a kernel that assigned instead of adding would pass a
// zero-seeded test and drop the main path's result in the real decode).
bool
run_kernel_(MetalCompute* mc, const char* fn, const std::vector<float>& in,
            std::vector<float>& out, const std::vector<int>& consts)
{
  metal_compute::ComputeLibrary lib = mc->load_library("llm_elementwise");
  metal_compute::ComputeFunction f = lib.function(fn);
  if (!f.valid()) { return false; }
  SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
  SharedBuffer ob = mc->make_shared_buffer(out.size() * 2);
  if (ib.empty() || ob.empty()) { return false; }
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < in.size(); ++i) { d[i] = (_Float16)in[i]; }
    auto* o = static_cast<_Float16*>(ob.contents());
    for (std::size_t i = 0; i < out.size(); ++i) { o[i] = (_Float16)out[i]; }
  }
  metal_compute::CommandStream stream = mc->make_command_stream();
  {
    metal_compute::ComputeEncoder enc = stream.begin_compute();
    if (!enc.valid()) { return false; }
    enc.set_function(f);
    enc.set_buffer(0, ib);
    enc.set_buffer(1, ob);
    for (std::size_t i = 0; i < consts.size(); ++i) {
      enc.set_constant((int)i + 2, consts[i]);
    }
    const int Cout = consts[3];
    const unsigned rows = (unsigned)(out.size() / (std::size_t)Cout);
    enc.dispatch({(unsigned)Cout, rows, 1}, {256, 1, 1});
  }
  metal_compute::CommandStream::Fence fence = stream.commit();
  if (!fence.valid()) { return false; }
  fence.wait();
  const auto* op = static_cast<const _Float16*>(ob.contents());
  for (std::size_t i = 0; i < out.size(); ++i) { out[i] = (float)op[i]; }
  return true;
}

}  // namespace

TEST(qwen_image_21_vae, dup_up_shortcut_matches_cpu)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // The three decoder shapes the real checkpoint uses, by (Cin, Cout, ft):
  // level 0/1 keep the width with a temporal factor, level 2 halves it
  // with one, level 3 halves it without. Each picks a different channel
  // out of the source, which is the whole risk.
  struct Case { int Cin, Cout, ft; };
  const Case cases[] = {{16, 16, 2}, {16, 8, 2}, {8, 4, 1}};
  const int H = 5, W = 3;                   // deliberately not square/even

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);

  for (const Case& c : cases) {
    std::vector<float> in((std::size_t)H * W * c.Cin);
    for (float& v : in) { v = u(rng); }
    const std::size_t orows = (std::size_t)(2 * H) * (2 * W);
    std::vector<float> seed(orows * (std::size_t)c.Cout);
    for (float& v : seed) { v = u(rng); }

    // CPU oracle, seeded with the same values so the ACCUMULATE is part
    // of what is compared.
    std::vector<float> want = seed;
    const int repeats = (c.Cout * c.ft * 4) / c.Cin;
    for (int oy = 0; oy < 2 * H; ++oy) {
      for (int ox = 0; ox < 2 * W; ++ox) {
        for (int o = 0; o < c.Cout; ++o) {
          const int j = ((o * c.ft + (c.ft - 1)) * 2 + (oy & 1)) * 2 + (ox & 1);
          const int sc = j / repeats;
          const std::size_t si =
              ((std::size_t)(oy / 2) * W + (ox / 2)) * c.Cin + sc;
          const std::size_t oi =
              ((std::size_t)oy * (2 * W) + ox) * c.Cout + o;
          want[oi] += in[si];
        }
      }
    }

    std::vector<float> got = seed;
    const bool ok = run_kernel_(mc, "vae_res_dup_up_f16", in, got,
                                {H, W, c.Cin, c.Cout, c.ft});
    Q21_REQUIRE(ok);
    const double r = rel_l2_(got.data(), want.data(), want.size());
    std::printf("[qwen_image_21_vae] dup_up Cin=%d Cout=%d ft=%d rel-L2 %.3e\n",
                c.Cin, c.Cout, c.ft, r);
    EXPECT_TRUE(r < 2e-3);                  // f16 round trip only
  }
}

TEST(qwen_image_21_vae, avg_down_shortcut_matches_cpu)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // The encoder's shapes: level 0 pools without a temporal factor, the
  // middle levels widen WITH one (so half their output channels come out
  // zero -- the padded frame), and the last level neither pools nor
  // widens, collapsing to a plain residual add.
  struct Case { int Cin, Cout, ft, fs; };
  const Case cases[] = {{8, 8, 1, 2}, {8, 16, 2, 2}, {16, 16, 1, 1}};
  const int H = 8, W = 6;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);

  for (const Case& c : cases) {
    std::vector<float> in((std::size_t)H * W * c.Cin);
    for (float& v : in) { v = u(rng); }
    const int OH = H / c.fs, OW = W / c.fs;
    std::vector<float> seed((std::size_t)OH * OW * c.Cout);
    for (float& v : seed) { v = u(rng); }

    std::vector<float> want = seed;
    const int factor = c.ft * c.fs * c.fs;
    const int G = (c.Cin * factor) / c.Cout;
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        for (int o = 0; o < c.Cout; ++o) {
          float acc = 0.0f;
          for (int g = 0; g < G; ++g) {
            const int n = o * G + g;
            const int sc = n / factor;
            const int rem = n % factor;
            const int t = rem / (c.fs * c.fs);
            if (t != c.ft - 1) { continue; }       // zero-padded frame
            const int sy = (rem / c.fs) % c.fs;
            const int sx = rem % c.fs;
            const std::size_t si =
                ((std::size_t)(oy * c.fs + sy) * W + (ox * c.fs + sx)) *
                    c.Cin + sc;
            acc += in[si];
          }
          want[((std::size_t)oy * OW + ox) * c.Cout + o] += acc / (float)G;
        }
      }
    }

    std::vector<float> got = seed;
    const bool ok = run_kernel_(mc, "vae_res_avg_down_f16", in, got,
                                {H, W, c.Cin, c.Cout, c.ft, c.fs});
    Q21_REQUIRE(ok);
    const double r = rel_l2_(got.data(), want.data(), want.size());
    std::printf(
        "[qwen_image_21_vae] avg_down Cin=%d Cout=%d ft=%d fs=%d rel-L2 %.3e\n",
        c.Cin, c.Cout, c.ft, c.fs, r);
    EXPECT_TRUE(r < 2e-3);
  }
}

// ---- the real checkpoint --------------------------------------------

// The round trip below, as a function so a second test can run it with
// the conv route forced: its PSNR, kSkipped where the model is not set,
// or kBroken where something failed before there was a picture (said).
constexpr double kSkipped = -1.0, kBroken = -2.0;
#define Q21_RT_REQUIRE(cond)                                               \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("[qwen_image_21_vae] %s:%d: %s\n", __FILE__, __LINE__,   \
                  #cond);                                                  \
      return kBroken;                                                      \
    }                                                                      \
  } while (0)
static double
round_trip_psnr_(const char* tag)
{
  const std::string root = model_root_();
  if (root.empty()) { return kSkipped; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return kSkipped; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return kSkipped; }

  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/true);
  Q21_RT_REQUIRE(m != nullptr);
  Q21_RT_REQUIRE(m->has_encoder());
  Q21_RT_REQUIRE(m->config().spatial_factor() == 16);

  // A picture the encoder can actually round-trip: smooth, so the test
  // is about the architecture rather than about high-frequency loss.
  // 64x64 is four latent cells a side at /16.
  const int H = 64, W = 64;
  const int IC = cfg.in_channels;
  std::vector<float> img((std::size_t)IC * H * W);
  for (int c = 0; c < IC; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float fx = (float)x / (float)(W - 1);
        const float fy = (float)y / (float)(H - 1);
        img[((std::size_t)c * H + y) * W + x] =
            0.6f * std::sin(6.2831853f * (fx + 0.13f * c)) * std::cos(
                3.1415927f * fy);
      }
    }
  }
  SharedBuffer ib = mc->make_shared_buffer(img.size() * 2);
  Q21_RT_REQUIRE(!ib.empty());
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < img.size(); ++i) { d[i] = (_Float16)img[i]; }
  }

  SharedBuffer lat = m->encode(ib, H, W);
  Q21_RT_REQUIRE(!lat.empty());
  const int hL = H / 16, wL = W / 16;
  // z_dim channels at the /16 grid, and nothing else would be this size.
  Q21_RT_REQUIRE(lat.byte_size() >=
              (std::size_t)cfg.z_dim * hL * wL * 2);

  // encode() whitens; decode() wants the raw latent back.
  SharedBuffer raw = m->unwhiten(lat, hL, wL);
  Q21_RT_REQUIRE(!raw.empty());
  std::string err;
  SharedBuffer rgb = m->decode(raw, hL, wL, &err);
  if (rgb.empty()) {
    std::printf("[qwen_image_21_vae] decode failed: %s\n", err.c_str());
  }
  Q21_RT_REQUIRE(!rgb.empty());
  Q21_RT_REQUIRE(rgb.byte_size() >= (std::size_t)cfg.out_channels * H * W * 2);

  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  std::vector<float> got((std::size_t)cfg.out_channels * H * W);
  bool finite = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    got[i] = (float)rp[i];
    if (!std::isfinite(got[i])) { finite = false; }
  }
  Q21_RT_REQUIRE(finite);

  // PSNR over the channels the input actually carried.
  double se = 0.0;
  const std::size_t n = (std::size_t)IC * H * W;
  for (std::size_t i = 0; i < n && i < got.size(); ++i) {
    const double d = (double)got[i] - (double)img[i];
    se += d * d;
  }
  const double mse = se / (double)n;
  const double psnr = mse > 0.0 ? 10.0 * std::log10(4.0 / mse) : 99.0;
  std::printf("[qwen_image_21_vae] round trip%s %dx%d -> [%d,%d,%d] PSNR "
              "%.2f dB\n", tag, H, W, cfg.z_dim, hL, wL, psnr);
  return psnr;
}
#undef Q21_RT_REQUIRE

// A loose floor on purpose: this says the architecture is wired, not
// that it is numerically right. The golden below is what pins that.
TEST(qwen_image_21_vae, loads_and_round_trips)
{
  const double psnr = round_trip_psnr_("");
  if (psnr == kSkipped) { return; }
  EXPECT_TRUE(psnr > 18.0);
}

// THE ON-CHIP CONV'S STRIDE 2, END TO END. Every conv the hardware op
// declines goes to conv2d_mma_3x3_* (VPIPE_KREA2_CONV2D) -- at 64x64 the
// encoder's last downsample has a 4x4 output, under the hardware op's 8.
// That kernel padded a downsample (1,1,1,1) where the encoders pad
// (0,1,0,1), so the round trip read 17.4 dB on an M5 whenever the timed
// route tune happened to pick it, and 35 when it did not.
TEST(qwen_image_21_vae, a_round_trip_on_the_on_chip_conv)
{
  ::setenv("VPIPE_KREA2_CONV2D", "1", 1);
  const double psnr = round_trip_psnr_(" (on-chip conv)");
  ::unsetenv("VPIPE_KREA2_CONV2D");
  if (psnr == kSkipped) { return; }
  EXPECT_TRUE(psnr > 18.0);
}

// A DECODE IS THE SAME RUN TO RUN. The materialized mid attention centres
// K by its column sums, and those were summed with atomic adds in
// whatever order the row chunks finished -- two identical generations
// came out ~70 dB apart. Two decodes of one latent through that member,
// BYTE-IDENTICAL, at a size whose mid has many chunks (64x64 latent
// cells: 4096 rows).
TEST(qwen_image_21_vae, a_decode_is_the_same_run_to_run)
{
  const std::string root = model_root_();
  if (root.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  // The materialized member, whatever the tune would say.
  ::setenv("VPIPE_VAE_MAT_ATTN_MMA2", "1", 1);
  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/false);
  ::unsetenv("VPIPE_VAE_MAT_ATTN_MMA2");
  Q21_REQUIRE(m != nullptr);
  const int hL = 64, wL = 64;
  const std::size_t n = (std::size_t)cfg.z_dim * hL * wL;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  Q21_REQUIRE(!z.empty());
  {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)nd(rng); }
  }
  std::string err;
  SharedBuffer a = m->decode(z, hL, wL, &err);
  Q21_REQUIRE(!a.empty());
  SharedBuffer b = m->decode(z, hL, wL, &err);
  Q21_REQUIRE(!b.empty());
  const std::size_t bytes =
      (std::size_t)cfg.out_channels * hL * 16 * wL * 16 * 2;
  Q21_REQUIRE(a.byte_size() >= bytes && b.byte_size() >= bytes);
  const bool same = std::memcmp(a.contents(), b.contents(), bytes) == 0;
  std::printf("[qwen_image_21_vae] two decodes of one latent (%dx%d): %s\n",
              hL * 16, wL * 16, same ? "IDENTICAL" : "DIFFERENT");
  EXPECT_TRUE(same);
}

TEST(qwen_image_21_vae, decode_matches_golden)
{
  const std::string root = model_root_();
  const std::string gdir = golden_dir_();
  if (root.empty() || gdir.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> lat = read_f32_(gdir + "/g_vae_declat.f32");
  const std::vector<float> want = read_f32_(gdir + "/g_vae_dec.f32");
  if (lat.empty() || want.empty()) { return; }

  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  Q21_REQUIRE(lat.size() % (std::size_t)cfg.z_dim == 0);
  const std::size_t cells = lat.size() / (std::size_t)cfg.z_dim;
  int hL = 1;
  while ((std::size_t)hL * hL < cells) { ++hL; }
  Q21_REQUIRE((std::size_t)hL * hL == cells);

  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/false);
  Q21_REQUIRE(m != nullptr);
  SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
  Q21_REQUIRE(!z.empty());
  {
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
  }
  std::string err;
  SharedBuffer rgb = m->decode(z, hL, hL, &err);
  Q21_REQUIRE(!rgb.empty());
  Q21_REQUIRE(rgb.byte_size() >= want.size() * 2);
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  std::vector<float> got(want.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)rp[i]; }

  dump_("vpipe_vae_dec", got);
  const double r = rel_l2_(got.data(), want.data(), got.size());
  std::printf("[qwen_image_21_vae] decode rel-L2 = %.6f ([%d,%d,%d] -> %dx%d)\n",
              r, cfg.z_dim, hL, hL, hL * 16, hL * 16);
  EXPECT_TRUE(r < 0.05);
}

TEST(qwen_image_21_vae, encode_matches_golden)
{
  const std::string root = model_root_();
  const std::string gdir = golden_dir_();
  if (root.empty() || gdir.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> in   = read_f32_(gdir + "/g_vae_input.f32");
  const std::vector<float> want = read_f32_(gdir + "/g_vae_enc.f32");
  if (in.empty() || want.empty()) { return; }

  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  Q21_REQUIRE(in.size() % (std::size_t)cfg.in_channels == 0);
  const std::size_t hw = in.size() / (std::size_t)cfg.in_channels;
  int H = 1;
  while ((std::size_t)H * H < hw) { ++H; }
  Q21_REQUIRE((std::size_t)H * H == hw);

  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/true);
  Q21_REQUIRE(m != nullptr);
  Q21_REQUIRE(m->has_encoder());
  SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
  Q21_REQUIRE(!ib.empty());
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < in.size(); ++i) { d[i] = (_Float16)in[i]; }
  }
  SharedBuffer lat = m->encode(ib, H, H);
  Q21_REQUIRE(!lat.empty());
  Q21_REQUIRE(lat.byte_size() >= want.size() * 2);
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  std::vector<float> got(want.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)lp[i]; }

  dump_("vpipe_vae_enc", got);
  const double r = rel_l2_(got.data(), want.data(), got.size());
  std::printf("[qwen_image_21_vae] encode rel-L2 = %.6f (%dx%d -> [%d,%d,%d])\n",
              r, H, H, cfg.z_dim, H / 16, H / 16);
  EXPECT_TRUE(r < 0.05);
}

// A TIMED DECODE AT A REAL OUTPUT SIZE.
//
// The golden above decodes whatever the dumper wrote, which is small.
// This one takes the size from VPIPE_QWEN_IMAGE21_VAE_BENCH (pixels,
// e.g. 1024) and times a decode at it, so the conv path can be A/B'd
// where it actually costs something. Random latents: the arithmetic
// does not care and this is a timing instrument, not a correctness
// one -- `decode_matches_golden` is that.
//
// Env-gated and off by default: a 1024 decode is seconds, and the
// suite should not pay for it.
TEST(qwen_image_21_vae, decode_bench)
{
  const char* bench = std::getenv("VPIPE_QWEN_IMAGE21_VAE_BENCH");
  if (bench == nullptr || *bench == '\0') { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);

  const int px = std::atoi(bench) > 0 ? std::atoi(bench) : 1024;
  const int hL = px / cfg.spatial_factor();
  Q21_REQUIRE(hL > 0);
  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/false);
  Q21_REQUIRE(m != nullptr);

  const std::size_t n = (std::size_t)cfg.z_dim * hL * hL;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  Q21_REQUIRE(!z.empty());
  {
    auto* d = static_cast<_Float16*>(z.contents());
    unsigned st = 7u;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)((st >> 8) & 0xffff) / 32768.0f - 1.0f));
    }
  }
  // One warm decode, then the timed pair: the first pays the conv and
  // attention autotunes, which are a property of the box and not of the
  // path under test.
  std::string err;
  SharedBuffer warm = m->decode(z, hL, hL, &err);
  Q21_REQUIRE(!warm.empty());
  warm = SharedBuffer{};

  double best = 0.0;
  for (int i = 0; i < 2; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer rgb = m->decode(z, hL, hL, &err);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    Q21_REQUIRE(!rgb.empty());
    if (best == 0.0 || ms < best) { best = ms; }
  }
  std::printf("[qwen_image_21_vae] decode %dx%d (latent %dx%d, /%d): "
              "%.0f ms\n", px, px, hL, hL, cfg.spatial_factor(), best);
}

// WHAT A DECODE HOLDS AGAINST decode_peak_bytes
// (VPIPE_QWEN_IMAGE21_VAE_PEAK=<side>[:<tier>], tier off | accurate |
// fast | token; side 0 loads and decodes nothing). Run each arm in a
// process of its own under /usr/bin/time -l and take the side-0 arm's
// peak footprint off: what is left is the decode's own, against the
// figure the preflight books, printed here.
TEST(qwen_image_21_vae, decode_peak_against_plan)
{
  const char* e = std::getenv("VPIPE_QWEN_IMAGE21_VAE_PEAK");
  if (e == nullptr || *e == '\0') { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/false);
  Q21_REQUIRE(m != nullptr);
  const int side = std::atoi(e);
  const char* colon = std::strchr(e, ':');
  const std::string tier = colon != nullptr ? colon + 1 : "off";
  if (tier != "off") {
    MetalKrea2Vae::Accel a;
    a.i8_gemm = true;
    a.i8_tier = tier == "accurate" ? 0 : tier == "fast" ? 1
              : tier == "token"    ? 2 : 3;
    m->set_accel(a);
  }
  if (side <= 0) {
    std::printf("[qwen_image_21_vae peak] load only (%s)\n", tier.c_str());
    return;
  }
  const int hL = side / cfg.spatial_factor();
  const std::size_t n = (std::size_t)cfg.z_dim * hL * hL;
  SharedBuffer z = mc->make_shared_buffer(n * 2);
  Q21_REQUIRE(!z.empty());
  {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)nd(rng); }
  }
  // The footprint sampled every half millisecond during each decode,
  // against what it was just before: the first decode carries the conv
  // route tune, the second is the steady state.
  auto footprint = []() -> std::size_t {
    task_vm_info_data_t vi{};
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &cnt) !=
        KERN_SUCCESS) {
      return 0;
    }
    return (std::size_t)vi.phys_footprint;
  };
  std::string err;
  std::size_t peaks[2] = {0, 0};
  for (int i = 0; i < 2; ++i) {
    // Freed Metal memory leaves the footprint asynchronously (~1 s for
    // 8 GB on an M5): settle first, or the previous decode's pages are
    // in this one's baseline.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    const std::size_t before = footprint();
    std::atomic<bool> done{false};
    std::atomic<std::size_t> hi{before};
    std::thread sampler([&] {
      while (!done.load()) {
        const std::size_t f = footprint();
        if (f > hi.load()) { hi.store(f); }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    });
    SharedBuffer rgb = m->decode(z, hL, hL, &err);
    done.store(true);
    sampler.join();
    Q21_REQUIRE(!rgb.empty());
    peaks[i] = hi.load() - before;
  }
  std::printf("[qwen_image_21_vae peak] %d^2 %s: decode_peak_bytes %zu MB, "
              "measured first %zu MB / warm %zu MB (%s)\n", side,
              tier.c_str(), m->decode_peak_bytes(hL, hL) >> 20,
              peaks[0] >> 20, peaks[1] >> 20, m->accel_note().c_str());
}

// INT8 AGAINST F16, ON A REAL PICTURE (MetalKrea2Vae::set_accel).
//
// VPIPE_QWEN_IMAGE21_VAE_IMAGE names a picture as raw f32, channel-first
// [in_channels, H, W] in [-1, 1] (a square side, a multiple of 16). It
// is encoded, then decoded in f16 -- the reference -- and again at each
// int8 tier, each decode timed (best of two after a warm one) and held
// to the reference by PSNR; the int8 ENCODE is held to the f16 one by
// the picture its latent decodes to. A real picture's latent, not noise:
// the activations' channel statistics are what the int8 route's scales
// are made of.
TEST(qwen_image_21_vae, int8_against_f16)
{
  const char* img_env = std::getenv("VPIPE_QWEN_IMAGE21_VAE_IMAGE");
  if (img_env == nullptr || *img_env == '\0') { return; }
  const std::string root = model_root_();
  if (root.empty()) { return; }
  const FlexData fd = vae_config_json_();
  if (!fd.is_object()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(fd, &cfg);
  const std::vector<float> img = read_f32_(img_env);
  const int IC = cfg.in_channels;
  Q21_REQUIRE(!img.empty() && img.size() % (std::size_t)IC == 0);
  int H = 1;
  while ((std::size_t)H * H < img.size() / IC) { ++H; }
  Q21_REQUIRE((std::size_t)H * H * IC == img.size());
  const int sf = cfg.spatial_factor();
  const int hL = H / sf;

  auto m = MetalKrea2Vae::load(root + "/vae", mc, cfg, /*with_encoder=*/true);
  Q21_REQUIRE(m != nullptr);
  SharedBuffer ib = mc->make_shared_buffer(img.size() * 2);
  Q21_REQUIRE(!ib.empty());
  {
    auto* d = static_cast<_Float16*>(ib.contents());
    for (std::size_t i = 0; i < img.size(); ++i) { d[i] = (_Float16)img[i]; }
  }
  const std::size_t nrgb = (std::size_t)cfg.out_channels * H * H;
  auto to_f32 = [&](const SharedBuffer& b, std::size_t n) {
    std::vector<float> v(n);
    const auto* p = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)p[i]; }
    return v;
  };
  // PSNR over [-1, 1] (peak-to-peak 2) on the colour channels.
  auto psnr = [&](const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t n = (std::size_t)3 * H * H;
    double se = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = (double)a[i] - (double)b[i];
      se += d * d;
    }
    const double mse = se / (double)n;
    return mse > 0.0 ? 10.0 * std::log10(4.0 / mse) : 99.0;
  };
  std::string err;
  auto decode_timed = [&](const SharedBuffer& z, double* ms) {
    SharedBuffer warm = m->decode(z, hL, hL, &err);
    if (warm.empty()) { return std::vector<float>{}; }
    double best = 0.0;
    SharedBuffer rgb;
    for (int i = 0; i < 2; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      rgb = m->decode(z, hL, hL, &err);
      const double t = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (rgb.empty()) { return std::vector<float>{}; }
      if (best == 0.0 || t < best) { best = t; }
    }
    *ms = best;
    return to_f32(rgb, nrgb);
  };

  // The f16 reference.
  SharedBuffer lat = m->encode(ib, H, H);
  Q21_REQUIRE(!lat.empty());
  SharedBuffer raw = m->unwhiten(lat, hL, hL);
  Q21_REQUIRE(!raw.empty());
  double ms16 = 0.0;
  const std::vector<float> ref = decode_timed(raw, &ms16);
  Q21_REQUIRE(!ref.empty());
  std::printf("[qwen_image_21_vae] %dx%d f16: decode %.0f ms, %.2f dB "
              "against the picture\n", H, H, ms16, psnr(ref, img));

  const char* tiers[] = {"accurate", "fast", "token", "token_plain"};
  for (int t = 0; t < 4; ++t) {
    m->set_accel({/*i8_gemm=*/true, /*i8_tier=*/t});
    double ms8 = 0.0;
    const std::vector<float> got = decode_timed(raw, &ms8);
    Q21_REQUIRE(!got.empty());
    const double p = psnr(got, ref);
    std::printf("[qwen_image_21_vae]   int8 %-11s decode %.0f ms (%.2fx), "
                "%.2f dB against f16 [%s]\n", tiers[t], ms8, ms16 / ms8, p,
                m->accel_note().c_str());
    // Lossy by design, but a picture: an int8 route gone wrong (a scale
    // misplaced, a channel dropped) lands far under this.
    EXPECT_TRUE(p > 30.0);
  }
  // The int8 ENCODE, decoded in f16: what an edit's reference latent
  // loses. Timed as the decodes are.
  auto encode_timed = [&](double* ms) {
    SharedBuffer warm = m->encode(ib, H, H);
    if (warm.empty()) { return SharedBuffer{}; }
    double best = 0.0;
    SharedBuffer l;
    for (int i = 0; i < 2; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      l = m->encode(ib, H, H);
      const double t = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      if (l.empty()) { return SharedBuffer{}; }
      if (best == 0.0 || t < best) { best = t; }
    }
    *ms = best;
    return l;
  };
  m->set_accel({});
  double enc16 = 0.0;
  Q21_REQUIRE(!encode_timed(&enc16).empty());
  for (int t = 0; t < 2; ++t) {
    m->set_accel({true, t});
    double enc8 = 0.0;
    SharedBuffer lat8 = encode_timed(&enc8);
    Q21_REQUIRE(!lat8.empty());
    m->set_accel({});
    SharedBuffer raw8 = m->unwhiten(lat8, hL, hL);
    double ms_d = 0.0;
    const std::vector<float> got8 = decode_timed(raw8, &ms_d);
    Q21_REQUIRE(!got8.empty());
    const double pe = psnr(got8, ref);
    std::printf("[qwen_image_21_vae]   int8 %-11s encode %.0f ms against "
                "%.0f (%.2fx), decoded in f16 %.2f dB against f16\n",
                tiers[t], enc8, enc16, enc16 / enc8, pe);
    EXPECT_TRUE(pe > 30.0);
  }
}
