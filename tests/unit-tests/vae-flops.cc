// The achieved-FLOP/s meter (common/perf-flops.h) as the VAE stages
// feed it: vae-decode, vae-encode and audio-vae-decode run these models,
// and each credits its own GPU arithmetic as it retires.
//
// Two things are held. EVERYTHING COUNTED IS FLUSHED AND CREDITED: a
// decode leaves nothing pending (a commit without a flush in front of
// it would) and the meter's GPU total grows by exactly what the model
// flushed. And for the image and video VAEs, THE COUNT IS THE MODEL'S:
// it equals the arithmetic derived here from the architecture --
// 2 * outputs * cout * taps * cin a conv (a causal conv's taps before
// the sequence are padding, not arithmetic), 2MNK a projection, and
// 4 * pairs * d an attention head. The audio decoders run everything
// through one GEMM helper, which is where they count, so for them the
// first property is the whole question.
//
// Each test is gated on its checkpoint's variable and skips without it.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace pf = vpipe::perf_flops;

const char*
env_(const char* name)
{
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? v : nullptr;
}

// A zeroed buffer of `n` 16-bit elements: what a value is does not
// change what a model multiplies.
SharedBuffer
zeros_(MetalCompute* mc, std::size_t n)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (!b.empty()) { std::memset(b.contents(), 0, n * 2); }
  return b;
}

// What one call added to the meter's GPU total: waits for the last
// buffer's handler, which may run after the wait that saw it.
struct Credit {
  std::uint64_t g0 = 0, f0 = 0;
  explicit Credit(const GpuFlopsTally& t)
    : g0(pf::totals(pf::kGpu).flops), f0(t.flushed())
  {
  }
  // {flushed by the model, credited to the meter} since construction.
  std::pair<std::uint64_t, std::uint64_t>
  take(const GpuFlopsTally& t) const
  {
    const std::uint64_t f = t.flushed() - f0;
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0 < f; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return {f, pf::totals(pf::kGpu).flops - g0};
  }
};

std::uint64_t
conv3_(long long hw, long long cout, long long cin)
{
  return gemm_flops(hw, cout, 9 * cin);
}

// A residual block of two 3x3 convs, with a 1x1 shortcut when it changes
// the width.
std::uint64_t
resblock_(long long hw, long long cin, long long cout)
{
  return conv3_(hw, cout, cin) + conv3_(hw, cout, cout) +
         (cin != cout ? gemm_flops(hw, cout, cin) : 0);
}

// The single-head spatial attention of a mid block: q, k, v and the
// output projection, and the attention itself.
std::uint64_t
mid_attn_(long long hw, long long c)
{
  return 4 * gemm_flops(hw, c, c) + attention_flops(1, hw * hw, c);
}

// The Wan 2.1 / Qwen-Image VAE (one architecture; MetalKrea2Vae runs it
// as 2D convs, MetalWanVae as causal 3D ones -- the same arithmetic for
// one frame). Decoder widths [base*m[-1]] + base*m[::-1]; a level's
// upsample conv halves its width, so the next level starts at half.
struct WanArch {
  int base, z, nres, in_ch, out_ch;
  std::vector<int> mult;

  std::uint64_t
  decode(long long hw) const
  {
    const int nlev = (int)mult.size();
    std::vector<int> dims{base * mult[(std::size_t)nlev - 1]};
    for (int i = nlev - 1; i >= 0; --i) {
      dims.push_back(base * mult[(std::size_t)i]);
    }
    std::uint64_t f = gemm_flops(hw, z, z) + conv3_(hw, dims[0], z) +
                      2 * resblock_(hw, dims[0], dims[0]) +
                      mid_attn_(hw, dims[0]);
    for (int i = 0; i < nlev; ++i) {
      int cin = i > 0 ? dims[(std::size_t)i] / 2 : dims[(std::size_t)i];
      const int out = dims[(std::size_t)i + 1];
      for (int r = 0; r <= nres; ++r) {
        f += resblock_(hw, cin, out);
        cin = out;
      }
      if (i != nlev - 1) {
        hw *= 4;
        f += conv3_(hw, out / 2, out);
      }
    }
    return f + conv3_(hw, out_ch, dims.back());
  }

  std::uint64_t
  encode(long long hw) const
  {
    const int nlev = (int)mult.size();
    std::vector<int> dims{base};
    for (int m : mult) { dims.push_back(base * m); }
    std::uint64_t f = conv3_(hw, dims[0], in_ch);
    for (int i = 0; i < nlev; ++i) {
      int cin = dims[(std::size_t)i];
      const int out = dims[(std::size_t)i + 1];
      for (int r = 0; r < nres; ++r) {
        f += resblock_(hw, cin, out);
        cin = out;
      }
      if (i != nlev - 1) {
        hw /= 4;
        f += conv3_(hw, out, out);   // stride 2
      }
    }
    const int top = dims.back();
    return f + 2 * resblock_(hw, top, top) + mid_attn_(hw, top) +
           conv3_(hw, 2 * z, top) + gemm_flops(hw, 2 * z, 2 * z);
  }
};

}  // namespace

// The Qwen-Image VAE Krea-2 ships, as vae-decode and vae-encode run it.
TEST(vae_flops, krea2_vae_counts_its_convs)
{
  const char* root = env_("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  // The checkpoint's own config, read as the VAE stages read it: the
  // encoder refuses to run without its latent statistics.
  const std::string vdir = std::string(root) + "/vae";
  MetalKrea2Vae::Config cfg;
  {
    std::ifstream in(vdir + "/config.json");
    ASSERT_TRUE((bool)in);
    const FlexData fd = FlexData::from_json(in);
    ASSERT_TRUE(fd.is_object());
    if (fd.is_object()) { MetalKrea2Vae::config_from_json(fd, &cfg); }
  }
  // The derivation below is the plain one, one width for both halves.
  ASSERT_TRUE(!cfg.is_residual && cfg.dec_base() == cfg.base_dim);
  auto m = MetalKrea2Vae::load(vdir, mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const WanArch arch{cfg.base_dim, cfg.z_dim, cfg.num_res_blocks,
                     cfg.in_channels, cfg.out_channels, cfg.dim_mult};

  const int h8 = 16, w8 = 16, H = h8 * 8, W = w8 * 8;
  SharedBuffer z = zeros_(mc, (std::size_t)cfg.z_dim * h8 * w8);
  ASSERT_TRUE(!z.empty());
  {
    const Credit c(m->gpu_flops());
    std::string err;
    ASSERT_TRUE(!m->decode(z, h8, w8, &err).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    const std::uint64_t want = arch.decode((long long)h8 * w8);
    std::printf("[vae_flops] krea2 decode %dx%d: counted %.4f GFLOP, "
                "derived %.4f, credited %.4f\n", W, H, f / 1e9, want / 1e9,
                g / 1e9);
    EXPECT_TRUE(f == want);
    EXPECT_TRUE(g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
  SharedBuffer img = zeros_(mc, (std::size_t)cfg.in_channels * H * W);
  ASSERT_TRUE(!img.empty());
  {
    const Credit c(m->gpu_flops());
    ASSERT_TRUE(!m->encode(img, H, W).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    const std::uint64_t want = arch.encode((long long)H * W);
    std::printf("[vae_flops] krea2 encode %dx%d: counted %.4f GFLOP, "
                "derived %.4f, credited %.4f\n", W, H, f / 1e9, want / 1e9,
                g / 1e9);
    EXPECT_TRUE(f == want);
    EXPECT_TRUE(g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
}

// FLUX.2's VAE (diffusers' AutoencoderKL layout: block_out widths,
// layers_per_block resnets a level going down and one more coming up,
// an upsample conv that keeps the width, a stride-2 conv going down).
TEST(vae_flops, flux2_vae_counts_its_convs)
{
  const char* root = env_("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalFlux2Vae::Config cfg;
  auto m = MetalFlux2Vae::load(std::string(root) + "/vae", mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  ASSERT_TRUE(m->ensure_encoder());
  const int* B = cfg.block_out;
  const int L = cfg.latent_channels, P = cfg.layers_per_block;

  const int h16 = 8, w16 = 8;
  const int h8 = h16 * cfg.patch, w8 = w16 * cfg.patch;
  const int H = h8 * 8, W = w8 * 8;
  SharedBuffer z = zeros_(mc, (std::size_t)cfg.dit_channels() * h16 * w16);
  ASSERT_TRUE(!z.empty());
  {
    long long hw = (long long)h8 * w8;
    std::uint64_t want =
        (cfg.use_post_quant_conv ? gemm_flops(hw, L, L) : 0) +
        conv3_(hw, B[3], L) + 2 * resblock_(hw, B[3], B[3]) +
        mid_attn_(hw, B[3]);
    int cin = B[3];
    for (int i = 0; i < 4; ++i) {
      const int out = B[3 - i];
      for (int r = 0; r <= P; ++r) {
        want += resblock_(hw, cin, out);
        cin = out;
      }
      if (i != 3) {
        hw *= 4;
        want += conv3_(hw, out, out);
      }
    }
    want += conv3_(hw, 3, B[0]);

    const Credit c(m->gpu_flops());
    std::string err;
    ASSERT_TRUE(!m->decode(z, h16, w16, &err).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    std::printf("[vae_flops] flux2 decode %dx%d: counted %.4f GFLOP, "
                "derived %.4f, credited %.4f\n", W, H, f / 1e9, want / 1e9,
                g / 1e9);
    EXPECT_TRUE(f == want);
    EXPECT_TRUE(g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
  SharedBuffer img = zeros_(mc, (std::size_t)cfg.in_channels * H * W);
  ASSERT_TRUE(!img.empty());
  {
    long long hw = (long long)H * W;
    std::uint64_t want = conv3_(hw, B[0], cfg.in_channels);
    int cin = B[0];
    for (int i = 0; i < 4; ++i) {
      for (int r = 0; r < P; ++r) {
        want += resblock_(hw, cin, B[i]);
        cin = B[i];
      }
      if (i != 3) {
        hw /= 4;
        want += conv3_(hw, B[i], B[i]);   // stride 2
      }
    }
    want += 2 * resblock_(hw, B[3], B[3]) + mid_attn_(hw, B[3]) +
            conv3_(hw, 2 * L, B[3]) +
            (cfg.use_quant_conv ? gemm_flops(hw, 2 * L, 2 * L) : 0);

    const Credit c(m->gpu_flops());
    ASSERT_TRUE(!m->encode(img, H, W).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    std::printf("[vae_flops] flux2 encode %dx%d: counted %.4f GFLOP, "
                "derived %.4f, credited %.4f\n", W, H, f / 1e9, want / 1e9,
                g / 1e9);
    EXPECT_TRUE(f == want);
    EXPECT_TRUE(g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
}

// The Wan 2.1 VAE (FlashVSR's), causal in time. One latent frame is the
// 2D arithmetic exactly: every causal conv has one live tap and the
// temporal resamples pass the first chunk through. Several frames are
// held to the first property only -- their live taps are the point of
// counting them, and a stream decode is pieces of one.
TEST(vae_flops, wan_vae_counts_its_live_taps)
{
  const char* root = env_("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(root, &layout) && !layout.vae.empty());
  if (layout.vae.empty()) { return; }
  MetalWanVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalWanVae::config_from_json(layout.vae, cfg, &err) ||
              MetalWanVae::config_for_native_checkpoint(layout.vae, cfg,
                                                        &err));
  auto m = MetalWanVae::load(layout.vae, mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const WanArch arch{cfg.base_dim, cfg.z_dim, cfg.num_res_blocks, 3, 3,
                     std::vector<int>(cfg.dim_mult, cfg.dim_mult + 4)};

  const int h8 = 16, w8 = 16, H = h8 * 8, W = w8 * 8;
  auto sink = [](const SharedBuffer&, int, int) { return true; };
  for (int T : {1, 3}) {
    SharedBuffer z = zeros_(mc, (std::size_t)cfg.z_dim * T * h8 * w8);
    ASSERT_TRUE(!z.empty());
    const Credit c(m->gpu_flops());
    ASSERT_TRUE(m->decode(z, T, h8, w8, sink, &err));
    const auto [f, g] = c.take(m->gpu_flops());
    const std::uint64_t want = arch.decode((long long)h8 * w8);
    std::printf("[vae_flops] wan decode %d latent frame(s) %dx%d: counted "
                "%.4f GFLOP, credited %.4f%s\n", T, W, H, f / 1e9, g / 1e9,
                T == 1 ? fmt(", derived {:.4f}", want / 1e9)().c_str()
                       : "");
    if (T == 1) { EXPECT_TRUE(f == want); }
    EXPECT_TRUE(f > 0 && g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
  SharedBuffer video = zeros_(mc, (std::size_t)3 * H * W);
  ASSERT_TRUE(!video.empty());
  {
    const Credit c(m->gpu_flops());
    ASSERT_TRUE(!m->encode(video, 1, H, W, &err).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    const std::uint64_t want = arch.encode((long long)H * W);
    std::printf("[vae_flops] wan encode 1 frame %dx%d: counted %.4f GFLOP, "
                "derived %.4f, credited %.4f\n", W, H, f / 1e9, want / 1e9,
                g / 1e9);
    EXPECT_TRUE(f == want);
    EXPECT_TRUE(g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
}

// MiniMax-H3's video VAE: a ViT decoder (register tokens beside the
// latent voxels) and a causal conv encoder.
TEST(vae_flops, minimax_h3_video_vae_counts_its_layers)
{
  const char* root = env_("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  MetalMiniMaxH3VideoVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(root, cfg, &err));
  cfg.ane_ffn = false;
  auto m = MetalMiniMaxH3VideoVae::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  const int T = 1, h = 4, w = 4;
  const long long voxels = (long long)T * h * w;
  const long long rows = voxels + cfg.n_register + 1;
  const long long D = cfg.dim, FF = cfg.ffn_inner, ZC = cfg.z_channels;
  const long long PE = (long long)cfg.out_channels * cfg.patch_t *
                       cfg.patch * cfg.patch;
  const std::uint64_t layer =
      gemm_flops(rows, 3 * D, D) +
      attention_flops(cfg.n_heads, rows * rows, cfg.head_dim) +
      gemm_flops(rows, D, D) + gemm_flops(rows, 2 * FF, D) +
      gemm_flops(rows, D, FF);
  const std::uint64_t want =
      gemm_flops(voxels, ZC, ZC) + gemm_flops(voxels, D, ZC) +
      (std::uint64_t)cfg.n_layers * layer + gemm_flops(voxels, PE, D);
  SharedBuffer z = zeros_(mc, (std::size_t)(voxels * ZC));
  ASSERT_TRUE(!z.empty());
  {
    const Credit c(m->gpu_flops());
    ASSERT_TRUE(!m->decode(z, T, h, w, &err).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    std::printf("[vae_flops] h3 video decode %lld voxels, %lld rows: "
                "counted %.4f GFLOP, derived %.4f, credited %.4f\n",
                voxels, rows, f / 1e9, want / 1e9, g / 1e9);
    EXPECT_TRUE(f == want);
    EXPECT_TRUE(g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
  const int ET = 5, EH = 64, EW = 64;
  SharedBuffer x = zeros_(mc, (std::size_t)cfg.in_channels * ET * EH * EW);
  ASSERT_TRUE(!x.empty());
  {
    const Credit c(m->gpu_flops());
    ASSERT_TRUE(!m->encode(x, ET, EH, EW, &err).empty());
    const auto [f, g] = c.take(m->gpu_flops());
    std::printf("[vae_flops] h3 video encode %dx%dx%d: counted %.4f GFLOP, "
                "credited %.4f\n", ET, EW, EH, f / 1e9, g / 1e9);
    EXPECT_TRUE(f > 0 && g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
}

// MiniMax-H3's audio VAE, as audio-vae-decode (and its encode) run it.
TEST(vae_flops, minimax_h3_audio_vae_credits_what_it_counts)
{
  const char* root = env_("VPIPE_MINIMAX_H3_AVAE_PATH");
  if (root == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string dir = MetalMiniMaxH3AudioVae::resolve_vae_dir(root);
  MetalMiniMaxH3AudioVae::Config cfg;
  std::string err;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(dir, cfg, &err));
  auto m = MetalMiniMaxH3AudioVae::load(dir, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const int stereo = 2, frames = 12;
  std::vector<float> z((std::size_t)stereo * cfg.latent_channels * frames,
                       0.0f);
  std::vector<float> pcm;
  {
    const Credit c(m->gpu_flops());
    ASSERT_TRUE(m->decode(z.data(), stereo, frames, &pcm, &err));
    const auto [f, g] = c.take(m->gpu_flops());
    std::printf("[vae_flops] h3 audio decode %d frames: counted %.4f "
                "GFLOP, credited %.4f\n", frames, f / 1e9, g / 1e9);
    EXPECT_TRUE(f > 0 && g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
  {
    std::vector<float> lat;
    int got = 0;
    const int samples = cfg.sample_rate / 4;
    std::vector<float> in((std::size_t)stereo * samples, 0.0f);
    const Credit c(m->gpu_flops());
    if (!m->encode(in.data(), stereo, samples, &lat, &got, &err)) {
      std::printf("[vae_flops] h3 audio encode skipped: %s\n", err.c_str());
      return;
    }
    const auto [f, g] = c.take(m->gpu_flops());
    std::printf("[vae_flops] h3 audio encode %d samples: counted %.4f "
                "GFLOP, credited %.4f\n", samples, f / 1e9, g / 1e9);
    EXPECT_TRUE(f > 0 && g == f);
    EXPECT_TRUE(m->gpu_flops().pending() == 0);
  }
}

// YuE2's Oobleck decoder, the "yue2" family audio-vae-decode runs.
TEST(vae_flops, oobleck_decoder_credits_what_it_counts)
{
  const char* root = env_("VPIPE_YUE2_VAE_TEST_MODEL_PATH");
  if (root == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto m = MetalOobleckDecoder::load(root, mc, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  // Longer than a tile, so more than one commit carries the count.
  const int T = m->core_frames() + 40;
  std::vector<float> z((std::size_t)m->config().latent_dim * T, 0.0f);
  std::vector<float> pcm;
  const Credit c(m->gpu_flops());
  ASSERT_TRUE(m->decode(z.data(), T, &pcm, /*clamp=*/false, {}, &err));
  const auto [f, g] = c.take(m->gpu_flops());
  std::printf("[vae_flops] oobleck decode %d frames: counted %.4f GFLOP, "
              "credited %.4f\n", T, f / 1e9, g / 1e9);
  EXPECT_TRUE(f > 0 && g == f);
  EXPECT_TRUE(m->gpu_flops().pending() == 0);
}
