// FlashVSR+ (Turbo and Base) on the FlashVSR transformer, held to its
// reference.
//
// FlashVSR+ is v1.1's transformer and source projection retrained, and
// what makes it its own model is the SCHEDULE it streams the stack with
// -- four latent frames a chunk from the first, a top-k taken over the
// whole chunk and not scaled by the frame area, a cache trimmed to three
// windows, and on Turbo 25 "hybrid" layers on a 3x3 band keeping one.
// None of that is in a tensor, and every row of it is silent when wrong:
// the other rule produces a latent of the right shape. So besides the
// golden comparisons, `the_schedule_is_load_bearing` breaks each row in
// turn and requires the latent to move off the reference.
//
// Goldens: tools/dump_flashvsr_plus_golden.py, the reference's own
// modules on the CPU in f32 (and on MPS in its own bf16, for the bar).
//
// Env:
//   VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH  the repository root (turbo/ and
//                                        base/ beside Wan2.1_VAE.pth and
//                                        posi_prompt.pth)
//   VPIPE_FLASHVSR_PLUS_GOLDEN           a directory holding the dumps:
//                                        turbo-384x640/, base-512/ and,
//                                        for the bar, their -bf16 twins
// The first two tests need neither.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/video-model-registry.h"
#include "generative-models/weight-set.h"
#include "stages/model-detect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;
using T = MetalFlashVsrTransformer;

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

double
rel_l2_(const float* got, const float* ref, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& ref)
{
  if (got.size() != ref.size() || got.empty()) { return 1e9; }
  return rel_l2_(got.data(), ref.data(), got.size());
}

std::vector<float>
read_f32_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<float> v((std::size_t)n / sizeof(float));
  in.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

// One golden: the variant, its geometry, and the reference's own bf16
// latent error when the twin was dumped (0 when not).
struct Golden {
  fs::path dir;
  std::string mode;
  int height = 0, width = 0, frames = 0, latents = 0, chunks = 0;
  double ref_bf16 = 0.0;
  bool ok = false;
};

Golden
read_golden_(const fs::path& dir)
{
  Golden g;
  std::ifstream in(dir / "config.json");
  if (!in) { return g; }
  FlexData c = FlexData::from_json(in);
  if (!c.is_object()) { return g; }
  auto o = c.as_object();
  g.dir = dir;
  g.mode = std::string(o.at("mode").as_string(""));
  g.height = (int)o.at("height").as_int(0);
  g.width = (int)o.at("width").as_int(0);
  g.frames = (int)o.at("frames").as_int(0);
  g.latents = (int)o.at("latent_frames").as_int(0);
  g.chunks = (int)o.at("chunks").as_int(0);
  const auto want = read_f32_(dir / "latent.f32");
  const fs::path twin = dir.string() + "-bf16";
  const auto bf = read_f32_(twin / "latent.f32");
  if (!want.empty() && bf.size() == want.size()) {
    g.ref_bf16 = rel_l2_(bf, want);
  }
  g.ok = !g.mode.empty() && g.height > 0 && !want.empty();
  return g;
}

// The goldens present, or none when the env is unset.
std::vector<Golden>
goldens_()
{
  std::vector<Golden> out;
  const char* g = std::getenv("VPIPE_FLASHVSR_PLUS_GOLDEN");
  const char* m = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  if (g == nullptr || m == nullptr || *g == '\0' || *m == '\0') {
    return out;
  }
  for (const char* d : {"turbo-384x640", "base-512"}) {
    Golden one = read_golden_(fs::path(g) / d);
    if (one.ok) { out.push_back(std::move(one)); }
  }
  return out;
}

// The variant's denoiser, with the release's schedule (or `override`).
std::unique_ptr<T>
load_dit_(metal_compute::MetalCompute* mc, const std::string& dir,
          const T::Schedule* override = nullptr)
{
  FlashVsrLayout layout;
  if (!resolve_flashvsr_layout(dir, &layout)) { return nullptr; }
  T::Config cfg;
  std::string err;
  if (!T::config_from_checkpoint(layout.denoiser, &cfg, &err)) {
    return nullptr;
  }
  cfg.schedule = override != nullptr ? *override
                                     : flashvsr_schedule(layout);
  auto ws = WeightSet::open(layout.denoiser, nullptr);
  auto ctx_ws = WeightSet::open(layout.context, nullptr);
  if (ws == nullptr || ctx_ws == nullptr) { return nullptr; }
  auto dit = T::load(ws, ctx_ws, layout.context_name, mc, cfg, &err);
  if (dit == nullptr) { std::printf("  load: %s\n", err.c_str()); }
  return dit;
}

// The golden's projected rows, bf16, latent frame after latent frame.
std::vector<std::uint16_t>
golden_rows_(const Golden& g, std::size_t row_frame)
{
  std::vector<std::uint16_t> rows((std::size_t)g.latents * row_frame);
  for (int i = 0; i < g.latents; ++i) {
    const auto rf =
        read_f32_(g.dir / ("lq_proj" + std::to_string(i) + ".f32"));
    if (rf.size() != row_frame) { return {}; }
    for (std::size_t k = 0; k < rf.size(); ++k) {
      rows[(std::size_t)i * row_frame + k] = to_bf16_(rf[k]);
    }
  }
  return rows;
}

// One clip of the golden through the denoiser: its rows, its noise, its
// geometry, the release's default params.
bool
run_golden_(T& dit, const Golden& g, std::vector<float>* latent,
            std::string* err)
{
  const std::size_t row_frame =
      (std::size_t)(g.height / 16) * (g.width / 16) * dit.config().hidden;
  const auto rows = golden_rows_(g, row_frame);
  const auto noise = read_f32_(g.dir / "noise.f32");
  if (rows.empty() || noise.empty()) {
    *err = "golden rows or noise missing";
    return false;
  }
  T::Request req;
  req.rows = rows.data();
  req.row_frames = g.latents;
  req.frames = 1 + 4 * g.latents;
  req.height = g.height;
  req.width = g.width;
  req.init_noise = noise.data();
  req.params = T::Params::defaults(dit.config().schedule);
  std::vector<int> shape;
  return dit.generate(req, latent, &shape, err);
}

}  // namespace

// The schedule's arithmetic, against the reference runner's own numbers
// (inference/runners/stream.py): no model needed.
TEST(flashvsr_plus, the_schedule_follows_the_reference)
{
  const T::Schedule turbo = T::Schedule::plus(true);
  const T::Schedule base = T::Schedule::plus(false);
  // Turbo's hybrid layers: all but 0, 6, 12, 18 and 24 (HYBRID_B25).
  EXPECT_TRUE(turbo.sliding_layers.size() == 25);
  for (int i = 0; i < 30; ++i) {
    EXPECT_TRUE(turbo.sliding(i) == (i % 6 != 0));
    EXPECT_FALSE(base.sliding(i));
  }
  EXPECT_TRUE(turbo.is_plus() && base.is_plus());
  EXPECT_FALSE(T::Schedule::v1_1().is_plus());

  // 61 source frames: 16 latent frames, four chunks of four, read from 65
  // (the last four repeat the last frame) and restoring all 61.
  EXPECT_TRUE(T::align_frames(base, 61) == 65);
  EXPECT_TRUE(T::align_frames(base, 65) == 65);
  EXPECT_TRUE(T::align_frames(base, 66) == 81);
  EXPECT_TRUE(T::latent_frames(base, 65) == 16);
  EXPECT_TRUE(T::chunks_of(base, 65) == 4);
  EXPECT_TRUE(T::restored_frames(base, 65) == 61);
  // The cadence: 17 frames open the stream (one, then four per latent
  // frame), every later chunk takes 16; chunk 0 restores [0, 13).
  EXPECT_TRUE(base.frames_through(0) == 17 && base.frames_through(1) == 33);
  EXPECT_TRUE(base.restored_end(0) == 13 && base.restored_begin(1) == 13);
  EXPECT_TRUE(base.restored_end(3) == 61);
  // ...and v1.1's, unchanged: 25 then 8, restoring [0, 21) first.
  const T::Schedule v11 = T::Schedule::v1_1();
  EXPECT_TRUE(v11.frames_through(0) == 25 && v11.frames_through(1) == 33);
  EXPECT_TRUE(v11.restored_end(0) == 21 && v11.restored_end(2) == 37);
  EXPECT_TRUE(T::align_frames(v11, 42) == 49);

  // The windows: a full layer keeps 3 and runs a ring of 5 (its pair
  // writes past the end from the third chunk); a hybrid one keeps 1 in
  // a ring of 3. v1.1 is the ring of 4 it always was.
  T::Config ct;
  ct.schedule = turbo;
  EXPECT_TRUE(T::keep_windows(ct, 0, 3.0) == 3);
  EXPECT_TRUE(T::keep_windows(ct, 1, 3.0) == 1);
  EXPECT_TRUE(T::layer_ring(ct, 0, 3.0) == 5);
  EXPECT_TRUE(T::layer_ring(ct, 1, 3.0) == 3);
  EXPECT_TRUE(T::kv_windows(ct, 3.0) == 5);
  EXPECT_TRUE(T::kv_windows(3.0) == 4 && T::ring_windows(25, 3.0) == 3);
  // A stream holds a chunk less than each ring: Turbo 5x3 + 25x1 windows
  // where Base holds 30x3 -- the "45% lower memory" is mostly here.
  T::Config cb;
  cb.schedule = base;
  const T::Params p = T::Params::defaults(base);
  EXPECT_TRUE(p.sparse_ratio == 2.5 && p.kv_ratio == 3.0 &&
              p.local_range == 11);
  const auto wt = T::stream_window_bytes(ct, 1152, 1920, p);
  const auto wb = T::stream_window_bytes(cb, 1152, 1920, p);
  EXPECT_TRUE(wt * 90 == wb * 40);
  std::printf("  stream windows at 1920x1152: Turbo %.0f MB, Base %.0f MB\n",
              wt / 1048576.0, wb / 1048576.0);
}

// The repository's two variant folders, and the root, on stub files:
// what decides Turbo or Base is the folder, never the tensors.
TEST(flashvsr_plus, layouts_name_their_variant)
{
  const fs::path tmp = fs::temp_directory_path() /
                       ("vpipe-fvsr-plus-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  auto touch = [](const fs::path& p) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << "x";
  };
  const fs::path repo = tmp / "FlashVSR_plus";
  for (const char* v : {"turbo", "base"}) {
    touch(repo / v / "dit.safetensors");
    touch(repo / v / "lq_proj_in.safetensors");
  }
  touch(repo / "Wan2.1_VAE.pth");
  touch(repo / "posi_prompt.pth");
  std::ofstream(repo / "config.json")
      << "{\"model_name\": \"FlashVSR+\", \"variants\": {"
         "\"turbo\": {\"path\": \"turbo\"}, \"base\": {\"path\": \"base\"},"
         "\"fast\": {\"path\": \"x\"}}}";

  FlashVsrLayout l;
  ASSERT_TRUE(resolve_flashvsr_layout((repo / "turbo").string(), &l));
  EXPECT_TRUE(l.plus && l.variant == "turbo");
  EXPECT_TRUE(l.vae == (repo / "Wan2.1_VAE.pth").string());
  EXPECT_TRUE(l.context == (repo / "posi_prompt.pth").string());
  EXPECT_TRUE(l.denoiser == (repo / "turbo" / "dit.safetensors").string());
  EXPECT_TRUE(l.source_prefix.empty() && l.context_name == "posi_prompt");
  EXPECT_FALSE(flashvsr_schedule(l).sliding_layers.empty());

  ASSERT_TRUE(resolve_flashvsr_layout((repo / "base").string(), &l));
  EXPECT_TRUE(l.plus && l.variant == "base");
  EXPECT_TRUE(flashvsr_schedule(l).sliding_layers.empty());
  EXPECT_TRUE(flashvsr_schedule(l).is_plus());

  // The repository root is Turbo, the reference CLI's default.
  ASSERT_TRUE(resolve_flashvsr_layout(repo.string(), &l));
  EXPECT_TRUE(l.plus && l.variant == "turbo");
  EXPECT_TRUE(l.denoiser == (repo / "turbo" / "dit.safetensors").string());

  // A folder named otherwise takes its name from the config's map...
  touch(repo / "x" / "dit.safetensors");
  touch(repo / "x" / "lq_proj_in.safetensors");
  ASSERT_TRUE(resolve_flashvsr_layout((repo / "x").string(), &l));
  EXPECT_TRUE(l.variant.empty());    // "fast" is neither of the two
  // ...and one nothing names is sized as Base and refused at load.
  EXPECT_TRUE(flashvsr_schedule(l).is_plus());
  EXPECT_TRUE(flashvsr_schedule(l).sliding_layers.empty());

  // The VAE a decode stage resolves for a variant folder and for the
  // repository: the shared file, one level up or beside.
  EXPECT_TRUE(resolve_vae_dir((repo / "turbo").string()) ==
              (repo / "Wan2.1_VAE.pth").string());
  EXPECT_TRUE(resolve_vae_dir(repo.string()) ==
              (repo / "Wan2.1_VAE.pth").string());

  // The family claims all of them; the frame rule follows the folder.
  VideoModelFamily* fam = VideoModelRegistry::get().find("flashvsr");
  ASSERT_TRUE(fam != nullptr);
  if (fam != nullptr) {
    EXPECT_TRUE(fam->claims((repo / "base").string(), ""));
    EXPECT_TRUE(fam->align_frames((repo / "base").string(), 61) == 65);
    EXPECT_TRUE(fam->align_frames((repo / "turbo").string(), 42) == 49);
  }
  fs::remove_all(tmp, ec);
}

// The source projection, driven with FlashVSR+'s cadence over the
// golden's clip: frame 0 alone, then four frames per latent frame, the
// end padded with the last frame.
TEST(flashvsr_plus, projection_matches_reference)
{
  const auto gs = goldens_();
  if (gs.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string root = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  for (const Golden& g : gs) {
    FlashVsrLayout l;
    ASSERT_TRUE(resolve_flashvsr_layout(root + "/" + g.mode, &l));
    std::string err;
    auto proj = FlashVsrLqProj::load(WeightSet::open(l.source, nullptr), mc,
                                     &err, l.source_prefix);
    ASSERT_TRUE(proj != nullptr);
    if (proj == nullptr) { std::printf("  %s\n", err.c_str()); continue; }
    const std::size_t frame = (std::size_t)3 * g.height * g.width;
    std::vector<std::uint8_t> lq(frame * (std::size_t)g.frames);
    {
      std::ifstream in(g.dir / "lq_video_u8.bin", std::ios::binary);
      in.read(reinterpret_cast<char*>(lq.data()), (std::streamsize)lq.size());
      ASSERT_TRUE(in.good());
    }
    // Pad to what the latents read: 1 + 4 per latent frame.
    const int read = 1 + 4 * g.latents;
    lq.resize(frame * (std::size_t)read);
    for (int f = g.frames; f < read; ++f) {
      std::memcpy(lq.data() + (std::size_t)f * frame,
                  lq.data() + (std::size_t)(g.frames - 1) * frame, frame);
    }
    std::vector<metal_compute::SharedBuffer> rows;
    int rf = 0;
    ASSERT_TRUE(proj->stream_forward(lq.data(), 1, g.height, g.width, &rows,
                                     &rf, &err));
    EXPECT_TRUE(rows.empty());
    for (int i = 0; i < g.latents; ++i) {
      ASSERT_TRUE(proj->stream_forward(lq.data() + (1 + 4 * i) * frame, 4,
                                       g.height, g.width, &rows, &rf, &err));
    }
    ASSERT_TRUE((int)rows.size() == g.latents);
    if ((int)rows.size() != g.latents) { continue; }
    double worst = 0.0;
    const std::size_t n = (std::size_t)(g.height / 16) * (g.width / 16) *
                          (std::size_t)proj->config().out_dim;
    for (int i = 0; i < g.latents; ++i) {
      const auto want =
          read_f32_(g.dir / ("lq_proj" + std::to_string(i) + ".f32"));
      ASSERT_TRUE(want.size() == n);
      if (want.size() != n) { break; }
      std::vector<float> got(n);
      const auto* p = (const std::uint16_t*)rows[(std::size_t)i].contents();
      for (std::size_t k = 0; k < n; ++k) { got[k] = from_bf16_(p[k]); }
      worst = std::max(worst, rel_l2_(got, want));
    }
    std::printf("  %s: %d row sets, worst rel-L2 %.5f\n", g.mode.c_str(),
                g.latents, worst);
    // The reference's own bf16 run is 0.003 from its f32 one.
    EXPECT_TRUE(worst < 0.01);
  }
}

// The opening chunk's self-attention at block 0 (a full layer) and block
// 1 (Turbo's first hybrid layer, Base's second full one) -- where the
// release's top-k, its grouping and Turbo's band all meet.
TEST(flashvsr_plus, self_attention_matches_reference)
{
  const auto gs = goldens_();
  if (gs.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string root = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  for (const Golden& g : gs) {
    auto dit = load_dit_(mc, root + "/" + g.mode);
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { continue; }
    const int D = dit->config().hidden;
    const int h8 = g.height / 8, w8 = g.width / 8, f = 4;
    const std::size_t plane = (std::size_t)h8 * w8;
    // Block 0's input: the opening chunk's noise, patchified, plus its
    // source rows.
    const auto noise = read_f32_(g.dir / "noise.f32");
    std::vector<float> slice((std::size_t)16 * f * plane);
    for (int c = 0; c < 16; ++c) {
      std::memcpy(slice.data() + (std::size_t)c * f * plane,
                  noise.data() + (std::size_t)c * g.latents * plane,
                  (std::size_t)f * plane * sizeof(float));
    }
    std::vector<float> x;
    std::string err;
    ASSERT_TRUE(dit->patchify_tokens(slice.data(), f, h8, w8, &x, &err));
    const std::size_t row_frame = (std::size_t)(h8 / 2) * (w8 / 2) * D;
    for (int i = 0; i < f; ++i) {
      const auto r =
          read_f32_(g.dir / ("lq_proj" + std::to_string(i) + ".f32"));
      ASSERT_TRUE(r.size() == row_frame);
      if (r.size() != row_frame) { return; }
      for (std::size_t k = 0; k < row_frame; ++k) {
        x[(std::size_t)i * row_frame + k] += r[k];
      }
    }
    const T::Params prm = T::Params::defaults(dit->config().schedule);
    std::vector<float> sa;
    ASSERT_TRUE(dit->block_self_attention(0, x.data(), f, h8 / 2, w8 / 2, 0,
                                          prm, &sa, &err));
    const double r0 = rel_l2_(sa, read_f32_(g.dir / "chunk0_b0_selfattn.f32"));
    // Block 1 reads block 0's output, the golden's own.
    const auto x1 = read_f32_(g.dir / "chunk0_block0.f32");
    ASSERT_TRUE(dit->block_self_attention(1, x1.data(), f, h8 / 2, w8 / 2, 0,
                                          prm, &sa, &err));
    const double r1 = rel_l2_(sa, read_f32_(g.dir / "chunk0_b1_selfattn.f32"));
    std::printf("  %s: block 0 self-attn rel-L2 %.5f, block 1 %.5f%s\n",
                g.mode.c_str(), r0, r1,
                g.mode == "turbo" ? " (hybrid)" : "");
    // The reference's own bf16 run: 0.010 / 0.016 (Turbo), 0.009 / 0.028.
    EXPECT_TRUE(r0 < 0.05 && r1 < 0.05);
  }
}

// Four chunks through the whole stack -- the ring fills and its pair
// wraps -- against the reference's latent, chunk by chunk.
TEST(flashvsr_plus, latent_matches_reference)
{
  const auto gs = goldens_();
  if (gs.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string root = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  for (const Golden& g : gs) {
    auto dit = load_dit_(mc, root + "/" + g.mode);
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { continue; }
    std::vector<float> got;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = run_golden_(*dit, g, &got, &err);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok);
    if (!ok) { std::printf("  %s\n", err.c_str()); continue; }
    const auto want = read_f32_(g.dir / "latent.f32");
    ASSERT_TRUE(got.size() == want.size());
    if (got.size() != want.size()) { continue; }
    const double r = rel_l2_(got, want);
    // Per chunk, so a failure says where the stream went wrong.
    const std::size_t plane = (std::size_t)(g.height / 8) * (g.width / 8);
    std::string per;
    for (int c = 0; c < g.chunks; ++c) {
      double num = 0.0, den = 0.0;
      for (int ch = 0; ch < 16; ++ch) {
        const std::size_t o =
            ((std::size_t)ch * g.latents + 4 * c) * plane;
        for (std::size_t i = 0; i < 4 * plane; ++i) {
          const double d = (double)got[o + i] - want[o + i];
          num += d * d;
          den += (double)want[o + i] * want[o + i];
        }
      }
      per += fmt(" {:.4f}", std::sqrt(num / den))();
    }
    std::printf("  %s %dx%d, %d chunks: latent rel-L2 %.5f (per chunk%s) "
                "in %.0f ms; the reference's own bf16 %.5f; %llu empty "
                "query tiles; %s\n", g.mode.c_str(), g.width, g.height,
                g.chunks, r, per.c_str(), ms, g.ref_bf16,
                (unsigned long long)dit->empty_query_tiles(),
                dit->accel_summary().c_str());
    // The bar is the reference's own bf16 error where it was dumped:
    // vpipe's bf16 accumulation is a different order of the same work.
    EXPECT_TRUE(r < (g.ref_bf16 > 0.0 ? 1.5 * g.ref_bf16 : 0.05));
    // FlashVSR+ gives an empty row its first candidate: none is left.
    EXPECT_TRUE(dit->empty_query_tiles() == 0);
  }
}

namespace {

// Block `bi`'s input over the golden's opening chunk: for block 0 the
// chunk's noise, patchified, plus its source rows; for block 1 block 0's
// output, the golden's own.
std::vector<float>
block_input_(T& dit, const Golden& g, int bi)
{
  if (bi == 1) { return read_f32_(g.dir / "chunk0_block0.f32"); }
  const int D = dit.config().hidden;
  const int h8 = g.height / 8, w8 = g.width / 8, f = 4;
  const std::size_t plane = (std::size_t)h8 * w8;
  const auto noise = read_f32_(g.dir / "noise.f32");
  std::vector<float> slice((std::size_t)16 * f * plane);
  for (int c = 0; c < 16; ++c) {
    std::memcpy(slice.data() + (std::size_t)c * f * plane,
                noise.data() + (std::size_t)c * g.latents * plane,
                (std::size_t)f * plane * sizeof(float));
  }
  std::vector<float> x;
  std::string err;
  if (!dit.patchify_tokens(slice.data(), f, h8, w8, &x, &err)) { return {}; }
  const std::size_t row_frame = (std::size_t)(h8 / 2) * (w8 / 2) * D;
  for (int i = 0; i < f; ++i) {
    const auto r = read_f32_(g.dir / ("lq_proj" + std::to_string(i) + ".f32"));
    if (r.size() != row_frame) { return {}; }
    for (std::size_t k = 0; k < row_frame; ++k) {
      x[(std::size_t)i * row_frame + k] += r[k];
    }
  }
  return x;
}

// The fraction of block-mask entries the production routing of block
// `bi` decides differently from the reference, at the opening chunk.
double
mask_mismatch_(T& dit, const Golden& g, int bi, double* kept = nullptr)
{
  const auto x = block_input_(dit, g, bi);
  const auto want =
      read_f32_(g.dir / ("chunk0_mask_b" + std::to_string(bi) + ".f32"));
  std::vector<std::uint8_t> got;
  std::string err;
  if (x.empty() || want.empty() ||
      !dit.route_blocks(bi, x.data(), 4, g.height / 16, g.width / 16, 0,
                        T::Params::defaults(dit.config().schedule), &got,
                        &err) ||
      got.size() != want.size()) {
    std::printf("  routing failed: %s\n", err.c_str());
    return 1.0;
  }
  std::size_t diff = 0, on = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    diff += (got[i] != 0) != (want[i] != 0.0f);
    on += got[i] != 0;
  }
  if (kept != nullptr) { *kept = (double)on / (double)got.size(); }
  return (double)diff / (double)got.size();
}

}  // namespace

// The routing DECISION, block for block against the reference's own
// masks at the opening chunk: a full layer, and Turbo's first hybrid one.
// A latent or an attention output hides a wrong near-threshold rule
// behind bf16 noise; a mask entry does not.
TEST(flashvsr_plus, routing_matches_reference_masks)
{
  const auto gs = goldens_();
  if (gs.empty()) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string root = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  for (const Golden& g : gs) {
    auto dit = load_dit_(mc, root + "/" + g.mode);
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { continue; }
    for (int bi : {0, 1}) {
      double kept = 0.0;
      const double mm = mask_mismatch_(*dit, g, bi, &kept);
      std::printf("  %s block %d: %.3f%% of block pairs decided otherwise "
                  "(%.1f%% kept)\n", g.mode.c_str(), bi, 100.0 * mm,
                  100.0 * kept);
      // A near-threshold block may flip on bf16 block means.
      EXPECT_TRUE(mm < 0.01);
    }
  }
}

// EVERY ROW OF THE SCHEDULE IS LOAD-BEARING. Break one at a time: a
// routing rule must change the decision well past the published run's
// own disagreement with the reference, and a cache rule must leave the
// published run the closer to the reference latent, by a margin.
// (v1.1's one-window trim is not an arm: with two-window chunks the ring
// refuses it outright, which is the check.)
TEST(flashvsr_plus, the_schedule_is_load_bearing)
{
  const auto gs = goldens_();
  const Golden* g = nullptr;
  for (const Golden& one : gs) {
    if (one.mode == "turbo") { g = &one; }
  }
  if (g == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string dir =
      std::string(std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH")) +
      "/turbo";

  // ---- the routing, on block 0 (full) and block 1 (hybrid).
  struct Arm {
    const char* name;
    int block;
    void (*mutate)(T::Schedule&);
  };
  const Arm route_arms[] = {
      {"top-k per window, not per chunk", 0,
       [](T::Schedule& s) { s.group_windows = 1; }},
      {"top-k scaled by the frame area", 0,
       [](T::Schedule& s) { s.area_topk = true; }},
      {"hybrid layer on the full band", 1,
       [](T::Schedule& s) { s.sliding_local_range = 11; }},
      {"no hybrid layers (Base's layout)", 1,
       [](T::Schedule& s) { s.sliding_layers.clear(); }},
  };
  double published[2] = {0.0, 0.0};
  {
    auto dit = load_dit_(mc, dir);
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { return; }
    for (int bi : {0, 1}) { published[bi] = mask_mismatch_(*dit, *g, bi); }
  }
  for (const Arm& a : route_arms) {
    T::Schedule s = T::Schedule::plus(true);
    a.mutate(s);
    auto dit = load_dit_(mc, dir, &s);
    ASSERT_TRUE(dit != nullptr);
    if (dit == nullptr) { return; }
    const double mm = mask_mismatch_(*dit, *g, a.block);
    std::printf("  %-36s block %d: %.2f%% of pairs off (published "
                "%.2f%%)\n", a.name, a.block, 100.0 * mm,
                100.0 * published[a.block]);
    EXPECT_TRUE(mm > 0.05 && mm > 10.0 * published[a.block]);
  }

  // ---- the cache, on the latent: the hybrid layers' single window.
  const auto want = read_f32_(g->dir / "latent.f32");
  std::vector<float> pub, mut;
  std::string err;
  {
    auto dit = load_dit_(mc, dir);
    ASSERT_TRUE(dit != nullptr && run_golden_(*dit, *g, &pub, &err));
  }
  {
    T::Schedule s = T::Schedule::plus(true);
    s.sliding_keep = 3;
    auto dit = load_dit_(mc, dir, &s);
    ASSERT_TRUE(dit != nullptr && run_golden_(*dit, *g, &mut, &err));
  }
  const double r_pub = rel_l2_(pub, want), r_mut = rel_l2_(mut, want);
  const double moved = rel_l2_(mut, pub);
  std::printf("  hybrid layers keep three windows: latent rel-L2 %.5f "
              "against the published %.5f (moved %.5f)\n", r_mut, r_pub,
              moved);
  EXPECT_TRUE(r_mut > r_pub && moved > 0.01);
}

// A stream cut into requests is the stream: the per-layer windows -- a
// ring of five for a full layer, three for a hybrid one, each written
// past its end -- carried between requests, in both orders.
TEST(flashvsr_plus, stream_requests_equal_one_request)
{
  const char* m = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  if (m == nullptr || *m == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto dit = load_dit_(mc, std::string(m) + "/turbo");
  ASSERT_TRUE(dit != nullptr);
  if (dit == nullptr) { return; }
  const T::Schedule& sch = dit->config().schedule;

  constexpr int kH = 256, kW = 384;
  constexpr int kChunks = 5;
  const int Tn = sch.latents(kChunks);
  const std::size_t row_frame =
      (std::size_t)(kH / 16) * (kW / 16) * dit->config().hidden;
  std::mt19937_64 rng(0x9105u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<std::uint16_t> rows((std::size_t)Tn * row_frame);
  for (auto& v : rows) { v = to_bf16_(0.5f * nd(rng)); }

  struct Piece { int chunk0, chunks; };
  const std::vector<std::vector<Piece>> cuttings = {
      {{0, 1}, {1, 2}, {3, 2}},
      {{0, 2}, {2, 1}, {3, 1}, {4, 1}}};
  for (const bool chunk_outer : {false, true}) {
    if (chunk_outer) {
      ::setenv("VPIPE_FVSR_CHUNK_OUTER", "1", 1);
    } else {
      ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
    }
    T::Request one;
    one.rows = rows.data();
    one.row_frames = Tn;
    one.frames = 1 + 4 * Tn;
    one.height = kH;
    one.width = kW;
    one.seed = 4321;
    one.params = T::Params::defaults(sch);
    std::vector<float> whole;
    std::vector<int> shape;
    std::string err;
    ASSERT_TRUE(dit->generate(one, &whole, &shape, &err));
    for (const auto& pieces : cuttings) {
      std::vector<float> joined(whole.size(), 0.0f);
      bool ok = true;
      int at = 0;
      for (std::size_t i = 0; i < pieces.size() && ok; ++i) {
        const Piece& p = pieces[i];
        const int pt = p.chunk0 == 0 ? sch.latents(p.chunks)
                                     : sch.chunk_latents * p.chunks;
        T::Request r = one;
        r.rows = rows.data() + (std::size_t)at * row_frame;
        r.row_frames = pt;
        r.chunk0 = p.chunk0;
        r.chunks = p.chunks;
        r.keep_window = i + 1 < pieces.size();
        std::vector<float> lat;
        ok = dit->generate(r, &lat, &shape, &err);
        ASSERT_TRUE(ok);
        if (!ok) { std::printf("  %s\n", err.c_str()); break; }
        const std::size_t plane = (std::size_t)shape[2] * shape[3];
        for (int c = 0; c < shape[0]; ++c) {
          std::memcpy(joined.data() + ((std::size_t)c * Tn + at) * plane,
                      lat.data() + (std::size_t)c * pt * plane,
                      (std::size_t)pt * plane * sizeof(float));
        }
        at += pt;
      }
      if (!ok) { continue; }
      const bool same = std::memcmp(joined.data(), whole.data(),
                                    whole.size() * sizeof(float)) == 0;
      EXPECT_TRUE(same);
      std::printf("  %s %zu requests vs 1: %s (rel-L2 %.2e)\n",
                  chunk_outer ? "chunk-outer" : "block-outer", pieces.size(),
                  same ? "IDENTICAL" : "DIFFERENT", rel_l2_(joined, whole));
    }
  }
  ::unsetenv("VPIPE_FVSR_CHUNK_OUTER");
}
