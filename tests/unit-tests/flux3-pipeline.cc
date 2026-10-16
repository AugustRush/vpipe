// FLUX.3 Image Klein through the stages: the diffusion-conditioner's
// "flux3" family against BFL's TextEncoder.encode, FLUX.2's VAE as FLUX.3
// ships it, and generate-image end to end against the reference sampler.
//
// What is silent when wrong here:
//   * the template -- a bare user turn plus the generation prompt, no
//     system message and no <think> block;
//   * the PAD: right-padded with <|endoftext|> to a multiple of 80, and
//     the DiT reads the pad rows too -- each at POSITION 1, attending to
//     the real rows only;
//   * the eight taps, hidden_states[4, 8, .., 32], concatenated PER TOKEN
//     into [rows, 8 x 2560].
//
// Env: VPIPE_FLUX3_TEST_MODEL_PATH (the repo root) and
// VPIPE_FLUX3_GOLDEN (tools/dump_flux3_golden.py --stage tokenizer and
// --stage text). Skips vacuously without them -- check the printed
// lines.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/shared/torch-randn.h"
#include "generative-models/tokenizer.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/image-resample-stage.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

namespace fs = std::filesystem;

std::string
env_(const char* k)
{
  const char* g = std::getenv(k);
  return (g == nullptr || *g == '\0') ? std::string() : std::string(g);
}

bool
read_json_(const fs::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  try {
    *out = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  return out->is_object();
}

std::vector<float>
read_f32_(const fs::path& p)
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

double
rel_l2_(const float* got, const float* want, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = (double)got[i] - (double)want[i];
    num += e * e;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

class SourceText : public TypedStage<SourceText> {
public:
  static constexpr const char* kTypeName = "ut-source-text-flux3";
  SourceText(const SessionContextIntf* s, std::string id,
             std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::string> prompts;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next < prompts.size()) {
      co_await ctx.write(0, std::make_unique<FlexDataPayload>(
                                FlexData::make_string(prompts[next++])));
      co_return;
    }
    ctx.signal_done();
    co_return;
  }
};

// Emits one U8 [3, H, W] picture, then ends.
class SourcePicture : public TypedStage<SourcePicture> {
public:
  static constexpr const char* kTypeName = "ut-source-picture-flux3";
  SourcePicture(const SessionContextIntf* s, std::string id,
                std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::vector<std::uint8_t> chw;
  int H = 0, W = 0;
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = TensorBeat::DType::U8;
      tb->shape = {3, H, W};
      tb->resize_contiguous(chw.size());
      std::memcpy(tb->as_u8(), chw.data(), chw.size());
      co_await ctx.write(0, std::move(tb));
    }
    ctx.signal_done();
    co_return;
  }
};

class SinkCapture : public TypedStage<SinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-sink-capture-flux3";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> captured;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

}  // namespace

// The template's ids, against the reference processor's for every
// prompt the dumper covers (ASCII, CJK, emoji and NFD-sensitive text,
// the bounding-box JSON the model is prompted with, and the empty
// prompt). Splitting the template at its special tokens and encoding
// the runs between them is what the reference tokenizer does too.
TEST(flux3_cond, template_ids_match_the_reference)
{
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (root.empty() || g.empty()) { return; }
  FlexData idx;
  if (!read_json_(fs::path(g) / "tokenizer.json", &idx)) { return; }
  Session sess;
  auto tok = genai::Tokenizer::from_huggingface_json(
      (fs::path(root) / "text_encoder" / "tokenizer.json").string(), &sess);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  const std::int32_t im_start = tok->special_token_id("<|im_start|>");
  const std::int32_t im_end = tok->special_token_id("<|im_end|>");
  ASSERT_TRUE(im_start >= 0 && im_end >= 0);
  const FlexData prompts = idx.as_object().at("prompts");
  int checked = 0, exact = 0;
  for (const auto& kv : prompts.as_object()) {
    const std::string key(kv.first);
    FlexData one;
    if (!read_json_(fs::path(g) / "tokenizer" / (key + ".json"), &one)) {
      continue;
    }
    auto o = one.as_object();
    const std::string prompt(o.at("prompt").as_string(""));
    const int real = (int)o.at("real").as_int(0);
    const FlexData want_ids = o.at("ids");
    auto wa = want_ids.as_array();
    std::vector<std::int32_t> ids;
    ids.push_back(im_start);
    for (std::int32_t t : tok->encode("user\n" + prompt)) { ids.push_back(t); }
    ids.push_back(im_end);
    for (std::int32_t t : tok->encode("\n")) { ids.push_back(t); }
    ids.push_back(im_start);
    for (std::int32_t t : tok->encode("assistant\n")) { ids.push_back(t); }
    bool same = (int)ids.size() == real;
    for (int i = 0; same && i < real; ++i) {
      same = ids[(std::size_t)i] ==
             (std::int32_t)wa.at((std::size_t)i).as_int(-1);
    }
    ++checked;
    if (same) { ++exact; }
    else {
      std::printf("[flux3_cond] '%s': %zu ids, the reference %d\n",
                  key.c_str(), ids.size(), real);
    }
  }
  std::printf("[flux3_cond] template ids id-exact for %d / %d prompts\n",
              exact, checked);
  EXPECT_TRUE(checked > 0 && exact == checked);
}

// The stage, end to end: prompt in, [rows, 20480] bf16 out, against the
// reference's f32 conditioning -- the real rows and the PAD rows held to
// their own bars, since a pad row wrongly positioned or masked is a
// different vector entirely while the real rows stay right.
TEST(flux3_cond, conditioning_matches_the_reference)
{
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (root.empty() || g.empty()) { return; }
  FlexData man;
  if (!read_json_(fs::path(g) / "text.json", &man)) { return; }
  const FlexData prompts = man.as_object().at("prompts");
  std::vector<std::string> keys;
  for (const auto& kv : prompts.as_object()) {
    keys.emplace_back(kv.first);
  }
  // The reference's own prompt strings, from the tokenizer goldens.
  std::vector<std::string> texts;
  for (const std::string& k : keys) {
    FlexData one;
    if (!read_json_(fs::path(g) / "tokenizer" / (k + ".json"), &one)) {
      return;
    }
    texts.emplace_back(one.as_object().at("prompt").as_string(""));
  }

  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto pl = std::make_unique<Pipeline>("flux3-cond", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = texts;
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(c)));
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{cond, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  ASSERT_TRUE(sink->captured.size() == keys.size());
  if (sink->captured.size() != keys.size()) { return; }

  double pad_sum = 0.0, pad_bar = 0.0;
  int pads = 0;
  for (std::size_t i = 0; i < keys.size(); ++i) {
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sink->captured[i].get());
    ASSERT_TRUE(tb != nullptr);
    if (tb == nullptr) { continue; }
    // Bound to a local first: as_object() is a VIEW, and a view of the
    // temporary at() returns dangles.
    const FlexData pe = prompts.as_object().at(keys[i]);
    auto p = pe.as_object();
    const int real = (int)p.at("real").as_int(0);
    const int padded = (int)p.at("padded").as_int(0);
    EXPECT_TRUE(tb->dtype == TensorBeat::DType::Bf16);
    EXPECT_TRUE(tb->shape.size() == 2 && tb->shape[0] == padded &&
                tb->shape[1] == 20480);
    if (tb->shape.size() != 2 || tb->shape[0] != padded) { continue; }
    const std::size_t W = 20480;
    std::vector<float> got((std::size_t)padded * W);
    const auto* u = reinterpret_cast<const std::uint16_t*>(
        (const void*)tb->as_u8());
    for (std::size_t k = 0; k < got.size(); ++k) {
      const std::uint32_t b = (std::uint32_t)u[k] << 16;
      std::memcpy(&got[k], &b, 4);
    }
    const std::vector<float> want =
        read_f32_(fs::path(g) / ("text_" + keys[i] + "_f32.f32"));
    ASSERT_TRUE(want.size() == got.size());
    if (want.size() != got.size()) { continue; }
    const double r_real =
        rel_l2_(got.data(), want.data(), (std::size_t)real * W);
    const double b_real = p.at("bf16_vs_f32_real_rows").as_real(0.0);
    const double r_pad =
        padded > real
            ? rel_l2_(got.data() + (std::size_t)real * W,
                      want.data() + (std::size_t)real * W,
                      (std::size_t)(padded - real) * W)
            : 0.0;
    const double b_pad = padded > real
                             ? p.at("bf16_vs_f32_pad_rows").as_real(0.0)
                             : 0.0;
    std::printf("[flux3_cond] '%s' (%d real + %d pad rows): real rows "
                "%.5f (reference bf16 %.5f), pad rows %.5f (%.5f)\n",
                keys[i].c_str(), real, padded - real, r_real, b_real, r_pad,
                b_pad);
    EXPECT_TRUE(r_real <= b_real * 1.1);
    if (padded > real) {
      pad_sum += r_pad;
      pad_bar += b_pad;
      ++pads;
    }
  }
  // THE PAD ROWS ARE ONE VECTOR, repeated -- every pad row is the same
  // token at the same position attending the same keys -- so each
  // prompt's pad error is a single sample, and the reference's own
  // scatters 0.012-0.042 across these three. Held as an AGGREGATE, and a
  // loose one: vpipe's per-prompt ratio to that bar MEASURED 0.58 / 3.07
  // / 0.70 on the M4 Pro and 0.47 / 1.74 / 1.50 on the M5 (aggregates
  // 1.00x and 1.16x), the M5's real rows sitting 3x BELOW the reference's
  // at the same time. A wrong position or mask is not a few percent: it
  // is a different vector, an order of magnitude past 1.5x.
  if (pads > 0) {
    std::printf("[flux3_cond] pad rows over %d prompts: mean %.5f (the "
                "reference's own bf16 %.5f)\n", pads, pad_sum / pads,
                pad_bar / pads);
    EXPECT_TRUE(pad_sum <= pad_bar * 1.5);
  }
}

// ---- the sampler's noise and FLUX.2's VAE ------------------------------

namespace {

std::uint16_t
f2bf_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf2f_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// [h*w, C] token-major -> [C, h, w].
std::vector<float>
to_chw_(const std::vector<float>& tok, int C, int h, int w)
{
  std::vector<float> out((std::size_t)C * h * w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < C; ++c) {
        out[((std::size_t)c * h + y) * w + x] =
            tok[((std::size_t)y * w + x) * C + c];
      }
    }
  }
  return out;
}

}  // namespace

// The seed IS the reference's: torch.randn((h*w, 128)) from a CPU
// generator, token-major, rounded to bf16.
TEST(flux3_stage, the_noise_is_the_references)
{
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (g.empty()) { return; }
  FlexData man;
  if (!read_json_(fs::path(g) / "e2e.json", &man)) { return; }
  const int seed = (int)man.as_object().at("seed").as_int(0);
  const std::vector<float> want = read_f32_(fs::path(g) / "e2e_noise.f32");
  ASSERT_TRUE(!want.empty());
  genai::TorchCpuRandn rng((std::uint64_t)seed);
  const std::vector<float> z = rng.randn(want.size());
  ASSERT_TRUE(z.size() == want.size());
  std::size_t same = 0;
  for (std::size_t i = 0; i < z.size(); ++i) {
    if (bf2f_(f2bf_(z[i])) == want[i]) { ++same; }
  }
  std::printf("[flux3_stage] noise for seed %d: %zu / %zu values "
              "bit-identical to the reference's\n", seed, same, z.size());
  EXPECT_TRUE(same == z.size());
}

// FLUX.2's decoder over FLUX.3's F32 copy of the same VAE, against the
// reference's f32 decode of its own final latent.
TEST(flux3_stage, vae_decode_matches_reference)
{
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (root.empty() || g.empty()) { return; }
  FlexData man;
  if (!read_json_(fs::path(g) / "vae_e2e.json", &man)) { return; }
  const FlexData la = man.as_object().at("latent");
  const int h = (int)la.as_array().at(0).as_int(0);
  const int w = (int)la.as_array().at(1).as_int(0);
  const std::vector<float> tok = read_f32_(fs::path(g) / "e2e_out_f32.f32");
  const std::vector<float> want =
      read_f32_(fs::path(g) / "vae_e2e_image_f32.f32");
  ASSERT_TRUE(tok.size() == (std::size_t)h * w * 128 && !want.empty());
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  genai::MetalFlux2Vae::Config cfg;
  auto vae = genai::MetalFlux2Vae::load(
      (fs::path(root) / "vae").string(), mc, cfg);
  ASSERT_TRUE(vae != nullptr);
  if (vae == nullptr) { return; }
  const std::vector<float> z = to_chw_(tok, 128, h, w);
  metal_compute::SharedBuffer zb = mc->make_shared_buffer(z.size() * 2);
  auto* zp = static_cast<_Float16*>(zb.contents());
  for (std::size_t i = 0; i < z.size(); ++i) { zp[i] = (_Float16)z[i]; }
  std::string err;
  metal_compute::SharedBuffer img = vae->decode(zb, h, w, &err);
  ASSERT_TRUE(!img.empty());
  if (img.empty()) { return; }
  std::vector<float> got(want.size());
  const auto* ip = static_cast<const _Float16*>(img.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)ip[i]; }
  const double rel = rel_l2_(got.data(), want.data(), got.size());
  std::printf("[flux3_stage] VAE decode %dx%d: rel-L2 vs the reference's "
              "f32 decode = %.5f\n", w * 16, h * 16, rel);
  EXPECT_TRUE(rel < 0.01);
}

TEST(flux3_stage, vae_encode_matches_reference)
{
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (root.empty() || g.empty()) { return; }
  FlexData man;
  if (!read_json_(fs::path(g) / "e2e_edit.json", &man)) { return; }
  const FlexData rs = man.as_object().at("ref_size");
  const int W = (int)rs.as_array().at(0).as_int(0);
  const int H = (int)rs.as_array().at(1).as_int(0);
  std::vector<std::uint8_t> hwc((std::size_t)H * W * 3);
  {
    std::ifstream in(fs::path(g) / "e2e_edit_ref_rgb.u8", std::ios::binary);
    if (!in) { return; }
    in.read(reinterpret_cast<char*>(hwc.data()), (std::streamsize)hwc.size());
  }
  const std::vector<float> tok =
      read_f32_(fs::path(g) / "e2e_edit_ref_tokens.f32");
  ASSERT_TRUE(tok.size() == (std::size_t)(H / 16) * (W / 16) * 128);
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  genai::MetalFlux2Vae::Config cfg;
  auto vae = genai::MetalFlux2Vae::load(
      (fs::path(root) / "vae").string(), mc, cfg);
  // The encoder half loads on demand: a decode-only graph never reads it.
  ASSERT_TRUE(vae != nullptr && vae->ensure_encoder());
  if (vae == nullptr || !vae->has_encoder()) { return; }
  metal_compute::SharedBuffer ib =
      mc->make_shared_buffer((std::size_t)3 * H * W * 2);
  auto* ipx = static_cast<_Float16*>(ib.contents());
  // to_tensor: x / 127.5 - 1, channel-first.
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        ipx[((std::size_t)c * H + y) * W + x] = (_Float16)(
            (float)hwc[((std::size_t)y * W + x) * 3 + c] / 127.5f - 1.0f);
      }
    }
  }
  metal_compute::SharedBuffer lat = vae->encode(ib, H, W);
  ASSERT_TRUE(!lat.empty());
  if (lat.empty()) { return; }
  const std::vector<float> want = to_chw_(tok, 128, H / 16, W / 16);
  std::vector<float> got(want.size());
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)lp[i]; }
  const double rel = rel_l2_(got.data(), want.data(), got.size());
  std::printf("[flux3_stage] VAE encode %dx%d: rel-L2 vs the reference's "
              "f32 encode = %.5f\n", W, H, rel);
  EXPECT_TRUE(rel < 0.01);
}

// ---- generate-image, end to end ------------------------------------------

namespace {

// prompt -> diffusion-conditioner -> generate-image (-> the latent), with
// an optional reference picture through vae-encode on ref_latent0. The
// stage draws its own noise and schedule, so this is the plumbing's test;
// the forward's own accuracy is flux3_dit's.
std::vector<float>
run_stages_(Session& sess, const std::string& root, const std::string& prompt,
            int W, int H, int seed, const std::vector<std::uint8_t>* ref_chw,
            int rW, int rH)
{
  auto pl = std::make_unique<Pipeline>("flux3-stages", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = {prompt};
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));
  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(cc)));
  std::vector<InEdge> edges{{cond, 0}};
  if (ref_chw != nullptr) {
    auto pu = std::make_unique<SourcePicture>(&sess, "pic",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
    pu->chw = *ref_chw;
    pu->H = rH;
    pu->W = rW;
    auto* pic = pl->insert_stage(std::move(pu));
    FlexData vc = FlexData::make_object();
    vc.as_object().insert("hf_dir", FlexData::make_string(root));
    auto* venc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
        &sess, "venc", std::vector<InEdge>{{pic, 0}}, std::move(vc)));
    edges = {{cond, 0}, InEdge{nullptr, 0}, InEdge{nullptr, 0},
             InEdge{nullptr, 0}, InEdge{nullptr, 0}, {venc, 0}};
  }
  FlexData gc = FlexData::make_object();
  gc.as_object().insert("hf_dir", FlexData::make_string(root));
  gc.as_object().insert("width", FlexData::make_int(W));
  gc.as_object().insert("height", FlexData::make_int(H));
  gc.as_object().insert("seed", FlexData::make_int(seed));
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(&sess, "gen", edges,
                                           std::move(gc))));
  if (!gen->config_error().empty()) { return {}; }
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  if (sink->captured.size() != 1) { return {}; }
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  if (tb == nullptr || tb->dtype != TensorBeat::DType::F32) { return {}; }
  const float* f = tb->as_f32();
  return std::vector<float>(f, f + tb->element_count());
}

// ran false: skipped (a variable unset). rel < 0: the run failed.
struct StageRun {
  bool ran = false;
  double rel = -1.0, bar = 0.0;
};

StageRun
stage_e2e_(const char* tag, bool edit)
{
  StageRun out;
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  const std::string g = env_("VPIPE_FLUX3_GOLDEN");
  if (root.empty() || g.empty() || env_("VPIPE_FLUX3_E2E").empty()) {
    return out;
  }
  FlexData man;
  if (!read_json_(fs::path(g) / (std::string(tag) + ".json"), &man)) {
    return out;
  }
  auto o = man.as_object();
  const FlexData sz = o.at("size");
  const int W = (int)sz.as_array().at(0).as_int(0);
  const int H = (int)sz.as_array().at(1).as_int(0);
  const int seed = (int)o.at("seed").as_int(0);
  FlexData one;
  if (!read_json_(fs::path(g) / "tokenizer" /
                      (std::string(o.at("prompt").as_string("")) + ".json"),
                  &one)) {
    return out;
  }
  const std::string prompt(one.as_object().at("prompt").as_string(""));
  std::vector<std::uint8_t> chw;
  int rW = 0, rH = 0;
  if (edit) {
    const FlexData rs = o.at("ref_size");
    rW = (int)rs.as_array().at(0).as_int(0);
    rH = (int)rs.as_array().at(1).as_int(0);
    std::vector<std::uint8_t> hwc((std::size_t)rH * rW * 3);
    std::ifstream in(fs::path(g) / (std::string(tag) + "_ref_rgb.u8"),
                     std::ios::binary);
    if (!in) { return out; }
    in.read(reinterpret_cast<char*>(hwc.data()), (std::streamsize)hwc.size());
    chw.resize(hwc.size());
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < rH; ++y) {
        for (int x = 0; x < rW; ++x) {
          chw[((std::size_t)c * rH + y) * rW + x] =
              hwc[((std::size_t)y * rW + x) * 3 + c];
        }
      }
    }
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return out; }
  out.ran = true;
  const std::vector<float> got =
      run_stages_(sess, root, prompt, W, H, seed, edit ? &chw : nullptr, rW,
                  rH);
  if (got.size() != (std::size_t)128 * (H / 16) * (W / 16)) { return out; }
  const std::vector<float> want = to_chw_(
      read_f32_(fs::path(g) / (std::string(tag) + "_out_f32.f32")), 128,
      H / 16, W / 16);
  const std::vector<float> ref16 = to_chw_(
      read_f32_(fs::path(g) / (std::string(tag) + "_out_bf16.f32")), 128,
      H / 16, W / 16);
  const double rel = rel_l2_(got.data(), want.data(), got.size());
  const double bar = rel_l2_(ref16.data(), want.data(), want.size());
  std::printf("[flux3_stage] %s through the stages (%dx%d, seed %d%s): "
              "latent rel-L2 vs the f32 reference = %.5f (the reference's "
              "own bf16 %.5f)\n", tag, W, H, seed, edit ? ", 1 reference" : "",
              rel, bar);
  out.rel = rel;
  out.bar = bar;
  return out;
}

}  // namespace

// Opt-in (VPIPE_FLUX3_E2E): loads the encoder and the DiT through the
// stages.
//
// The stages add their OWN text encoding and VAE encode to the DiT's
// error, where the reference's bf16 run takes its f32 sibling's
// conditioning and reference tokens. So the bound is the same ORDER as
// the bar, not under it: a plumbing fault (noise, schedule, packing,
// layout) is a different latent, at ~1.
TEST(flux3_stage, text_to_image_matches_reference)
{
  const StageRun r = stage_e2e_("e2e", false);
  if (!r.ran) { return; }
  EXPECT_TRUE(r.rel >= 0.0 && r.rel < 2.0 * r.bar);
}

TEST(flux3_stage, edit_matches_reference)
{
  const StageRun r = stage_e2e_("e2e_edit", true);
  if (!r.ran) { return; }
  EXPECT_TRUE(r.rel >= 0.0 && r.rel < 2.0 * r.bar);
}

// SECOND GENERATION IS THE TEST. A one-shot run never reaches the
// release/reload regime -- on a box where the decode does not fit beside
// the DiT, generate-image frees it after every latent and reloads it for
// the next prompt -- so the same prompt runs twice with the seed held
// (`seed_sequence: keep`) and the two latents must be the same bits.
// Opt-in with the other stage runs (VPIPE_FLUX3_E2E).
TEST(flux3_stage, a_second_generation_reproduces_the_first)
{
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  if (root.empty() || env_("VPIPE_FLUX3_E2E").empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const int W = 256, H = 256;
  auto pl = std::make_unique<Pipeline>("flux3-twice", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = {"A red bicycle leaning against a brick wall",
                   "A red bicycle leaning against a brick wall"};
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));
  FlexData cc = FlexData::make_object();
  cc.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, std::move(cc)));
  FlexData gc = FlexData::make_object();
  gc.as_object().insert("hf_dir", FlexData::make_string(root));
  gc.as_object().insert("width", FlexData::make_int(W));
  gc.as_object().insert("height", FlexData::make_int(H));
  gc.as_object().insert("seed", FlexData::make_int(7));
  gc.as_object().insert("seed_sequence", FlexData::make_string("keep"));
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(
          &sess, "gen", std::vector<InEdge>{{cond, 0}}, std::move(gc))));
  ASSERT_TRUE(gen->config_error().empty());
  auto* sink = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "sink",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  ASSERT_TRUE(sink->captured.size() == 2);
  if (sink->captured.size() != 2) { return; }
  const auto* a =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  const auto* b =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[1].get());
  ASSERT_TRUE(a != nullptr && b != nullptr);
  if (a == nullptr || b == nullptr) { return; }
  const std::size_t n = a->element_count();
  ASSERT_TRUE(n == b->element_count() && n > 0);
  const bool same =
      std::memcmp(a->as_f32(), b->as_f32(), n * sizeof(float)) == 0;
  std::printf("[flux3_stage] two generations, seed kept: %s (%zu values)\n",
              same ? "bit-identical" : "DIFFERENT", n);
  EXPECT_TRUE(same);
}

// BFL'S SINGLE-REFERENCE EDIT, through the stages: a 200x150 photo is fitted
// onto a 208x160 canvas (image-resample `pad_to_grid`, mirror-padded),
// encoded, edited at the canvas's size (no width/height: the reference's),
// and decoded -- and the picture that comes out is the photo's 200x150,
// cut from the canvas where the photo sat, (4, 5). The box rides the
// sideband the whole way; this checks every hop carried it. Two steps,
// since the picture is not what is being checked. Opt-in with the other
// stage runs (VPIPE_FLUX3_E2E).
TEST(flux3_stage, a_padded_reference_is_cut_back)
{
  const std::string root = env_("VPIPE_FLUX3_TEST_MODEL_PATH");
  if (root.empty() || env_("VPIPE_FLUX3_E2E").empty()) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  const int rW = 200, rH = 150;
  auto pl = std::make_unique<Pipeline>("flux3-pad-crop", &sess);
  auto srcu = std::make_unique<SourceText>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  srcu->prompts = {"The same street at night, lit by lamps"};
  auto* src = static_cast<SourceText*>(pl->insert_stage(std::move(srcu)));
  auto pu = std::make_unique<SourcePicture>(&sess, "pic",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  pu->H = rH; pu->W = rW;
  pu->chw.resize((std::size_t)3 * rH * rW);
  for (std::size_t i = 0; i < pu->chw.size(); ++i) {
    pu->chw[i] = (std::uint8_t)((i * 37) % 251);
  }
  auto* pic = pl->insert_stage(std::move(pu));
  FlexData rc = FlexData::make_object();
  rc.as_object().insert("fit", FlexData::make_string("pad_to_grid"));
  auto* rs = pl->insert_stage(std::make_unique<ImageResampleStage>(
      &sess, "fit", std::vector<InEdge>{{pic, 0}}, std::move(rc)));
  auto hf = [&] {
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string(root));
    return c;
  };
  auto* cond = pl->insert_stage(std::make_unique<DiffusionConditionerStage>(
      &sess, "cond", std::vector<InEdge>{{src, 0}}, hf()));
  auto* venc = pl->insert_stage(std::make_unique<VaeEncodeStage>(
      &sess, "venc", std::vector<InEdge>{{rs, 0}}, hf()));
  FlexData gc = hf();
  gc.as_object().insert("seed", FlexData::make_int(3));
  gc.as_object().insert("steps", FlexData::make_int(2));
  auto* gen = static_cast<GenerateImageStage*>(pl->insert_stage(
      std::make_unique<GenerateImageStage>(
          &sess, "gen",
          std::vector<InEdge>{{cond, 0}, InEdge{nullptr, 0},
                              InEdge{nullptr, 0}, InEdge{nullptr, 0},
                              InEdge{nullptr, 0}, {venc, 0}},
          std::move(gc))));
  ASSERT_TRUE(gen->config_error().empty());
  auto* lat = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "lat",
                                    std::vector<InEdge>{{gen, 0}},
                                    FlexData::make_object())));
  auto* vdec = pl->insert_stage(std::make_unique<VaeDecodeStage>(
      &sess, "vdec", std::vector<InEdge>{{gen, 0}}, hf()));
  auto* img = static_cast<SinkCapture*>(pl->insert_stage(
      std::make_unique<SinkCapture>(&sess, "img",
                                    std::vector<InEdge>{{vdec, 0}},
                                    FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  ASSERT_TRUE(lat->captured.size() == 1 && img->captured.size() == 1);
  if (lat->captured.size() != 1 || img->captured.size() != 1) { return; }
  const auto* lt =
      dynamic_cast<const TensorBeatPayload*>(lat->captured[0].get());
  const auto* it =
      dynamic_cast<const TensorBeatPayload*>(img->captured[0].get());
  ASSERT_TRUE(lt != nullptr && it != nullptr);
  if (lt == nullptr || it == nullptr) { return; }
  // The latent is the canvas's (208 / 16 x 160 / 16) and carries the box.
  std::vector<std::int64_t> box;
  if (lt->sideband.is_object()) {
    FlexData sb = lt->sideband;              // as_object() is a view
    auto o = sb.as_object();
    if (o.contains(sideband::kCropBox)) {
      FlexData b = o.at(sideband::kCropBox);
      for (const FlexData& e : b.as_array()) { box.push_back(e.as_int(-1)); }
    }
  }
  std::printf("[flux3_stage] padded reference: latent [%lld, %lld, %lld], "
              "box %s, picture [%lld, %lld, %lld]\n",
              (long long)lt->shape[0], (long long)lt->shape[1],
              (long long)lt->shape[2], box.size() == 4 ? "carried" : "MISSING",
              (long long)it->shape[0], (long long)it->shape[1],
              (long long)it->shape[2]);
  EXPECT_TRUE(lt->shape == (std::vector<std::int64_t>{128, 10, 13}));
  EXPECT_TRUE(box == (std::vector<std::int64_t>{4, 5, 204, 155}));
  EXPECT_TRUE(it->shape == (std::vector<std::int64_t>{3, rH, rW}));
}
