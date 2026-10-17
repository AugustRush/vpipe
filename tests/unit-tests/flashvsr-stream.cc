// FlashVSR restoring a long source AS ONE STREAM, through the real
// stages: frames -> temporal-stack -> flashvsr-src-encoder ->
// generate-video -> vae-decode.
//
// The question is the one the user sees: does the picture change where a
// group ends? So the same source is restored twice, once cut into groups
// of 25 with the 4-frame overlap a clip graph needs, once as a single
// group, and the frames must be the SAME BYTES. That holds only if every
// stage carries its state across the cut -- the projection's carries, the
// denoiser's key/value windows, the decode's causal carries -- and if the
// overlap is dropped rather than restored twice. The frame COUNT must be
// the source's: the end is padded and the padding's frames dropped.
//
// And the colour fix (sideband::kColorRef, applied by vae-decode) must
// move every frame's channel means toward its source frame's.
//
// A box too small for a stream restores overlapping CLIPS whose shared
// frames vae-decode cross-fades (sideband::kBlendHead / kBlendTail); that
// is held to the seam it exists to soften, measured as the worst
// frame-to-frame change over the median.
//
// Env: VPIPE_FLASHVSR_TEST_MODEL_PATH, or the test skips.
//
//   vpipe_test --filter 'flashvsr_stream*'

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/flashvsr-tiny-decoder.h"
#include "generative-models/weight-set.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/temporal-stack-stage.h"
#include "stages/flashvsr-src-encoder-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/vae-decode-stage.h"
#include "stages/wavelet-color-fix.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

constexpr int kSize = 256;
constexpr int kFrames = 50;

// A source with structure that MOVES, so a seam has something to show:
// a drifting pattern of soft blobs over a gradient, one frame a beat.
std::vector<std::vector<std::uint8_t>>
source_()
{
  std::vector<std::vector<std::uint8_t>> out;
  const int n = kSize * kSize;
  for (int f = 0; f < kFrames; ++f) {
    std::vector<std::uint8_t> px((std::size_t)3 * n);
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        const double u = (x + 1.7 * f) / 23.0, v = (y + 0.9 * f) / 31.0;
        const double s = 0.5 + 0.25 * std::sin(u) * std::cos(v) +
                         0.2 * std::sin(0.11 * (x + y) + 0.05 * f);
        for (int c = 0; c < 3; ++c) {
          const double g = s * (0.7 + 0.15 * c) + 0.1 * c * y / kSize;
          px[(std::size_t)c * n + (std::size_t)y * kSize + x] =
              (std::uint8_t)std::lround(255.0 * std::clamp(g, 0.0, 1.0));
        }
      }
    }
    out.push_back(std::move(px));
  }
  return out;
}

// Emits the source a frame per beat, with its rate, then ends.
class Frames : public TypedStage<Frames> {
public:
  static constexpr const char* kTypeName = "ut-flashvsr-stream-frames";
  using TypedStage::TypedStage;
  std::vector<std::vector<std::uint8_t>> frames;
  std::size_t next = 0;
  int h = kSize, w = kSize;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= frames.size()) { ctx.signal_done(); co_return; }
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::U8;
    b->shape = {3, h, w};
    b->resize_contiguous(frames[next].size());
    std::memcpy(b->as_u8(), frames[next].data(), frames[next].size());
    b->sideband = FlexData::make_object();
    b->sideband.as_object().insert_or_assign(sideband::kFps,
                                             FlexData::make_real(24.0));
    ++next;
    co_await ctx.write(0, std::move(b));
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec op[] = {{.name = "out", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .oports = op};
    return s;
  }
};

class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-flashvsr-stream-sink";
  using TypedStage::TypedStage;
  std::vector<std::vector<std::uint8_t>> frames;
  // Conditioning beats, when the sink sits after the source encoder:
  // their bf16 rows and their sideband.
  std::vector<std::vector<std::uint16_t>> rows;
  std::vector<FlexData> sidebands;
  // Latents, when the sink taps generate-video: data, shape, sideband.
  std::vector<std::vector<float>> latents;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<FlexData> latent_sb;
  struct Beats {
    std::vector<std::vector<std::uint16_t>> rows;
    std::vector<FlexData> sidebands;
  };
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (tb != nullptr && tb->dtype == TensorBeat::DType::U8) {
      const auto px = tb->materialize_contiguous();
      frames.emplace_back(px.begin(), px.end());
    } else if (tb != nullptr && tb->dtype == TensorBeat::DType::Bf16) {
      const auto px = tb->materialize_contiguous();
      std::vector<std::uint16_t> r(px.size() / 2);
      std::memcpy(r.data(), px.data(), r.size() * 2);
      rows.push_back(std::move(r));
      sidebands.push_back(tb->sideband);
    } else if (tb != nullptr && tb->dtype == TensorBeat::DType::F32) {
      const auto px = tb->materialize_contiguous();
      std::vector<float> v(px.size() / 4);
      std::memcpy(v.data(), px.data(), v.size() * 4);
      latents.push_back(std::move(v));
      shapes.emplace_back(tb->shape.begin(), tb->shape.end());
      latent_sb.push_back(tb->sideband);
    }
  }
  const StageSpec&
  spec() const noexcept override
  {
    static const PortSpec ip[] = {{.name = "in", .doc = "", .type = nullptr}};
    static const StageSpec s = {.type_name = kTypeName, .doc = "",
                                .display_name = "", .iports = ip};
    return s;
  }
};

FlexData
obj_(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData o = FlexData::make_object();
  for (const auto& [k, v] : kv) { o.as_object().insert_or_assign(k, v); }
  return o;
}

struct Run {
  std::vector<std::vector<std::uint8_t>> frames;
  FlashVsrSrcEncoderStage::Continuity mode =
      FlashVsrSrcEncoderStage::Continuity::kAuto;
  // generate-video's latents, when tapped.
  std::vector<std::vector<float>> latents;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<FlexData> latent_sb;
};

// The graph, run to the end.
// `color_fix` < 0 leaves the key unset, for the release's own default.
// `extra` adds encoder keys; `refs` wires the encoder's color_reference
// output to vae-decode's input, which the wavelet colour fix and the tiny
// decoder read; `tap` also collects generate-video's latents.
Run
restore_(const std::string& root,
         const std::vector<std::vector<std::uint8_t>>& src, int group,
         int overlap, const char* continuity, int color_fix,
         int clip_frames = 41, int blend_frames = 8,
         const FlexData& extra = FlexData{}, bool refs = false,
         bool tap = false)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto fs = std::make_unique<Frames>(&sess, "src", std::vector<InEdge>{},
                                     FlexData::make_object());
  fs->frames = src;
  fs->allocate_oports(1);
  Stage* s_src = pl->insert_stage(std::move(fs));

  Stage* s_stack = pl->insert_stage(std::make_unique<TemporalStackStage>(
      &sess, "stack", std::vector<InEdge>{{s_src, 0}},
      obj_({{"mode", FlexData::make_string("video")},
            {"group_size", FlexData::make_int(group)},
            {"overlap", FlexData::make_int(overlap)}})));

  FlexData ecfg =
      obj_({{"hf_dir", FlexData::make_string(root)},
            {"continuity", FlexData::make_string(continuity)},
            {"clip_frames", FlexData::make_int(clip_frames)},
            {"blend_frames", FlexData::make_int(blend_frames)}});
  if (color_fix >= 0) {
    ecfg.as_object().insert_or_assign("color_fix",
                                      FlexData::make_bool(color_fix != 0));
  }
  if (extra.is_object()) {
    FlexData x = extra;                    // as_object() is a view
    for (const auto& [k, v] : x.as_object()) {
      ecfg.as_object().insert_or_assign(k, v);
    }
  }
  auto enc_up = std::make_unique<FlashVsrSrcEncoderStage>(
      &sess, "enc", std::vector<InEdge>{{s_stack, 0}}, std::move(ecfg));
  auto* enc = enc_up.get();
  Stage* s_enc = pl->insert_stage(std::move(enc_up));

  std::vector<InEdge> gin{{s_enc, 0}};
  Stage* s_gen = nullptr;
  {
    auto g = std::make_unique<GenerateVideoStage>(
        &sess, "gen", std::vector<InEdge>{},
        obj_({{"hf_dir", FlexData::make_string(root)},
              {"width", FlexData::make_int(kSize)},
              {"height", FlexData::make_int(kSize)},
              {"frames", FlexData::make_int(25)},
              {"seed", FlexData::make_int(0)}}));
    gin.resize(g->spec().iports.size(), InEdge{nullptr, 0});
    s_gen = pl->insert_stage(std::make_unique<GenerateVideoStage>(
        &sess, "gen", std::move(gin),
        obj_({{"hf_dir", FlexData::make_string(root)},
              {"width", FlexData::make_int(kSize)},
              {"height", FlexData::make_int(kSize)},
              {"frames", FlexData::make_int(25)},
              {"seed", FlexData::make_int(0)}})));
  }
  std::vector<InEdge> vin{{s_gen, 0}};
  Stage* s_vae = nullptr;
  {
    auto v = std::make_unique<VaeDecodeStage>(
        &sess, "vae", std::vector<InEdge>{},
        obj_({{"hf_dir", FlexData::make_string(root)}}));
    vin.resize(v->spec().iports.size(), InEdge{nullptr, 0});
    if (refs && vin.size() > 2) { vin[2] = InEdge{s_enc, 1}; }
    s_vae = pl->insert_stage(std::make_unique<VaeDecodeStage>(
        &sess, "vae", std::move(vin),
        obj_({{"hf_dir", FlexData::make_string(root)}})));
  }
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::make_unique<Sink>(
      &sess, "sink", std::vector<InEdge>{{s_vae, 0}},
      FlexData::make_object())));
  Sink* lat = nullptr;
  if (tap) {
    lat = static_cast<Sink*>(pl->insert_stage(std::make_unique<Sink>(
        &sess, "latents", std::vector<InEdge>{{s_gen, 0}},
        FlexData::make_object())));
  }
  {
    PipelineRuntime rt(pl.get(), &sess);
    if (rt.launch()) {
      rt.wait_idle();
      rt.stop();
    }
  }
  Run r;
  r.frames = std::move(sink->frames);
  r.mode = enc->continuity();
  if (lat != nullptr) {
    r.latents = std::move(lat->latents);
    r.shapes = std::move(lat->shapes);
    r.latent_sb = std::move(lat->latent_sb);
  }
  return r;
}

// Per-channel mean of a planar [3, H, W] U8 frame.
void
means_(const std::vector<std::uint8_t>& px, double out[3])
{
  const std::size_t n = px.size() / 3;
  for (int c = 0; c < 3; ++c) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) { s += px[(std::size_t)c * n + i]; }
    out[c] = n > 0 ? s / (double)n : 0.0;
  }
}

}  // namespace

TEST(flashvsr_stream, groups_restore_as_one_pass)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const auto src = source_();

  // ONE group: the stream in one beat (and its end in another).
  const Run one = restore_(root, src, kFrames, 0, "stream", 1);
  // Groups of 25 overlapping by 4, as a clip graph cuts them -- AUTO, which
  // must choose the stream at this size on any box this test runs on.
  const Run cut = restore_(root, src, 25, 4, "auto", 1);

  ASSERT_TRUE(cut.mode == FlashVsrSrcEncoderStage::Continuity::kStream);
  std::printf("  one group: %zu frames; groups of 25+4: %zu frames (source "
              "%d)\n", one.frames.size(), cut.frames.size(), kFrames);
  EXPECT_TRUE((int)one.frames.size() == kFrames);
  EXPECT_TRUE((int)cut.frames.size() == kFrames);
  bool same = one.frames.size() == cut.frames.size() && !one.frames.empty();
  int first_diff = -1;
  for (std::size_t i = 0; same && i < one.frames.size(); ++i) {
    if (one.frames[i] != cut.frames[i]) {
      same = false;
      first_diff = (int)i;
    }
  }
  EXPECT_TRUE(same);
  std::printf("  groups vs one pass: %s%s\n", same ? "IDENTICAL" : "DIFFERENT",
              first_diff >= 0
                  ? (" from frame " + std::to_string(first_diff)).c_str()
                  : "");

  // THE COLOUR FIX: without it the same stream decodes to other bytes, and
  // with it every frame's channel means sit closer to its source frame's.
  const Run raw = restore_(root, src, kFrames, 0, "stream", 0);
  ASSERT_TRUE(raw.frames.size() == one.frames.size());
  if (raw.frames.size() != one.frames.size()) { return; }
  double err_fix = 0.0, err_raw = 0.0;
  for (std::size_t i = 0; i < one.frames.size(); ++i) {
    // FlashVSR's source is already at the output size, so frame i of it
    // is the colour reference of restored frame i.
    double ms[3], mf[3], mr[3];
    means_(src[i], ms);
    means_(one.frames[i], mf);
    means_(raw.frames[i], mr);
    for (int c = 0; c < 3; ++c) {
      err_fix += std::fabs(mf[c] - ms[c]);
      err_raw += std::fabs(mr[c] - ms[c]);
    }
  }
  const double n = 3.0 * (double)one.frames.size();
  std::printf("  channel-mean error against the source: %.3f with the colour "
              "fix, %.3f without (8-bit levels)\n", err_fix / n, err_raw / n);
  EXPECT_TRUE(raw.frames != one.frames);
  EXPECT_TRUE(err_fix < err_raw);
  EXPECT_TRUE(err_fix / n < 1.0);
}

// THE SEAM, measured: the mean absolute change from frame t-1 to frame
// t, against the median change over the whole run. A cut shows as a
// spike at the cut; a stream shows none anywhere.
namespace {
struct Seams {
  double max_ratio = 0.0;   // the largest change over the median
  int    at = -1;           // ...and where
};

Seams
seams_(const std::vector<std::vector<std::uint8_t>>& f)
{
  std::vector<double> d;
  for (std::size_t t = 1; t < f.size(); ++t) {
    double s = 0.0;
    for (std::size_t i = 0; i < f[t].size(); ++i) {
      s += std::fabs((double)f[t][i] - (double)f[t - 1][i]);
    }
    d.push_back(s / (double)f[t].size());
  }
  Seams r;
  if (d.empty()) { return r; }
  std::vector<double> sorted = d;
  std::sort(sorted.begin(), sorted.end());
  const double med = std::max(1e-6, sorted[sorted.size() / 2]);
  for (std::size_t t = 0; t < d.size(); ++t) {
    if (d[t] / med > r.max_ratio) {
      r.max_ratio = d[t] / med;
      r.at = (int)t + 1;
    }
  }
  return r;
}
}  // namespace

// CLIPS, the fallback for a box too small for a stream: the source cut
// into overlapping clips, each restored cold, the shared frames
// cross-faded by the decode. Every source frame comes out exactly once
// either way, and the cross-fade must make the worst frame-to-frame jump
// smaller than hard cuts do -- the stream, with no cut at all, smallest.
TEST(flashvsr_stream, clips_cross_fade_their_cuts)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const auto src = source_();
  const Run hard = restore_(root, src, 25, 0, "clips", 1, 25, 0);
  const Run soft = restore_(root, src, 25, 0, "clips", 1, 25, 8);
  const Run one = restore_(root, src, 25, 0, "stream", 1);
  ASSERT_TRUE(hard.mode == FlashVsrSrcEncoderStage::Continuity::kClips);
  EXPECT_TRUE((int)hard.frames.size() == kFrames);
  EXPECT_TRUE((int)soft.frames.size() == kFrames);
  EXPECT_TRUE((int)one.frames.size() == kFrames);
  const Seams sh = seams_(hard.frames), ss = seams_(soft.frames),
              so = seams_(one.frames);
  std::printf("  worst frame-to-frame change over the median: hard cuts "
              "%.2fx (at %d), cross-faded %.2fx (at %d), stream %.2fx (at "
              "%d)\n", sh.max_ratio, sh.at, ss.max_ratio, ss.at,
              so.max_ratio, so.at);
  // Hard clips of 25 restore 21 frames each: the cuts are at 21 and 42.
  EXPECT_TRUE(sh.at == 21 || sh.at == 42);
  EXPECT_TRUE(ss.max_ratio < sh.max_ratio);
  EXPECT_TRUE(so.max_ratio < sh.max_ratio);
}

// WHERE AUTO STREAMS. The rule is total RAM against what a stream holds,
// so VPIPE_RAM_LIMIT_MB stands in for the boxes: 1920x1152 streams on 64
// GB and not on 16 (~20 GB of carries alone), 512x512 streams on both.
// The figures are printed because they ARE the decision.
TEST(flashvsr_stream, auto_streams_where_the_carries_fit)
{
  const char* root = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const char* prev = std::getenv("VPIPE_RAM_LIMIT_MB");
  const std::string keep = prev != nullptr ? prev : "";
  struct Case { int w, h, ram_gb; bool fits; };
  const Case cases[] = {{1920, 1152, 64, true}, {1920, 1152, 16, false},
                        {512, 512, 16, true},   {512, 512, 64, true}};
  for (const Case& c : cases) {
    ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(c.ram_gb * 1024).c_str(),
             1);
    std::size_t need = 0;
    const bool fits = genai::flashvsr_stream_fits(root, c.w, c.h, &need);
    std::printf("  %dx%d on %d GB: carries %zu MB, needs %zu MB -> %s\n",
                c.w, c.h, c.ram_gb,
                genai::flashvsr_stream_bytes(root, c.w, c.h) >> 20,
                need >> 20, fits ? "stream" : "clips");
    EXPECT_TRUE(fits == c.fits);
  }
  if (prev != nullptr) {
    ::setenv("VPIPE_RAM_LIMIT_MB", keep.c_str(), 1);
  } else {
    ::unsetenv("VPIPE_RAM_LIMIT_MB");
  }
}

// ---- FlashVSR+ ---------------------------------------------------------
//
// The same graph and the same promises on FlashVSR+ Turbo, whose chunks
// take 16 source frames where v1.1's take 8 (17 to open, not 25), whose
// stream carries a window per LAYER of two sizes, and whose reference
// restores every source frame with no colour fix. Env:
// VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH (the repository root), and for the
// rows VPIPE_FLASHVSR_PLUS_GOLDEN (tools/dump_flashvsr_plus_golden.py).

namespace {

std::string
plus_turbo_()
{
  const char* m = std::getenv("VPIPE_FLASHVSR_PLUS_TEST_MODEL_PATH");
  return m != nullptr && *m != '\0' ? std::string(m) + "/turbo" : "";
}

// The source through temporal-stack and the encoder alone: its beats.
Sink::Beats
encode_(const std::string& root,
        const std::vector<std::vector<std::uint8_t>>& src, int h, int w,
        int group)
{
  Session sess;
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto fs = std::make_unique<Frames>(&sess, "src", std::vector<InEdge>{},
                                     FlexData::make_object());
  fs->frames = src;
  fs->h = h;
  fs->w = w;
  fs->allocate_oports(1);
  Stage* s_src = pl->insert_stage(std::move(fs));
  Stage* s_stack = pl->insert_stage(std::make_unique<TemporalStackStage>(
      &sess, "stack", std::vector<InEdge>{{s_src, 0}},
      obj_({{"mode", FlexData::make_string("video")},
            {"group_size", FlexData::make_int(group)},
            {"overlap", FlexData::make_int(0)}})));
  Stage* s_enc = pl->insert_stage(std::make_unique<FlashVsrSrcEncoderStage>(
      &sess, "enc", std::vector<InEdge>{{s_stack, 0}},
      obj_({{"hf_dir", FlexData::make_string(root)},
            {"continuity", FlexData::make_string("stream")}})));
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::make_unique<Sink>(
      &sess, "sink", std::vector<InEdge>{{s_enc, 0}},
      FlexData::make_object())));
  {
    PipelineRuntime rt(pl.get(), &sess);
    if (rt.launch()) {
      rt.wait_idle();
      rt.stop();
    }
  }
  return {std::move(sink->rows), std::move(sink->sidebands)};
}

}  // namespace

TEST(flashvsr_stream, plus_groups_restore_as_one_pass)
{
  const std::string root = plus_turbo_();
  if (root.empty()) { return; }
  const auto src = source_();
  // The colour fix left to the release: off for FlashVSR+, as its
  // reference has it -- so the default and an explicit `false` agree.
  const Run one = restore_(root, src, kFrames, 0, "stream", -1, 0);
  const Run off = restore_(root, src, kFrames, 0, "stream", 0, 0);
  const Run cut = restore_(root, src, 20, 0, "stream", -1, 0);
  std::printf("  FlashVSR+ Turbo: one group %zu frames, groups of 20 %zu "
              "(source %d)\n", one.frames.size(), cut.frames.size(),
              kFrames);
  // EVERY source frame comes back: the end is padded past the look-ahead.
  EXPECT_TRUE((int)one.frames.size() == kFrames);
  EXPECT_TRUE(one.frames == cut.frames);
  EXPECT_TRUE(one.frames == off.frames);
  // Clips at the release's own length (49): every frame, exactly once.
  const Run clips = restore_(root, src, 25, 0, "clips", -1, 0, 8);
  ASSERT_TRUE(clips.mode == FlashVsrSrcEncoderStage::Continuity::kClips);
  EXPECT_TRUE((int)clips.frames.size() == kFrames);
  const Seams sc = seams_(clips.frames), so = seams_(one.frames);
  std::printf("  clips of 49 blending 8: %zu frames, worst change %.2fx "
              "the median (stream %.2fx)\n", clips.frames.size(),
              sc.max_ratio, so.max_ratio);
}

// The encoder's own cadence, through the stage, against the reference's
// rows: 17 frames to open and 16 a chunk, the end padded with the last
// frame, and the beats' bookkeeping -- chunks in order, every real frame
// counted once, the last beat final.
TEST(flashvsr_stream, plus_encoder_rows_match_reference)
{
  const std::string root = plus_turbo_();
  const char* g = std::getenv("VPIPE_FLASHVSR_PLUS_GOLDEN");
  if (root.empty() || g == nullptr || *g == '\0') { return; }
  const std::string dir = std::string(g) + "/turbo-384x640";
  constexpr int H = 384, W = 640, F = 61, L = 16;
  const std::size_t frame = (std::size_t)3 * H * W;
  std::vector<std::uint8_t> all(frame * F);
  {
    std::FILE* f = std::fopen((dir + "/lq_video_u8.bin").c_str(), "rb");
    if (f == nullptr) { return; }
    const std::size_t n = std::fread(all.data(), 1, all.size(), f);
    std::fclose(f);
    if (n != all.size()) { return; }
  }
  std::vector<std::vector<std::uint8_t>> src;
  for (int i = 0; i < F; ++i) {
    src.emplace_back(all.begin() + (std::ptrdiff_t)(i * frame),
                     all.begin() + (std::ptrdiff_t)((i + 1) * frame));
  }
  const Sink::Beats b = encode_(root, src, H, W, 20);
  const std::size_t row_frame = (std::size_t)(H / 16) * (W / 16) * 1536;
  std::vector<std::uint16_t> rows;
  int next = 0;
  long long valid = 0;
  bool final_seen = false;
  for (std::size_t i = 0; i < b.rows.size(); ++i) {
    FlexData sb = b.sidebands[i];
    auto o = sb.as_object();
    const int c0 = (int)o.at(genai::flashvsr_stream::kChunk0).as_int(-1);
    const int n = (int)o.at(genai::flashvsr_stream::kChunks).as_int(0);
    EXPECT_TRUE(c0 == next);
    EXPECT_TRUE(b.rows[i].size() == (std::size_t)4 * n * row_frame);
    next = c0 + n;
    valid += o.at(sideband::kValidFrames).as_int(0);
    final_seen = o.at(sideband::kStreamFinal).as_bool(false);
    EXPECT_FALSE(o.contains(sideband::kColorRef));
    rows.insert(rows.end(), b.rows[i].begin(), b.rows[i].end());
  }
  std::printf("  %zu beats, %d chunks, %lld valid frames (source %d)%s\n",
              b.rows.size(), next, valid, F, final_seen ? ", final" : "");
  EXPECT_TRUE(next == L / 4 && valid == F && final_seen);
  ASSERT_TRUE(rows.size() == (std::size_t)L * row_frame);
  if (rows.size() != (std::size_t)L * row_frame) { return; }
  double worst = 0.0;
  for (int i = 0; i < L; ++i) {
    std::FILE* f = std::fopen(
        (dir + "/lq_proj" + std::to_string(i) + ".f32").c_str(), "rb");
    if (f == nullptr) { return; }
    std::vector<float> want(row_frame);
    const std::size_t n = std::fread(want.data(), 4, row_frame, f);
    std::fclose(f);
    if (n != row_frame) { return; }
    double num = 0.0, den = 0.0;
    for (std::size_t k = 0; k < row_frame; ++k) {
      std::uint32_t u = (std::uint32_t)rows[(std::size_t)i * row_frame + k]
                        << 16;
      float got;
      std::memcpy(&got, &u, 4);
      num += ((double)got - want[k]) * ((double)got - want[k]);
      den += (double)want[k] * want[k];
    }
    worst = std::max(worst, std::sqrt(num / den));
  }
  std::printf("  encoder rows vs the reference: worst rel-L2 %.5f\n", worst);
  EXPECT_TRUE(worst < 0.01);
}

// ---- temporal compression and the wavelet colour fix ------------------

namespace {

// The source with every frame but the first repeated r times: what the
// encoder feeds the model under temporal compression 4 / r.
std::vector<std::vector<std::uint8_t>>
expand_(const std::vector<std::vector<std::uint8_t>>& src, int r)
{
  std::vector<std::vector<std::uint8_t>> out;
  for (std::size_t i = 0; i < src.size(); ++i) {
    for (int k = 0; k < (i == 0 ? 1 : r); ++k) { out.push_back(src[i]); }
  }
  return out;
}

// The reference's merge (inference/preprocessing/temporal.py), over
// decoded U8 frames: the first alone, each later run of r weighted by a
// gaussian of sigma 1 over its positions.
std::vector<std::vector<std::uint8_t>>
merge_(const std::vector<std::vector<std::uint8_t>>& dec, int r)
{
  std::vector<double> w((std::size_t)r);
  double sum = 0.0;
  for (int k = 0; k < r; ++k) {
    const double x = k - (r - 1) / 2.0;
    w[(std::size_t)k] = std::exp(-0.5 * x * x);
    sum += w[(std::size_t)k];
  }
  for (double& x : w) { x /= sum; }
  std::vector<std::vector<std::uint8_t>> out;
  if (dec.empty()) { return out; }
  out.push_back(dec[0]);
  for (std::size_t j = 1; j + r <= dec.size(); j += (std::size_t)r) {
    std::vector<std::uint8_t> f(dec[j].size());
    for (std::size_t i = 0; i < f.size(); ++i) {
      double v = 0.0;
      for (int k = 0; k < r; ++k) {
        v += w[(std::size_t)k] * dec[j + (std::size_t)k][i];
      }
      f[i] = (std::uint8_t)std::lround(std::clamp(v, 0.0, 255.0));
    }
    out.push_back(std::move(f));
  }
  return out;
}

// The largest and the mean absolute difference over two runs of frames,
// in 8-bit levels.
void
level_diff_(const std::vector<std::vector<std::uint8_t>>& a,
            const std::vector<std::vector<std::uint8_t>>& b, int* worst,
            double* mean)
{
  *worst = 255;
  *mean = 255.0;
  if (a.size() != b.size() || a.empty()) { return; }
  long long w = 0;
  double s = 0.0;
  std::size_t n = 0;
  for (std::size_t f = 0; f < a.size(); ++f) {
    if (a[f].size() != b[f].size()) { return; }
    for (std::size_t i = 0; i < a[f].size(); ++i) {
      const int d = std::abs((int)a[f][i] - (int)b[f][i]);
      w = std::max<long long>(w, d);
      s += d;
      ++n;
    }
  }
  *worst = (int)w;
  *mean = s / (double)n;
}

FlexData
tc_(int tc)
{
  return obj_({{"temporal_compression", FlexData::make_int(tc)}});
}

}  // namespace

// TEMPORAL COMPRESSION 2 and 1: the source fed 2 or 4 times a frame and
// merged back. It has to be EXACTLY the reference's chain -- expand, run,
// merge -- so the same source expanded by hand and run at 4 must decode to
// frames whose merge is this run's output, to the one 8-bit level the two
// quantisations can differ by (here the merge is in f32 before the frame
// is rounded; there after). Every source frame comes back once, and a
// stream cut into groups is still one pass.
TEST(flashvsr_stream, plus_temporal_compression_merges_runs)
{
  const std::string root = plus_turbo_();
  if (root.empty()) { return; }
  const auto src = source_();
  for (int tc : {2, 1}) {
    const int r = 4 / tc;
    const Run a = restore_(root, src, kFrames, 0, "stream", 0, 0, 8, tc_(tc));
    const Run cut = restore_(root, src, 20, 0, "stream", 0, 0, 8, tc_(tc));
    const auto big = expand_(src, r);
    const Run b = restore_(root, big, (int)big.size(), 0, "stream", 0);
    const auto merged = merge_(b.frames, r);
    int worst = 0;
    double mean = 0.0;
    level_diff_(a.frames, merged, &worst, &mean);
    std::printf("  temporal compression %d: %zu frames (source %d, model "
                "%zu); against the hand-expanded run merged: worst %d "
                "level(s), mean %.4f; groups of 20 %s\n", tc,
                a.frames.size(), kFrames, b.frames.size(), worst, mean,
                cut.frames == a.frames ? "identical" : "DIFFERENT");
    EXPECT_TRUE((int)a.frames.size() == kFrames);
    EXPECT_TRUE(b.frames.size() == big.size());
    EXPECT_TRUE(worst <= 1);
    EXPECT_TRUE(cut.frames == a.frames);
  }
  // `first` merges with weights [1, 0]: each output frame IS the first
  // decoded frame of its run, to the byte -- which pins which decoded
  // frames make a run (the lead, the cut, the order) exactly.
  {
    FlexData x = tc_(2);
    x.as_object().insert_or_assign("temporal_merge",
                                   FlexData::make_string("first"));
    const Run first = restore_(root, src, kFrames, 0, "stream", 0, 0, 8, x);
    const auto big = expand_(src, 2);
    const Run b = restore_(root, big, (int)big.size(), 0, "stream", 0);
    // The lead frame, then each run [1, 2], [3, 4], ...'s first.
    std::vector<std::vector<std::uint8_t>> picked;
    if (!b.frames.empty()) { picked.push_back(b.frames[0]); }
    for (std::size_t j = 1; j + 1 < b.frames.size(); j += 2) {
      picked.push_back(b.frames[j]);
    }
    std::printf("  merge `first` at compression 2: %s the hand-expanded "
                "run's run starts\n",
                first.frames == picked ? "IDENTICAL to" : "DIFFERENT from");
    EXPECT_TRUE(first.frames == picked);
  }
  // CLIPS cut on whole runs: every source frame, once.
  const Run clips = restore_(root, src, 25, 0, "clips", 0, 0, 8, tc_(2));
  ASSERT_TRUE(clips.mode == FlashVsrSrcEncoderStage::Continuity::kClips);
  EXPECT_TRUE((int)clips.frames.size() == kFrames);
  const Seams sc = seams_(clips.frames);
  std::printf("  clips at compression 2: %zu frames, worst change %.2fx "
              "the median\n", clips.frames.size(), sc.max_ratio);
}

// THE WAVELET COLOUR FIX through the graph: FlashVSR+'s own (`color_fix`
// on, method auto), its pictures carried from the encoder to vae-decode.
// The output must be the fix applied to the uncorrected run's frames
// against their source frames -- to the levels the two roundings allow --
// and its low frequencies must sit nearer the source's than without it.
// (The arithmetic itself is held to the reference's own corrector by
// flashvsr_plus.wavelet_fix_matches_reference; this is the plumbing.)
// Unwired, the fix is said to be missing and the frames are the raw ones.
TEST(flashvsr_stream, plus_wavelet_color_fix)
{
  const std::string root = plus_turbo_();
  if (root.empty()) { return; }
  // A source whose colour FLICKERS frame to frame, so a picture paired
  // with the wrong frame shows: the drift alone moves the low band too
  // little between neighbours to tell.
  auto src = source_();
  for (std::size_t f = 0; f < src.size(); ++f) {
    const int tint = (int)(f % 5) * 9 - 18;
    const std::size_t plane = (std::size_t)kSize * kSize;
    for (std::size_t i = 0; i < plane; ++i) {
      std::uint8_t& v = src[f][i];               // the red plane
      v = (std::uint8_t)std::clamp((int)v + tint, 0, 255);
    }
  }
  // The VAE, pinned: wired, FlashVSR+ would decode with its tiny decoder
  // (plus_tiny_decoder_in_the_graph holds the two together).
  const FlexData vae = obj_({{"decoder", FlexData::make_string("vae")}});
  const Run raw = restore_(root, src, kFrames, 0, "stream", 0, 0, 8, vae);
  const Run fix = restore_(root, src, kFrames, 0, "stream", 1, 0, 8, vae,
                           /*refs=*/true);
  const Run unwired = restore_(root, src, kFrames, 0, "stream", 1, 0, 8,
                               vae);
  ASSERT_TRUE((int)raw.frames.size() == kFrames &&
              (int)fix.frames.size() == kFrames);
  if ((int)fix.frames.size() != kFrames) { return; }
  EXPECT_TRUE(unwired.frames == raw.frames);
  EXPECT_TRUE(fix.frames != raw.frames);

  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  WaveletColorFix wf(mc);
  ASSERT_TRUE(wf.valid());
  const std::size_t n = (std::size_t)3 * kSize * kSize;
  std::vector<std::vector<std::uint8_t>> want;
  double low_fix = 0.0, low_raw = 0.0;
  for (int f = 0; f < kFrames; ++f) {
    std::vector<float> x(n), ref(n), lf(n), lr(n);
    for (std::size_t i = 0; i < n; ++i) {
      x[i] = raw.frames[(std::size_t)f][i] / 127.5f - 1.0f;
      ref[i] = src[(std::size_t)f][i] / 127.5f - 1.0f;
    }
    std::vector<float> y = x;
    ASSERT_TRUE(wf.apply(y.data(), ref.data(), kSize, kSize, 5));
    std::vector<std::uint8_t> q(n);
    for (std::size_t i = 0; i < n; ++i) {
      q[i] = (std::uint8_t)std::lround(
          std::clamp((y[i] + 1.0f) * 0.5f, 0.0f, 1.0f) * 255.0f);
    }
    want.push_back(std::move(q));
    // The LOW FREQUENCIES of each frame's difference from its source, by
    // the fix's own low-pass: apply(0, d) is clamp(low(d)).
    for (std::size_t i = 0; i < n; ++i) {
      lf[i] = (fix.frames[(std::size_t)f][i] / 127.5f - 1.0f) - ref[i];
      lr[i] = x[i] - ref[i];
    }
    std::vector<float> zf(n, 0.0f), zr(n, 0.0f);
    ASSERT_TRUE(wf.apply(zf.data(), lf.data(), kSize, kSize, 5));
    ASSERT_TRUE(wf.apply(zr.data(), lr.data(), kSize, kSize, 5));
    for (std::size_t i = 0; i < n; ++i) {
      low_fix += std::fabs(zf[i]);
      low_raw += std::fabs(zr[i]);
    }
  }
  int worst = 0;
  double mean = 0.0;
  level_diff_(fix.frames, want, &worst, &mean);
  const double nn = (double)n * kFrames;
  std::printf("  wavelet fix in the graph vs applied to the raw run: worst "
              "%d level(s), mean %.4f; low-frequency error against the "
              "source %.4f fixed, %.4f raw ([-1, 1])\n", worst, mean,
              low_fix / nn, low_raw / nn);
  // Two roundings apart: the graph fixes the f32 frame and rounds once,
  // the check fixes the rounded raw frame and rounds again.
  EXPECT_TRUE(worst <= 2 && mean < 0.5);
  // Nearer, not equal: low(out) = low(x) - low(low(x)) + low(low(ref)),
  // and a low-pass applied twice is not the low-pass -- so the fix pulls
  // the low band toward the source's (0.0576 -> 0.0215 here) without
  // landing on it.
  EXPECT_TRUE(low_fix < 0.75 * low_raw);
}

// ---- the tiny decoder -------------------------------------------------

namespace {

// The tapped latents through the tiny decoder BY HAND: beats that continue
// one another are one stream, and the stream's decoded frame g reads the
// source frame the encoder expanded into it -- frame 0 once, then each
// source frame r times (the last standing in past the end) -- the rule
// written out here, not the stages'. Each beat is trimmed to its
// valid_frames and its runs merged with its own merge_weights, in f32 on
// the decoder's UNCLAMPED output as the reference does (a decoder that
// overshoots [-1, 1] merges differently once clamped): what vae-decode
// holds before its colour fix and its rounding.
std::vector<std::vector<float>>
by_hand_(const std::string& root, const Run& run,
         const std::vector<std::vector<std::uint8_t>>& src, int r)
{
  std::vector<std::vector<float>> out;
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout) ||
      layout.tiny_decoder.empty()) {
    return out;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return out; }
  std::string err;
  auto dec = genai::FlashVsrTinyDecoder::load(
      genai::WeightSet::open(layout.tiny_decoder, nullptr), mc, &err);
  if (dec == nullptr) { return out; }
  genai::FlashVsrTinyDecoder::Stream s;
  const std::size_t n = (std::size_t)3 * kSize * kSize;
  const long long last = (long long)src.size() - 1;
  for (std::size_t b = 0; b < run.latents.size(); ++b) {
    FlexData sb = run.latent_sb[b];
    auto o = sb.as_object();
    if (!(o.contains(sideband::kStreamContinues) &&
          o.at(sideband::kStreamContinues).as_bool(false))) {
      s = {};
    }
    const long long valid = o.contains(sideband::kValidFrames)
                                ? o.at(sideband::kValidFrames).as_int(0)
                                : (long long)1 << 40;
    std::vector<float> w;
    if (o.contains(sideband::kMergeWeights)) {
      FlexData mw = o.at(sideband::kMergeWeights);
      for (const FlexData& x : mw.as_array()) {
        w.push_back((float)x.as_real(0.0));
      }
    }
    const int lead = s.frames == 0 && w.size() > 1 ? 1 : 0;
    const long long f0 = s.frames;
    const auto pic = [&](long long d) {
      const long long g = f0 + d;
      const long long i = g == 0 ? 0 : 1 + (g - 1) / r;
      genai::FlashVsrTinyDecoder::Picture p;
      p.u8 = src[(std::size_t)std::min(i, last)].data();
      return p;
    };
    std::vector<float> acc(n, 0.0f);
    const auto sink = [&](long long d, float* x) {
      if (d >= valid) { return true; }
      if (w.size() <= 1 || d < lead) {
        out.emplace_back(x, x + n);
        return true;
      }
      const int slot = (int)((d - lead) % (long long)w.size());
      for (std::size_t i = 0; i < n; ++i) {
        acc[i] = (slot == 0 ? 0.0f : acc[i]) + w[(std::size_t)slot] * x[i];
      }
      if (slot + 1 == (int)w.size()) { out.push_back(acc); }
      return true;
    };
    const auto& sh = run.shapes[b];
    if (sh.size() != 4 ||
        !dec->decode(run.latents[b].data(), (int)sh[1], (int)sh[2],
                     (int)sh[3], pic, sink, &s, &err)) {
      return {};
    }
  }
  return out;
}

// [-1, 1] frames rounded as vae-decode rounds them.
std::vector<std::vector<std::uint8_t>>
round_(const std::vector<std::vector<float>>& f)
{
  std::vector<std::vector<std::uint8_t>> out;
  for (const auto& x : f) {
    std::vector<std::uint8_t> q(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      float u = (x[i] + 1.0f) * 0.5f;
      if (!(u > 0.0f)) { u = 0.0f; }
      if (u > 1.0f) { u = 1.0f; }
      q[i] = (std::uint8_t)std::lround(u * 255.0f);
    }
    out.push_back(std::move(q));
  }
  return out;
}

FlexData
decoder_(const char* d)
{
  return obj_({{"decoder", FlexData::make_string(d)}});
}

}  // namespace

// FlashVSR+'s TINY DECODER through the graph: the release's default once
// its source pictures are wired (decoder auto), each latent decoded with
// the pictures its decode covers. Held to: one pass whatever the groups;
// the frames a decode by hand makes from the same latents and the source
// frames paired by the rule above (to the byte -- this is the plumbing:
// which picture reaches which decoded frame, the stream's first and the
// padding past its end included); temporal compression 2 as the
// hand-expanded run merged; clips; and with the wavelet fix on top, that
// fix applied to the tiny run. Unwired, the decode falls back to the VAE.
// (The decoder's arithmetic is held to the reference by
// flashvsr_plus.tiny_decoder_matches_reference.)
TEST(flashvsr_stream, plus_tiny_decoder_in_the_graph)
{
  const std::string root = plus_turbo_();
  if (root.empty()) { return; }
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout) ||
      layout.tiny_decoder.empty()) {
    std::printf("  no TCDecoder.ckpt beside '%s'; skipped\n", root.c_str());
    return;
  }
  const auto src = source_();
  const Run one = restore_(root, src, kFrames, 0, "stream", 0, 0, 8,
                           FlexData{}, /*refs=*/true, /*tap=*/true);
  const Run cut = restore_(root, src, 20, 0, "stream", 0, 0, 8, FlexData{},
                           true);
  const Run vae = restore_(root, src, kFrames, 0, "stream", 0, 0, 8,
                           decoder_("vae"), true);
  const Run unwired = restore_(root, src, kFrames, 0, "stream", 0, 0, 8,
                               decoder_("tiny"), false);
  ASSERT_TRUE((int)one.frames.size() == kFrames);
  if ((int)one.frames.size() != kFrames) { return; }
  EXPECT_TRUE(cut.frames == one.frames);
  // Engaged: not the VAE's frames -- and not far from them either.
  EXPECT_TRUE(one.frames != vae.frames);
  EXPECT_TRUE(unwired.frames == vae.frames);
  int worst = 0;
  double mean = 0.0;
  level_diff_(one.frames, vae.frames, &worst, &mean);
  const auto hand = round_(by_hand_(root, one, src, 1));
  std::printf("  tiny decoder: %zu frames (%zu latent beats), groups of 20 "
              "%s, by hand %s; against the VAE's decode mean %.2f levels; "
              "unwired %s the VAE\n", one.frames.size(), one.latents.size(),
              cut.frames == one.frames ? "identical" : "DIFFERENT",
              hand == one.frames ? "IDENTICAL" : "DIFFERENT", mean,
              unwired.frames == vae.frames ? "is" : "is NOT");
  EXPECT_TRUE(hand == one.frames);
  EXPECT_TRUE(mean < 12.0);

  // Temporal compression 2: the decoded frames are the EXPANDED ones, each
  // reading its own source picture, merged back in runs -- by hand from
  // the run's own latents to the byte; and against the source expanded by
  // hand and run at 4, merged after rounding (the decoder overshoots
  // [-1, 1], so the order of clamp and merge moves a few pixels by a few
  // levels -- the mean is the bar there).
  {
    FlexData x = tc_(2);
    x.as_object().insert_or_assign("decoder", FlexData::make_string("tiny"));
    const Run a = restore_(root, src, kFrames, 0, "stream", 0, 0, 8, x,
                           true, /*tap=*/true);
    const auto a_hand = round_(by_hand_(root, a, src, 2));
    const auto big = expand_(src, 2);
    const Run b = restore_(root, big, (int)big.size(), 0, "stream", 0, 0, 8,
                           decoder_("tiny"), true);
    int w2 = 0;
    double m2 = 0.0;
    level_diff_(a.frames, merge_(b.frames, 2), &w2, &m2);
    std::printf("  temporal compression 2: %zu frames; by hand %s; against "
                "the hand-expanded run merged: worst %d level(s), mean "
                "%.4f\n", a.frames.size(),
                a_hand == a.frames ? "IDENTICAL" : "DIFFERENT", w2, m2);
    EXPECT_TRUE((int)a.frames.size() == kFrames);
    EXPECT_TRUE(a_hand == a.frames);
    EXPECT_TRUE(m2 < 0.5);
  }
  // Clips: each its own stream, every frame once.
  {
    const Run clips = restore_(root, src, 25, 0, "clips", 0, 0, 8,
                               decoder_("tiny"), true);
    ASSERT_TRUE(clips.mode == FlashVsrSrcEncoderStage::Continuity::kClips);
    EXPECT_TRUE((int)clips.frames.size() == kFrames);
    const Seams sc = seams_(clips.frames), so = seams_(one.frames);
    std::printf("  clips of 49 blending 8: %zu frames, worst change %.2fx "
                "the median (stream %.2fx)\n", clips.frames.size(),
                sc.max_ratio, so.max_ratio);
  }
  // The wavelet fix on top: its pictures and the decoder's both ride
  // color_reference, in that order. The fix of the hand decode -- the
  // decoder's unclamped output, as the reference corrects it.
  {
    const Run wf = restore_(root, src, kFrames, 0, "stream", 1, 0, 8,
                            decoder_("tiny"), true, /*tap=*/true);
    ASSERT_TRUE((int)wf.frames.size() == kFrames);
    if ((int)wf.frames.size() != kFrames) { return; }
    auto raw = by_hand_(root, wf, src, 1);
    Session sess;
    auto* mc = sess.metal_compute();
    if (mc == nullptr || !mc->valid() || raw.size() != src.size()) {
      ASSERT_TRUE(raw.size() == src.size());
      return;
    }
    WaveletColorFix fix(mc);
    const std::size_t n = (std::size_t)3 * kSize * kSize;
    for (int f = 0; f < kFrames; ++f) {
      std::vector<float> ref(n);
      for (std::size_t i = 0; i < n; ++i) {
        ref[i] = src[(std::size_t)f][i] / 127.5f - 1.0f;
      }
      ASSERT_TRUE(fix.apply(raw[(std::size_t)f].data(), ref.data(), kSize,
                            kSize, 5));
    }
    int w3 = 0;
    double m3 = 0.0;
    const auto want = round_(raw);
    level_diff_(wf.frames, want, &w3, &m3);
    std::printf("  with the wavelet fix: against the fix of the hand "
                "decode, worst %d level(s), mean %.4f%s\n", w3, m3,
                want == wf.frames ? " (IDENTICAL)" : "");
    EXPECT_TRUE(wf.frames != one.frames);
    EXPECT_TRUE(w3 <= 1);
  }
}
