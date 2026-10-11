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
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/temporal-stack-stage.h"
#include "stages/flashvsr-src-encoder-stage.h"
#include "stages/generate-video-stage.h"
#include "stages/vae-decode-stage.h"

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
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= frames.size()) { ctx.signal_done(); co_return; }
    auto b = std::make_unique<TensorBeatPayload>();
    b->dtype = TensorBeat::DType::U8;
    b->shape = {3, kSize, kSize};
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
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (tb != nullptr && tb->dtype == TensorBeat::DType::U8) {
      const auto px = tb->materialize_contiguous();
      frames.emplace_back(px.begin(), px.end());
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
};

// The graph, run to the end.
Run
restore_(const std::string& root,
         const std::vector<std::vector<std::uint8_t>>& src, int group,
         int overlap, const char* continuity, bool color_fix,
         int clip_frames = 41, int blend_frames = 8)
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

  auto enc_up = std::make_unique<FlashVsrSrcEncoderStage>(
      &sess, "enc", std::vector<InEdge>{{s_stack, 0}},
      obj_({{"hf_dir", FlexData::make_string(root)},
            {"continuity", FlexData::make_string(continuity)},
            {"color_fix", FlexData::make_bool(color_fix)},
            {"clip_frames", FlexData::make_int(clip_frames)},
            {"blend_frames", FlexData::make_int(blend_frames)}}));
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
    s_vae = pl->insert_stage(std::make_unique<VaeDecodeStage>(
        &sess, "vae", std::move(vin),
        obj_({{"hf_dir", FlexData::make_string(root)}})));
  }
  auto* sink = static_cast<Sink*>(pl->insert_stage(std::make_unique<Sink>(
      &sess, "sink", std::vector<InEdge>{{s_vae, 0}},
      FlexData::make_object())));
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
  const Run one = restore_(root, src, kFrames, 0, "stream", true);
  // Groups of 25 overlapping by 4, as a clip graph cuts them -- AUTO, which
  // must choose the stream at this size on any box this test runs on.
  const Run cut = restore_(root, src, 25, 4, "auto", true);

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
  const Run raw = restore_(root, src, kFrames, 0, "stream", false);
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
  const Run hard = restore_(root, src, 25, 0, "clips", true, 25, 0);
  const Run soft = restore_(root, src, 25, 0, "clips", true, 25, 8);
  const Run one = restore_(root, src, 25, 0, "stream", true);
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
