#ifndef VPIPE_STAGES_FLASHVSR_SRC_ENCODER_STAGE_H
#define VPIPE_STAGES_FLASHVSR_SRC_ENCODER_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/tensor-beat.h"
#endif

// The source projection is a from-scratch, MLX-free metal-compute module
// on the VPIPE_BUILD_APPLE_SILICON axis; an inert stub off it.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "stages/model-memory.h"
#endif

#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// FlashVSR's source encoder: a low-quality CLIP in pixels -> the
// conditioning rows its denoiser adds into the residual stream.
//
// WHY ITS OWN STAGE, and not a branch of `diffusion-conditioner`. That
// stage's own warning says it: a family needing another encoder ships
// its own conditioner. This family needs one that shares nothing with
// the text path -- no prompt, no tokenizer, no negative, no guidance,
// and no text encoder at all, because its text context is a CONSTANT
// baked into the checkpoint. What it has instead is a learned causal
// convolution over pixels. Branching the shared conditioner on that
// would have cost it a pixel-shaped input port and a fourth family, to
// express something that is not a variant of what it does.
//
// So this stage is MODEL-SPECIFIC in the same sense the `*-model-config`
// stages are, and carries the same category: its applicability is
// decided by the checkpoint rather than by the graph. A
// `flashvsr-src-encoder` is meaningless anywhere else, which is exactly
// what lets tooling offer it only where it belongs.
//
// WHAT IT EMITS IS ORDINARY CONDITIONING. The projection's output is
// [tokens, hidden] bf16 -- the shape a conditioning beat already has --
// so it feeds `generate-video`'s existing conditioning port and NOTHING
// in that stage changed to accept it. That is the whole point of putting
// the projection here: the graph reads like every other generative
// graph, conditioner into generator.
//
//   iport0  source   TensorBeatPayload, planar U8 RGB [frames, 3, H, W]
//                    with `fps` on the sideband -- the shape
//                    `temporal-stack` builds. REQUIRED: this family
//                    conditions on its source and nothing else, so an
//                    unwired port is a refusal rather than a fallback.
//                    Read fresh every beat, never latched: a restoration
//                    graph hands a different clip each time.
//   iport1  model    OPTIONAL FlexDataPayload model reference (from a
//                    `model-select` source); OVERRIDES the hf_dir config
//                    so the encoder, the denoiser and the VAE of one
//                    graph share a single model choice. When wired, the
//                    load is DEFERRED to the first process() -- the beat
//                    only arrives after the init barrier.
//
//   oport0  conditioning  TensorBeatPayload bf16 [row_frames * tokens,
//                    hidden]: one row frame per FOUR source frames (the
//                    VAE's temporal compression, which the projection
//                    matches with two stride-2 causal convolutions) and
//                    one token per cell of the denoiser's own grid (the
//                    frame over 16). The sideband carries the geometry
//                    the rows belong to -- source width, height, frame
//                    count and row frame count -- because a row count
//                    alone cannot say which of those produced it.
//
// THE PROJECTION IS CAUSAL, which shapes how this stage drives it. It
// carries the last two input frames of each convolution across calls,
// and its OPENING call seeds those carries and produces no rows at all.
// So a clip is fed in the reference's own irregular pattern rather than
// one shot: seven slices to open, the first of them a single frame, then
// two per denoise chunk. One call over the whole clip returns nothing.
//
// A LONG SOURCE IS ONE STREAM. The groups temporal-stack cuts are memory
// units, not clips: the projection keeps its carries across them, each
// beat's rows say which of the stream's chunks they hold, and the
// denoiser and the VAE decode carry their own state from one beat to the
// next -- so the restoration is the reference's single pass, with no
// change of texture where a group ends. Groups need no overlap (one is
// dropped, by temporal-stack's sideband::kRepeated), and any group size
// works: frames
// that do not complete a chunk wait for the next group. The source's end
// is padded with its last frame, as the reference pads it, and the
// padding's frames are dropped after the decode.
//
// Config (FlexData object):
//   hf_dir     (string, OPTIONAL) -- the FlashVSR model directory. A
//              model-select source on iport1 overrides it; required only
//              when iport1 is unwired.
//   continuity ("auto" | "stream" | "clips", default auto) -- see
//              Continuity.
//   clip_frames, blend_frames (ints, 41 and 8) -- clips only: source
//              frames per clip, and restored frames each shares with the
//              next for the decode to cross-fade (sideband::kBlendHead /
//              kBlendTail). What SAnodes-Streaming does for ComfyUI.
//   color_fix  (bool, default true) -- carry each restored frame's
//              source channel statistics for the decode's colour fix.
class FlashVsrSrcEncoderStage final
    : public TypedStage<FlashVsrSrcEncoderStage> {
public:
  static constexpr const char* kTypeName = "flashvsr-src-encoder";

  FlashVsrSrcEncoderStage(const SessionContextIntf* session,
                          std::string               id,
                          std::vector<InEdge>       iports,
                          FlexData                  config);
  ~FlashVsrSrcEncoderStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The projection is ~0.6 GB and is resident while a clip is encoded.
  // It is the only model this stage holds, and nothing else declares it
  // -- the denoiser stopped owning it when the projection moved here, so
  // a missing declaration here is a real hole rather than a duplicate.
  std::vector<ResourceClaim> declare_resources() const override;
  // ...and on the memory plan, where it read 0 MB.
  StageMemory declare_memory() const override;

  // Latch a `model-select` constant before the planning phase, so the
  // claim above is made against the model this graph will actually run
  // rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job  process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()   const noexcept { return _hf_dir; }
  std::uint64_t      emitted()  const noexcept { return _emitted; }

  // How the groups this stage is given relate. STREAM: they are
  // consecutive pieces of one source, restored as one pass -- the
  // projection keeps its carries across them, the rows say which of the
  // stream's chunks they hold, and the denoiser and the decode carry
  // their state too. CLIPS: the source is cut into overlapping clips,
  // each restored cold, whose shared frames the decode cross-fades.
  // AUTO (the default) streams when the box can hold a stream's carries
  // at the source's geometry (genai::flashvsr_stream_fits) and restores
  // clips otherwise.
  enum class Continuity { kAuto, kStream, kClips };
  // What the first group decided, for tests. kAuto until then.
  Continuity continuity() const noexcept { return _mode; }

private:
#ifdef VPIPE_BUILD_APPLE_SILICON
  bool ensure_loaded_();

  std::unique_ptr<genai::FlashVsrLqProj> _proj;

  // ---- the SOURCE, between groups ---------------------------------------
  //
  // Both modes cut the source themselves out of one buffer of frames
  // received and not yet restored: a stream into chunks, clips into
  // overlapping clips.
  struct ColorStats { float v[6]; };
  void reset_stream_();
  // Every beat the frames received so far complete; with `flush`, the
  // source has ended and its end is padded with its last frame, as the
  // reference pads it, so every real frame is restored.
  std::vector<std::unique_ptr<TensorBeatPayload>> drain_(bool flush,
                                                         std::string* err);
  // A stream: run every chunk the frames received complete, appending
  // their rows.
  bool drive_chunks_(bool flush,
                     std::vector<metal_compute::SharedBuffer>* rows,
                     int* chunk0, int* chunks, std::string* err);
  // Clips: the next one, or null when its frames are not here yet.
  std::unique_ptr<TensorBeatPayload> next_clip_(bool flush,
                                                std::string* err);
  // Chunk `c` of a run that starts at source frame `origin`, through the
  // projection in the reference's call pattern.
  bool feed_chunk_(long long origin, int c,
                   std::vector<metal_compute::SharedBuffer>* rows,
                   std::string* err);
  void pad_to_(long long end);
  void drop_before_(long long g);          // frames and their statistics
  void drop_before_stats_(long long g);
  std::unique_ptr<TensorBeatPayload>
  rows_beat_(const std::vector<metal_compute::SharedBuffer>& rows) const;
  FlexData color_refs_(long long frame0, long long n) const;
  std::unique_ptr<TensorBeatPayload>
  stream_beat_(const std::vector<metal_compute::SharedBuffer>& rows,
               int chunk0, int chunks, bool final_beat);
  std::unique_ptr<TensorBeatPayload>
  clip_beat_(const std::vector<metal_compute::SharedBuffer>& rows,
             int chunks, long long S, long long valid, int head, int tail);
  // A frame's channel statistics in [-1, 1], as the reference's colour
  // fix computes them.
  static ColorStats frame_stats_(const std::uint8_t* px, int H, int W);

  bool                      _open = false;    // a source is open
  int                       _sw = 0, _sh = 0; // ...and its geometry
  // Frames received and not yet done with, from source frame `_pend_g0`.
  std::vector<std::uint8_t> _pending;
  long long                 _pend_g0 = 0;
  long long                 _received = 0;   // real frames, this source
  long long                 _padded = 0;     // ...and the end's padding
  std::vector<std::uint8_t> _last_frame;
  // The colour statistics of frames not yet restored, from `_stats_g0`.
  std::vector<ColorStats>   _stats;
  long long                 _stats_g0 = 0;
  bool                      _overlap_noted = false;
  // A stream: the next chunk.
  int                       _next_chunk = 0;
  // Clips: where the next starts, where the last one's restored frames
  // end, and how many have gone.
  long long                 _clip_s = 0;
  long long                 _covered = 0;
  int                       _clips = 0;
#endif
  std::string   _hf_dir;
  bool          _model_latched = false;
  bool          _load_attempted = false;
  std::uint64_t _emitted = 0;
  Continuity    _cfg_mode = Continuity::kAuto;
  Continuity    _mode = Continuity::kAuto;
  bool          _color_fix = true;
  int           _clip_frames = 41;
  int           _blend_frames = 8;
};

}  // namespace vpipe

#endif
