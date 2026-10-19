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
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
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
//   oport1  color_reference  the source pictures a beat's decode needs:
//                    the wavelet colour fix's, then the tiny decoder's.
//                    It BACKPRESSURES (a few beats deep): vae-decode
//                    reads them as its latents arrive, so the source
//                    cannot run ahead of the restoration by more.
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
//   clip_frames, blend_frames (ints, 0 and 8) -- clips only: source
//              frames per clip (0: the release's own, 41 for v1.1 and 49
//              for FlashVSR+; rounded up to whole chunks), and restored
//              frames each shares with the next for the decode to
//              cross-fade (sideband::kBlendHead / kBlendTail). What
//              SAnodes-Streaming does for ComfyUI.
//   color_fix  (bool, default true) -- correct each restored frame's
//              colour toward its source frame, by the release's own fix
//              (color_fix_method): v1.1's AdaIN, FlashVSR+'s wavelet. On
//              for both: FlashVSR+'s reference makes it opt-in, and
//              without it the restored colours drift from the source's.
//   decoder    ("auto" | "vae" | "tiny") -- what vae-decode decodes the
//              latents with: the Wan VAE, or the reference's TINY
//              decoder (TCDecoder.ckpt beside the model), which reads the
//              source pictures this stage sends on `color_reference`
//              beside each latent. auto: the release's reference -- tiny
//              for FlashVSR+ when the file is there, the VAE for v1.1.
//
// BOTH RELEASES. FlashVSR-v1.1 and FlashVSR+ (Turbo, Base) project their
// source the same way and differ in its cadence -- the frames a chunk
// takes -- which the loaded checkpoint's schedule supplies
// (genai::flashvsr_schedule).
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

  // The colour fix: as configured (kAuto: the release's own method), and
  // as resolved once the release is loaded (kOff when the fix is off).
  enum class ColorFix { kAuto, kOff, kAdain, kWavelet };
  // What the loaded release resolved, for tests.
  ColorFix color_fix() const noexcept { return _method; }

private:
#ifdef VPIPE_BUILD_APPLE_SILICON
  bool ensure_loaded_();

  std::unique_ptr<genai::FlashVsrLqProj> _proj;

  // ---- the SOURCE, between groups ---------------------------------------
  //
  // Both modes cut the source themselves out of one buffer of frames
  // received and not yet restored: a stream into chunks, clips into
  // overlapping clips.
  //
  // TWO TIMELINES. What the model reads is the EXPANDED one: with temporal
  // compression below 4, every source frame but the first is repeated
  // r = 4 / compression times, so each latent frame covers fewer source
  // frames and fast motion gets more of them. Frame counts here --
  // pending, received, padded, chunk and clip bounds -- are expanded;
  // what is restored is the SOURCE, frame for frame, so the colour
  // references, the cross-fade and the output frames count source frames.
  // src_of_() maps an expanded boundary to the source boundary; cuts are
  // only ever made where the two agree (on whole runs of r).
  struct ColorStats { float v[6]; };
  // One beat for each output: the rows; for the wavelet colour fix the
  // source pictures they restore; for the tiny decoder the source
  // pictures its decode covers.
  struct Out {
    std::unique_ptr<TensorBeatPayload> rows;
    std::unique_ptr<TensorBeatPayload> refs;
    std::unique_ptr<TensorBeatPayload> cond;
  };
  void reset_stream_();
  // Every beat the frames received so far complete; with `flush`, the
  // source has ended and its end is padded with its last frame, as the
  // reference pads it, so every real frame is restored.
  std::vector<Out> drain_(bool flush, std::string* err);
  // A stream: run every chunk the frames received complete, appending
  // their rows.
  bool drive_chunks_(bool flush,
                     std::vector<metal_compute::SharedBuffer>* rows,
                     int* chunk0, int* chunks, std::string* err);
  // Clips: the next one, or nothing when its frames are not here yet.
  Out next_clip_(bool flush, std::string* err);
  // Chunk `c` of a run that starts at expanded frame `origin`, through
  // the projection in the reference's call pattern.
  bool feed_chunk_(long long origin, int c,
                   std::vector<metal_compute::SharedBuffer>* rows,
                   std::string* err);
  void pad_to_(long long end);
  // Expanded frames before `g`, and the per-SOURCE-frame records before
  // its source frame.
  void drop_before_(long long g);
  void drop_records_before_(long long s);
  long long src_of_(long long e) const noexcept
  {
    return e <= 0 ? 0 : 1 + (e - 1) / _r;
  }
  std::unique_ptr<TensorBeatPayload>
  rows_beat_(const std::vector<metal_compute::SharedBuffer>& rows) const;
  FlexData color_refs_(long long frame0, long long n) const;
  // The output frames [s0, s0 + n) of a beat whose decode starts at
  // expanded frame `e0` and covers `decoded` frames, `valid` of them
  // real: the merge, the colour fix's references and the tiny decoder's
  // conditioning, announced on the rows and set on `out`.
  void finish_beat_(long long e0, long long valid, long long decoded,
                    long long s0, long long n, Out* out) const;
  // Pictures [s0, s0 + n) of the records, as one planar U8 beat; null
  // (and said) when they are no longer held.
  std::unique_ptr<TensorBeatPayload> pictures_(long long s0, long long n,
                                               const char* what) const;
  // Whether the tiny decoder runs: the release asks for it and the
  // pictures it reads have a consumer.
  bool tiny_on_() const noexcept { return _tiny && _refs_wired; }
  Out stream_beat_(const std::vector<metal_compute::SharedBuffer>& rows,
                   int chunk0, int chunks, bool final_beat);
  Out clip_beat_(const std::vector<metal_compute::SharedBuffer>& rows,
                 int chunks, long long S, long long valid, int head,
                 int tail);
  // A frame's channel statistics in [-1, 1], as the reference's colour
  // fix computes them.
  static ColorStats frame_stats_(const std::uint8_t* px, int H, int W);

  bool                      _open = false;    // a source is open
  int                       _sw = 0, _sh = 0; // ...and its geometry
  // Expanded frames received and not yet done with, from `_pend_g0`.
  std::vector<std::uint8_t> _pending;
  long long                 _pend_g0 = 0;
  long long                 _received = 0;   // real frames, expanded
  long long                 _src_received = 0;  // ...and as the source
  long long                 _padded = 0;     // ...and the end's padding
  std::vector<std::uint8_t> _last_frame;
  // Per SOURCE frame not yet restored, from `_rec_g0`: its colour
  // statistics (AdaIN) or its picture (the wavelet fix).
  std::vector<ColorStats>   _stats;
  std::vector<std::vector<std::uint8_t>> _pics;
  long long                 _rec_g0 = 0;
  bool                      _overlap_noted = false;
  // A stream: the next chunk.
  int                       _next_chunk = 0;
  // Clips: where the next starts, where the last one's restored frames
  // end (both expanded), and how many have gone.
  long long                 _clip_s = 0;
  long long                 _covered = 0;
  int                       _clips = 0;
  // Whether the color_reference output has a consumer, and whether its
  // absence has been said.
  bool                      _refs_wired = false;
  bool                      _refs_noted = false;
  bool                      _tiny_noted = false;
#endif
  std::string   _hf_dir;
  bool          _model_latched = false;
  bool          _load_attempted = false;
  std::uint64_t _emitted = 0;
  Continuity    _cfg_mode = Continuity::kAuto;
  Continuity    _mode = Continuity::kAuto;
  bool          _color_fix = true;
  bool          _color_fix_set = false;   // the graph said, either way
  int           _cfg_clip_frames = 0;     // 0: the release's own
  int           _clip_frames = 41;
  int           _blend_frames = 8;
  // The colour fix's method, as configured and as resolved for the
  // loaded release (off when the fix is), and the wavelet's levels.
  ColorFix      _cfg_method = ColorFix::kAuto;
  ColorFix      _method = ColorFix::kAdain;
  int           _wavelet_levels = 5;
  // The decoder: as configured, and resolved for the loaded release --
  // the tiny one where it asks for it and TCDecoder.ckpt is there.
  enum class Decoder { kAuto, kVae, kTiny };
  Decoder       _cfg_decoder = Decoder::kAuto;
  bool          _tiny = false;
  // TEMPORAL COMPRESSION: input frames per latent frame (4, 2 or 1); each
  // source frame but the first is fed r = 4 / it times, and the decode
  // merges every run of r back with `_merge_w`.
  int           _tc = 4;
  int           _r = 1;
  std::vector<float> _merge_w;
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The loaded release's cadence; v1.1's until a model loads.
  genai::MetalFlashVsrTransformer::Schedule _sch;
#endif
};

}  // namespace vpipe

#endif
