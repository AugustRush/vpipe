#include "stages/flashvsr-src-encoder-stage.h"

#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/weight-set.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

constexpr unsigned kSourcePort = 0;
constexpr unsigned kModelPort  = 1;
constexpr unsigned kPicturePort = 1;
// Beats the pictures output holds before this stage waits for vae-decode:
// a beat's colour-fix pictures and its decoder's, for a couple of beats.
constexpr unsigned kPictureDepth = 4;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the FlashVSR model directory (its converted layout: the denoiser, "
          "the source projection and the fixed context as one checkpoint). A "
          "model-select source on iport1 overrides it, which is how the "
          "shipped graph shares one model choice across the encoder, the "
          "denoiser and the VAE"},
  {.key = "continuity", .type = ConfigType::String, .required = false,
   .doc = "how the groups this stage is given relate: auto | stream | "
          "clips. `stream`: they are consecutive pieces of ONE source, "
          "restored as one pass -- the projection, the denoiser's key/value "
          "windows and the VAE decode all carry their state from one group "
          "to the next, so there is no change of texture where a group "
          "ends; groups need no overlap. `clips`: the source is cut into "
          "overlapping CLIPS (clip_frames, blend_frames), each restored "
          "cold, whose shared frames the decode cross-fades -- a gradual "
          "transition rather than a jump, for a box too small for a "
          "stream. `auto` (default) streams when the box can "
          "hold a stream's carries at the source's size -- ~20 GB at "
          "1920x1152, ~2.5 GB at 512x512 -- and restores clips otherwise",
   .def_str = "auto"},
  {.key = "clip_frames", .type = ConfigType::Int, .required = false,
   .doc = "when restoring CLIPS: source frames per clip -- whole chunks, "
          "8k+1 and at least 25 for FlashVSR-v1.1, 16k+1 for FlashVSR+ "
          "(rounded up to that). Each restores its first clip_frames - 4 "
          "(the last four are the model's look-ahead). Longer clips cost "
          "the overlap less often and hold more while they run (0, the "
          "default: 41 for v1.1, 49 for FlashVSR+)",
   .def_int = 0},
  {.key = "blend_frames", .type = ConfigType::Int, .required = false,
   .doc = "when restoring CLIPS: restored frames each clip shares with the "
          "next, which the decode cross-fades from one to the other (a "
          "raised cosine) so a cut is a transition rather than a jump; 0 "
          "cuts hard. Each costs a frame restored twice (8)",
   .def_int = 8},
  {.key = "color_fix", .type = ConfigType::Bool, .required = false,
   .doc = "correct each restored frame's colour toward its source frame "
          "(color_fix_method says how). Unset, it follows the release's "
          "own reference: on for FlashVSR-v1.1, which applies its fix by "
          "default, and off for FlashVSR+, whose fix is opt-in (its --cf)",
   .def_bool = true},
  {.key = "color_fix_method", .type = ConfigType::String, .required = false,
   .doc = "auto | adain | wavelet. `adain`: each channel's mean and spread "
          "moved to the source frame's -- v1.1's fix, six numbers a frame "
          "on the sideband. `wavelet`: the frame's LOW FREQUENCIES taken "
          "from the source picture, its detail kept -- FlashVSR+'s fix, "
          "which corrects a cast where it is; the pictures travel to "
          "vae-decode on this stage's color_reference output, which must be "
          "wired to vae-decode's color_reference input. `auto` (default): "
          "the release's own, adain for v1.1 and wavelet for FlashVSR+",
   .def_str = "auto"},
  {.key = "wavelet_levels", .type = ConfigType::Int, .required = false,
   .doc = "the wavelet fix's levels: low-pass passes of a 3x3 Gaussian "
          "with taps 1, 2, 4, ... pixels apart. More moves coarser colour "
          "only (5, the reference's)",
   .def_int = 5},
  {.key = "decoder", .type = ConfigType::String, .required = false,
   .doc = "auto | vae | tiny: what vae-decode decodes the restored latents "
          "with. `vae`: the Wan VAE. `tiny`: the reference's tiny decoder "
          "(TCDecoder.ckpt beside the model), a small TAEHV decoder that "
          "reads the source pictures beside each latent -- several times "
          "faster than the VAE and a fraction of its memory, which lets a "
          "smaller box restore a source as one stream. Its pictures travel "
          "on this stage's color_reference output, which must be wired to "
          "vae-decode's color_reference input. `auto` (default): the "
          "release's own -- tiny for FlashVSR+ when the file is there, the "
          "VAE for v1.1",
   .def_str = "auto"},
  {.key = "temporal_compression", .type = ConfigType::Int,
   .required = false,
   .doc = "source frames per latent frame: 4 (default), 2 or 1. Below 4 "
          "every source frame but the first is fed to the model 2 or 4 "
          "times, so fast motion gets more latent frames, and the decode "
          "merges each run back into one frame -- FlashVSR+'s "
          "large-motion option, at 2x or 4x the denoise and decode work",
   .def_int = 4},
  {.key = "temporal_merge", .type = ConfigType::String, .required = false,
   .doc = "how the decode merges a run of repeated frames back: gaussian "
          "(default; weights over the run, temporal_sigma wide) or first "
          "(the run's first frame)",
   .def_str = "gaussian"},
  {.key = "temporal_sigma", .type = ConfigType::Real, .required = false,
   .doc = "the gaussian merge's width, in frames of the run (1.0)",
   .def_real = 1.0},
};
const PortSpec kIports[] = {
  {.name = "source",
   .doc = "the low-quality CLIP in pixels: planar U8 RGB [frames, 3, H, W] "
          "with `fps` on the sideband, which is the shape temporal-stack "
          "builds. REQUIRED -- this family conditions on its source and "
          "nothing else, so an unwired port is a refusal and not a fallback: "
          "a super-resolver handed no source would produce a confident "
          "upscale of noise",
   .type = &typeid(TensorBeatPayload),
   .tags = "video", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "conditioning",
   .doc = "bf16 [row_frames * tokens, hidden]: the source projected onto the "
          "denoiser's own token grid, one row frame per FOUR source frames. "
          "This is ORDINARY conditioning -- it feeds generate-video's "
          "existing conditioning iport, which needed no change to accept it. "
          "The sideband carries the geometry the rows belong to "
          "{source_width, source_height, row_frames} and where they sit: "
          "the stream chunks they hold, whether they continue the last "
          "beat or end the source, how many of the frames they restore "
          "are real, how they overlap their neighbours, and each frame's "
          "colour reference",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "color_reference",
   .doc = "the source pictures a conditioning beat's decode needs, planar "
          "U8 [n, 3, H, W], right after it: those it restores, for the "
          "WAVELET colour fix, then those its decode covers, for the TINY "
          "decoder. Wire it to vae-decode's color_reference input; written "
          "only when one of the two runs and this is wired. It waits for "
          "vae-decode a few beats deep, so the source cannot run ahead of "
          "the restoration",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "flashvsr-src-encoder",
  .doc       = "FlashVSR's source encoder: a low-quality video clip in "
               "PIXELS through the model's learned causal projection, into "
               "the conditioning rows its denoiser adds to the residual "
               "stream. Model-specific -- this family has no text encoder at "
               "all, because its text context is a constant its checkpoint "
               "carries, and its conditioning is the clip itself. Feed "
               "generate-video's conditioning iport.",
  .display_name = "FlashVSR Source Encoder",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

FlashVsrSrcEncoderStage::FlashVsrSrcEncoderStage(
    const SessionContextIntf* session, std::string id,
    std::vector<InEdge> iports, FlexData config)
    : TypedStage<FlashVsrSrcEncoderStage>(session, std::move(id),
                                          std::move(iports),
                                          std::move(config))
{
  // The spec DESCRIBES the oport; this is what CREATES it. Without it
  // the stage has zero oports, and every graph that wires its
  // conditioning onward -- the shipped flashvsr-restore one included --
  // is refused at pipeline_from_spec before anything loads.
  allocate_oports(spec().oports.size());
  // The pictures are a frame's worth each at the output size, and nothing
  // else paces this stage: without backpressure it would decode the whole
  // source ahead of the restoration and queue it here.
  set_oport_policy(kPicturePort,
                   {kPictureDepth, OverrunPolicy::Backpressure});
  // Deferred validation: constructors never throw here, so a bad config
  // is recorded and the runtime skips the stage at launch.
  _hf_dir = attr_str("hf_dir");
  const std::string c = attr_str("continuity");
  if (c.empty() || c == "auto") {
    _cfg_mode = Continuity::kAuto;
  } else if (c == "stream") {
    _cfg_mode = Continuity::kStream;
  } else if (c == "clips") {
    _cfg_mode = Continuity::kClips;
  } else {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): continuity must be "
                    "auto, stream or clips; got '{}'", this->id(), c));
  }
  // Unset, the colour fix follows the release, which is known only once
  // the model is: see ensure_loaded_().
  {
    // this->: the constructor's own `config` parameter, moved from by
    // now, shadows the accessor.
    const FlexData& c = this->config();
    _color_fix_set = c.is_object() && c.as_object().contains("color_fix");
  }
  _color_fix = attr_bool("color_fix");
  _cfg_clip_frames = (int)attr_int("clip_frames");
  _blend_frames = (int)attr_int("blend_frames");
  // The clip's frame rule is the release's, checked at load; what any
  // release needs is a whole chunk past the look-ahead.
  if (_cfg_clip_frames != 0 && _cfg_clip_frames < 17) {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): clip_frames must be at "
                    "least 17, or 0 for the release's own; got {}",
                    this->id(), _cfg_clip_frames));
  } else if (_blend_frames < 0) {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): blend_frames must be "
                    ">= 0; got {}", this->id(), _blend_frames));
  }
  const std::string m = attr_str("color_fix_method");
  if (m.empty() || m == "auto") {
    _cfg_method = ColorFix::kAuto;
  } else if (m == "adain") {
    _cfg_method = ColorFix::kAdain;
  } else if (m == "wavelet") {
    _cfg_method = ColorFix::kWavelet;
  } else {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): color_fix_method must "
                    "be auto, adain or wavelet; got '{}'", this->id(), m));
  }
  const std::string d = attr_str("decoder");
  if (d.empty() || d == "auto") {
    _cfg_decoder = Decoder::kAuto;
  } else if (d == "vae") {
    _cfg_decoder = Decoder::kVae;
  } else if (d == "tiny") {
    _cfg_decoder = Decoder::kTiny;
  } else {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): decoder must be auto, "
                    "vae or tiny; got '{}'", this->id(), d));
  }
  _wavelet_levels = (int)attr_int("wavelet_levels");
  if (_wavelet_levels < 1 || _wavelet_levels > 12) {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): wavelet_levels must be "
                    "in [1, 12]; got {}", this->id(), _wavelet_levels));
  }
  // THE MERGE WEIGHTS, the reference's own: a gaussian over the run's
  // positions centred on its middle, normalised -- or the first frame.
  _tc = (int)attr_int("temporal_compression");
  if (_tc != 4 && _tc != 2 && _tc != 1) {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): temporal_compression "
                    "must be 4, 2 or 1; got {}", this->id(), _tc));
    _tc = 4;
  }
  _r = 4 / _tc;
  const std::string merge = attr_str("temporal_merge");
  const double sigma = attr_real("temporal_sigma");
  if (merge != "gaussian" && merge != "first" && !merge.empty()) {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): temporal_merge must be "
                    "gaussian or first; got '{}'", this->id(), merge));
  } else if (!(sigma > 0.0)) {
    fail_config(fmt("FlashVsrSrcEncoderStage('{}'): temporal_sigma must be "
                    "> 0; got {}", this->id(), sigma));
  } else if (_r > 1) {
    _merge_w.assign((std::size_t)_r, 0.0f);
    if (merge == "first") {
      _merge_w[0] = 1.0f;
    } else {
      // In f32, as the reference computes them.
      float sum = 0.0f;
      for (int k = 0; k < _r; ++k) {
        const float x =
            ((float)k - (float)(_r - 1) / 2.0f) / (float)sigma;
        _merge_w[(std::size_t)k] = std::exp(-0.5f * (x * x));
        sum += _merge_w[(std::size_t)k];
      }
      for (float& w : _merge_w) { w /= sum; }
    }
  }
}

FlashVsrSrcEncoderStage::~FlashVsrSrcEncoderStage() = default;

const StageSpec&
FlashVsrSrcEncoderStage::spec() const noexcept
{
  return kSpec;
}

void
FlashVsrSrcEncoderStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // The pre-launch twin of the runtime latch in process(): the same beat
  // and the same parse, early enough that declare_resources() sees the
  // model. Bookkeeping only -- nothing loads here.
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

std::vector<ResourceClaim>
FlashVsrSrcEncoderStage::declare_resources() const
{
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (root.empty()) { return {}; }
  // The projection's OWN set, which is the name the manager keys its load
  // by: the root when converted, LQ_proj_in.ckpt as published.
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout)) {
    return model_memory::weight_claims({root});
  }
  return model_memory::weight_claims({layout.source});
}

StageMemory
FlashVsrSrcEncoderStage::declare_memory() const
{
  // The projection, under the set it loads through -- the root when
  // converted, where the denoiser's holding names it too and the two merge
  // (that holding's floor counts the projection), and LQ_proj_in.ckpt as
  // published. It does not stream, so its floor is what it weighs. Its
  // WORKING SET is booked by the denoiser's family, which knows the clip
  // geometry this stage only sees at its first beat.
  StageMemory m;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (root.empty()) { return m; }
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout)) { return m; }
  const std::size_t b = genai::FlashVsrLqProj::weight_bytes(
      layout.source, layout.source_prefix);
  m.hold(layout.source, b, b);
  return m;
}

void
FlashVsrSrcEncoderStage::reset_run_state()
{
  // Let a relaunch load again if the weights are gone; keep the guard
  // set while they are still held, because reloading on top of a
  // resident copy is what doubles peak memory.
  if (_proj == nullptr) { _load_attempted = false; }
  _model_latched = false;
  _emitted = 0;
  // A relaunch is a new source, and re-decides how to restore it.
  reset_stream_();
  _mode = Continuity::kAuto;
  _overlap_noted = false;
  _refs_noted = false;
  _tiny_noted = false;
}

bool
FlashVsrSrcEncoderStage::ensure_loaded_()
{
  if (_proj != nullptr) { return true; }
  if (_load_attempted) { return false; }
  _load_attempted = true;
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (root.empty()) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): no model directory -- set hf_dir or "
        "wire a model-select source", this->id()));
    return false;
  }
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout)) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): '{}' is not a FlashVSR checkpoint -- "
        "neither the converted layout nor the published files; inert",
        this->id(), root));
    return false;
  }
  std::string err;
  _proj = genai::FlashVsrLqProj::load(
      genai::open_weight_set(layout.source, session()), mc, &err,
      layout.source_prefix);
  if (_proj == nullptr) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): source projection load failed for "
        "'{}': {}; inert", this->id(), root, err));
    return false;
  }
  // THE RELEASE SETS THE CADENCE: how many source frames open the
  // stream and how many each later chunk takes -- 25 then 8 for v1.1, 17
  // then 16 for FlashVSR+.
  _sch = genai::flashvsr_schedule(layout);
  // THE COLOUR FIX the release's reference applies: v1.1 its AdaIN, by
  // default; FlashVSR+ its wavelet one, only when asked (--cf).
  if (!_color_fix_set) { _color_fix = !_sch.is_plus(); }
  _method = !_color_fix ? ColorFix::kOff
          : _cfg_method != ColorFix::kAuto
              ? _cfg_method
              : (_sch.is_plus() ? ColorFix::kWavelet : ColorFix::kAdain);
  // THE DECODER the release's reference uses by default: FlashVSR+'s tiny
  // one, when its file came with the model; v1.1's VAE.
  if (_cfg_decoder == Decoder::kTiny && layout.tiny_decoder.empty()) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): decoder `tiny` asks for {} beside "
        "the model and '{}' has none; decoding with the VAE", this->id(),
        "TCDecoder.ckpt", root));
  }
  _tiny = !layout.tiny_decoder.empty() &&
          (_cfg_decoder == Decoder::kTiny ||
           (_cfg_decoder == Decoder::kAuto && _sch.is_plus()));
  if (_cfg_decoder == Decoder::kAuto && _sch.is_plus() &&
      layout.tiny_decoder.empty()) {
    // A download from before the catalogue pinned it.
    session()->info(fmt(
        "FlashVsrSrcEncoderStage('{}'): FlashVSR+ decodes with its tiny "
        "decoder by default, and {} is not beside '{}'; decoding with the "
        "VAE (fetching the model again adds it)", this->id(),
        "TCDecoder.ckpt", root));
  }
  // A clip is whole chunks: rounded UP to the release's rule, so a graph
  // written for one release runs on the other.
  _clip_frames = _cfg_clip_frames > 0 ? _cfg_clip_frames
                                      : (_sch.is_plus() ? 49 : 41);
  const int aligned = genai::MetalFlashVsrTransformer::align_frames(
      _sch, std::max(_clip_frames, (int)_sch.frames_through(0)));
  if (aligned != _clip_frames && _cfg_clip_frames > 0) {
    session()->info(fmt(
        "FlashVsrSrcEncoderStage('{}'): clip_frames {} is not whole chunks "
        "of this release; restoring clips of {}", this->id(), _clip_frames,
        aligned));
  }
  _clip_frames = aligned;
  // A clip restores (clip_frames - 4) / r source frames past the first
  // clip's lead; it must keep at least one of them its own.
  const int restores = (_clip_frames - 4) / _r;
  if (_blend_frames >= restores) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): blend_frames {} leaves a {}-frame "
        "clip (restoring {} source frames) nothing of its own; blending {}",
        this->id(), _blend_frames, _clip_frames, restores, restores - 1));
    _blend_frames = restores - 1;
  }
  session()->info(fmt(
      "FlashVsrSrcEncoderStage('{}'): {} source projection, {} wide, {} "
      "output layer(s), {}x{} pixel unshuffle -- no prompt and no text "
      "encoder; chunks of {} source frames{}{}{}", this->id(),
      layout.plus ? "FlashVSR+ " + layout.variant : std::string("v1.1"),
      _proj->config().out_dim, _proj->config().n_out_layers,
      _proj->config().shuffle_w, _proj->config().shuffle_h,
      4 * _sch.chunk_latents,
      _method == ColorFix::kAdain     ? ", AdaIN colour fix"
      : _method == ColorFix::kWavelet ? ", wavelet colour fix"
                                      : "",
      _r > 1 ? fmt(", temporal compression {} (each source frame fed {} "
                   "times, merged back by the decode)", _tc, _r)()
             : std::string(),
      _tiny ? ", the tiny decoder" : ""));
  return true;
}

Job
FlashVsrSrcEncoderStage::initialize(RuntimeContext& ctx)
{
  // With a model-select wired the choice only lands after the init
  // barrier, so the load waits for the first beat. Without one, hf_dir
  // is all there is and loading now keeps the first clip from paying
  // for it.
  if (!(ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort))) {
    ensure_loaded_();
  }
  co_return;
}

Job
FlashVsrSrcEncoderStage::process(RuntimeContext& ctx)
{
  if (ctx.num_iports() <= kSourcePort || !ctx.iport_connected(kSourcePort)) {
    session()->error(fmt(
        "FlashVsrSrcEncoderStage('{}'): nothing is wired to the source "
        "iport; inert", this->id()));
    ctx.signal_done();
    co_return;
  }
  // Latch the shared model once, before anything is read: the load needs
  // it and the first clip is already on its way.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      apply_model_select_beat(mfd->data, _hf_dir);
    }
    ensure_loaded_();
  }

  // The wavelet fix's and the tiny decoder's pictures go out beside the
  // rows, and only where vae-decode will read them.
  _refs_wired = ctx.has_consumers(kPicturePort);
  if (_tiny && !_refs_wired && !_tiny_noted && _proj != nullptr) {
    _tiny_noted = true;
    const auto m = fmt(
        "FlashVsrSrcEncoderStage('{}'): the tiny decoder reads the source "
        "pictures this stage sends on its color_reference output, and "
        "nothing is wired to it; vae-decode decodes with the VAE",
        this->id());
    if (_cfg_decoder == Decoder::kTiny) {
      session()->warn(m);
    } else {
      session()->info(m);
    }
  }
  if (_method == ColorFix::kWavelet && !_refs_wired && !_refs_noted &&
      _proj != nullptr) {
    _refs_noted = true;
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): the wavelet colour fix needs this "
        "stage's color_reference output wired to vae-decode's "
        "color_reference input; nothing is, so the source restores "
        "without a colour fix", this->id()));
  }

  auto cb = co_await ctx.read(kSourcePort);
  if (!cb) {
    // THE END OF THE SOURCE. What is still owed is restored now: every
    // chunk reads four frames past what it restores, so the end is padded
    // with the last frame, as the reference pads it.
    if (_open && _proj != nullptr) {
      std::string err;
      auto beats = drain_(/*flush=*/true, &err);
      if (!err.empty()) {
        session()->warn(fmt(
            "FlashVsrSrcEncoderStage('{}'): the source's end failed: {}",
            this->id(), err));
      }
      for (auto& b : beats) {
        co_await ctx.write(0, std::move(b.rows));
        if (b.refs) { co_await ctx.write(kPicturePort, std::move(b.refs)); }
        if (b.cond) { co_await ctx.write(kPicturePort, std::move(b.cond)); }
        ++_emitted;
      }
      reset_stream_();
    }
    ctx.signal_done();
    co_return;
  }
  if (_proj == nullptr && !ensure_loaded_()) {
    // Inert, but the clip is consumed: returning without reading makes
    // the runtime re-invoke immediately and the stage spins a core.
    co_return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(cb.get());
  if (tb == nullptr || tb->dtype != TensorBeat::DType::U8 ||
      tb->shape.size() != 4 || tb->shape[1] != 3) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): expected a planar U8 RGB "
        "[frames, 3, H, W] clip, got {}; dropping beat", this->id(),
        cb->describe()));
    co_return;
  }

  const int frames = (int)tb->shape[0];
  const int H = (int)tb->shape[2], W = (int)tb->shape[3];
  const int sh = _proj->config().shuffle_h, sw = _proj->config().shuffle_w;
  if (H % sh != 0 || W % sw != 0) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): clip is {}x{}; the projection "
        "unshuffles by {}x{}, so both axes must divide by it",
        this->id(), W, H, sw, sh));
    co_return;
  }
  const std::size_t frame_px = (std::size_t)3 * H * W;
  const auto bytes = tb->materialize_contiguous();
  if (bytes.size() < frame_px * (std::size_t)frames) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): clip is short of its own shape",
        this->id()));
    co_return;
  }

  // STREAM OR CLIPS, settled by the first group. Auto asks the one rule
  // the family's memory plan asks too, at this source's size.
  if (_mode == Continuity::kAuto) {
    if (_cfg_mode != Continuity::kAuto) {
      _mode = _cfg_mode;
    } else {
      // ...with the decode it will run: the tiny decoder's carries are a
      // fraction of the VAE's whole-plane ones.
      const std::string root = resolve_model_dir(session(), _hf_dir);
      std::size_t need = 0;
      const bool fits =
          genai::flashvsr_stream_fits(root, W, H, &need, tiny_on_());
      _mode = fits ? Continuity::kStream : Continuity::kClips;
      const std::size_t carry =
          genai::flashvsr_stream_bytes(root, W, H, 3.0, tiny_on_());
      const std::size_t ram = model_memory::phys_ram();
      if (fits) {
        session()->info(fmt(
            "FlashVsrSrcEncoderStage('{}'): restoring the source as ONE "
            "STREAM at {}x{} -- the projection, the denoiser and the decode "
            "carry ~{} MB of state from group to group (~{} MB in all of "
            "{} MB)", this->id(), W, H, carry >> 20, need >> 20, ram >> 20));
      } else {
        session()->info(fmt(
            "FlashVsrSrcEncoderStage('{}'): restoring the source as "
            "OVERLAPPING CLIPS of {} frames at {}x{}, each sharing {} with "
            "the next for the decode to cross-fade -- a stream carries ~{} "
            "MB between groups and needs ~{} MB in all, more than this {} MB "
            "box holds with headroom", this->id(), _clip_frames, W, H,
            _blend_frames, carry >> 20, need >> 20, ram >> 20));
      }
    }
  }

  // A source of another size is another source: the open one ends.
  if (_open && (W != _sw || H != _sh)) {
    std::string err;
    auto beats = drain_(/*flush=*/true, &err);
    for (auto& b : beats) {
      co_await ctx.write(0, std::move(b.rows));
      if (b.refs) { co_await ctx.write(kPicturePort, std::move(b.refs)); }
      if (b.cond) { co_await ctx.write(kPicturePort, std::move(b.cond)); }
      ++_emitted;
    }
    reset_stream_();
  }
  if (!_open) {
    _open = true;
    _sw = W;
    _sh = H;
  }
  // An OVERLAPPING group repeats frames the source already gave
  // (temporal-stack says how many); they are dropped rather than taken
  // twice. Both modes cut the source themselves and need no overlap.
  int repeated = 0;
  if (tb->sideband.is_object()) {
    FlexData sb = tb->sideband;              // as_object() is a view: keep it
    auto o = sb.as_object();
    if (o.contains(sideband::kRepeated)) {
      repeated = (int)o.at(sideband::kRepeated).as_int(0);
    }
  }
  const int skip = _received > 0 ? std::clamp(repeated, 0, frames) : 0;
  if (skip > 0 && !_overlap_noted) {
    _overlap_noted = true;
    session()->info(fmt(
        "FlashVsrSrcEncoderStage('{}'): groups overlap by {} frame(s); this "
        "stage cuts the source itself, so they are dropped rather than "
        "restored twice (temporal-stack overlap 0 saves reading them)",
        this->id(), skip));
  }
  for (int i = skip; i < frames; ++i) {
    const std::uint8_t* px = bytes.data() + (std::size_t)i * frame_px;
    // EXPANDED: the source's first frame once, every later one r times
    // (temporal compression below 4) -- the reference's own expansion.
    const int copies = _src_received == 0 ? 1 : _r;
    for (int k = 0; k < copies; ++k) {
      _pending.insert(_pending.end(), px, px + frame_px);
    }
    _received += copies;
    ++_src_received;
    // ...while what is restored is the source, a frame each.
    if (_method == ColorFix::kAdain) {
      _stats.push_back(frame_stats_(px, H, W));
    }
    if ((_method == ColorFix::kWavelet || _tiny) && _refs_wired) {
      _pics.emplace_back(px, px + frame_px);
    }
  }
  if (skip < frames) {
    const std::uint8_t* last =
        bytes.data() + (std::size_t)(frames - 1) * frame_px;
    _last_frame.assign(last, last + frame_px);
  }
  std::string err;
  auto beats = drain_(/*flush=*/false, &err);
  if (!err.empty()) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): source projection failed: {}; the "
        "source ends here", this->id(), err));
    reset_stream_();
  }
  for (auto& b : beats) {
    co_await ctx.write(0, std::move(b.rows));
    if (b.refs) { co_await ctx.write(kPicturePort, std::move(b.refs)); }
    if (b.cond) { co_await ctx.write(kPicturePort, std::move(b.cond)); }
    ++_emitted;
  }
}

std::vector<FlashVsrSrcEncoderStage::Out>
FlashVsrSrcEncoderStage::drain_(bool flush, std::string* err)
{
  std::vector<Out> out;
  if (_mode == Continuity::kStream) {
    std::vector<metal_compute::SharedBuffer> rows;
    int c0 = 0, n = 0;
    if (!drive_chunks_(flush, &rows, &c0, &n, err)) { return out; }
    if (n > 0) {
      Out b = stream_beat_(rows, c0, n, flush);
      if (b.rows) { out.push_back(std::move(b)); }
    }
    return out;
  }
  for (;;) {
    Out b = next_clip_(flush, err);
    if (!b.rows) { break; }
    out.push_back(std::move(b));
  }
  return out;
}

void
FlashVsrSrcEncoderStage::reset_stream_()
{
  _open = false;
  _sw = _sh = 0;
  _pending.clear();
  _pending.shrink_to_fit();
  _pend_g0 = 0;
  _received = 0;
  _src_received = 0;
  _padded = 0;
  _next_chunk = 0;
  _last_frame.clear();
  _stats.clear();
  _pics.clear();
  _rec_g0 = 0;
  _clip_s = 0;
  _covered = 0;
  _clips = 0;
}

FlashVsrSrcEncoderStage::ColorStats
FlashVsrSrcEncoderStage::frame_stats_(const std::uint8_t* px, int H, int W)
{
  // The reference's _calc_mean_std over the frame mapped to [-1, 1]: the
  // mean, and the square root of the UNBIASED variance plus 1e-5.
  ColorStats out{};
  const std::size_t n = (std::size_t)H * W;
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* p = px + (std::size_t)c * n;
    std::uint64_t sum = 0, sq = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += p[i];
      sq += (std::uint64_t)p[i] * p[i];
    }
    // x = u * 2/255 - 1, so the mean maps affinely and the variance by
    // (2/255)^2.
    const double mu = (double)sum / (double)n;
    const double var_u =
        n > 1 ? ((double)sq - (double)n * mu * mu) / (double)(n - 1) : 0.0;
    const double k = 2.0 / 255.0;
    out.v[c] = (float)(mu * k - 1.0);
    out.v[3 + c] = (float)std::sqrt(std::max(0.0, var_u) * k * k + 1e-5);
  }
  return out;
}

void
FlashVsrSrcEncoderStage::pad_to_(long long end)
{
  while (_received + _padded < end) {
    _pending.insert(_pending.end(), _last_frame.begin(), _last_frame.end());
    ++_padded;
  }
}

void
FlashVsrSrcEncoderStage::drop_before_(long long g)
{
  const std::size_t frame_px = (std::size_t)3 * _sh * _sw;
  if (g > _pend_g0) {
    const long long n = std::min<long long>(
        g - _pend_g0, (long long)(_pending.size() / frame_px));
    _pending.erase(_pending.begin(),
                   _pending.begin() +
                       (std::ptrdiff_t)(n * (long long)frame_px));
    _pend_g0 += n;
  }
  drop_records_before_(src_of_(g));
}

void
FlashVsrSrcEncoderStage::drop_records_before_(long long s)
{
  if (s <= _rec_g0) { return; }
  const long long n = s - _rec_g0;
  if (!_stats.empty()) {
    _stats.erase(_stats.begin(),
                 _stats.begin() + (std::ptrdiff_t)std::min<long long>(
                                      n, (long long)_stats.size()));
  }
  if (!_pics.empty()) {
    _pics.erase(_pics.begin(),
                _pics.begin() + (std::ptrdiff_t)std::min<long long>(
                                    n, (long long)_pics.size()));
  }
  _rec_g0 = s;
}

bool
FlashVsrSrcEncoderStage::feed_chunk_(long long origin, int c,
                                     std::vector<metal_compute::SharedBuffer>*
                                         rows,
                                     std::string* err)
{
  // THE REFERENCE'S CALL PATTERN, from `origin`: the opening chunk feeds
  // a single frame that seeds the carries and yields nothing, then four
  // frames per latent frame -- frames [0, 25) as seven slices for v1.1,
  // [0, 17) as five for FlashVSR+; every later chunk feeds four per
  // latent frame -- v1.1's chunk c [8c + 17, 8c + 25) as two slices,
  // FlashVSR+'s [16c + 1, 16c + 17) as four. Both releases' references
  // drive their projection exactly so.
  const std::size_t frame_px = (std::size_t)3 * _sh * _sw;
  auto feed = [&](long long lo, long long hi) {
    int rf = 0;
    return _proj->stream_forward(
        _pending.data() + (std::size_t)(origin + lo - _pend_g0) * frame_px,
        (int)(hi - lo), _sh, _sw, rows, &rf, err);
  };
  long long b = c == 0 ? 1 : _sch.frames_through(c - 1);
  if (c == 0 && !feed(0, 1)) { return false; }
  const int n = c == 0 ? _sch.open_latents : _sch.chunk_latents;
  for (int j = 0; j < n; ++j, b += 4) {
    if (!feed(b, b + 4)) { return false; }
  }
  return true;
}

// A STREAM's chunks: every chunk the frames received so far complete.
bool
FlashVsrSrcEncoderStage::drive_chunks_(
    bool flush, std::vector<metal_compute::SharedBuffer>* rows, int* chunk0,
    int* chunks, std::string* err)
{
  *chunk0 = _next_chunk;
  *chunks = 0;
  if (!_open || _received == 0) { return true; }
  if (flush) {
    // Enough chunks that the restored frames reach the last real one:
    // chunk c restores [restored_begin(c), restored_end(c)) -- v1.1's
    // opening one [0, 21), its later ones eight more each; FlashVSR+'s
    // [0, 13), then sixteen more -- and reads four frames past that.
    const long long open_end = _sch.restored_end(0);
    const long long step = 4LL * _sch.chunk_latents;
    const int last = _received <= open_end
                         ? 0
                         : (int)((_received - open_end + step - 1) / step);
    if (last < _next_chunk) { return true; }
    pad_to_(_sch.frames_through(last));
  }
  const long long avail = _received + _padded;
  for (;;) {
    const int c = _next_chunk;
    const long long end = _sch.frames_through(c);
    if (end > avail) { break; }
    if (!feed_chunk_(0, c, rows, err)) { return false; }
    ++_next_chunk;
    ++*chunks;
    // Everything before `end` is through the projection; its carries
    // hold what the next chunk reads of it.
    const std::size_t frame_px = (std::size_t)3 * _sh * _sw;
    const long long n = end - _pend_g0;
    _pending.erase(_pending.begin(),
                   _pending.begin() +
                       (std::ptrdiff_t)(n * (long long)frame_px));
    _pend_g0 = end;
  }
  return true;
}

// The next of a source's OVERLAPPING CLIPS, or null when its frames are
// not here yet (or, flushing, when nothing is left to restore). Clip k
// reads frames [S, S + L) and restores [S, S + L - 4); the next starts
// `blend` restored frames before this one's end, so the decode can
// cross-fade them -- the same frames twice, cold and warm, which turns a
// cut into a transition. That is what a box too small for a stream gets;
// it costs blend + 4 frames of every clip twice.
FlashVsrSrcEncoderStage::Out
FlashVsrSrcEncoderStage::next_clip_(bool flush, std::string* err)
{
  if (!_open || _received == 0) { return {}; }
  const int L = _clip_frames;
  const long long S = _clip_s;
  // What a clip restores, in decoded (expanded) frames: all but its four
  // of look-ahead -- cut back, after the source's first frame, to WHOLE
  // RUNS of r, so the next clip starts on a run and no merge straddles a
  // cut. With no temporal compression nothing is cut.
  const long long lead = S == 0 ? 1 : 0;
  const long long out_len = lead + ((L - 4 - lead) / _r) * _r;
  if (_received + _padded < S + L) {
    // Not here yet -- unless the source has ended and some of it is
    // still unrestored, in which case its end is padded.
    if (!flush || _received <= _covered) { return {}; }
    pad_to_(S + L);
  }
  // LAST only if its restored frames reach the source's end. A clip can
  // need padding for its four frames of look-ahead alone and still leave
  // real frames to the next -- which then needs its tail to blend into.
  const bool last = flush && S + out_len >= _received;
  const int chunks =
      genai::MetalFlashVsrTransformer::chunks_of(_sch, L);
  std::vector<metal_compute::SharedBuffer> rows;
  _proj->reset();
  for (int c = 0; c < chunks; ++c) {
    if (!feed_chunk_(S, c, &rows, err)) { return {}; }
  }
  const long long valid = std::min(S + out_len, _received) - S;
  // The cross-fade counts SOURCE frames: what the decode emits.
  const int head = _clips > 0 ? (int)((_covered - S) / _r) : 0;
  // Whether a clip follows: one does whenever restored frames remain past
  // this one's end, which a clip that ran on real frames cannot know yet
  // -- it says so, and the decode lets the frames go if none comes.
  const int tail = last ? 0 : _blend_frames;
  Out out = clip_beat_(rows, chunks, S, valid, head, tail);
  ++_clips;
  _covered = S + out_len;
  _clip_s = S + out_len - (long long)_blend_frames * _r;
  drop_before_(_clip_s);
  if (last) {
    // Nothing is owed past this clip; the padding is done with.
    _pending.clear();
  }
  return out;
}

std::unique_ptr<TensorBeatPayload>
FlashVsrSrcEncoderStage::rows_beat_(
    const std::vector<metal_compute::SharedBuffer>& rows) const
{
  const int dim = _proj->config().out_dim;
  const std::size_t tok = (std::size_t)(_sh / _proj->config().shuffle_h) *
                          (_sw / _proj->config().shuffle_w);
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::Bf16;
  out->shape = {(std::int64_t)(rows.size() * tok), (std::int64_t)dim};
  out->resize_contiguous(rows.size() * tok * (std::size_t)dim);
  auto* dst = (std::uint8_t*)out->as_bf16();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::memcpy(dst + i * tok * (std::size_t)dim * 2, rows[i].contents(),
                tok * (std::size_t)dim * 2);
  }
  // A fresh payload's sideband is NULL, and as_object() on it throws.
  out->sideband = FlexData::make_object();
  auto o = out->sideband.as_object();
  o.insert_or_assign("source_width", FlexData::make_int(_sw));
  o.insert_or_assign("source_height", FlexData::make_int(_sh));
  o.insert_or_assign("row_frames", FlexData::make_int((int)rows.size()));
  return out;
}

FlexData
FlashVsrSrcEncoderStage::color_refs_(long long frame0, long long n) const
{
  FlexData refs = FlexData::make_array();
  for (long long f = frame0; f < frame0 + n; ++f) {
    const long long i = f - _rec_g0;
    if (i < 0 || i >= (long long)_stats.size()) { break; }
    FlexData one = FlexData::make_array();
    for (float v : _stats[(std::size_t)i].v) {
      one.as_array().push_back(FlexData::make_real(v));
    }
    refs.as_array().push_back(one);
  }
  return refs;
}

std::unique_ptr<TensorBeatPayload>
FlashVsrSrcEncoderStage::pictures_(long long s0, long long n,
                                   const char* what) const
{
  const std::size_t frame_px = (std::size_t)3 * _sh * _sw;
  const long long i0 = s0 - _rec_g0;
  if (n <= 0 || i0 < 0 || i0 + n > (long long)_pics.size()) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): source frames {}..{} are no longer "
        "held for {}; this beat goes without", this->id(), s0, s0 + n - 1,
        what));
    return nullptr;
  }
  auto p = std::make_unique<TensorBeatPayload>();
  p->dtype = TensorBeat::DType::U8;
  p->shape = {(std::int64_t)n, 3, _sh, _sw};
  p->resize_contiguous((std::size_t)n * frame_px);
  for (long long k = 0; k < n; ++k) {
    std::memcpy(p->as_u8() + (std::size_t)k * frame_px,
                _pics[(std::size_t)(i0 + k)].data(), frame_px);
  }
  p->sideband = FlexData::make_object();
  return p;
}

void
FlashVsrSrcEncoderStage::finish_beat_(long long e0, long long valid,
                                      long long decoded, long long s0,
                                      long long n, Out* out) const
{
  auto o = out->rows->sideband.as_object();
  // How many DECODED frames are real; the decode trims to them first.
  o.insert_or_assign(sideband::kValidFrames,
                     FlexData::make_int((std::int64_t)std::max(0LL, valid)));
  // ...then merges each run of r back into a source frame, the source's
  // first frame (fed once) passing alone.
  if (_r > 1) {
    FlexData w = FlexData::make_array();
    for (float x : _merge_w) {
      w.as_array().push_back(FlexData::make_real(x));
    }
    o.insert_or_assign(sideband::kMergeWeights, w);
    o.insert_or_assign(sideband::kMergeLead,
                       FlexData::make_int(e0 == 0 ? 1 : 0));
  }
  if (_method == ColorFix::kAdain) {
    o.insert_or_assign(sideband::kColorRef, color_refs_(s0, n));
  } else if (_method == ColorFix::kWavelet && _refs_wired && n > 0) {
    // The source pictures of the frames this beat restores, for the
    // decode's wavelet fix: one beat beside this one.
    out->refs = pictures_(s0, n, "the wavelet colour fix");
    if (out->refs) {
      o.insert_or_assign(sideband::kColorRefFrames,
                         FlexData::make_int((std::int64_t)n));
      o.insert_or_assign(sideband::kColorRefLevels,
                         FlexData::make_int(_wavelet_levels));
    }
  }
  if (tiny_on_() && decoded > 0) {
    // THE TINY DECODER's pictures: the source of every frame this beat's
    // decode covers -- past the frames it restores, to the end of its last
    // latent frame, the source's last frame standing in for its padding.
    // One per SOURCE frame; the decode maps its expanded frames onto them
    // by the merge's own rule.
    const long long last =
        std::min(src_of_(e0 + decoded - 1), _src_received - 1);
    const long long m = last - s0 + 1;
    out->cond = pictures_(s0, m, "the tiny decoder");
    if (out->cond) {
      o.insert_or_assign(sideband::kDecoder, FlexData::make_string("tiny"));
      o.insert_or_assign(sideband::kDecoderCondFrames,
                         FlexData::make_int((std::int64_t)m));
    }
  }
}

FlashVsrSrcEncoderStage::Out
FlashVsrSrcEncoderStage::clip_beat_(
    const std::vector<metal_compute::SharedBuffer>& rows, int chunks,
    long long S, long long valid, int head, int tail)
{
  if ((int)rows.size() != _sch.latents(chunks)) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): a {}-chunk clip projected {} row "
        "frames, not {}; dropping it", this->id(), chunks, rows.size(),
        _sch.latents(chunks)));
    return {};
  }
  Out out;
  out.rows = rows_beat_(rows);
  auto o = out.rows->sideband.as_object();
  // A clip is a stream of its own, opened and ended in one beat.
  o.insert_or_assign(genai::flashvsr_stream::kChunk0, FlexData::make_int(0));
  o.insert_or_assign(genai::flashvsr_stream::kChunks,
                     FlexData::make_int(chunks));
  o.insert_or_assign(sideband::kStreamContinues, FlexData::make_bool(false));
  o.insert_or_assign(sideband::kStreamFinal, FlexData::make_bool(true));
  if (head > 0) {
    o.insert_or_assign(sideband::kBlendHead, FlexData::make_int(head));
  }
  if (tail > 0) {
    o.insert_or_assign(sideband::kBlendTail, FlexData::make_int(tail));
  }
  const long long s0 = src_of_(S);
  const long long n = src_of_(S + valid) - s0;
  // The decode covers every latent frame of the clip: 4T - 3 frames.
  finish_beat_(S, valid, 4LL * _sch.latents(chunks) - 3, s0, n, &out);
  session()->log_debug(fmt(
      "FlashVsrSrcEncoderStage('{}'): clip {} restores frames {}..{} "
      "(blend {} in, {} out)", this->id(), _clips, s0, s0 + n - 1, head,
      tail));
  return out;
}

FlashVsrSrcEncoderStage::Out
FlashVsrSrcEncoderStage::stream_beat_(
    const std::vector<metal_compute::SharedBuffer>& rows, int chunk0,
    int chunks, bool final_beat)
{
  // The opening chunk's row frames (v1.1 six, FlashVSR+ four) and each
  // later chunk's (two, four): a short count means the projection did
  // not do what the pattern says.
  const std::size_t want =
      (std::size_t)(chunk0 == 0 ? _sch.latents(chunks)
                                : _sch.chunk_latents * chunks);
  if (rows.size() != want) {
    session()->warn(fmt(
        "FlashVsrSrcEncoderStage('{}'): chunks {}+{} projected {} row "
        "frames, not {}; dropping them", this->id(), chunk0, chunks,
        rows.size(), want));
    return {};
  }
  Out out;
  out.rows = rows_beat_(rows);
  // The frames these chunks restore, in the stream: [frame0, end)
  // decoded, of which the real ones end at the last frame received --
  // and as the SOURCE, [s0, s0 + n). A chunk ends on a whole run of r
  // (four frames a latent frame, past the source's first), so the two
  // timelines agree at every cut.
  const long long frame0 = _sch.restored_begin(chunk0);
  const long long end = _sch.restored_end(chunk0 + chunks - 1);
  const long long valid = std::max(0LL, std::min(end, _received) - frame0);
  const long long s0 = src_of_(frame0);
  const long long n = src_of_(frame0 + valid) - s0;
  auto o = out.rows->sideband.as_object();
  o.insert_or_assign(genai::flashvsr_stream::kChunk0,
                     FlexData::make_int(chunk0));
  o.insert_or_assign(genai::flashvsr_stream::kChunks,
                     FlexData::make_int(chunks));
  o.insert_or_assign(sideband::kStreamContinues,
                     FlexData::make_bool(chunk0 > 0));
  o.insert_or_assign(sideband::kStreamFinal, FlexData::make_bool(final_beat));
  finish_beat_(frame0, valid, end - frame0, s0, n, &out);
  // What has been restored no longer needs its records.
  drop_records_before_(src_of_(end));
  session()->log_debug(fmt(
      "FlashVsrSrcEncoderStage('{}'): stream chunks {}..{} -> source frames "
      "{}..{}{}", this->id(), chunk0, chunk0 + chunks - 1, s0, s0 + n - 1,
      final_beat ? " (final)" : ""));
  return out;
}

#else   // !VPIPE_BUILD_APPLE_SILICON

std::vector<ResourceClaim>
FlashVsrSrcEncoderStage::declare_resources() const
{
  return {};
}

StageMemory
FlashVsrSrcEncoderStage::declare_memory() const
{
  return {};
}

void
FlashVsrSrcEncoderStage::reset_run_state()
{
}

Job
FlashVsrSrcEncoderStage::initialize(RuntimeContext&)
{
  session()->error(fmt(
      "FlashVsrSrcEncoderStage('{}'): the source projection runs on the "
      "metal-compute backend; this build has none", this->id()));
  co_return;
}

Job
FlashVsrSrcEncoderStage::process(RuntimeContext& ctx)
{
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(FlashVsrSrcEncoderStage)
VPIPE_REGISTER_SPEC(FlashVsrSrcEncoderStage, kSpec)

}  // namespace vpipe
