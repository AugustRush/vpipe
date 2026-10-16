#include "stages/audio-separate-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include "generative-models/bs-roformer/bs-roformer-config.h"
#include "generative-models/bs-roformer/metal-bs-roformer.h"
#include "generative-models/shared/overlap-demix.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

constexpr unsigned kModelPort = 1;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the separator: a models-DB key, a directory holding an MSST "
          "checkpoint and its YAML, or the .ckpt itself. "
          "pcunwa/BS-Roformer-Resurrection is catalogued as two models, "
          "vocals and instrumental. OPTIONAL: a model-select source on "
          "the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "bs-roformer"},
  {.key = "checkpoint", .type = ConfigType::String, .required = false,
   .doc = "the checkpoint FILE, for a directory holding several (the "
          "Resurrection repository holds two) when the registry record "
          "does not pin one",
   .def_str = ""},
  {.key = "config", .type = ConfigType::String, .required = false,
   .doc = "the MSST YAML, when it is not named for the checkpoint "
          "(<stem>-Config.yaml, <stem>.yaml, ...) or the only one beside "
          "it. Relative to the checkpoint's directory",
   .def_str = ""},
  {.key = "stem", .type = ConfigType::String, .required = false,
   .doc = "which source goes to the `stem` oport, by name (from the "
          "YAML's training section) or index. Default: the model's first "
          "-- for a single-stem model, its target",
   .def_str = ""},
  {.key = "num_overlap", .type = ConfigType::Int, .required = false,
   .doc = "how many chunks cover each sample. 0 (default) keeps the "
          "YAML's (2 for Resurrection). The cost grows linearly with it; "
          "more overlap smooths the seams between chunks",
   .def_int = 0},
};

const PortSpec kIports[] = {
  {.name = "pcm",
   .doc = "f32 PCM, [N] mono or PLANAR [C, N], at the model's sample rate "
          "(sideband `sample_rate`). Streamed: any beat length",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples, audio-pcm", .clock_group = 0},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source; "
          "overrides hf_dir",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "stem",
   .doc = "the separated source, planar [C, N] f32 at the input's rate. "
          "Trails the input by about one model chunk, in beats of the "
          "stage's own length",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples, audio-pcm", .clock_group = 1},
  {.name = "residual",
   .doc = "the mix minus that source -- the instrumental of a vocals "
          "model. Same layout and beats as `stem`",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples, audio-pcm", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "audio-separate",
  .doc       = "Separates a source -- vocals, say -- from music with a "
               "BS-RoFormer (an MSST checkpoint + YAML), emitting the "
               "source and the mix without it. Streams: the track is cut "
               "into the model's overlapping chunks and cross-faded back "
               "together, identically to separating it whole.",
  .display_name = "Audio Separate",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

bool
sb_num_(const FlexData& sb, const char* key, double* out)
{
  if (!sb.is_object()) { return false; }
  const auto o = sb.as_object();
  if (!o.contains(key)) { return false; }
  const FlexData v = o.at(key);
  if (v.is_int() || v.is_uint() || v.is_real()) {
    *out = v.as_real(0.0);
    return true;
  }
  return false;
}

}  // namespace

AudioSeparateStage::AudioSeparateStage(const SessionContextIntf* s,
                                       std::string id,
                                       std::vector<InEdge> ip, FlexData cfg)
  : TypedStage<AudioSeparateStage>(s, std::move(id), std::move(ip),
                                   std::move(cfg))
{
  allocate_oports(spec().oports.size());
  _hf_dir = attr_str("hf_dir");
  _checkpoint = attr_str("checkpoint");
  _config = attr_str("config");
  _stem_sel = attr_str("stem");
  _num_overlap = (int)attr_int("num_overlap");
  if (_num_overlap < 0 || _num_overlap > 64) {
    fail_config(fmt("AudioSeparateStage('{}'): num_overlap {} is outside "
                    "[0, 64] (0 keeps the YAML's)", this->id(),
                    _num_overlap));
  }
}

AudioSeparateStage::~AudioSeparateStage() = default;

const StageSpec&
AudioSeparateStage::spec() const noexcept
{
  return kSpec;
}

void
AudioSeparateStage::apply_constant(unsigned iport, const FlexData& beat)
{
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

bool
AudioSeparateStage::resolve_files_(std::string* ckpt, std::string* yaml,
                                   std::string* err) const
{
  if (_hf_dir.empty()) {
    if (err != nullptr) { *err = "no model (hf_dir or the model iport)"; }
    return false;
  }
  // The record's pinned files say WHICH of a directory's models was
  // registered; a directory scan cannot.
  const ResolvedModel rm = resolve_model(session(), _hf_dir);
  genai::BsRoformerFiles f;
  if (!genai::resolve_bs_roformer_files(rm.dir, rm.files, _checkpoint,
                                        _config, &f, err)) {
    return false;
  }
  *ckpt = f.ckpt;
  *yaml = f.yaml;
  return true;
}

std::vector<ResourceClaim>
AudioSeparateStage::declare_resources() const
{
  std::vector<ResourceClaim> out;
  std::string ckpt, yaml, err;
  if (!resolve_files_(&ckpt, &yaml, &err)) { return out; }
  // The FILE, not the directory: a directory holding two models would
  // book both.
  for (auto& c : model_memory::weight_claims({ckpt})) {
    out.push_back(std::move(c));
  }
  genai::BsRoformerConfig cfg;
  if (genai::bs_roformer_config_from_file(yaml, &cfg, &err)) {
    // One forward's scratch plus the stream's buffers: a chunk and a
    // step of input, estimates and weights, and one chunk in and out.
    const std::size_t C = (std::size_t)cfg.chunk_size;
    const std::size_t S = (std::size_t)cfg.channels();
    const std::size_t K = (std::size_t)cfg.num_stems;
    const std::size_t stream = (1 + K) * S * (C + C) * 4 + (S + K * S) * C * 4;
    for (auto& c : model_memory::scratch_claims(
             fmt("audio-separate:{}", this->id())(),
             genai::MetalBsRoformer::scratch_bytes(cfg) + stream, {})) {
      out.push_back(std::move(c));
    }
  }
  return out;
}

StageMemory
AudioSeparateStage::declare_memory() const
{
  StageMemory m;
  std::string ckpt, yaml, err;
  if (!resolve_files_(&ckpt, &yaml, &err)) { return m; }
  // Held for the stage's life, as read: the f32 file converts to f16,
  // so this books twice what stays resident -- the read itself is a
  // transient at the file's size.
  m.hold(ckpt, model_memory::dir_weights_bytes(ckpt));
  genai::BsRoformerConfig cfg;
  if (genai::bs_roformer_config_from_file(yaml, &cfg, &err)) {
    m.scratch = genai::MetalBsRoformer::scratch_bytes(cfg);
  }
  return m;
}

void
AudioSeparateStage::ensure_loaded_()
{
  if (_load_attempted) { return; }
  _load_attempted = true;
  std::string err;
  if (!resolve_files_(&_ckpt, &_yaml, &err)) {
    session()->error(fmt("AudioSeparateStage('{}'): {}; inert", this->id(),
                         err));
    return;
  }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    session()->error(fmt("AudioSeparateStage('{}'): no metal-compute "
                         "backend; inert", this->id()));
    return;
  }
  genai::BsRoformerConfig cfg;
  if (!genai::bs_roformer_config_from_file(_yaml, &cfg, &err)) {
    session()->error(fmt("AudioSeparateStage('{}'): {}; inert", this->id(),
                         err));
    return;
  }
  if (_num_overlap > 0) { cfg.num_overlap = _num_overlap; }
  if (cfg.normalize) {
    // MSST's whole-track normalisation needs the whole track's mean and
    // deviation before the first chunk, which a stream does not have.
    // The band split normalises every band of every frame itself, so
    // the network is indifferent to the input's scale; what is not
    // applied is the subtraction of the track's DC offset.
    session()->info(fmt("AudioSeparateStage('{}'): the YAML asks for "
                        "inference.normalize; a stream cannot know the "
                        "whole track's statistics, so it is not applied "
                        "(the model is scale-invariant; only a DC offset "
                        "would differ)", this->id()));
  }
  // Which stem goes to oport0.
  _stem = 0;
  if (!_stem_sel.empty()) {
    const auto it = std::find(cfg.stems.begin(), cfg.stems.end(), _stem_sel);
    if (it != cfg.stems.end()) {
      _stem = (int)(it - cfg.stems.begin());
    } else {
      char* end = nullptr;
      const long v = std::strtol(_stem_sel.c_str(), &end, 10);
      if (end == nullptr || *end != '\0' || v < 0 || v >= cfg.num_stems) {
        std::string names;
        for (const auto& n : cfg.stems) {
          names += names.empty() ? n : ", " + n;
        }
        session()->error(fmt("AudioSeparateStage('{}'): stem '{}' is not "
                             "one of the model's ({}); inert", this->id(),
                             _stem_sel, names));
        return;
      }
      _stem = (int)v;
    }
  }
  _stem_name = cfg.stems[(std::size_t)_stem];

  const auto t0 = std::chrono::steady_clock::now();
  _ws = genai::open_weight_set(_ckpt, session());
  if (_ws == nullptr) {
    session()->error(fmt("AudioSeparateStage('{}'): cannot open '{}'; "
                         "inert", this->id(), _ckpt));
    return;
  }
  _model = genai::MetalBsRoformer::load(_ws, mc, cfg, &err);
  if (_model == nullptr) {
    _ws.reset();
    session()->error(fmt("AudioSeparateStage('{}'): '{}': {}; inert",
                         this->id(), _ckpt, err));
    return;
  }
  const double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
  session()->info(fmt(
      "AudioSeparateStage('{}'): BS-RoFormer loaded in {:.2f} s -- {} "
      "layers of width {}, {} bands, {} stem(s); {:.1f} s chunks at {} Hz, "
      "overlap {}; {} MB of weights; GEMMs on {}, attention on {}. "
      "stem = {}, residual = the mix without it", this->id(), s, cfg.depth,
      cfg.dim, cfg.bands(), cfg.num_stems,
      (double)cfg.chunk_size / cfg.sample_rate, cfg.sample_rate,
      cfg.num_overlap, _model->weight_bytes() >> 20,
      _model->matrix_core_gemm() ? "matmul2d" : "steel",
      _model->nax_attention() ? "NAX" : "steel", _stem_name));
}

Job
AudioSeparateStage::initialize(RuntimeContext& ctx)
{
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

void
AudioSeparateStage::reset_run_state()
{
  // A relaunch is a new track: nothing of the last one's stream may
  // leak into it. The model stays loaded.
  _demix.reset();
  _model_latched = false;
  _sb_seed = FlexData{};
  _have_seed = _have_ts = _failed = _mono_noted = false;
  _first_ts_us = 0;
  _rate = 0;
  _emitted = 0;
  _chunks = 0;
  _gpu_ms = 0.0;
}

bool
AudioSeparateStage::accept_(const TensorBeatPayload& tb,
                            std::vector<float>* planar, std::int64_t* n)
{
  const genai::BsRoformerConfig& c = _model->config();
  if (tb.dtype != TensorBeat::DType::F32) {
    session()->warn(fmt("AudioSeparateStage('{}'): PCM must be f32, got "
                        "{}; dropping a beat", this->id(),
                        TensorBeat::name_of(tb.dtype)));
    return false;
  }
  const int rank = (int)tb.shape.size();
  const int ch = rank == 2 ? (int)tb.shape[0] : 1;
  const std::int64_t len = rank == 2 ? tb.shape[1]
                         : rank == 1 ? tb.shape[0] : 0;
  if ((rank != 1 && rank != 2) || ch < 1 || ch > 2 || len <= 0) {
    session()->warn(fmt("AudioSeparateStage('{}'): PCM must be [N] or "
                        "[C, N] with one or two channels; dropping a beat",
                        this->id()));
    return false;
  }
  double rv = 0.0;
  if ((!sb_num_(tb.sideband, "sample_rate", &rv) &&
       !sb_num_(tb.sideband, "sr", &rv)) || !(rv > 0.0)) {
    session()->error(fmt("AudioSeparateStage('{}'): the beat carries no "
                         "`sample_rate`, and a separator fed the wrong rate "
                         "hears the wrong instruments; stopping",
                         this->id()));
    _failed = true;
    return false;
  }
  if ((int)rv != c.sample_rate) {
    session()->error(fmt("AudioSeparateStage('{}'): the input is {} Hz and "
                         "the model was trained at {} Hz. Resample upstream "
                         "(audio-to-pcm / audio-temporal-resample "
                         "output_sample_rate: {}); stopping", this->id(),
                         (int)rv, c.sample_rate, c.sample_rate));
    _failed = true;
    return false;
  }
  if (_rate != 0 && (int)rv != _rate) {
    _failed = true;
    return false;
  }
  _rate = (int)rv;
  const auto host = tb.materialize_contiguous();
  const float* p = reinterpret_cast<const float*>(host.data());
  const int S = c.channels();
  planar->assign((std::size_t)S * len, 0.0f);
  for (int s = 0; s < S; ++s) {
    float* d = planar->data() + (std::size_t)s * len;
    if (ch == S) {
      std::memcpy(d, p + (std::size_t)s * len, (std::size_t)len * 4);
    } else if (ch == 1) {
      // Mono into a stereo model: the same channel twice, as MSST does.
      std::memcpy(d, p, (std::size_t)len * 4);
    } else {
      // Stereo into a mono model: their mean.
      for (std::int64_t i = 0; i < len; ++i) {
        d[i] = 0.5f * (p[i] + p[len + i]);
      }
    }
  }
  if (ch != S && !_mono_noted) {
    _mono_noted = true;
    session()->info(fmt("AudioSeparateStage('{}'): {}-channel input to a "
                        "{}-channel model; {}", this->id(), ch, S,
                        ch == 1 ? "duplicated" : "averaged"));
  }
  if (!_have_seed) {
    _sb_seed = tb.sideband;
    _have_seed = true;
    double ts = 0.0;
    _have_ts = sb_num_(tb.sideband, "timestamp_us", &ts) && ts >= 0.0;
    _first_ts_us = _have_ts ? (std::uint64_t)ts : 0;
  }
  *n = len;
  return true;
}

Job
AudioSeparateStage::emit_(RuntimeContext& ctx)
{
  if (!_demix || _demix->available() == 0) { co_return; }
  std::vector<float> stems, mix;
  const std::int64_t n = _demix->take(&stems, &mix);
  if (n <= 0) { co_return; }
  const genai::BsRoformerConfig& c = _model->config();
  const int S = c.channels();
  const std::size_t plane = (std::size_t)S * n;
  auto beat = [&](const float* src, bool minus_mix) {
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = TensorBeat::DType::F32;
    t->shape = S == 1 ? std::vector<std::int64_t>{n}
                      : std::vector<std::int64_t>{S, n};
    t->resize_contiguous(plane);
    float* d = t->as_f32();
    for (std::size_t i = 0; i < plane; ++i) {
      d[i] = minus_mix ? mix[i] - src[i] : src[i];
    }
    FlexData sb = _have_seed && _sb_seed.is_object() ? _sb_seed
                                                     : FlexData::make_object();
    auto o = sb.as_object();
    o.insert_or_assign("sample_rate", FlexData::make_int(_rate));
    o.insert_or_assign("duration_us",
                       FlexData::make_uint((std::uint64_t)n * 1000000ULL /
                                           (std::uint64_t)_rate));
    if (_have_ts) {
      o.insert_or_assign("timestamp_us",
                         FlexData::make_uint(_first_ts_us +
                                             _emitted * 1000000ULL /
                                                 (std::uint64_t)_rate));
    }
    o.erase("stacked");
    t->sideband = std::move(sb);
    return t;
  };
  const float* stem = stems.data() + (std::size_t)_stem * plane;
  auto a = beat(stem, false);
  auto b = beat(stem, true);
  _emitted += (std::uint64_t)n;
  co_await ctx.write(0, std::move(a));
  co_await ctx.write(1, std::move(b));
}

Job
AudioSeparateStage::process(RuntimeContext& ctx)
{
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
  auto fwd = [this](const float* in, float* out) {
    std::string err;
    const bool ok = _model->forward(in, out, &err);
    if (!ok) {
      session()->error(fmt("AudioSeparateStage('{}'): {}", this->id(), err));
    }
    _gpu_ms += _model->last_timing().gpu_ms;
    return ok;
  };
  auto in = co_await ctx.read(0);
  if (!in) {
    if (_model && _demix && !_failed) {
      _failed = !_demix->finish(fwd);
      _chunks = _demix->chunks();
      co_await emit_(ctx);
      session()->info(fmt(
          "AudioSeparateStage('{}'): {:.1f} s separated in {} chunks, "
          "{:.1f} s of GPU", this->id(),
          _rate > 0 ? (double)_emitted / _rate : 0.0, _chunks,
          _gpu_ms / 1000.0));
    }
    ctx.signal_done();
    co_return;
  }
  if (!_model || _failed) { co_return; }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tb == nullptr) {
    session()->warn(fmt("AudioSeparateStage('{}'): expected a TensorBeat, "
                        "got {}; skipping", this->id(), in->describe()));
    co_return;
  }
  std::vector<float> planar;
  std::int64_t n = 0;
  if (!accept_(*tb, &planar, &n)) { co_return; }
  if (!_demix) {
    const genai::BsRoformerConfig& c = _model->config();
    genai::OverlapDemix::Params p;
    p.chunk = c.chunk_size;
    p.num_overlap = c.num_overlap;
    p.channels = c.channels();
    p.stems = c.num_stems;
    _demix = std::make_unique<genai::OverlapDemix>(p);
  }
  if (!_demix->push(planar.data(), n, n, fwd)) {
    _failed = true;
    co_return;
  }
  _chunks = _demix->chunks();
  co_await emit_(ctx);
}

VPIPE_REGISTER_STAGE(AudioSeparateStage)
VPIPE_REGISTER_SPEC(AudioSeparateStage, kSpec)

}  // namespace vpipe
