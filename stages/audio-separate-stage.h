#ifndef VPIPE_STAGES_AUDIO_SEPARATE_STAGE_H
#define VPIPE_STAGES_AUDIO_SEPARATE_STAGE_H

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace genai {
class WeightSet;
class MetalBsRoformer;
class OverlapDemix;
}

// Music source separation: PCM in, one SOURCE out -- vocals, say --
// and what is left when it is taken away.
//
// The separator is a BS-RoFormer (generative-models/bs-roformer/) in
// the format MSST (ZFTurbo/Music-Source-Separation-Training) publishes:
// a torch .ckpt beside the YAML that describes it.
// pcunwa/BS-Roformer-Resurrection is the catalogued pair -- a vocals
// model and an instrumental one in one repository.
//
//   iport0  pcm    f32 PCM TensorBeat, [N] mono or PLANAR [C, N], with
//                  `sample_rate` on the sideband. It must be the model's
//                  rate (44100 for every MSST model so far); resample
//                  upstream -- `audio-to-pcm` / `audio-temporal-resample`
//                  `output_sample_rate` -- because a separator fed the
//                  wrong rate hears the wrong instruments. Mono is
//                  duplicated into a stereo model's two channels, as
//                  MSST does.
//   iport1  model  OPTIONAL model-select reference; overrides `hf_dir`.
//
//   oport0  stem      the separated source, planar [C, N] at the input's
//                     rate: the model's own target (the Resurrection
//                     vocals model: vocals; its Inst model: the
//                     instrumental), or the one named by `stem`.
//   oport1  residual  the mix minus that source -- the instrumental of a
//                     vocals model, the vocals of an instrumental one.
//                     MSST's --extract_instrumental.
//
// STREAMED. The model sees a fixed chunk (17.8 s for Resurrection) and
// is bidirectional within it, so a track is cut into overlapping chunks
// and the estimates cross-faded back together by MSST's own rules
// (shared/overlap-demix.h). Chunks run as soon as their samples are in
// and each sample is emitted once every chunk covering it has run: the
// output trails the input by about a chunk, comes out in the stage's own
// rhythm (its oports are on a clock of their own), and is identical to
// separating the whole track at once -- however the input was cut into
// beats. A track shorter than the padding needs is handled at its end.
//
// Config:
//   hf_dir       a models-DB key, a directory or a .ckpt
//   checkpoint   the .ckpt in a directory holding several, when the
//                registry record does not pin one
//   config       the MSST YAML, when it is not named for the checkpoint
//   stem         which of a multi-stem model's sources goes to oport0;
//                default the model's first (its target)
//   num_overlap  chunks covering each sample; 0 (default) keeps the
//                YAML's. Cost grows with it linearly; MSST users run
//                2-8, trading time for smoother seams.
class AudioSeparateStage final : public TypedStage<AudioSeparateStage> {
public:
  static constexpr const char* kTypeName = "audio-separate";

  AudioSeparateStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);
  ~AudioSeparateStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;
  void reset_run_state() override;
  void apply_constant(unsigned iport, const FlexData& beat) override;

  std::vector<ResourceClaim> declare_resources() const override;
  StageMemory declare_memory() const override;

  const StageSpec& spec() const noexcept override;

  // Test-only.
  const std::string& checkpoint_path() const noexcept { return _ckpt; }
  const std::string& stem_name() const noexcept { return _stem_name; }
  std::int64_t chunks_run() const noexcept { return _chunks; }
  std::uint64_t samples_emitted() const noexcept { return _emitted; }

private:
  bool resolve_files_(std::string* ckpt, std::string* yaml,
                      std::string* err) const;
  void ensure_loaded_();
  bool accept_(const class TensorBeatPayload& tb, std::vector<float>* planar,
               std::int64_t* n);
  Job emit_(RuntimeContext& ctx);

  std::string _hf_dir, _checkpoint, _config, _stem_sel;
  int _num_overlap = 0;

  bool _model_latched = false, _load_attempted = false;
  std::string _ckpt, _yaml, _stem_name;
  int _stem = 0;
  std::shared_ptr<genai::WeightSet>       _ws;
  std::unique_ptr<genai::MetalBsRoformer> _model;
  std::unique_ptr<genai::OverlapDemix>    _demix;

  // The stream.
  FlexData _sb_seed;
  bool _have_seed = false, _have_ts = false, _failed = false;
  bool _mono_noted = false;
  std::uint64_t _first_ts_us = 0;
  int _rate = 0;
  std::uint64_t _emitted = 0;
  std::int64_t _chunks = 0;
  double _gpu_ms = 0.0;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_AUDIO_SEPARATE_STAGE_H
