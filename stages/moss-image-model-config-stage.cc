#include "stages/moss-image-model-config-stage.h"
#include "stages/latent-preview.h"
#include "stages/lora-slots.h"
#include "stages/model-registry.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "guidance_scale", .type = ConfigType::Real, .required = false,
   .doc = "classifier-free guidance: v = v_neg + g (v - v_neg), against "
          "the negative conditioning (the EMPTY prompt's unless one is "
          "wired). The reference's default is 3; 1 runs one forward a step "
          "and no negative at all. Unset = the default"},
  {.key = "uncond_skip_layers", .type = ConfigType::Array, .required = false,
   .doc = "zero-based main blocks the NEGATIVE branch runs without -- the "
          "reference's default [22] (one block of 42 skipped in the uncond "
          "forward, still two forwards a step). [] disables the skip. Unset "
          "= the default"},
  {.key = "slg_layers", .type = ConfigType::Array, .required = false,
   .doc = "skip-layer guidance on the POSITIVE branch: a THIRD forward a "
          "step that runs the prompt without these zero-based blocks, and "
          "v += slg_scale (v_cond - v_skipped). The reference's guide "
          "suggests [24] at scale 3 between 0.01 and 1. Unset or [] = off"},
  {.key = "slg_scale", .type = ConfigType::Real, .required = false,
   .doc = "the skip-layer guidance's strength; 0 (the default) is off. "
          "With guidance_scale 1 the prediction is the skipped forward "
          "plus slg_scale times the difference"},
  {.key = "slg_start", .type = ConfigType::Real, .required = false,
   .doc = "where in the denoise (step / steps, from the noise) both the "
          "negative branch's skip AND the skip-layer guidance start: "
          "outside [slg_start, slg_stop) plain CFG runs. Default 0"},
  {.key = "slg_stop", .type = ConfigType::Real, .required = false,
   .doc = "where they stop (exclusive). Default 1"},
  {.key = "shift", .type = ConfigType::Real, .required = false,
   .doc = "the schedule's shift: a shift-s grid over the steps, snapped to "
          "the training grid's nearest nodes. Unset = the checkpoint's own "
          "(5, scheduler/scheduler_config.json). Higher spends more of the "
          "steps at high noise. The step COUNT is generate-image's `steps` "
          "(or a scheduler-select's); the reference defaults it to 50"},
  {.key = "lora", .type = ConfigType::String, .required = false,
   .doc = "a LoRA applied AT RUNTIME -- every adapted projection computes "
          "W x + scale * B (A x). A registered model, a directory holding "
          "one .safetensors, or a path to one. Module names are the "
          "checkpoint's own (layers.N.proj.q / k / v / o / ffn_up_gate / "
          "ffn_down), optionally under a `diffusion_model.` or "
          "`transformer.` prefix; an adapter for another model binds "
          "nothing and says so. LOAD-time: read before the DiT is built, so "
          "a beat that changes it afterwards is reported and ignored (the "
          "strength stays live)",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "moss-image-lora",
   .is_path = true, .path_filter = "weights"},
  {.key = "lora_scale", .type = ConfigType::Real, .required = false,
   .doc = "the adapter's strength, applied PER FORWARD: live, so it can be "
          "swept without a reload. 1.0 = as trained; 0 skips both adapter "
          "GEMMs, so off is exactly off",
   .def_real = 1.0},
  lora_slots::path_key(1, "moss-image-lora"), lora_slots::scale_key(1),
  lora_slots::path_key(2, "moss-image-lora"), lora_slots::scale_key(2),
  lora_slots::path_key(3, "moss-image-lora"), lora_slots::scale_key(3),
  lora_slots::path_key(4, "moss-image-lora"), lora_slots::scale_key(4),
  lora_slots::path_key(5, "moss-image-lora"), lora_slots::scale_key(5),
  lora_slots::path_key(6, "moss-image-lora"), lora_slots::scale_key(6),
  lora_slots::path_key(7, "moss-image-lora"), lora_slots::scale_key(7),
  // MOSS-Image decodes through FLUX.2's VAE (its small decoder reads the
  // same latent), so FLUX.2's TAE (madebyollin's taef2) previews it.
  {.key = latent_preview::kVaeKey, .type = ConfigType::String,
   .required = false, .doc = latent_preview::kVaeDoc,
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "flux2-tae"},
  {.key = latent_preview::kEveryKey, .type = ConfigType::Int,
   .required = false, .doc = latent_preview::kEveryDoc, .def_int = 1},
  {.key = latent_preview::kMaxEdgeKey, .type = ConfigType::Int,
   .required = false, .doc = latent_preview::kMaxEdgeDoc, .def_int = 512},
};
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc = "OPTIONAL beat that gates re-emitting the config (a chrono tick, a "
          "prompt source, a feedback loop). Any payload -- receipt is the "
          "signal. Unwired, the stage emits once for the run",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "model_config",
   .doc = "MOSS-Image parameters as one FlexData object {model_family: "
          "moss-image, +guidance_scale, +uncond_skip_layers, +slg_*, "
          "+shift, +lora "
          ".. +lora8 and their scales, +preview_vae and its knobs}, for a "
          "generate-image model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "moss-image-model-config",
  .doc       = "Source: the MOSS-Image parameters -- guidance, the negative "
               "branch's skipped blocks, skip-layer guidance, the schedule's "
               "shift, runtime LoRA slots and the live preview. None is "
               "needed for an ordinary "
               "run: generate-image runs the reference's defaults. One beat "
               "then done; with a trigger iport, one beat per inbound beat.",
  .display_name = "MOSS-Image Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

MossImageModelConfigStage::MossImageModelConfigStage(
    const SessionContextIntf* s, std::string id, std::vector<InEdge> iports,
    FlexData config)
  : ModelConfigSourceStage<MossImageModelConfigStage>(
        s, std::move(id), std::move(iports), std::move(config))
{
  auto in = this->config().as_object();
  if (in.contains("guidance_scale") &&
      in.at("guidance_scale").as_real(-1.0) < 0.0) {
    fail_config(fmt("MossImageModelConfigStage('{}'): guidance_scale must "
                    "be >= 0", this->id()));
  }
  if (in.contains("shift") && in.at("shift").as_real(0.0) <= 0.0) {
    fail_config(fmt("MossImageModelConfigStage('{}'): shift must be > 0",
                    this->id()));
  }
  if (in.contains("slg_scale") && in.at("slg_scale").as_real(-1.0) < 0.0) {
    fail_config(fmt("MossImageModelConfigStage('{}'): slg_scale must be "
                    ">= 0", this->id()));
  }
  {
    const double a = in.contains("slg_start")
                         ? in.at("slg_start").as_real(-1.0) : 0.0;
    const double b = in.contains("slg_stop")
                         ? in.at("slg_stop").as_real(-1.0) : 1.0;
    if (!(0.0 <= a && a <= b && b <= 1.0)) {
      fail_config(fmt("MossImageModelConfigStage('{}'): need 0 <= "
                      "slg_start <= slg_stop <= 1", this->id()));
    }
  }
  for (const char* key : {"uncond_skip_layers", "slg_layers"}) {
    if (!in.contains(key)) { continue; }
    const FlexData a = in.at(key);
    bool ok = a.is_array();
    if (ok) {
      auto arr = a.as_array();
      for (std::size_t i = 0; i < arr.size(); ++i) {
        ok = ok && arr.at(i).as_int(-1) >= 0;
      }
    }
    if (!ok) {
      fail_config(fmt("MossImageModelConfigStage('{}'): {} must be a list "
                      "of block indices >= 0", this->id(), key));
    }
  }

  _preview.vae      = attr_str(latent_preview::kVaeKey);
  _preview.every    = (int)attr_int(latent_preview::kEveryKey);
  _preview.max_edge = (int)attr_int(latent_preview::kMaxEdgeKey);
  if (_preview.every < 0 || _preview.max_edge < 0) {
    fail_config(fmt(
        "MossImageModelConfigStage('{}'): preview_every and "
        "preview_max_edge must be >= 0 (got {} / {})", this->id(),
        _preview.every, _preview.max_edge));
  }
  allocate_oports(spec().oports.size());
}

const StageSpec&
MossImageModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
MossImageModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("moss-image");
  // Each key written only when SET: a default emitted as though it were
  // a choice reads downstream as "the graph asked for it".
  auto in = config().as_object();
  auto o  = fd.as_object();
  for (const char* k : {"guidance_scale", "uncond_skip_layers", "shift",
                        "slg_layers", "slg_scale", "slg_start",
                        "slg_stop"}) {
    if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
  }
  for (int i = 0; i < lora_slots::kSlots; ++i) {
    for (std::string_view k : {lora_slots::kPath[i], lora_slots::kScale[i]}) {
      if (in.contains(k)) { o.insert_or_assign(k, in.at(k)); }
    }
  }
  _preview.emit(fd);
  return fd;
}

VPIPE_REGISTER_STAGE(MossImageModelConfigStage)
VPIPE_REGISTER_SPEC(MossImageModelConfigStage, kSpec)

}  // namespace vpipe
