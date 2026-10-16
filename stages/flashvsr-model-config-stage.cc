#include "stages/flashvsr-model-config-stage.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"

#include <string>
#include <utility>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "kv_ratio", .type = ConfigType::Real, .required = false,
   .doc = "temporal windows of EARLIER chunks each block keeps to attend "
          "across (the reference's 3 when unset). The largest allocation "
          "the model makes when it streams: v1.1 carries kv_ratio windows "
          "of keys and values per block between groups -- 9.6 GB at "
          "1920x1152 and 3. FlashVSR+ Turbo's 25 hybrid layers keep one "
          "whatever this says, so its stream carries 4.2 GB there. Lower it "
          "to fit a box; it shortens how far back the restoration can look "
          "for consistency",
   .def_real = 3.0},
  {.key = "sparse_ratio", .type = ConfigType::Real, .required = false,
   .doc = "how many key blocks the routing keeps, in the release's own "
          "terms. FlashVSR-v1.1 scales it by the frame area as its "
          "reference does, so one value means one selectivity at every "
          "resolution (2.0 when unset; 1.5 is faster, 2.0 more stable); "
          "FlashVSR+ takes it as it is (2.5 when unset)",
   .def_real = 2.0},
  {.key = "local_range", .type = ConfigType::Int, .required = false,
   .doc = "the always-attended spatial band, in window blocks (11 when "
          "unset; 9 for sharper detail, 11 for stability). FlashVSR+ "
          "Turbo's hybrid layers keep their own 3",
   .def_int = 11},
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
   .doc = "FlashVSR parameters as one FlexData object {model_family: "
          "flashvsr, +kv_ratio, +sparse_ratio, +local_range}, for a "
          "generate-video model_config iport",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "flashvsr-model-config",
  .doc       = "Source: the FlashVSR-specific parameters -- the key/value "
               "window the denoiser attends across chunks, which is the "
               "memory it needs, and the routing that decides which key "
               "blocks it reads. One beat then done; with a trigger iport, "
               "one beat per inbound beat.",
  .display_name = "FlashVSR Model Config",
  .category  = StageCategory::ModelSpecificConfig,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

FlashVsrModelConfigStage::FlashVsrModelConfigStage(
    const SessionContextIntf* s, std::string id, std::vector<InEdge> iports,
    FlexData config)
  : ModelConfigSourceStage<FlashVsrModelConfigStage>(s, std::move(id),
                                                     std::move(iports),
                                                     std::move(config))
{
  allocate_oports(spec().oports.size());
  // Refused at config time rather than clamped downstream: a window or a
  // band the routing cannot use is a graph mistake, and silently running
  // the default instead would report a run the graph did not ask for.
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) { return; }
  auto o = cfg.as_object();
  if (o.contains("kv_ratio") && o.at("kv_ratio").as_real(0.0) < 0.0) {
    fail_config(fmt("FlashVsrModelConfigStage('{}'): kv_ratio must be >= 0",
                    this->id()));
  }
  if (o.contains("sparse_ratio") &&
      o.at("sparse_ratio").as_real(0.0) <= 0.0) {
    fail_config(fmt(
        "FlashVsrModelConfigStage('{}'): sparse_ratio must be > 0",
        this->id()));
  }
  if (o.contains("local_range") && o.at("local_range").as_int(0) < 1) {
    fail_config(fmt(
        "FlashVsrModelConfigStage('{}'): local_range must be >= 1",
        this->id()));
  }
}

const StageSpec&
FlashVsrModelConfigStage::spec() const noexcept
{
  return kSpec;
}

FlexData
FlashVsrModelConfigStage::resolved_config() const
{
  FlexData fd = model_config::make_config("flashvsr");
  // ONLY WHAT THE GRAPH SET, for the reason the VOSR source gives: a
  // default emitted as a choice is indistinguishable from one.
  const FlexData& cfg = config();
  if (cfg.is_object()) {
    auto in = cfg.as_object();
    auto out = fd.as_object();
    for (const char* k : {"kv_ratio", "sparse_ratio", "local_range"}) {
      if (in.contains(k)) { out.insert_or_assign(k, in.at(k)); }
    }
  }
  return fd;
}

VPIPE_REGISTER_STAGE(FlashVsrModelConfigStage)
VPIPE_REGISTER_SPEC(FlashVsrModelConfigStage, kSpec)

}  // namespace vpipe
