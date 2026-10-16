#ifndef VPIPE_STAGES_FLUX3_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_FLUX3_MODEL_CONFIG_STAGE_H

#include "stages/latent-preview.h"
#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the FLUX.3 Image Klein parameters of a `generate-image`
// request.
//
// The schedule's shape (`alpha`), runtime LoRA slots and the live
// preview. None is needed for an ordinary run: the checkpoint states its
// own sampler (alpha 6.93, 16 steps) and generate-image reads it; this
// stage is for a graph that wants something else.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
class Flux3ModelConfigStage final
  : public ModelConfigSourceStage<Flux3ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "flux3-model-config";

  Flux3ModelConfigStage(const SessionContextIntf* session,
                        std::string               id,
                        std::vector<InEdge>       iports,
                        FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;

private:
  double _alpha = 0.0;           // 0: the checkpoint's own
  LatentPreviewSpec _preview;
};

}  // namespace vpipe

#endif
