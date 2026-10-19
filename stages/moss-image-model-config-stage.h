#ifndef VPIPE_STAGES_MOSS_IMAGE_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_MOSS_IMAGE_MODEL_CONFIG_STAGE_H

#include "stages/latent-preview.h"
#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the MOSS-Image parameters of a `generate-image` request.
//
// Guidance, the negative branch's skipped blocks, a shift override,
// runtime LoRA slots and the live preview. None is needed for an
// ordinary run: generate-image runs the reference's defaults (CFG 3
// against the empty prompt with block 22 skipped, the checkpoint's shift
// 5); this stage is for a graph that wants something else.
//
// See stages/model-config-source.h for the beat contract and the trigger
// rule (unwired = one beat for the run; wired = one per inbound beat).
class MossImageModelConfigStage final
  : public ModelConfigSourceStage<MossImageModelConfigStage> {
public:
  static constexpr const char* kTypeName = "moss-image-model-config";

  MossImageModelConfigStage(const SessionContextIntf* session,
                            std::string               id,
                            std::vector<InEdge>       iports,
                            FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;

private:
  LatentPreviewSpec _preview;
};

}  // namespace vpipe

#endif
