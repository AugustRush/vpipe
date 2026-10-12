#ifndef VPIPE_STAGES_LORA_SLOTS_H
#define VPIPE_STAGES_LORA_SLOTS_H

// lora-slots.h -- the config keys of the runtime-LoRA SLOTS, written once
// for every stage that names them (generate-image, generate-video and the
// families' model-config sources).
//
// A DiT carries up to kSlots runtime adapters at once, each applied as its
// own pair of skinny GEMMs beside the base projection, each with its own
// LIVE strength. Slot 0 is the plain `lora` keys and slot i the
// `lora<i+1>` ones, so a graph that names one adapter reads exactly as it
// always did and every further one is additive. Which slot holds what is
// the caller's order, not a role the model knows.
//
// The number is the stages' half of a contract whose other half is
// genai::lora::Stack::kMax (generative-models/shared/runtime-lora.h). It
// is repeated rather than referenced because a stage's slot bookkeeping
// lives outside the Apple-Silicon guard, where that header does not
// exist; the stages that load a DiT static_assert the two agree.

#include "pipeline/stage-config.h"
#include "stages/model-registry.h"

#include <string_view>

namespace vpipe {
namespace lora_slots {

inline constexpr int kSlots = 8;

// Per slot: the adapter (a registry key, a directory holding one
// .safetensors, or a path to one), its strength, and -- MiniMax-H3 only
// -- the row order of its fused qkv projection.
inline constexpr std::string_view kPath[kSlots] = {
  "lora", "lora2", "lora3", "lora4", "lora5", "lora6", "lora7", "lora8",
};
inline constexpr std::string_view kScale[kSlots] = {
  "lora_scale",  "lora2_scale", "lora3_scale", "lora4_scale",
  "lora5_scale", "lora6_scale", "lora7_scale", "lora8_scale",
};
inline constexpr std::string_view kQkvLayout[kSlots] = {
  "lora_qkv_layout",  "lora2_qkv_layout", "lora3_qkv_layout",
  "lora4_qkv_layout", "lora5_qkv_layout", "lora6_qkv_layout",
  "lora7_qkv_layout", "lora8_qkv_layout",
};

// ---- the ConfigKey rows of slots 3..8 ----------------------------------
//
// The first two slots keep their own rows in each stage's table, with the
// docs that explain the pairing; the rest read exactly as `lora2` does,
// so one doc serves every stage and these build them. Each stage lists
// them right after its `lora2` rows, so the editor shows the slots in
// order.

inline constexpr std::string_view kPathDoc =
    "one more runtime LoRA slot, read exactly as `lora2` is: applied "
    "alongside the others (every adapted projection adds s B (A x) per "
    "slot), the same accepted forms, LOAD-time, with its own live "
    "strength in the matching `_scale` key. Up to eight adapters run at "
    "once; each costs one more pair of skinny GEMMs, and the base weight "
    "is still read once";
inline constexpr std::string_view kScaleDoc =
    "this slot's strength, per FORWARD and independent of every other "
    "slot's. 0 skips its two GEMMs";
inline constexpr std::string_view kQkvLayoutDoc =
    "the row order of this slot's FUSED attn.qkv_proj, exactly as "
    "`lora_qkv_layout` is for `lora`. Per slot because the order is a "
    "property of the FILE";

// Slot `i`'s adapter: a model picker typed to `suggest_type` AND a file
// browser, as every LoRA field is (a LoRA is as often a download in the
// sandbox as a catalogued model). `doc` must be static storage.
constexpr ConfigKey
path_key(int i, std::string_view suggest_type,
         std::string_view doc = kPathDoc)
{
  ConfigKey k{};
  k.key             = kPath[i];
  k.type            = ConfigType::String;
  k.doc             = doc;
  k.suggest_db      = kModelRegistryDb;
  k.suggest_db_type = suggest_type;
  k.is_path         = true;
  k.path_filter     = "weights";
  return k;
}

constexpr ConfigKey
scale_key(int i)
{
  ConfigKey k{};
  k.key      = kScale[i];
  k.type     = ConfigType::Real;
  k.doc      = kScaleDoc;
  k.def_real = 1.0;
  return k;
}

constexpr ConfigKey
qkv_layout_key(int i)
{
  ConfigKey k{};
  k.key     = kQkvLayout[i];
  k.type    = ConfigType::String;
  k.doc     = kQkvLayoutDoc;
  k.def_str = "auto";
  return k;
}

}  // namespace lora_slots
}  // namespace vpipe

#endif
