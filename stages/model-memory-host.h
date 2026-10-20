// model-memory-host.h -- sizing for the host's own models, beside the
// shared rules of model-memory.h.
//
// NOT INSTALLED AND NOT EXPORTED, on purpose. What is declared here sizes
// machinery no plugin can reach (the routed-expert cache lives inside the
// host's Qwen stack, which the SDK does not ship), so putting it in
// model-memory.h would export a symbol no plugin can use -- and a plugin
// that called it anyway would fail to load on every ABI-8 host built
// before it, ahead of any version or feature check. Keep it here until the
// machinery it sizes is offered to plugins, and then behind a feature.

#ifndef VPIPE_STAGES_MODEL_MEMORY_HOST_H
#define VPIPE_STAGES_MODEL_MEMORY_HOST_H

#include "stages/model-memory.h"

#include <cstddef>
#include <string>

namespace vpipe {
class SessionContextIntf;
namespace model_memory {

// THE ROUTED-EXPERT CACHE a Mixture-of-Experts model takes on this box
// (MetalQwenModel::Config::expert_cache_bytes): 0 when its whole preload
// fits beside its peers -- plan_streaming's rule, total RAM and the
// irreversible headroom, so every stage answers alike -- else its experts'
// bytes LESS the shortfall, never below one layer's worth (`layer_experts`:
// a prefill step may route to every expert of a layer at once). `dir` is
// counted at its preload whether or not it is declared yet; `experts` is
// what its routed experts hold resident, `scratch` what it allocates to
// run (K/V, activations) beside them.
std::size_t moe_expert_cache(const SessionContextIntf* session,
                             const std::string& dir, std::size_t preload,
                             std::size_t experts, std::size_t layer_experts,
                             std::size_t scratch,
                             std::size_t headroom = kStreamHeadroom);

}  // namespace model_memory
}  // namespace vpipe

#endif  // VPIPE_STAGES_MODEL_MEMORY_HOST_H
