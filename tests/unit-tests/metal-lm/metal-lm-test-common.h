// metal-lm/ -- the metal-compute LM (language / vision-language /
// speech) test group. These tests compile and run in the
// VPIPE_BUILD_APPLE_SILICON test block and are the end-to-end proof
// that the language-model subsystem loads + generates on the metal
// backend. They never reference any other inference runtime.
//
// Env-gated: set VPIPE_METAL_LM_SMOKE_MODEL to a metal-supported
// model dir (Qwen3.5-4B / Llama / Qwen3-ASR text decoder); the
// per-theme files name the extra vars they need. Loads force
// VPIPE_LLM_BACKEND=metal for the duration.
//
// This header carries the include block and the `using namespace
// vpipe` every file in the group shares. It is test-only -- nothing
// outside tests/unit-tests/metal-lm/ may include it.

#ifndef VPIPE_TESTS_INTERNAL_METAL_LM_TEST_COMMON_H
#define VPIPE_TESTS_INTERNAL_METAL_LM_TEST_COMMON_H

#include "minitest.h"
#include "generative-models/chat-template.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/shared/coreml-vision-encoder.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/model-loader.h"
#include "generative-models/sampler.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#include "generative-models/video-turn.h"
#include "generative-models/weight-set.h"
#include "generative-models/shared/gguf-file.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/media-line.h"
#include "common/session.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <mach/mach.h>

using namespace vpipe;

// The speculative verify's pooled scratch against what a stage plans for
// it (MetalQwenModel::verify_scratch_bytes): never past the plan, and
// short of it only by the share these rounds may not have taken.
inline bool
verify_scratch_planned(const genai::MetalQwenModel& m, int rows,
                       bool mtp_head)
{
  std::size_t cond = 0;
  const std::size_t plan = genai::MetalQwenModel::verify_scratch_bytes(
      m.config(), rows, mtp_head, &cond);
  const std::size_t held = m.verify_scratch_held();
  std::printf("[verify-scratch] %d rows%s: held %.2f MB, planned %.2f MB "
              "(%.2f MB of it conditional)\n", rows,
              mtp_head ? " + MTP head" : "", held / 1048576.0,
              plan / 1048576.0, cond / 1048576.0);
  return held > 0 && held <= plan && held + cond >= plan;
}

// The first of positions [pos0, pos0 + n) where a row a speculative
// decode on `spec` committed is not the plain decode's on `ref` byte for
// byte (MetalQwenModel::set_row_hashes, on before both ran); -1 if none.
// A token-exact run can hide such a row until a near-tie meets it.
inline int
first_row_unlike_decode(const genai::MetalQwenModel& m, genai::ContextId ref,
                        genai::ContextId spec, int pos0, int n)
{
  for (int p = pos0; p < pos0 + n; ++p) {
    const std::uint64_t a = m.row_hash(ref, p);
    if (a == 0 || a != m.row_hash(spec, p)) { return p; }
  }
  return -1;
}

#endif  // VPIPE_TESTS_INTERNAL_METAL_LM_TEST_COMMON_H
