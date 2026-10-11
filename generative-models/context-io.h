#ifndef VPIPE_GENERATIVE_MODELS_CONTEXT_IO_H
#define VPIPE_GENERATIVE_MODELS_CONTEXT_IO_H

// A language-model exec's contexts written out and read back: what lets
// a conversation outlive the model that holds it (LoadedLanguageModel::
// save_context / load_context, which a host reaches through the model
// manager's kept contexts -- generative-model-manager.h).
//
// An exec that keeps its own K/V (owns_kv) answers ModelExec::
// query_extension(kExtensionId) with a ContextIoExt*. Not part of the
// plugin SDK: only libvpipe's own execs implement it today (the metal
// Qwen family, through its ContextManager), and the stream format is
// theirs. An exec without it simply cannot spill, and a host falls back
// to prefilling the conversation again.

#include "generative-models/context-manager.h"

#include <string_view>

namespace vpipe::genai {

namespace context_io {

// ModelExec::query_extension id; the answer is a ContextIoExt*.
inline constexpr std::string_view kExtensionId = "vpipe.context-io/1";

}  // namespace context_io

class ContextIoExt {
public:
  // Everything the exec holds for `ctx`, streamed to `out`; false when it
  // cannot (an unknown context, a layout it does not write).
  virtual bool export_context(ContextId ctx,
                              const ContextManager::ByteSink& out) = 0;
  // `ctx` -- a context id the exec has never seen -- made to hold what
  // export_context wrote; false (and `ctx` left untouched) when the bytes
  // do not fit this model.
  virtual bool import_context(ContextId ctx,
                              const ContextManager::ByteSource& in) = 0;

protected:
  ~ContextIoExt() = default;
};

}  // namespace vpipe::genai

#endif
