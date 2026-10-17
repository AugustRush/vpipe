#pragma once

// A kernel tuner's answers kept for the session (KernelTuningCache,
// interfaces/session-services-intf.h; feature kernel-tuning/1).
//
// Every tuner here keeps its winner on the object that tuned -- a model,
// an I8GemmContext, a kernel set -- and those die with the model. A host
// that unloads its models between jobs (to give their memory back) then
// measured everything again at every load: MiniMax-H3's GEMM routes took
// 2.8 s a clip at 832x480 on an M5 Pro (12-16 s on the plain M5),
// Qwen's head-dim-128 prefill attention 1.2 s on an M5,
// and every i8 shape a picture model met cost its second step a vote.
// So a tuner FILES its answer in the session under a key and asks for it
// before measuring. A model loaded again in the same session, or another
// with the same shapes, finds it; one at a similar size finds it too,
// because the tuners already bucket the row counts their answers hold
// for (I8GemmContext's 512 / 1024 / 2048 / 4096, the DiT's ceiling).
//
// THE KEY IS THE WHOLE QUESTION. Whatever changes which candidates were
// timed or what they cost -- the shape, the weight format, the kernels
// that loaded, the tuner's own env knobs, the rows it measured at -- goes
// in it, or a run under other settings would read an answer to another
// question. Too much in a key costs a re-measurement; too little runs the
// wrong kernel without a word. Version a tuner's name ("i8-split/1") when
// what its answer means changes.
//
// Session-scoped by design: an answer is the GPU's, measured under this
// process's conditions, and is not carried to disk.
//
// TOOLKIT, header-only. Through the stable surface alone (service lookup
// is a virtual every ABI-8 host has), so a plugin built with it loads on
// an older host and simply tunes as before.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <charconv>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace vpipe::genai {

// The tuning answers of the session `mc` belongs to; null when there are
// none to keep (no session, an older host, the cache switched off).
inline KernelTuningCache*
tuning_cache(const metal_compute::MetalCompute* mc)
{
  if (mc == nullptr) { return nullptr; }
  const SessionContextIntf* s = mc->session();
  if (s == nullptr) { return nullptr; }
  return s->services()->kernel_tuning_cache();
}

// A key: the tuner's versioned name, then name=value fields in the order
// given -- "i8-split/1 bf16=1 group=64 N=3072 K=9216 M=4096".
class TuneKey {
 public:
  explicit TuneKey(std::string_view tuner) : _s(tuner) {}

  TuneKey& operator()(std::string_view name, std::string_view v)
  {
    _s += ' ';
    _s += name;
    _s += '=';
    _s += v;
    return *this;
  }
  // Integers and bools (as 0 / 1). A template, so a literal 0 is a
  // number here rather than an ambiguous null string.
  template <class Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0>
  TuneKey& operator()(std::string_view name, Int v)
  {
    return (*this)(name, std::string_view(std::to_string((long long)v)));
  }

  const std::string& str() const { return _s; }

 private:
  std::string _s;
};

// The integers filed under `key`, into `out`; false when the session has
// none (or keeps no answers). An answer that does not read as integers is
// treated as absent: measured again, it is filed over.
inline bool
find_tuned(KernelTuningCache* cache, const TuneKey& key, std::vector<int>& out)
{
  if (cache == nullptr) { return false; }
  std::string text;
  if (!cache->find(key.str(), text)) { return false; }
  std::vector<int> v;
  const char* p = text.data();
  const char* const end = p + text.size();
  while (p < end) {
    int x = 0;
    const auto r = std::from_chars(p, end, x);
    if (r.ec != std::errc()) { return false; }
    v.push_back(x);
    p = r.ptr;
    if (p < end) {
      if (*p != ',') { return false; }
      ++p;
    }
  }
  out = std::move(v);
  return true;
}

inline void
store_tuned(KernelTuningCache* cache, const TuneKey& key,
            const std::vector<int>& answer)
{
  if (cache == nullptr) { return; }
  std::string text;
  for (std::size_t i = 0; i < answer.size(); ++i) {
    if (i != 0) { text += ','; }
    text += std::to_string(answer[i]);
  }
  cache->store(key.str(), text);
}

// One integer, the common case (a split, a tile, a member).
inline bool
find_tuned(KernelTuningCache* cache, const TuneKey& key, int& out)
{
  std::vector<int> v;
  if (!find_tuned(cache, key, v) || v.size() != 1) { return false; }
  out = v[0];
  return true;
}

inline void
store_tuned(KernelTuningCache* cache, const TuneKey& key, int answer)
{
  store_tuned(cache, key, std::vector<int>{answer});
}

}  // namespace vpipe::genai
