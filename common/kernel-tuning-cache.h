#ifndef VPIPE_COMMON_KERNEL_TUNING_CACHE_H
#define VPIPE_COMMON_KERNEL_TUNING_CACHE_H

// The session's KernelTuningCache (interfaces/session-services-intf.h):
// a map behind a mutex. Host-only -- a plugin reaches it through the
// interface, never this class.
//
// Never evicted. An answer is a few integers under a key of a hundred
// bytes or so, one per tuned shape, and a session meets a bounded set
// of shapes: the tuners already bucket their row counts, which is what
// lets a run at a similar size find the last one's answer.

#include "interfaces/session-services-intf.h"

#include <map>
#include <mutex>
#include <string>
#include <string_view>

namespace vpipe {

class SessionKernelTuningCache final : public KernelTuningCache {
public:
  bool find(std::string_view key, std::string& answer) const override
  {
    std::lock_guard lk(_mu);
    const auto it = _answers.find(key);
    if (it == _answers.end()) {
      ++_stats.misses;
      return false;
    }
    ++_stats.hits;
    answer = it->second;
    return true;
  }

  void store(std::string_view key, std::string_view answer) override
  {
    std::lock_guard lk(_mu);
    _answers.insert_or_assign(std::string(key), std::string(answer));
    _stats.entries = _answers.size();
  }

  Stats stats() const override
  {
    std::lock_guard lk(_mu);
    return _stats;
  }

private:
  mutable std::mutex _mu;
  std::map<std::string, std::string, std::less<>> _answers;
  mutable Stats _stats;
};

}  // namespace vpipe

#endif
