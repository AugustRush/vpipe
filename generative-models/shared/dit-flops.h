#ifndef VPIPE_GENERATIVE_MODELS_SHARED_DIT_FLOPS_H
#define VPIPE_GENERATIVE_MODELS_SHARED_DIT_FLOPS_H

// A DiT's arithmetic, credited to common/perf-flops.h on the GPU's clock.
//
// COUNT AT ENCODE, CREDIT AT RETIRE. A forward is one deferred command
// stream: the block loop encodes the whole stack in milliseconds and the
// GPU spends seconds on it, so a count credited where it is encoded would
// say the GPU did a step's work in a millisecond and nothing after. So
// the GEMM and attention helpers ADD to a tally as they encode (one
// integer add, on the encode thread), and at each block boundary flush()
// hangs what was added on the command buffer open at that point; the
// count is credited when that buffer RETIRES. That is the mechanism
// dit-gpu-progress.h uses for the progress bar, and for the same reason.
//
// THE INTERVAL. A flush credits its FLOPs over [t0, retire], where t0 is
// the later of when this batch began to be encoded and when the previous
// batch retired. On a held forward (encoded at once) that chains the
// blocks end to end on the GPU's timeline; on a streamed one (committed
// and waited per block) it starts each block where its encode did; and an
// idle gap between forwards stays empty, because the first batch of the
// next forward cannot start before it was encoded.
//
// Flush before EVERY commit -- a streamed block's, a drain's, the last --
// and at each block boundary of a stack that commits nothing between
// blocks. A batch flushed onto a buffer committed later than its own work
// is credited a little late, never twice and never lost.
//
// On a host without the meter (a plugin built against a newer SDK; see
// common/perf-flops.h) flush() and SolFlops credit nothing.

#include "apple-silicon/metal-compute/command-stream.h"
#include "common/perf-flops.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe::genai {

// FLOPs of one projection, x[M,K] W^T[K,N].
inline std::uint64_t
gemm_flops(long long m, long long n, long long k) noexcept
{
  if (m <= 0 || n <= 0 || k <= 0) { return 0; }
  return 2ull * (std::uint64_t)m * (std::uint64_t)n * (std::uint64_t)k;
}

// FLOPs of attention over `pairs` (query, key) pairs per head: QK^T and
// PV, each 2 * d per pair.
inline std::uint64_t
attention_flops(long long heads, long long pairs, long long d) noexcept
{
  if (heads <= 0 || pairs <= 0 || d <= 0) { return 0; }
  return 4ull * (std::uint64_t)heads * (std::uint64_t)pairs *
         (std::uint64_t)d;
}

// (query, key) pairs of a causal run of `q` rows whose first row sits at
// sequence position `q_off`: row i sees keys [0, q_off + i].
inline long long
causal_pairs(long long q, long long q_off) noexcept
{
  if (q <= 0) { return 0; }
  return q * q_off + q * (q + 1) / 2;
}

class GpuFlopsTally {
 public:
  // At the start of a forward: what an aborted forward added and never
  // flushed is dropped -- its buffers were never committed.
  void begin() noexcept { _pending = 0; }

  // On the encode thread, as the work is encoded.
  void add(std::uint64_t flops) noexcept
  {
    if (flops == 0) { return; }
    if (_pending == 0) { _since = perf_flops::now_ns(); }
    _pending += flops;
  }
  void gemm(long long m, long long n, long long k) noexcept
  {
    add(gemm_flops(m, n, k));
  }
  void attention(long long heads, long long pairs, long long d) noexcept
  {
    add(attention_flops(heads, pairs, d));
  }

  // Hang what was added since the last flush on `stream`'s open command
  // buffer (opening one if none is), credited when it retires. With no
  // stream to hang it on, credited now.
  void flush(metal_compute::CommandStream& stream)
  {
    if (_pending == 0) { return; }
    const std::uint64_t f     = _pending;
    const std::uint64_t since = _since;
    _pending = 0;
    if (!perf_flops::available()) { return; }   // a host without the meter
    _flushed += f;
    if (!stream.valid()) {
      const std::uint64_t t1 = perf_flops::now_ns();
      perf_flops::credit(perf_flops::kGpu, f, since, t1);
      return;
    }
    std::shared_ptr<std::atomic<std::uint64_t>> chain = _chain;
    stream.on_completion([chain, f, since]() {
      const std::uint64_t t1 = perf_flops::now_ns();
      // The previous retire on this tally's timeline, advanced to this
      // one. Buffers of one stream retire in order; should two ever
      // cross, the later finds `prev` past its own end and keeps a
      // zero-length interval rather than a negative one.
      std::uint64_t prev = chain->load(std::memory_order_relaxed);
      while (prev < t1 &&
             !chain->compare_exchange_weak(prev, t1,
                                           std::memory_order_relaxed)) {
      }
      const std::uint64_t t0 = std::max(since, std::min(prev, t1));
      perf_flops::credit(perf_flops::kGpu, f, t0, t1);
    });
  }

  // Everything ever flushed -- what the meter will have been credited
  // once the buffers retire, for a test to hold the two against.
  std::uint64_t flushed() const noexcept { return _flushed; }
  // Added and not yet flushed: zero once a forward has flushed before
  // its last commit -- what a test asks to find a commit left out.
  std::uint64_t pending() const noexcept { return _pending; }

 private:
  std::uint64_t _pending = 0;
  std::uint64_t _since   = 0;
  std::uint64_t _flushed = 0;
  // Shared with the completion handlers, which may outlive the model:
  // a buffer completes asynchronously, after the wait that saw it.
  std::shared_ptr<std::atomic<std::uint64_t>> _chain =
      std::make_shared<std::atomic<std::uint64_t>>(0);
};

// Sol-Attn's exact half, counted by what it KEPT.
//
// How many key blocks the routing keeps is a property of the
// activations, read back from the GPU only once the forward has run
// (MetalSolAttention::exact_blocks()). So a routed attention is counted,
// as it is encoded, at the ratio the PREVIOUS forward realized -- which
// moves slowly from step to step -- and the first forward, which has no
// predecessor, is credited at finish() with its own measured count,
// spread over the forward. The approximate half and the routing passes
// are a small fraction of the exact one and are not counted.
class SolFlops {
 public:
  // At the start of a forward, with the GPU counter reset.
  void begin() noexcept
  {
    _blocks = 0;
    _dense  = 0;
    _t0     = perf_flops::now_ns();
  }
  // One routed encode: `dense` is what the attention would cost dense,
  // `blocks` the key blocks it routed among
  // (MetalSolAttention::total_blocks()). Returns the FLOPs to add to the
  // GPU tally now.
  std::uint64_t routed(std::uint64_t dense, long long blocks) noexcept
  {
    _blocks += blocks;
    _dense  += dense;
    if (_ratio < 0.0) { return 0; }
    return (std::uint64_t)((double)dense * _ratio);
  }
  // After the forward's final wait, with the GPU counter complete:
  // `kept` is MetalSolAttention::exact_blocks().
  void finish(long long kept) noexcept
  {
    if (_blocks <= 0) { return; }
    const double r =
        std::clamp((double)kept / (double)_blocks, 0.0, 1.0);
    if (_ratio < 0.0 && _dense > 0 && perf_flops::available()) {
      perf_flops::credit(perf_flops::kGpu,
                         (std::uint64_t)((double)_dense * r), _t0,
                         perf_flops::now_ns());
    }
    _ratio  = r;
    _blocks = 0;
    _dense  = 0;
  }
  // The ratio routed attention is counted at; < 0 before any forward
  // has finished.
  double ratio() const noexcept { return _ratio; }

 private:
  double        _ratio  = -1.0;
  long long     _blocks = 0;
  std::uint64_t _dense  = 0;
  std::uint64_t _t0     = 0;
};

}  // namespace vpipe::genai

#endif
