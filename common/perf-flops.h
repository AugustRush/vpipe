#ifndef VPIPE_COMMON_PERF_FLOPS_H
#define VPIPE_COMMON_PERF_FLOPS_H

// perf-flops.h -- the FLOP/s a compute unit actually achieved, for a
// live plot.
//
// The event profiler (perf-event.h, PerfSinkIntf) records WHEN things
// happen, per session, and is read once a capture is over. This is the
// other question -- HOW MUCH arithmetic the GPU and the ANE did over the
// last few seconds -- asked continuously by a UI that redraws a graph
// every second or ten. It shares nothing with the event machinery: no
// lanes, no gvids, no enable switch. It is always on, because what it
// costs is one ring write per credited interval.
//
// A PRODUCER credits a count of FLOPs to a unit together with the
// interval of steady_clock time it was executed over: a DiT credits each
// block when the command buffer holding it RETIRES (the GPU's clock, not
// the encode thread's -- see generative-models/shared/dit-flops.h), and
// the ANE tier credits each predict as it returns. Credits are
// lock-free and wait-free: a fetch_add claims a slot of a fixed ring and
// a sequence word publishes it. Any thread -- a Metal completion
// handler, the ANE worker, a pool worker -- may credit at any time.
//
// A READER asks for a SERIES: FLOPs per bin over the last n bins, each
// interval's FLOPs spread evenly across the time it covered, so a block
// that ran for four seconds lands as four seconds of rate rather than a
// spike in the bin it retired in. Reading takes no lock either; a slot
// being written while it is read is skipped and seen on the next read.
//
// WHAT COUNTS. Matrix arithmetic -- GEMMs at 2*M*N*K and attention at
// 4*heads*(query, key pairs it attends)*head_dim -- whatever the
// precision (an int8 GEMM's operations are counted like a bf16 one's).
// Normalisations, activations and other elementwise passes are not: they
// are a rounding error of a DiT's arithmetic and would make the figure
// depend on how kernels happen to be fused. A sparse attention counts
// what it EXECUTED, not the dense figure it stands in for, so turning on
// a sparse tier does not read as the GPU getting faster.
//
// THE NEWEST BINS FILL IN LATE. Work is credited when it finishes, so the
// last block or chunk in flight is not in the series yet; Totals::
// latest_ns says how far the credits reach. A UI that redraws the whole
// series every poll sees those bins settle on the next poll, which is the
// latency this design trades for costing the producer nothing.
//
// Process-wide, like the hardware it measures: two sessions sharing the
// GPU share its figure.
//
// OPTIONAL TO A PLUGIN. Outside libvpipe these functions are WEAK
// imports: on a host built before the meter (no perf-flops/1) they
// resolve to null, available() says false, and a plugin that credits the
// meter still loads there and simply reports nothing. The toolkit's
// helpers (dit-flops.h, the ANE tier) check it; code calling credit()
// directly checks it first. Never list perf-flops/1 as REQUIRED for this.

#include "common/vpipe-api.h"

#include <chrono>
#include <cstddef>
#include <cstdint>

#if defined(VPIPE_BUILDING_LIBRARY)
#define VPIPE_PERF_FLOPS_IMPORT
#else
#define VPIPE_PERF_FLOPS_IMPORT __attribute__((weak_import))
#endif

VPIPE_API_BEGIN

namespace vpipe {
namespace perf_flops {

// The compute units, each with its own ring and totals.
enum Unit : unsigned {
  kGpu       = 0u,
  kAne       = 1u,
  kUnitCount = 2u,
};

inline constexpr const char* kUnitName[kUnitCount] = { "gpu", "ane" };

// Intervals each unit's ring holds. At a credit per DiT block that is
// minutes of history; a reader polling every few seconds never laps it.
inline constexpr std::size_t kRingRecords = 8192;

// The clock every interval is stated in: steady_clock, in nanoseconds
// since its epoch. Inline, so a producer pays no call for it.
inline std::uint64_t
now_ns() noexcept
{
  return (std::uint64_t)std::chrono::duration_cast<
      std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// PRODUCER. `flops` executed on `unit` over [t0_ns, t1_ns]. A zero-length
// interval is allowed (its FLOPs land in the bin holding t1_ns); t0_ns
// after t1_ns is taken as t1_ns. A zero count or an unknown unit is
// ignored.
VPIPE_PERF_FLOPS_IMPORT void credit(Unit unit, std::uint64_t flops,
                                    std::uint64_t t0_ns,
                                    std::uint64_t t1_ns) noexcept;

// Whether this host has the meter at all: always inside libvpipe; in a
// plugin, whether the weak import above resolved.
inline bool
available() noexcept
{
#if defined(VPIPE_BUILDING_LIBRARY)
  return true;
#else
  return &credit != nullptr;
#endif
}

// READER. Everything credited to `unit` since the process started.
// `flops` wraps at 2^64 (a week at 30 TFLOP/s): take differences.
struct Totals {
  std::uint64_t flops     = 0;
  std::uint64_t records   = 0;   // intervals credited
  std::uint64_t latest_ns = 0;   // the latest t1 credited; 0 = none yet
};
VPIPE_PERF_FLOPS_IMPORT Totals totals(Unit unit) noexcept;

// READER. `out[i]` gets the FLOPs `unit` executed over bin i of `n_bins`
// bins of `bin_ns`, the last one ending at `end_ns`: bin i covers
// [end_ns - (n_bins - i) * bin_ns, end_ns - (n_bins - 1 - i) * bin_ns).
// Divide by the bin's seconds for a rate.
//
// Returns the time from which the series is COMPLETE: 0 while the ring
// has never wrapped, else the end of the oldest interval it still holds
// -- intervals overwritten before it ended before it, so a bin wholly
// after it is complete and one before it may be short.
VPIPE_PERF_FLOPS_IMPORT std::uint64_t series(Unit unit,
                                             std::uint64_t end_ns,
                                             std::uint64_t bin_ns,
                                             std::size_t n_bins,
                                             double* out) noexcept;

}  // namespace perf_flops
}  // namespace vpipe

VPIPE_API_END

#endif
