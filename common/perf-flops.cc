#include "common/perf-flops.h"

#include <algorithm>
#include <atomic>
#include <limits>

namespace vpipe {
namespace perf_flops {

namespace {

static_assert((kRingRecords & (kRingRecords - 1)) == 0,
              "the ring is indexed by mask");

// One credited interval. `seq` is a sequence word: 2i+1 while record i
// is being written into this slot, 2i+2 once it is complete. A reader
// that wants record i accepts the slot only if it reads 2i+2 both before
// and after the fields -- anything else is a write in progress or a
// newer record, and the slot is skipped. Every field is an atomic read
// and written relaxed, so a torn read is a rejected read and never a
// data race.
struct Slot {
  std::atomic<std::uint64_t> seq{0};
  std::atomic<std::uint64_t> flops{0};
  std::atomic<std::uint64_t> t0{0};
  std::atomic<std::uint64_t> t1{0};
};

struct Ring {
  std::atomic<std::uint64_t> head{0};     // records ever claimed
  std::atomic<std::uint64_t> total{0};
  std::atomic<std::uint64_t> records{0};
  std::atomic<std::uint64_t> latest{0};
  Slot slot[kRingRecords];
};

// Zero-initialised static storage, so constant-initialised: a credit
// from a static constructor elsewhere finds the rings ready.
Ring g_ring[kUnitCount];

constexpr std::uint64_t kMask = kRingRecords - 1;

}  // namespace

void
credit(Unit unit, std::uint64_t flops, std::uint64_t t0_ns,
       std::uint64_t t1_ns) noexcept
{
  if (unit >= kUnitCount || flops == 0) { return; }
  if (t0_ns > t1_ns) { t0_ns = t1_ns; }
  Ring& r = g_ring[unit];
  const std::uint64_t i = r.head.fetch_add(1, std::memory_order_relaxed);
  Slot& s = r.slot[i & kMask];
  // Odd first, and fenced, so no reader can see a new field under the
  // previous record's even word.
  s.seq.store(2 * i + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  s.flops.store(flops, std::memory_order_relaxed);
  s.t0.store(t0_ns, std::memory_order_relaxed);
  s.t1.store(t1_ns, std::memory_order_relaxed);
  s.seq.store(2 * i + 2, std::memory_order_release);
  r.total.fetch_add(flops, std::memory_order_relaxed);
  r.records.fetch_add(1, std::memory_order_relaxed);
  std::uint64_t prev = r.latest.load(std::memory_order_relaxed);
  while (prev < t1_ns &&
         !r.latest.compare_exchange_weak(prev, t1_ns,
                                         std::memory_order_relaxed)) {
  }
}

Totals
totals(Unit unit) noexcept
{
  Totals t;
  if (unit >= kUnitCount) { return t; }
  const Ring& r = g_ring[unit];
  t.flops     = r.total.load(std::memory_order_relaxed);
  t.records   = r.records.load(std::memory_order_relaxed);
  t.latest_ns = r.latest.load(std::memory_order_relaxed);
  return t;
}

std::uint64_t
series(Unit unit, std::uint64_t end_ns, std::uint64_t bin_ns,
       std::size_t n_bins, double* out) noexcept
{
  if (out == nullptr || n_bins == 0) { return 0; }
  std::fill(out, out + n_bins, 0.0);
  if (unit >= kUnitCount || bin_ns == 0) { return 0; }
  // Positions are int64 nanoseconds from the window's start, so the
  // window may begin before the clock's epoch and nothing is lost to a
  // double's mantissa however long the machine has been up.
  const std::int64_t bin  = (std::int64_t)bin_ns;
  const std::int64_t span = bin * (std::int64_t)n_bins;
  const std::int64_t base = (std::int64_t)end_ns - span;
  Ring& r = g_ring[unit];
  const std::uint64_t h  = r.head.load(std::memory_order_acquire);
  const bool wrapped     = h > kRingRecords;
  const std::uint64_t lo = wrapped ? h - kRingRecords : 0;
  std::uint64_t oldest   = std::numeric_limits<std::uint64_t>::max();
  for (std::uint64_t j = lo; j < h; ++j) {
    Slot& s = r.slot[j & kMask];
    const std::uint64_t want = 2 * j + 2;
    if (s.seq.load(std::memory_order_acquire) != want) { continue; }
    const std::uint64_t f  = s.flops.load(std::memory_order_relaxed);
    const std::uint64_t a  = s.t0.load(std::memory_order_relaxed);
    const std::uint64_t b  = s.t1.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s.seq.load(std::memory_order_relaxed) != want) { continue; }
    oldest = std::min(oldest, b);
    const std::int64_t pa = (std::int64_t)a - base;
    const std::int64_t pb = (std::int64_t)b - base;
    if (pb < 0 || pa >= span) { continue; }
    if (pa == pb) {
      // A point: all of it in the bin holding it (half-open bins, so a
      // point exactly at end_ns falls outside).
      if (pb < span) { out[pb / bin] += (double)f; }
      continue;
    }
    const double rate = (double)f / (double)(pb - pa);
    const std::int64_t x0 = std::max<std::int64_t>(pa, 0);
    const std::int64_t x1 = std::min<std::int64_t>(pb, span);
    if (x1 <= x0) { continue; }
    const std::int64_t i0 = x0 / bin;
    const std::int64_t i1 = std::min<std::int64_t>((x1 - 1) / bin,
                                                   (std::int64_t)n_bins - 1);
    for (std::int64_t i = i0; i <= i1; ++i) {
      const std::int64_t c0 = std::max(x0, i * bin);
      const std::int64_t c1 = std::min(x1, (i + 1) * bin);
      if (c1 > c0) { out[i] += rate * (double)(c1 - c0); }
    }
  }
  if (!wrapped) { return 0; }
  return oldest == std::numeric_limits<std::uint64_t>::max() ? end_ns
                                                             : oldest;
}

}  // namespace perf_flops
}  // namespace vpipe
