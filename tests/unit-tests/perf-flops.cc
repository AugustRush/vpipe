// The achieved-FLOP/s meter (common/perf-flops.h) and its GPU tally
// (generative-models/shared/dit-flops.h).
//
// The meter is process-wide and the test binary runs many tests in one
// process, so each case here credits into a window of its OWN -- days
// away from the real clock -- and reads totals as differences.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/perf-flops-doc.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/shared/dit-flops.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using namespace vpipe;
namespace pf = vpipe::perf_flops;

namespace {

constexpr std::uint64_t kSec = 1000000000ull;
constexpr std::uint64_t kMs  = 1000000ull;

// A window nobody else credits into: `days` days past now.
std::uint64_t
far_(int days)
{
  return pf::now_ns() + (std::uint64_t)days * 86400ull * kSec;
}

double
sum_(const std::vector<double>& v)
{
  double s = 0.0;
  for (double x : v) { s += x; }
  return s;
}

// Credits land from a Metal completion handler, after the wait that saw
// the buffer finish; give them a moment.
bool
settle_(pf::Unit u, std::uint64_t want_flops, std::uint64_t base)
{
  for (int i = 0; i < 400; ++i) {
    if (pf::totals(u).flops - base >= want_flops) { return true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

}  // namespace

// An interval's FLOPs are spread evenly over the time it covered, a
// point lands in the bin holding it, and nothing leaks into another unit.
TEST(perf_flops, series_spreads_an_interval_over_its_bins)
{
  const std::uint64_t B = far_(10);
  const pf::Totals g0 = pf::totals(pf::kGpu);
  const pf::Totals a0 = pf::totals(pf::kAne);

  // 3e12 over [B+0.5s, B+3.5s]: 1e12 a second.
  pf::credit(pf::kGpu, 3000000000000ull, B + kSec / 2, B + 7 * kSec / 2);
  // A point of 7 at B+2.5s.
  pf::credit(pf::kGpu, 7, B + 5 * kSec / 2, B + 5 * kSec / 2);
  // Ignored: no FLOPs, no such unit.
  pf::credit(pf::kGpu, 0, B, B + kSec);
  pf::credit((pf::Unit)7, 5, B, B + kSec);
  // The ANE's, in the same window.
  pf::credit(pf::kAne, 4000, B + kSec, B + 2 * kSec);

  std::vector<double> bins(4);
  const std::uint64_t from =
      pf::series(pf::kGpu, B + 4 * kSec, kSec, bins.size(), bins.data());
  std::printf("  gpu bins %.4g %.4g %.4g %.4g (complete from %llu)\n",
              bins[0], bins[1], bins[2], bins[3],
              (unsigned long long)from);
  EXPECT_TRUE(bins[0] == 5e11);
  EXPECT_TRUE(bins[1] == 1e12);
  EXPECT_TRUE(bins[2] == 1e12 + 7.0);
  EXPECT_TRUE(bins[3] == 5e11);

  // Half the window: only what overlaps it.
  std::vector<double> half(2);
  pf::series(pf::kGpu, B + 2 * kSec, kSec, half.size(), half.data());
  EXPECT_TRUE(half[0] == 5e11 && half[1] == 1e12);

  // The ANE kept its own.
  std::vector<double> ane(4);
  pf::series(pf::kAne, B + 4 * kSec, kSec, ane.size(), ane.data());
  EXPECT_TRUE(ane[0] == 0.0 && ane[1] == 4000.0 && ane[2] == 0.0);

  const pf::Totals g1 = pf::totals(pf::kGpu);
  const pf::Totals a1 = pf::totals(pf::kAne);
  EXPECT_TRUE(g1.flops - g0.flops == 3000000000007ull);
  EXPECT_TRUE(g1.records - g0.records == 2);
  EXPECT_TRUE(g1.latest_ns >= B + 7 * kSec / 2);
  EXPECT_TRUE(a1.flops - a0.flops == 4000);

  // A point exactly at the window's end is outside it (half-open bins),
  // and t0 after t1 is a point at t1.
  const std::uint64_t C = far_(11);
  pf::credit(pf::kGpu, 11, C + kSec, C + kSec);
  pf::credit(pf::kGpu, 13, C + 3 * kSec, C + kSec / 2);
  std::vector<double> edge(1);
  pf::series(pf::kGpu, C + kSec, kSec, 1, edge.data());
  EXPECT_TRUE(edge[0] == 13.0);
}

// Credits from many threads at once are each counted once, and a reader
// running beside them sees a sane, growing figure.
TEST(perf_flops, concurrent_credits_are_counted_once)
{
  const std::uint64_t B = far_(20);
  constexpr int kThreads = 8, kEach = 900;   // < kRingRecords in all
  const pf::Totals t0 = pf::totals(pf::kGpu);
  std::atomic<bool> done{false};
  std::atomic<int> bad_reads{0};
  std::thread reader([&]() {
    std::vector<double> bins(10);
    double last = 0.0;
    while (!done.load()) {
      pf::series(pf::kGpu, B + 10 * kSec, kSec, bins.size(), bins.data());
      const double s = sum_(bins);
      if (!std::isfinite(s) || s < last) { ++bad_reads; }
      last = s;
    }
  });
  std::vector<std::thread> ws;
  std::uint64_t want = 0;
  for (int t = 0; t < kThreads; ++t) {
    for (int i = 0; i < kEach; ++i) { want += 1000 + (std::uint64_t)t; }
    ws.emplace_back([t, B]() {
      for (int i = 0; i < kEach; ++i) {
        const std::uint64_t at = B + (std::uint64_t)(i % 10) * kSec + 1;
        pf::credit(pf::kGpu, 1000 + (std::uint64_t)t, at, at + kSec / 4);
      }
    });
  }
  for (auto& w : ws) { w.join(); }
  done = true;
  reader.join();

  std::vector<double> bins(10);
  pf::series(pf::kGpu, B + 10 * kSec, kSec, bins.size(), bins.data());
  const double got = sum_(bins);
  std::printf("  %d threads x %d credits: %.0f of %llu FLOPs in the "
              "series, %d bad reads\n", kThreads, kEach, got,
              (unsigned long long)want, bad_reads.load());
  EXPECT_TRUE(pf::totals(pf::kGpu).flops - t0.flops == want);
  EXPECT_TRUE(got == (double)want);
  EXPECT_TRUE(bad_reads.load() == 0);
}

// Past the ring's capacity the oldest intervals are gone, and the series
// says from when it is complete.
TEST(perf_flops, a_wrapped_ring_says_where_its_history_starts)
{
  const std::uint64_t B = far_(30);
  const std::size_t n = pf::kRingRecords + 100;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint64_t at = B + (std::uint64_t)i * kMs;
    pf::credit(pf::kGpu, 1, at, at);
  }
  std::vector<double> bins(1);
  const std::uint64_t from = pf::series(
      pf::kGpu, B + (std::uint64_t)n * kMs, (std::uint64_t)n * kMs, 1,
      bins.data());
  std::printf("  %zu credits into %zu slots: %.0f kept, complete from "
              "+%llu ms\n", n, pf::kRingRecords, bins[0],
              (unsigned long long)((from - B) / kMs));
  EXPECT_TRUE(bins[0] == (double)pf::kRingRecords);
  EXPECT_TRUE(from == B + 100 * kMs);
}

// The document the web UI and Python serve: both units, the bins asked
// for, rates in TFLOP/s.
TEST(perf_flops, the_document_carries_both_units)
{
  const std::uint64_t now = pf::now_ns();
  // 2e12 over the last half second, within the newest 1 s bin -- placed
  // a little in the past so the window that ends "now" holds it.
  pf::credit(pf::kGpu, 2000000000000ull, now - 900 * kMs, now - 400 * kMs);
  FlexData doc = perf_flops_document(1000, 5, 0);
  ASSERT_TRUE(doc.is_object());
  FlexData units = doc.as_object().at("units");
  ASSERT_TRUE(units.is_object());
  EXPECT_TRUE(units.as_object().contains("gpu"));
  EXPECT_TRUE(units.as_object().contains("ane"));
  FlexData gpu = units.as_object().at("gpu");
  FlexData tf = gpu.as_object().at("tflops");
  ASSERT_TRUE(tf.is_array() && tf.as_array().size() == 5);
  double s = 0.0;
  for (std::size_t i = 0; i < 5; ++i) { s += tf.as_array()[i].as_real(0); }
  std::printf("  document: %s\n", doc.to_json().substr(0, 160).c_str());
  // At least the 2 TFLOP just credited, as TFLOP/s over 1 s bins.
  EXPECT_TRUE(s >= 2.0 - 1e-9);
  EXPECT_TRUE(doc.as_object().at("bins").as_int(0) == 5);
}

// The tally credits a flush when the command buffer holding it RETIRES,
// not when it was encoded -- and loses nothing on the way.
TEST(perf_flops, gpu_tally_credits_when_the_buffer_retires)
{
  Session sess;
  metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  using genai::GpuFlopsTally;
  GpuFlopsTally tally;
  const pf::Totals t0 = pf::totals(pf::kGpu);

  metal_compute::CommandStream stream = mc->make_command_stream();
  ASSERT_TRUE(stream.valid());
  tally.begin();
  tally.gemm(64, 128, 256);                 // 2*64*128*256
  tally.attention(4, 100, 32);              // 4*4*100*32
  const std::uint64_t want = 2ull * 64 * 128 * 256 + 4ull * 4 * 100 * 32;
  const std::uint64_t enc_t = pf::now_ns();
  tally.flush(stream);
  EXPECT_TRUE(tally.flushed() == want);
  // Not before the buffer is even committed.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_TRUE(pf::totals(pf::kGpu).flops - t0.flops == 0);
  ASSERT_TRUE(stream.commit().wait_ok());
  EXPECT_TRUE(settle_(pf::kGpu, want, t0.flops));
  EXPECT_TRUE(pf::totals(pf::kGpu).flops - t0.flops == want);
  EXPECT_TRUE(pf::totals(pf::kGpu).latest_ns >= enc_t);

  // An unflushed batch an aborted forward left is dropped by begin().
  tally.gemm(8, 8, 8);
  tally.begin();
  metal_compute::CommandStream s2 = mc->make_command_stream();
  tally.flush(s2);
  ASSERT_TRUE(s2.commit().wait_ok());
  EXPECT_TRUE(tally.flushed() == want);

  // With no stream to ride, a flush credits at once.
  const pf::Totals t1 = pf::totals(pf::kGpu);
  metal_compute::CommandStream none;
  tally.gemm(2, 3, 4);
  tally.flush(none);
  EXPECT_TRUE(pf::totals(pf::kGpu).flops - t1.flops == 2ull * 2 * 3 * 4);
}

// Sol's exact half: counted at the previous forward's keep ratio, and
// the first forward -- which has none -- credited at finish().
TEST(perf_flops, sol_counts_what_it_kept)
{
  genai::SolFlops sol;
  const pf::Totals t0 = pf::totals(pf::kGpu);
  sol.begin();
  EXPECT_TRUE(sol.routed(1000000, 100) == 0);   // nothing known yet
  EXPECT_TRUE(sol.routed(1000000, 100) == 0);
  sol.finish(50);                               // kept 50 of 200 blocks
  EXPECT_TRUE(pf::totals(pf::kGpu).flops - t0.flops == 500000);
  EXPECT_TRUE(std::fabs(sol.ratio() - 0.25) < 1e-12);
  // The next forward is counted as it is encoded, at 25%.
  sol.begin();
  EXPECT_TRUE(sol.routed(1000000, 100) == 250000);
  const pf::Totals t1 = pf::totals(pf::kGpu);
  sol.finish(10);
  EXPECT_TRUE(pf::totals(pf::kGpu).flops == t1.flops);   // no double count
  EXPECT_TRUE(std::fabs(sol.ratio() - 0.1) < 1e-12);
  EXPECT_TRUE(genai::causal_pairs(3, 2) == 2 * 3 + 6);   // keys 3, 4, 5
}

// What a producer pays. A DiT credits once per block and the ANE once per
// predict -- tens a second -- so this only has to be small against a
// block; it is measured to say how small. Runs last (the suite is
// alphabetical): its credits wrap the ring.
TEST(perf_flops, what_a_credit_costs)
{
  const std::uint64_t B = far_(40);
  constexpr int kN = 200000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kN; ++i) {
    pf::credit(pf::kGpu, 1, B + (std::uint64_t)i, B + (std::uint64_t)i + 1);
  }
  const double ns = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - t0).count() / kN;
  genai::GpuFlopsTally tally;
  const auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < kN; ++i) { tally.gemm(4096, 4096, 4096); }
  const double add_ns = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - t1).count() / kN;
  std::printf("  credit %.1f ns, tally add %.2f ns (single thread)\n", ns,
              add_ns);
  EXPECT_TRUE(ns < 1000.0);
}
