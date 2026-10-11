#include "common/perf-flops-doc.h"
#include "common/perf-flops.h"

#include <chrono>
#include <vector>

namespace vpipe {

FlexData
perf_flops_document(std::uint64_t bin_ms, std::size_t bins,
                    std::uint64_t lag_ms)
{
  namespace pf = perf_flops;
  constexpr std::uint64_t kMs = 1000000;
  if (bin_ms == 0) { bin_ms = 1; }
  if (bins == 0) { bins = 1; }
  const std::uint64_t now = pf::now_ns();
  const std::uint64_t end = now > lag_ms * kMs ? now - lag_ms * kMs : 0;
  const auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();

  FlexData doc = FlexData::make_object();
  auto o = doc.as_object();
  o.insert("bin_ms", FlexData::make_uint(bin_ms));
  o.insert("bins", FlexData::make_uint(bins));
  o.insert("end_ms", FlexData::make_uint(end / kMs));
  o.insert("end_unix_ms",
           FlexData::make_int((std::int64_t)wall - (std::int64_t)lag_ms));
  FlexData units = FlexData::make_object();
  auto uo = units.as_object();
  std::vector<double> flops(bins);
  const double per_bin = (double)bin_ms / 1000.0 * 1e12;   // -> TFLOP/s
  for (unsigned u = 0; u < pf::kUnitCount; ++u) {
    const auto unit = (pf::Unit)u;
    const std::uint64_t from =
        pf::series(unit, end, bin_ms * kMs, bins, flops.data());
    const pf::Totals t = pf::totals(unit);
    FlexData one = FlexData::make_object();
    auto oo = one.as_object();
    FlexData rates = FlexData::make_array();
    auto ra = rates.as_array();
    for (double f : flops) { ra.push_back(f / per_bin); }
    oo.insert("tflops", std::move(rates));
    oo.insert("total_tflop", FlexData::make_real((double)t.flops / 1e12));
    oo.insert("records", FlexData::make_uint(t.records));
    oo.insert("latest_ms", FlexData::make_uint(t.latest_ns / kMs));
    oo.insert("complete_from_ms", FlexData::make_uint(from / kMs));
    uo.insert(pf::kUnitName[u], std::move(one));
  }
  o.insert("units", std::move(units));
  return doc;
}

}  // namespace vpipe
