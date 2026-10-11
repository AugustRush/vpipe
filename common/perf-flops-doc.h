#ifndef VPIPE_COMMON_PERF_FLOPS_DOC_H
#define VPIPE_COMMON_PERF_FLOPS_DOC_H

// The FLOP/s meter (common/perf-flops.h) as one document, for a UI that
// polls it: the web UI's GET /api/profiler/flops and the Python
// binding's perf_flops() both return exactly this.
//
//   {
//     "bin_ms": 1000, "bins": 60,
//     "end_ms": ...,        // the last bin's end, steady clock, ms
//     "end_unix_ms": ...,   // the same instant on the wall clock
//     "units": {
//       "gpu": {
//         "tflops": [ ... ],      // TFLOP/s per bin, oldest first
//         "total_tflop": ...,     // everything credited, in TFLOP
//         "records": ...,         // intervals credited
//         "latest_ms": ...,       // how far credits reach; 0 = none
//         "complete_from_ms": ... // 0, or where the ring's history starts
//       },
//       "ane": { ... }
//     }
//   }
//
// Every *_ms but end_unix_ms is on the meter's steady clock, which only
// orders and spaces things; place the bins with end_ms and bin_ms. Bins
// after latest_ms are still filling in (work is credited when it
// finishes), so a plot redraws the whole series each poll.
//
// `lag_ms` ends the window that long before now -- a reader that would
// rather show settled bins than fresh ones.

#include "common/flex-data.h"
#include "common/vpipe-api.h"

#include <cstddef>
#include <cstdint>

VPIPE_HOST_API_BEGIN

namespace vpipe {

FlexData perf_flops_document(std::uint64_t bin_ms, std::size_t bins,
                             std::uint64_t lag_ms);

}  // namespace vpipe

VPIPE_HOST_API_END

#endif
