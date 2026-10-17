#ifndef VPIPE_STAGES_WAVELET_COLOR_FIX_H
#define VPIPE_STAGES_WAVELET_COLOR_FIX_H

#ifdef VPIPE_BUILD_APPLE_SILICON

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <string>

namespace vpipe {

// THE WAVELET COLOUR FIX: a restored frame keeps its own detail and takes
// its LOW FREQUENCIES from the picture it restores --
//
//   out = clamp(x - low(x) + low(ref), -1, 1)
//
// over samples in [-1, 1], with `low` a stack of `levels` passes of a 3x3
// Gaussian whose taps are 1, 2, 4, ... pixels apart (edge replicated). It
// is FlashVSR+'s reference colour correction (its --cf), and where an
// AdaIN fix moves a frame's six channel statistics this moves its whole
// low-frequency picture, so a colour cast confined to part of the frame
// is corrected where it is.
//
// The passes run on the GPU (`wavelet_blur_f32`); the combine is one pass
// on the host. Buffers are kept for the geometry last used.
class WaveletColorFix {
 public:
  explicit WaveletColorFix(metal_compute::MetalCompute* mc);
  bool valid() const noexcept { return _fn.valid(); }

  // `x` planar [3, H, W] f32, corrected in place; `ref` the same shape.
  bool apply(float* x, const float* ref, int H, int W, int levels,
             std::string* err = nullptr);

 private:
  bool ensure_(int H, int W);

  metal_compute::MetalCompute*   _mc = nullptr;
  metal_compute::ComputeLibrary  _lib;
  metal_compute::ComputeFunction _fn;
  int _h = 0, _w = 0;
  metal_compute::SharedBuffer _a, _b, _c, _d;   // x, ref, two scratches
};

}  // namespace vpipe

#endif  // VPIPE_BUILD_APPLE_SILICON
#endif
