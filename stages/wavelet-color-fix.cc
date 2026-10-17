#include "stages/wavelet-color-fix.h"

#ifdef VPIPE_BUILD_APPLE_SILICON

#include <algorithm>
#include <cstring>

namespace vpipe {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

WaveletColorFix::WaveletColorFix(metal_compute::MetalCompute* mc) : _mc(mc)
{
  if (_mc == nullptr) { return; }
  _lib = _mc->load_library("llm_elementwise_bf16");
  if (_lib.valid()) { _fn = _lib.function("wavelet_blur_f32"); }
}

bool
WaveletColorFix::ensure_(int H, int W)
{
  if (H == _h && W == _w && !_a.empty()) { return true; }
  const std::size_t bytes = (std::size_t)3 * H * W * sizeof(float);
  _a = _mc->make_shared_buffer(bytes);
  _b = _mc->make_shared_buffer(bytes);
  _c = _mc->make_shared_buffer(bytes);
  _d = _mc->make_shared_buffer(bytes);
  if (_a.empty() || _b.empty() || _c.empty() || _d.empty()) {
    _h = _w = 0;
    return false;
  }
  _h = H;
  _w = W;
  return true;
}

bool
WaveletColorFix::apply(float* x, const float* ref, int H, int W, int levels,
                       std::string* err)
{
  auto fail = [&](const char* m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!valid()) { return fail("wavelet_blur_f32 unavailable"); }
  if (x == nullptr || ref == nullptr || H <= 0 || W <= 0 || levels <= 0) {
    return fail("nothing to correct");
  }
  if (!ensure_(H, W)) { return fail("wavelet buffer allocation failed"); }
  const std::size_t n = (std::size_t)3 * H * W;
  std::memcpy(_a.contents(), x, n * sizeof(float));
  std::memcpy(_b.contents(), ref, n * sizeof(float));
  // Each picture through `levels` passes, ping-ponging through its own
  // scratch: x in _a/_c, ref in _b/_d. Odd levels leave the result in the
  // scratch, even ones back in the input -- which the input no longer
  // needs: the combine reads `x` from the caller's own copy.
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(_fn);
    for (int pic = 0; pic < 2; ++pic) {
      SharedBuffer* from = pic == 0 ? &_a : &_b;
      SharedBuffer* to = pic == 0 ? &_c : &_d;
      for (int l = 0; l < levels; ++l) {
        enc.set_buffer(0, *from);
        enc.set_buffer(1, *to);
        enc.set_constant(2, H);
        enc.set_constant(3, W);
        enc.set_constant(4, 1 << l);
        enc.dispatch({(unsigned)W, (unsigned)H, 3u}, {32, 8, 1});
        std::swap(from, to);
      }
    }
    enc.end();
  }
  std::string ge;
  if (!stream.commit().wait_ok(&ge)) {
    if (err != nullptr) { *err = "wavelet low-pass failed: " + ge; }
    return false;
  }
  const bool odd = (levels % 2) != 0;
  const auto* lx = static_cast<const float*>(odd ? _c.contents()
                                                 : _a.contents());
  const auto* lr = static_cast<const float*>(odd ? _d.contents()
                                                 : _b.contents());
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::clamp(x[i] - lx[i] + lr[i], -1.0f, 1.0f);
  }
  return true;
}

}  // namespace vpipe

#endif  // VPIPE_BUILD_APPLE_SILICON
