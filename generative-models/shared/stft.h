#ifndef VPIPE_GENERATIVE_MODELS_SHARED_STFT_H
#define VPIPE_GENERATIVE_MODELS_SHARED_STFT_H

// torch.stft / torch.istft on the CPU, keeping the complex spectrum.
//
// The tree's other spectral front ends (WhisperFeatureExtractor, the
// Gemma-4 audio extractor) square the spectrum into power and drop the
// phase. A mask-based separator multiplies a complex mask into the
// complex spectrum and inverts it, so it needs both directions with the
// phase intact -- and with torch's exact conventions, because the
// network was trained on torch's frames and its mask is applied to
// them:
//
//   * a PERIODIC Hann window (torch.hann_window's default) of
//     win_length, zero-padded to n_fft about its centre;
//   * center: the signal reflect-padded by n_fft / 2 on each side
//     (torch's pad_mode default), giving 1 + n / hop frames;
//   * onesided: n_fft / 2 + 1 bins; normalized multiplies the forward
//     by n_fft^-1/2 and the inverse by its reciprocal;
//   * the inverse windows each irfft frame again, overlap-adds, divides
//     by the overlap-added SQUARED window, drops the centre padding and
//     returns exactly `length` samples -- the samples past the frames'
//     reach are zero, as torch pads them.
//
// f32 throughout; pocketfft does the transforms. One channel per call:
// a caller with several runs one call per channel, writing each into
// its own slot of whatever layout it keeps (see Layout).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vpipe::genai::stft {

struct Params {
  int  n_fft      = 2048;
  int  hop        = 512;
  int  win_length = 0;        // 0 = n_fft
  bool center     = true;
  bool normalized = false;
};

// Where bin `f` of frame `t` lives, in FLOATS: its real part at
// spec[t * t_stride + f * f_stride] and its imaginary part one after.
struct Layout {
  std::ptrdiff_t t_stride = 0;
  std::ptrdiff_t f_stride = 2;
};

class Stft {
 public:
  explicit Stft(const Params& p);

  // False for a geometry torch would refuse: n_fft < 2, hop < 1,
  // win_length outside [1, n_fft].
  bool valid() const noexcept { return _ok; }
  const Params& params() const noexcept { return _p; }
  const std::vector<float>& window() const noexcept { return _win; }

  int bins() const noexcept { return _p.n_fft / 2 + 1; }
  // Frames of an n-sample signal; 0 when it is too short (with center,
  // the reflect padding needs n > n_fft / 2, as torch requires).
  int frames(std::int64_t n) const noexcept;

  // Forward transform of x[0, n) into `spec`. False when frames(n) == 0.
  bool forward(const float* x, std::int64_t n, float* spec,
               const Layout& l) const;

  // Inverse of `frames` frames into y[0, length). False when the
  // overlap-added window falls below 1e-11 anywhere in the kept range
  // (torch's NOLA check; a valid Hann geometry never trips it).
  bool inverse(const float* spec, const Layout& l, int frames,
               std::int64_t length, float* y) const;

 private:
  Params             _p;
  std::vector<float> _win;    // n_fft long, the window centred in it
  bool               _ok = false;
};

}  // namespace vpipe::genai::stft

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_STFT_H
