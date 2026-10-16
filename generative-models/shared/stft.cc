#include "generative-models/shared/stft.h"

#include "3rd-party/pocketfft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace vpipe::genai::stft {

namespace {

// Frames are transformed in batches of this many, so the scratch stays
// a few MB however long the signal is.
constexpr int kBatch = 256;

}  // namespace

Stft::Stft(const Params& p) : _p(p)
{
  if (_p.win_length <= 0) { _p.win_length = _p.n_fft; }
  _ok = _p.n_fft >= 2 && _p.hop >= 1 && _p.win_length <= _p.n_fft;
  if (!_ok) { return; }
  // torch.hann_window(win_length), periodic, then centred in n_fft as
  // torch.stft pads a shorter window. Computed in double, rounded once.
  _win.assign((std::size_t)_p.n_fft, 0.0f);
  const int left = (_p.n_fft - _p.win_length) / 2;
  for (int i = 0; i < _p.win_length; ++i) {
    const double a = 2.0 * std::numbers::pi * (double)i /
                     (double)_p.win_length;
    _win[(std::size_t)(left + i)] = (float)(0.5 - 0.5 * std::cos(a));
  }
}

int
Stft::frames(std::int64_t n) const noexcept
{
  if (!_ok || n <= 0) { return 0; }
  if (_p.center) {
    if (n <= _p.n_fft / 2) { return 0; }
    return (int)(1 + n / _p.hop);
  }
  if (n < _p.n_fft) { return 0; }
  return (int)(1 + (n - _p.n_fft) / _p.hop);
}

bool
Stft::forward(const float* x, std::int64_t n, float* spec,
              const Layout& l) const
{
  const int T = frames(n);
  if (T == 0 || x == nullptr || spec == nullptr) { return false; }
  const int N = _p.n_fft, F = bins(), half = _p.center ? N / 2 : 0;
  // Sample `j` of the padded signal: reflected about the first and last
  // samples (torch's "reflect", which does not repeat the edge).
  auto at = [&](std::int64_t j) -> float {
    j -= half;
    if (j < 0) { j = -j; }
    if (j >= n) { j = 2 * (n - 1) - j; }
    return x[j];
  };
  const float scale = _p.normalized ? (float)(1.0 / std::sqrt((double)N))
                                    : 1.0f;
  std::vector<float> frm((std::size_t)kBatch * N);
  std::vector<std::complex<float>> out((std::size_t)kBatch * F);
  for (int t0 = 0; t0 < T; t0 += kBatch) {
    const int nb = std::min(kBatch, T - t0);
    for (int b = 0; b < nb; ++b) {
      const std::int64_t s0 = (std::int64_t)(t0 + b) * _p.hop;
      float* d = frm.data() + (std::size_t)b * N;
      // Interior frames read straight; only the edges reflect.
      if (s0 >= half && s0 - half + N <= n) {
        const float* src = x + (s0 - half);
        for (int i = 0; i < N; ++i) { d[i] = src[i] * _win[(std::size_t)i]; }
      } else {
        for (int i = 0; i < N; ++i) {
          d[i] = at(s0 + i) * _win[(std::size_t)i];
        }
      }
    }
    pocketfft::shape_t shape{(std::size_t)nb, (std::size_t)N};
    pocketfft::stride_t sin{(std::ptrdiff_t)(N * sizeof(float)),
                            (std::ptrdiff_t)sizeof(float)};
    pocketfft::stride_t sout{
        (std::ptrdiff_t)(F * sizeof(std::complex<float>)),
        (std::ptrdiff_t)sizeof(std::complex<float>)};
    pocketfft::r2c<float>(shape, sin, sout, /*axis=*/1, /*forward=*/true,
                          frm.data(), out.data(), scale);
    for (int b = 0; b < nb; ++b) {
      float* dst = spec + (std::ptrdiff_t)(t0 + b) * l.t_stride;
      const std::complex<float>* src = out.data() + (std::size_t)b * F;
      for (int f = 0; f < F; ++f) {
        dst[(std::ptrdiff_t)f * l.f_stride]     = src[f].real();
        dst[(std::ptrdiff_t)f * l.f_stride + 1] = src[f].imag();
      }
    }
  }
  return true;
}

bool
Stft::inverse(const float* spec, const Layout& l, int T,
              std::int64_t length, float* y) const
{
  if (!_ok || spec == nullptr || y == nullptr || T <= 0 || length <= 0) {
    return false;
  }
  const int N = _p.n_fft, F = bins();
  const std::int64_t full = (std::int64_t)N + (std::int64_t)_p.hop * (T - 1);
  const std::int64_t start = _p.center ? N / 2 : 0;
  const std::int64_t end = std::min(full, start + length);
  // Overlap-add in double: the envelope and the signal are each a sum of
  // up to n_fft / hop terms, and the division below is where their
  // rounding would meet.
  std::vector<double> acc((std::size_t)full, 0.0);
  std::vector<double> env((std::size_t)full, 0.0);
  const float scale = _p.normalized ? (float)std::sqrt((double)N)
                                    : 1.0f;
  std::vector<std::complex<float>> in((std::size_t)kBatch * F);
  std::vector<float> frm((std::size_t)kBatch * N);
  for (int t0 = 0; t0 < T; t0 += kBatch) {
    const int nb = std::min(kBatch, T - t0);
    for (int b = 0; b < nb; ++b) {
      const float* src = spec + (std::ptrdiff_t)(t0 + b) * l.t_stride;
      std::complex<float>* d = in.data() + (std::size_t)b * F;
      for (int f = 0; f < F; ++f) {
        d[f] = {src[(std::ptrdiff_t)f * l.f_stride],
                src[(std::ptrdiff_t)f * l.f_stride + 1]};
      }
    }
    pocketfft::shape_t shape{(std::size_t)nb, (std::size_t)N};
    pocketfft::stride_t sin{
        (std::ptrdiff_t)(F * sizeof(std::complex<float>)),
        (std::ptrdiff_t)sizeof(std::complex<float>)};
    pocketfft::stride_t sout{(std::ptrdiff_t)(N * sizeof(float)),
                             (std::ptrdiff_t)sizeof(float)};
    // irfft's "backward" normalisation: 1 / n_fft.
    pocketfft::c2r<float>(shape, sin, sout, /*axis=*/1, /*forward=*/false,
                          in.data(), frm.data(),
                          scale / (float)N);
    for (int b = 0; b < nb; ++b) {
      const std::int64_t s0 = (std::int64_t)(t0 + b) * _p.hop;
      const float* src = frm.data() + (std::size_t)b * N;
      for (int i = 0; i < N; ++i) {
        const double w = _win[(std::size_t)i];
        acc[(std::size_t)(s0 + i)] += (double)src[i] * w;
        env[(std::size_t)(s0 + i)] += w * w;
      }
    }
  }
  for (std::int64_t j = start; j < end; ++j) {
    if (std::fabs(env[(std::size_t)j]) < 1e-11) { return false; }
  }
  for (std::int64_t j = 0; j < length; ++j) {
    const std::int64_t s = start + j;
    y[j] = s < end ? (float)(acc[(std::size_t)s] / env[(std::size_t)s])
                   : 0.0f;
  }
  return true;
}

}  // namespace vpipe::genai::stft
