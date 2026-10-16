#include "generative-models/shared/overlap-demix.h"

#include <algorithm>
#include <cmath>

namespace vpipe::genai {

namespace {

// torch.linspace(start, end, n) in f32, element for element: the first
// half counts up from `start`, the second down from `end`.
std::vector<float>
linspace_(float start, float end, int n)
{
  std::vector<float> v((std::size_t)std::max(n, 0));
  if (n == 1) { v[0] = start; }
  if (n <= 1) { return v; }
  const float step = (end - start) / (float)(n - 1);
  const int half = n / 2;
  for (int i = 0; i < n; ++i) {
    v[(std::size_t)i] = i < half ? start + step * (float)i
                                 : end - step * (float)(n - 1 - i);
  }
  return v;
}

}  // namespace

OverlapDemix::OverlapDemix(const Params& p) : _p(p)
{
  _ok = p.chunk > 0 && p.num_overlap >= 1 && p.channels > 0 && p.stems > 0;
  if (!_ok) { return; }
  _step = p.chunk / p.num_overlap;
  if (_step <= 0) {
    _ok = false;
    return;
  }
  _border = p.chunk - _step;
  _fade = p.chunk / 10;
  _window.assign((std::size_t)p.chunk, 1.0f);
  const std::vector<float> in = linspace_(0.0f, 1.0f, _fade);
  const std::vector<float> out = linspace_(1.0f, 0.0f, _fade);
  for (int k = 0; k < _fade; ++k) {
    _window[(std::size_t)(p.chunk - _fade + k)] = out[(std::size_t)k];
  }
  for (int k = 0; k < _fade; ++k) {
    _window[(std::size_t)k] = in[(std::size_t)k];
  }
  _pending.assign((std::size_t)p.channels, {});
  _tail.assign((std::size_t)p.channels, {});
  _sig.assign((std::size_t)p.channels, {});
  _acc.assign((std::size_t)p.stems * p.channels, {});
  _in.assign((std::size_t)p.channels * p.chunk, 0.0f);
  _out.assign((std::size_t)p.stems * p.channels * p.chunk, 0.0f);
}

void
OverlapDemix::append_p_(const float* planar, std::int64_t n,
                        std::int64_t stride)
{
  for (int c = 0; c < _p.channels; ++c) {
    const float* src = planar + (std::int64_t)c * stride;
    _sig[(std::size_t)c].insert(_sig[(std::size_t)c].end(), src, src + n);
  }
  _p_len += n;
}

void
OverlapDemix::decide_padding_(bool at_end)
{
  if (_decided) { return; }
  const bool long_enough = _border > 0 && _received > 2 * (std::int64_t)_border;
  if (!long_enough && !at_end) { return; }
  _decided = true;
  _padded = long_enough;
  const std::int64_t n = (std::int64_t)_pending[0].size();
  if (_padded) {
    // torch's reflect padding: P[j] = x[border - j], the first sample
    // itself not repeated.
    std::vector<float> left((std::size_t)_p.channels * _border);
    for (int c = 0; c < _p.channels; ++c) {
      const std::vector<float>& x = _pending[(std::size_t)c];
      for (int j = 0; j < _border; ++j) {
        left[(std::size_t)c * _border + j] = x[(std::size_t)(_border - j)];
      }
    }
    append_p_(left.data(), _border, _border);
  }
  std::vector<float> flat((std::size_t)_p.channels * n);
  for (int c = 0; c < _p.channels; ++c) {
    std::copy(_pending[(std::size_t)c].begin(),
              _pending[(std::size_t)c].end(),
              flat.begin() + (std::ptrdiff_t)c * n);
    _pending[(std::size_t)c].clear();
    _pending[(std::size_t)c].shrink_to_fit();
  }
  append_p_(flat.data(), n, n);
}

bool
OverlapDemix::push(const float* planar, std::int64_t n, std::int64_t stride,
                   const Forward& fwd)
{
  if (!_ok || _eos) { return false; }
  if (n <= 0) { return true; }
  const std::size_t keep = (std::size_t)_border + 1;
  for (int c = 0; c < _p.channels; ++c) {
    const float* src = planar + (std::int64_t)c * stride;
    std::vector<float>& t = _tail[(std::size_t)c];
    t.insert(t.end(), src, src + n);
    if (t.size() > keep) {
      t.erase(t.begin(), t.end() - (std::ptrdiff_t)keep);
    }
  }
  _received += n;
  if (_decided) {
    append_p_(planar, n, stride);
  } else {
    for (int c = 0; c < _p.channels; ++c) {
      const float* src = planar + (std::int64_t)c * stride;
      _pending[(std::size_t)c].insert(_pending[(std::size_t)c].end(), src,
                                      src + n);
    }
    decide_padding_(false);
  }
  return run_ready_(fwd);
}

bool
OverlapDemix::finish(const Forward& fwd)
{
  if (!_ok) { return false; }
  if (_eos) { return true; }
  _eos = true;
  decide_padding_(true);
  if (_padded) {
    // P[border + L0 + j] = x[L0 - 2 - j]; the tail holds x's last
    // border + 1 samples.
    std::vector<float> right((std::size_t)_p.channels * _border);
    for (int c = 0; c < _p.channels; ++c) {
      const std::vector<float>& t = _tail[(std::size_t)c];
      for (int j = 0; j < _border; ++j) {
        right[(std::size_t)c * _border + j] =
            t[(std::size_t)(_border - 1 - j)];
      }
    }
    append_p_(right.data(), _border, _border);
  }
  return run_ready_(fwd);
}

bool
OverlapDemix::run_ready_(const Forward& fwd)
{
  if (!_decided) { return true; }
  const std::int64_t C = _p.chunk;
  while (true) {
    const std::int64_t i = _next;
    bool last = false;
    if (_eos) {
      if (i >= _p_len) { break; }
      last = i + _step >= _p_len;
    } else if (!(_p_len >= i + C && _p_len > i + _step)) {
      // Not all of it yet -- or all of it but possibly the last chunk,
      // which only the end of the input can say.
      break;
    }
    if (!run_chunk_(fwd, last)) { return false; }
    _next += _step;
    finalize_to_(last ? _p_len : _next);
    if (last) { break; }
  }
  return true;
}

bool
OverlapDemix::run_chunk_(const Forward& fwd, bool last)
{
  const std::int64_t C = _p.chunk, i = _next;
  const std::int64_t cl = std::min(C, _p_len - i);
  const std::int64_t off = i - _base;
  for (int c = 0; c < _p.channels; ++c) {
    float* d = _in.data() + (std::size_t)c * C;
    const float* s = _sig[(std::size_t)c].data() + off;
    std::copy(s, s + cl, d);
    if (cl < C) {
      if (cl > C / 2) {
        for (std::int64_t j = 0; j < C - cl; ++j) { d[cl + j] = d[cl - 2 - j]; }
      } else {
        std::fill(d + cl, d + C, 0.0f);
      }
    }
  }
  if (!fwd(_in.data(), _out.data())) { return false; }

  std::vector<float> w = _window;
  if (i == 0) {
    std::fill(w.begin(), w.begin() + _fade, 1.0f);
  } else if (last) {
    std::fill(w.end() - _fade, w.end(), 1.0f);
  }
  const std::size_t need = (std::size_t)(off + cl);
  if (_cnt.size() < need) {
    _cnt.resize(need, 0.0f);
    for (auto& a : _acc) { a.resize(need, 0.0f); }
  }
  const std::size_t sc = (std::size_t)_p.stems * _p.channels;
  for (std::size_t k = 0; k < sc; ++k) {
    float* a = _acc[k].data() + off;
    const float* y = _out.data() + k * (std::size_t)C;
    for (std::int64_t j = 0; j < cl; ++j) { a[j] += y[j] * w[(std::size_t)j]; }
  }
  float* n = _cnt.data() + off;
  for (std::int64_t j = 0; j < cl; ++j) { n[j] += w[(std::size_t)j]; }
  ++_chunks;
  return true;
}

void
OverlapDemix::finalize_to_(std::int64_t p_end)
{
  _final = std::max(_final, p_end);
}

std::int64_t
OverlapDemix::available() const noexcept
{
  if (!_decided) { return 0; }
  const std::int64_t lo = _padded ? _border : 0;
  const std::int64_t hi = _eos ? lo + _received : _final;
  const std::int64_t a = std::min(_final, hi) - std::max(_taken, lo);
  return std::max<std::int64_t>(a, 0);
}

std::int64_t
OverlapDemix::take(std::vector<float>* stems, std::vector<float>* mix)
{
  const std::int64_t n = available();
  const std::int64_t lo = _padded ? _border : 0;
  const std::int64_t p0 = std::max(_taken, lo);
  const std::size_t sc = (std::size_t)_p.stems * _p.channels;
  if (stems != nullptr) { stems->assign(sc * (std::size_t)n, 0.0f); }
  if (mix != nullptr) {
    mix->assign((std::size_t)_p.channels * n, 0.0f);
  }
  for (std::int64_t j = 0; j < n; ++j) {
    const std::size_t at = (std::size_t)(p0 + j - _base);
    const float cnt = _cnt[at];
    for (std::size_t k = 0; k < sc && stems != nullptr; ++k) {
      // MSST divides and then maps NaN (0 / 0, where no window reached)
      // to 0.
      const float v = _acc[k][at] / cnt;
      (*stems)[k * (std::size_t)n + (std::size_t)j] = std::isnan(v) ? 0.0f : v;
    }
    for (int c = 0; c < _p.channels && mix != nullptr; ++c) {
      (*mix)[(std::size_t)c * n + (std::size_t)j] =
          _sig[(std::size_t)c][at];
    }
  }
  if (n > 0) { _taken = p0 + n; }
  // Drop what nothing will read again: below what was taken (the
  // output) and below the next chunk's start (the input).
  const std::int64_t drop = std::min(std::max(_taken, _decided ? lo : 0),
                                     _next) - _base;
  if (drop > 0) {
    for (auto& s : _sig) {
      s.erase(s.begin(), s.begin() + std::min<std::int64_t>(
                                         drop, (std::int64_t)s.size()));
    }
    const std::int64_t da = std::min<std::int64_t>(drop,
                                                   (std::int64_t)_cnt.size());
    for (auto& a : _acc) { a.erase(a.begin(), a.begin() + da); }
    _cnt.erase(_cnt.begin(), _cnt.begin() + da);
    _base += drop;
  }
  return n;
}

}  // namespace vpipe::genai
