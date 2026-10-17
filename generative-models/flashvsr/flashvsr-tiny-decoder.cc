#include "generative-models/flashvsr/flashvsr-tiny-decoder.h"

#include "common/vpipe-format.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace vpipe {
namespace genai {

namespace {

// The conditioning's geometry, the reference's PixelShuffle3d(4, 8, 8):
// four frames a latent frame, an 8x8 block of pixels a latent cell.
constexpr int kFf = 4;
constexpr int kBlk = 8;
constexpr int kLatentChannels = 16;

}  // namespace

std::unique_ptr<FlashVsrTinyDecoder>
FlashVsrTinyDecoder::load(std::shared_ptr<WeightSet> ws,
                          metal_compute::MetalCompute* mc, std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<FlashVsrTinyDecoder> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  std::string e;
  auto tae = MetalTaeDecoder::load(std::move(ws), mc, &e);
  if (tae == nullptr) { return fail(e); }
  // The conditioning rule fixes the shape: 4 frames and 8x8 pixels a
  // latent cell, 16 latent channels after the source's.
  const MetalTaeDecoder::Info& in = tae->info();
  if (in.latent_channels != kCondChannels + kLatentChannels ||
      in.time_upscale != kFf || in.spatial != kBlk || in.patch != 1 ||
      in.image_channels != 3 || !in.temporal) {
    return fail(fmt("not FlashVSR's tiny decoder: reads {} channels, {} "
                    "frames and {} pixels a latent cell, {} image channels "
                    "(want {} + {}, {}, {}, 3)", in.latent_channels,
                    in.time_upscale, in.spatial, in.image_channels,
                    kCondChannels, kLatentChannels, kFf, kBlk)());
  }
  std::unique_ptr<FlashVsrTinyDecoder> d(new FlashVsrTinyDecoder());
  const bool clamp = tae->clamps_input();
  for (int u = 0; u < 256; ++u) {
    // In f32, as the reference scales and clamps it.
    float v = (float)u / 127.5f - 1.0f;
    if (clamp) { v = std::tanh(v / 3.0f) * 3.0f; }
    d->_lut[u] = (_Float16)v;
  }
  d->_tae = std::move(tae);
  return d;
}

bool
FlashVsrTinyDecoder::plan(const std::string& file, int T, int h, int w,
                          std::size_t* weight_bytes,
                          std::size_t* scratch_bytes, std::string* err)
{
  MetalTaeDecoder::Plan p;
  if (!MetalTaeDecoder::plan(file, T, h, w, MetalTaeDecoder::Trim::kLeading,
                             &p, err)) {
    return false;
  }
  if (p.info.latent_channels != kCondChannels + kLatentChannels) {
    if (err != nullptr) { *err = "not FlashVSR's tiny decoder"; }
    return false;
  }
  if (weight_bytes != nullptr) { *weight_bytes = p.weight_bytes; }
  if (scratch_bytes != nullptr) { *scratch_bytes = p.scratch_bytes; }
  return true;
}

bool
FlashVsrTinyDecoder::decode(const float* z, int T, int h, int w,
                            const PictureOf& pic, const FrameSink& sink,
                            Stream* s, std::string* err)
{
  if (z == nullptr || s == nullptr || !pic || !sink) {
    if (err != nullptr) { *err = "nothing to decode"; }
    return false;
  }
  const int H = h * kBlk, W = w * kBlk;
  const std::size_t hw = (std::size_t)h * w;
  const std::size_t plane = (std::size_t)H * W;
  // Where this call sits in the stream: its first latent frame, and its
  // first decoded frame (decoded frames are counted from the stream's).
  const long long k0 = s->latent_frames;
  const long long f0 = s->frames;
  const bool clamp = _tae->clamps_input();
  bool short_pic = false;
  const MetalTaeDecoder::Input input = [&](int t, _Float16* d, int stride) {
    // THE SOURCE: the four frames latent frame K decodes to, 4K - 3 ..
    // 4K, the stream's frame 0 standing in for the ones before it; each
    // 8x8 block of a frame is 64 channels of its latent cell, channel
    // ((c * 4 + frame) * 8 + row) * 8 + column.
    const long long K = k0 + t;
    for (int ff = 0; ff < kFf; ++ff) {
      const long long g = std::max(0LL, 4 * K - 3 + ff);
      const Picture p = pic(g - f0);
      if (p.u8 == nullptr && p.f16 == nullptr && p.f32 == nullptr) {
        short_pic = true;
        return false;
      }
      for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
          for (int hh = 0; hh < kBlk; ++hh) {
            const std::size_t row =
                (std::size_t)c * plane + (std::size_t)(y * kBlk + hh) * W;
            _Float16* dp = d + (std::size_t)y * w * stride +
                           (std::size_t)((c * kFf + ff) * kBlk + hh) * kBlk;
            for (int x = 0; x < w; ++x) {
              _Float16* o = dp + (std::size_t)x * stride;
              const std::size_t at = row + (std::size_t)x * kBlk;
              if (p.u8 != nullptr) {
                const std::uint8_t* q = p.u8 + at;
                for (int k = 0; k < kBlk; ++k) { o[k] = _lut[q[k]]; }
                continue;
              }
              for (int k = 0; k < kBlk; ++k) {
                float v = 2.0f * (p.f16 != nullptr ? (float)p.f16[at + k]
                                                   : p.f32[at + k]) -
                          1.0f;
                if (clamp) { v = std::tanh(v / 3.0f) * 3.0f; }
                o[k] = (_Float16)v;
              }
            }
          }
        }
      }
    }
    // ...then the latent, as the denoiser left it.
    for (std::size_t px = 0; px < hw; ++px) {
      _Float16* o = d + px * (std::size_t)stride + kCondChannels;
      for (int c = 0; c < kLatentChannels; ++c) {
        float v = z[((std::size_t)c * T + t) * hw + px];
        if (clamp) { v = std::tanh(v / 3.0f) * 3.0f; }
        o[c] = (_Float16)v;
      }
    }
    return true;
  };
  const MetalTaeDecoder::FrameSink out = [&](long long frame, float* chw) {
    // The reference's own output scale: x * 2 - 1, unclamped.
    for (std::size_t i = 0; i < 3 * plane; ++i) {
      chw[i] = chw[i] * 2.0f - 1.0f;
    }
    return sink(frame - f0, chw);
  };
  const bool ok = _tae->decode_stream(T, h, w, MetalTaeDecoder::Trim::kLeading,
                                      input, out, s, err);
  if (!ok && short_pic && err != nullptr) {
    *err = "a source picture the decode needs was not supplied";
  }
  return ok;
}

}  // namespace genai
}  // namespace vpipe
