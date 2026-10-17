#ifndef VPIPE_GENERATIVE_MODELS_FLASHVSR_FLASHVSR_TINY_DECODER_H
#define VPIPE_GENERATIVE_MODELS_FLASHVSR_FLASHVSR_TINY_DECODER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/shared/metal-tae-decoder.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vpipe {
namespace genai {

class WeightSet;

// FlashVSR's TINY DECODER: TCDecoder.ckpt, the same file for v1.1 and
// FlashVSR+, and the FlashVSR+ reference's default decoder (the Wan VAE is
// its documented alternative).
//
// WHAT IT IS. A TAEHV video decoder (madebyollin's blocks: MemBlocks with
// one remembered frame, TGrow for time, nearest 2x upsamples) widened to
// [512, 256, 128, 128] channels, with an extra 3x3 conv + ReLU after each
// of its two top-level ReLUs -- all of which MetalTaeDecoder reads off the
// tensor names as it reads any TAEHV. What it adds is CONDITIONING: its
// first conv reads 784 channels a latent frame, the restored clip's own
// SOURCE at the output size pixel-shuffled 4 x 8 x 8 (768 channels: the
// four frames that latent frame decodes to, each 8x8 block of pixels)
// AHEAD of the 16 latent channels. So it does not have to invent detail
// the source already shows, and it is ~3.6 TFLOP an output frame at
// 1920x1152 against several times that for the Wan VAE.
//
// THE LATENT IS THE DENOISER'S OWN, decoded as it comes: no un-whitening
// (that is the Wan VAE's input convention, not this decoder's). Output is
// [-1, 1] -- the network's ~[0, 1] doubled, as the reference does, NOT
// clamped; the caller clamps where its output format needs it.
//
// THE CONDITIONING'S FRAMES. Latent frame K of a stream decodes to frames
// 4K - 3 .. 4K, the first latent frame to frame 0 alone; its 768 channels
// are those four frames' source pictures, frame 0 standing in for the
// three before it (the reference pads the clip's front with its first
// frame). A Stream continues across calls exactly as one call over the
// whole latent would: the MemBlocks' memories, and the front trim only
// at the stream's start.
class FlashVsrTinyDecoder {
public:
  // The file, in the layout's directory (FlashVsrLayout::tiny_decoder).
  static constexpr const char* kFile = "TCDecoder.ckpt";
  // Source channels per latent frame: RGB x 4 frames x an 8x8 block.
  static constexpr int kCondChannels = 3 * 4 * 8 * 8;

  // Null + *err for a file that is not this decoder (any other TAE, or a
  // geometry the conditioning rule does not fit).
  static std::unique_ptr<FlashVsrTinyDecoder>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       std::string* err);

  // A source picture at the output size, planar [3, H, W]: U8 (0..255,
  // the reference's x / 127.5 - 1), or F16 / F32 over [0, 1].
  struct Picture {
    const std::uint8_t* u8  = nullptr;
    const _Float16*     f16 = nullptr;
    const float*        f32 = nullptr;
  };
  // The picture for this call's decoded frame `d` (0 = the first frame
  // this call decodes): what the caller holds for that frame's source.
  using PictureOf = std::function<Picture(long long d)>;
  using Stream = MetalTaeDecoder::Stream;
  // A decoded frame: its index among this call's decoded frames, and the
  // frame planar f32 [3, H, W] in [-1, 1], unclamped -- the decoder's
  // buffer, the sink's to work on until it returns. False stops.
  using FrameSink = std::function<bool(long long d, float* chw)>;

  // Decode `z` [16, T, h, w] f32 -- this call's latent frames of the
  // stream `s` -- conditioned on the pictures, into H = 8h by W = 8w
  // frames: 4T - 3 at the stream's start, 4T after it. On false (an
  // error, or a sink that stopped) the stream cannot continue.
  bool decode(const float* z, int T, int h, int w, const PictureOf& pic,
              const FrameSink& sink, Stream* s, std::string* err);

  // Frames a call of T latent frames decodes: 4T - 3 opening a stream,
  // 4T continuing one.
  static int frames_for(int T, bool opens) noexcept
  {
    return T <= 0 ? 0 : (opens ? 4 * T - 3 : 4 * T);
  }

  // What the decoder holds: weights plus the scratch of the last geometry
  // it decoded, all of it kept between a stream's calls.
  std::size_t resident_bytes() const noexcept
  {
    return _tae ? _tae->resident_bytes() : 0;
  }

  // Weights and decode scratch for T latent frames of h x w, off the
  // file's tensor headers (nothing read or allocated): what a plan books.
  // False when the file is not readable as this decoder.
  static bool plan(const std::string& file, int T, int h, int w,
                   std::size_t* weight_bytes, std::size_t* scratch_bytes,
                   std::string* err = nullptr);

  const MetalTaeDecoder& tae() const noexcept { return *_tae; }

private:
  FlashVsrTinyDecoder() = default;

  std::unique_ptr<MetalTaeDecoder> _tae;
  // The source byte through the reference's scale and the decoder's input
  // clamp: f16(tanh((u / 127.5 - 1) / 3) * 3), one entry per byte.
  _Float16 _lut[256] = {};
};

}  // namespace genai
}  // namespace vpipe

#endif
