#ifndef VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_FISH_CODEC_H
#define VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_FISH_CODEC_H

// MetalFishCodec -- Fish Audio's modded DAC, the codec.pth that ships with
// Fish Audio Drama 3 (and, byte for byte, S2 Pro): 44.1 kHz mono audio <->
// ten codebooks at 21.5 frames a second (2048 samples a frame).
//
// DECODE (the synthesis path): the ten codes of a frame through their
// codebooks and out_projs (one table lookup each -- a semantic codebook of
// 4096, nine residual ones of 1024), an 8-layer causal transformer with a
// 128-frame window, two (transposed conv x2 + ConvNeXt) upsamples to
// 86 Hz, then the DAC decoder: a 7-tap conv to 1536 channels and four
// Snake + transposed-conv (x8, x8, x4, x2) blocks, each followed by three
// residual units at dilations 1 / 3 / 9, to one tanh'd channel.
//
// ENCODE (voice cloning's reference audio): the mirror -- four strided
// DAC encoder blocks to 1024 channels (the last with a 4-layer transformer,
// window 512), two (strided conv + ConvNeXt) downsamples, the 8-layer
// pre-transformer, then the residual VQ: nearest normalised codebook entry
// per quantizer, the semantic one first.
//
// EVERY convolution is CAUSAL (left-padded), so a frame's audio depends on
// the frames before it only. The reference: fish-speech's
// fish_speech/models/dac/{modded_dac,rvq}.py + descript-audio-codec's
// layers, the yaml modded_dac_vq.yaml. Two details a port gets wrong
// without failing: the transformers' rotary table is rounded to BF16 (it is
// built in bf16 and kept so even in an fp32 model), and the encoder's
// transformer window is 512 (its config passes none; the default is 512),
// the quantizer's two 128.
//
// Activations are f16 with f32 arithmetic in every kernel; weights are
// folded (weight norm) and narrowed to f16 at load, as the Oobleck and
// BigVGAN decoders here do.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class WeightSet;

class MetalFishCodec {
public:
  static constexpr int kSampleRate = 44100;
  static constexpr int kHop = 2048;            // samples a frame
  static constexpr int kCodebooks = 10;

  // codec.pth, opened as a weight set (a torch zip maps as a shard).
  // `with_encoder` also loads the analysis path (~0.8 GB more at f16).
  static std::unique_ptr<MetalFishCodec> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      bool with_encoder, std::string* err = nullptr);
  ~MetalFishCodec();

  bool has_encoder() const { return _enc_ready; }

  // codes [frames][10] (codebook-fastest) -> mono f32 PCM, frames * 2048
  // samples at 44.1 kHz, the reference's tanh'd output.
  bool decode(const std::int32_t* codes, int frames, std::vector<float>* pcm,
              std::string* err = nullptr);
  // The decode's stages, channel-LAST [T, 1024] f32 (tests): after the
  // quantizer lookup, after the post-transformer, after the upsamples.
  bool decode_stages(const std::int32_t* codes, int frames,
                     std::vector<float>* vq, std::vector<float>* post,
                     std::vector<float>* up, std::string* err = nullptr);

  // STREAMED decode: the frames of one utterance in chunks, each chunk's
  // PCM out as it is decoded -- and the concatenation is decode() of all
  // of them. The codec is causal throughout, so a chunk needs only what
  // came before it: each post-transformer layer's keys and values of the
  // last 127 frames (its window), each ConvNeXt's last 6 input rows, and
  // the decoder's last 32 rows of 86 Hz input (more than its receptive
  // field). A stream is reset() to start the next utterance.
  struct DecodeStream {
    int max_chunk = 0;
    int pos = 0;                 // frames decoded so far
    int base = 0;                // the position of K/V row 0
    int cap = 0;                 // K/V rows a layer holds
    std::vector<metal_compute::SharedBuffer> k, v;   // per layer, f16
    metal_compute::SharedBuffer nx_hist[2];   // f16 [6, 1024] each
    int nx_rows[2] = {0, 0};
    metal_compute::SharedBuffer z_hist;       // f16 [32, 1024]
    int z_rows = 0;
    // Each ConvNeXt's input, its history rows first.
    metal_compute::SharedBuffer nx_buf[2];
    void reset() { pos = base = 0; nx_rows[0] = nx_rows[1] = z_rows = 0; }
  };
  std::unique_ptr<DecodeStream> decode_stream_begin(int max_chunk_frames);
  // `frames` (<= max_chunk) more frames -> their frames * 2048 samples.
  bool decode_stream_chunk(DecodeStream& st, const std::int32_t* codes,
                           int frames, std::vector<float>* pcm,
                           std::string* err = nullptr);

  // Mono f32 PCM at 44.1 kHz -> codes [ceil(n / 2048)][10]. The tail is
  // zero-padded to a whole frame, as the reference pads it.
  // Tests: `latent` the encoder's output [4 * frames, 1024], `down` after
  // the downsamples and `pre` the quantizer's input [frames, 1024], all
  // channel-last.
  bool encode(const float* pcm, int n, std::vector<std::int32_t>* codes,
              std::string* err = nullptr, std::vector<float>* pre = nullptr,
              std::vector<float>* latent = nullptr,
              std::vector<float>* down = nullptr);

  std::size_t resident_bytes() const;
  // What a load will hold, from codec.pth alone: the decoder's weights at
  // f16 and its codebook tables (+ the encoder's with `with_encoder`).
  // 0 when the file does not open.
  static std::size_t plan_bytes(const std::string& path, bool with_encoder);
  // The scratch a decode of `frames` frames (or an encode of as many)
  // allocates.
  static std::size_t decode_scratch_bytes(int frames);
  static std::size_t encode_scratch_bytes(int frames);

private:
  MetalFishCodec() = default;

  // A convolution folded (weight norm) into the GEMM layout, f16:
  // [cout, k * cin] in (k, cin) order, or [k * cout, cin] transposed.
  struct Conv {
    metal_compute::SharedBuffer w, b;
    int cin = 0, cout = 0, k = 1, dilation = 1, stride = 1;
    bool transposed = false;
    bool empty() const { return w.empty(); }
  };
  struct Snake {
    metal_compute::SharedBuffer ealpha, rbeta;   // f32 [C]
    int c = 0;
    bool empty() const { return ealpha.empty(); }
  };
  struct ResUnit { Snake a1, a2; Conv c1, c2; };
  struct DecBlock { Snake act; Conv up; ResUnit res[3]; };
  struct EncBlock { ResUnit res[3]; Snake act; Conv down; };
  struct TLayer {
    metal_compute::SharedBuffer wqkv, wo, w1, w3, w2;   // f16
    metal_compute::SharedBuffer anorm, fnorm;           // f16 [dim]
    metal_compute::SharedBuffer ascale, fscale;         // f32 [dim]
  };
  struct Transformer {
    std::vector<TLayer> layers;
    metal_compute::SharedBuffer norm;                   // f16 [dim]
    int window = 0;
    bool empty() const { return layers.empty(); }
  };
  struct ConvNeXt {
    metal_compute::SharedBuffer dw_w, dw_b;   // f32 [C, 7], [C]
    metal_compute::SharedBuffer ln_w, ln_b;   // f16 [C]
    metal_compute::SharedBuffer pw1, pw1_b, pw2, pw2_b;   // f16
    metal_compute::SharedBuffer gamma;        // f32 [C]
  };
  struct Resample { Conv conv; ConvNeXt nx; };

  bool init_(const std::shared_ptr<WeightSet>& ws, bool with_encoder,
             std::string* err);
  bool init_encoder_(std::string* err);
  Conv conv_(const std::string& nm, bool transposed, int dilation,
             int stride);
  Snake snake_(const std::string& nm);
  Transformer transformer_(const std::string& nm, int n_layer, int window);
  ConvNeXt convnext_(const std::string& nm);
  metal_compute::SharedBuffer f16_(const std::string& nm);
  metal_compute::SharedBuffer f32_(const std::string& nm);

  bool ensure_rope_(int frames);
  // y[ye..] = x[xe..] @ w^T (+ bias); offsets in elements.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer* bias,
             const metal_compute::SharedBuffer& y, std::size_t ye,
             int M, int N, int K, std::size_t xe = 0);
  // copy `rows` rows of 1024 from src (from row s0) to dst (at row d0).
  void copy_rows_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& src, int s0,
                  const metal_compute::SharedBuffer& dst, int d0, int rows);
  // A (strided) causal conv over T rows of `in` -> `out` (T / stride rows).
  void conv_run_(metal_compute::ComputeEncoder& enc, const Conv& c,
                 const metal_compute::SharedBuffer& in,
                 const metal_compute::SharedBuffer& out, int T);
  // A causal transposed conv: T rows (from row in0) -> T * stride (at row
  // out0).
  void convt_run_(metal_compute::ComputeEncoder& enc, const Conv& c,
                  const metal_compute::SharedBuffer& in,
                  const metal_compute::SharedBuffer& out, int T,
                  int in0 = 0, int out0 = 0);
  void snake_run_(metal_compute::ComputeEncoder& enc, const Snake& s,
                  const metal_compute::SharedBuffer& in,
                  const metal_compute::SharedBuffer& out, int T);
  // x += unit(x) over T rows (scratch a, b).
  void resunit_run_(metal_compute::ComputeEncoder& enc, const ResUnit& r,
                    const metal_compute::SharedBuffer& x,
                    const metal_compute::SharedBuffer& a,
                    const metal_compute::SharedBuffer& b, int T);
  // In place over x [T, 1024]. With `st`, x is the next T frames of a
  // stream: rows at positions st->pos.., their keys and values appended to
  // the stream's history (room made beforehand by the caller).
  bool transformer_run_(metal_compute::ComputeEncoder& enc,
                        const Transformer& tf,
                        const metal_compute::SharedBuffer& x, int T,
                        DecodeStream* st = nullptr);
  // The conv decoder over `rows` 86 Hz rows in _d -> rows * 512 samples
  // in _c (f16); the caller keeps what it needs.
  void decoder_run_(metal_compute::ComputeEncoder& enc, int rows);
  // x [T, C] -> x + ConvNeXt(x), in place.
  void convnext_run_(metal_compute::ComputeEncoder& enc, const ConvNeXt& n,
                     const metal_compute::SharedBuffer& x, int T);
  bool ensure_scratch_(std::size_t act, std::size_t tf_rows);
  bool ensure_fold_(std::size_t elems);

  metal_compute::MetalCompute* _mc = nullptr;
  std::shared_ptr<WeightSet> _ws;
  int _dim = 1024;                 // the latent / transformer width
  int _sem = 4096, _res = 1024;    // codebook sizes

  // decode
  metal_compute::SharedBuffer _tables;      // f32 [4096 + 9*1024, 1024]
  Transformer _post;
  Resample _up[2];
  Conv _dec_in, _dec_out;
  DecBlock _dec[4];
  Snake _dec_act;
  // encode
  bool _enc_ready = false;
  Conv _enc_in, _enc_out;
  EncBlock _enc[4];
  Transformer _enc_tf, _pre;
  Snake _enc_act;
  Resample _down[2];
  metal_compute::SharedBuffer _win, _bin, _cbn;   // VQ in_projs, codebooks

  // The rotary table, cos | sin per pair, bf16-rounded, f32 [rows, 64].
  metal_compute::SharedBuffer _rope;
  int _rope_rows = 0;

  // Scratch.
  std::size_t _act_cap = 0, _tf_cap = 0, _col_cap = 0;
  metal_compute::SharedBuffer _a, _b, _c, _d, _col, _fold;
  metal_compute::SharedBuffer _tq, _tk, _tv, _tatt, _tqkv, _th, _tf1, _tf3;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_avae, _lib_fd,
      _lib_rms, _lib_sdpa, _lib_mma;
  metal_compute::ComputeFunction _fn_gemm, _fn_im2col, _fn_col2im,
      _fn_snake, _fn_add, _fn_vq_dec, _fn_vq_enc, _fn_qkv, _fn_heads,
      _fn_scale_add, _fn_dw, _fn_ln, _fn_gelu, _fn_rms, _fn_swiglu,
      _fn_flash, _fn_mma, _fn_mma_deep, _fn_bias_rows, _fn_copy;
  bool _use_mma = false;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_FISH_CODEC_H
