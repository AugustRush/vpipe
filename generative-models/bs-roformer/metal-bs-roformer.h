#ifndef VPIPE_GENERATIVE_MODELS_BS_ROFORMER_METAL_BS_ROFORMER_H
#define VPIPE_GENERATIVE_MODELS_BS_ROFORMER_METAL_BS_ROFORMER_H

// MetalBsRoformer -- the Band-Split RoPE Transformer music source
// separator (Lu et al. 2023, "Music Source Separation with Band-Split
// RoPE Transformer"), as MSST trains and ships it, on metal-compute.
//
// ONE FORWARD IS ONE CHUNK: planar PCM [channels][chunk_size] in, one
// estimate per stem out, [stems][channels][chunk_size]. Running a whole
// track is OverlapDemix's job (shared/overlap-demix.h), which is what
// the `audio-separate` stage drives.
//
// THE SHAPE OF IT. The chunk's complex STFT is cut into bands of
// adjacent bins; each band (all its bins, both channels, real and
// imaginary) is normalised and projected into the model width, giving a
// [frames, bands, dim] grid. `depth` layers of axial attention follow --
// a transformer along TIME for each band, then one along the BANDS for
// each frame, both with rotary positions -- and a per-band MLP head per
// stem turns the grid back into a complex mask over the same bins, which
// multiplies the mixture's spectrum before the inverse STFT.
//
// WHERE IT RUNS. The STFT, the mask multiply and the inverse are f32 on
// the host (shared/stft.h): they are cheap -- tens of milliseconds a
// chunk -- and keep the phase in f32. Everything between runs on the
// GPU in f16 with f32 accumulation, on the tree's shared kernels: the
// dense GEMMs (matmul2d on matrix cores, the steel tiles elsewhere),
// rms_norm_f16, gelu_erf_f16, and the steel flash attention
// (attn_steel_nax on M5, attn_steel on M4) at head dim 64. The grid
// stays TOKEN-MAJOR, [frames, bands]: both attentions read q, k and v
// out of the one qkv projection through strides -- time steps are
// `bands` rows apart, bands are adjacent -- and write back the same way,
// so no layer transposes anything. What only this model needs is in
// band_split_roformer.metal.
//
// The network is MSST's (see bs-roformer-config.h): plain pre-norm
// residual blocks, per-head sigmoid gates on the attention output,
// no output norm per transformer, one final RMSNorm.

#include "generative-models/bs-roformer/bs-roformer-config.h"
#include "generative-models/shared/dit-flops.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; class ComputeEncoder; }

namespace vpipe::genai {

class WeightSet;

class MetalBsRoformer {
 public:
  using Config = BsRoformerConfig;

  // Load from `ws`, which must hold the checkpoint `cfg` describes. The
  // model KEEPS the set. Null with `err` set on a missing tensor, a
  // shape that disagrees with the config, or a kernel that would not
  // load.
  static std::unique_ptr<MetalBsRoformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err);

  ~MetalBsRoformer();

  // What a test compares against the reference's intermediates, host
  // f32 in the golden's layouts. Filling them costs a sync per tap.
  struct Taps {
    std::vector<float> spec;          // [frames, frame_features]
    std::vector<float> band_split;    // [frames, bands, dim]
    std::vector<float> layer0_time;   // [frames, bands, dim]
    std::vector<float> layer0_freq;   // [frames, bands, dim]
    std::vector<float> final_norm;    // [frames, bands, dim]
    std::vector<float> mask;          // [stems, frames, frame_features]
  };

  // One chunk: `in` planar [channels][chunk_size], `out` planar
  // [stems][channels][chunk_size]. False with `err` set when the GPU
  // reported a failure.
  bool forward(const float* in, float* out, std::string* err,
               Taps* taps = nullptr);

  const Config& config() const noexcept { return _cfg; }

  // Bytes this model holds in weights, as converted (f16, the band
  // split f32).
  std::size_t weight_bytes() const noexcept { return _weight_bytes; }
  // Bytes one forward needs besides the weights, GPU and host.
  static std::size_t scratch_bytes(const Config& c);
  // FLOPs one forward counts: every projection at 2MNK, attention at
  // 4 * heads * pairs * head_dim. What the meter is credited per chunk.
  static std::uint64_t forward_flops(const Config& c);

  // Engagement, for tests: the GEMMs ran on matmul2d / the attention on
  // the NAX steel kernel.
  bool matrix_core_gemm() const noexcept { return _use_mma2; }
  bool nax_attention() const noexcept { return _attn_nax; }

  // What the forwards counted, for a test to hold against forward_flops
  // and the meter.
  const GpuFlopsTally& flops() const noexcept { return _flops; }

  // The last forward, split.
  struct Timing {
    double stft_ms = 0.0, gpu_ms = 0.0, istft_ms = 0.0;
  };
  const Timing& last_timing() const noexcept { return _timing; }

 private:
  MetalBsRoformer() = default;

  struct Linear {
    metal_compute::SharedBuffer w;    // [N, K] f16
    metal_compute::SharedBuffer b;    // [N] f16, may be empty
  };
  struct Attn {
    metal_compute::SharedBuffer norm;    // [D] f16
    metal_compute::SharedBuffer qkv;     // [3 * inner, D]
    metal_compute::SharedBuffer gate_w;  // [H, D]
    metal_compute::SharedBuffer gate_b;  // [H]
    metal_compute::SharedBuffer out;     // [D, inner]
    metal_compute::SharedBuffer freqs;   // [dim_head / 2] f32
  };
  struct Ff {
    metal_compute::SharedBuffer norm;    // [D]
    Linear up, down;
  };
  struct Block { Attn attn; Ff ff; };
  struct Layer { std::vector<Block> time, freq; };
  // One stem's head: per Linear of the MLP, every band's weight stacked
  // in band order -- [bands * hidden, K] for a hidden layer, [2 * Fin,
  // hidden] for the last.
  struct MaskHead { std::vector<Linear> layers; };

  bool load_weights_(std::string* err);
  bool alloc_scratch_(std::string* err);
  // y = x W^T (+ b). With `bias_done` set, a route whose GEMM takes no
  // bias (matmul2d) leaves it to the caller -- to fuse into its next
  // pass -- and says so through it.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const metal_compute::SharedBuffer& w, std::size_t we,
             const metal_compute::SharedBuffer* b, std::size_t be,
             const metal_compute::SharedBuffer& y, std::size_t ye,
             int M, int N, int K, bool* bias_done = nullptr);

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  std::shared_ptr<WeightSet> _ws;
  std::size_t _weight_bytes = 0;

  metal_compute::ComputeLibrary _lib_bsr, _lib_gemm, _lib_mma, _lib_elt,
      _lib_rms, _lib_attn;
  metal_compute::ComputeFunction _fn_split, _fn_rope, _fn_gate,
      _fn_bias_tanh, _fn_glu, _fn_bias_gelu, _fn_bias_res, _fn_rms,
      _fn_gelu, _fn_residual,
      _fn_transpose, _fn_bias_add, _fn_gemm_t, _fn_gemm_64, _fn_mma,
      _fn_mma_wide;
  metal_compute::ComputeFunction _fn_attn_time, _fn_attn_freq;
  bool _use_mma2 = false, _attn_nax = false;
  int  _mma_min_m = 64;

  // Weights.
  metal_compute::SharedBuffer _band_tab;     // int [bands, 2]
  metal_compute::SharedBuffer _band_gamma;   // f32 [Fin]
  metal_compute::SharedBuffer _band_wt;      // f32 [Fin, D]
  metal_compute::SharedBuffer _band_bias;    // f16 [bands, D]
  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _final_norm;   // [D]
  std::vector<MaskHead> _heads;
  std::vector<int> _band_off;                // feature offset per band
  std::vector<int> _band_len;                // features per band

  // Scratch, allocated once for the chunk geometry.
  metal_compute::SharedBuffer _spec;         // f32 [T, Fin], host-written
  metal_compute::SharedBuffer _x, _nx, _tmp; // f16 [N, D]
  metal_compute::SharedBuffer _big;          // qkv / FF / mask hidden
  metal_compute::SharedBuffer _big2;         // mask hidden, depth > 2
  metal_compute::SharedBuffer _att;          // [N, max(inner, D)]
  metal_compute::SharedBuffer _mo;           // f16 [T, 2 * Fin]
  metal_compute::SharedBuffer _mask;         // f32 [stems, T, Fin]
  metal_compute::SharedBuffer _skip;         // f16 [N, D], skip only
  metal_compute::SharedBuffer _ap_time, _ap_freq;   // attention params
  // Rotary (cos, sin) per (position, pair): frames for the time
  // attention, bands for the frequency one.
  metal_compute::SharedBuffer _rope_time, _rope_freq;
  std::vector<float> _yspec;                 // host [T, Fin]

  GpuFlopsTally _flops;
  Timing _timing;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_BS_ROFORMER_METAL_BS_ROFORMER_H
