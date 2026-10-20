#ifndef VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_DRAMA3_TOWER_H
#define VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_DRAMA3_TOWER_H

// MetalDrama3Tower -- Fish Audio Drama 3's codebook-MTP audio decoder: the
// thirteen layers that turn the backbone's hidden state at a position into
// codes 1..9 of the frame there (code 0 is the backbone's own pick).
//
// Each layer MIXES one code embedding into the hidden state from below,
// u = eh_proj([enorm(e); hnorm(h)]), then runs one pre-norm transformer
// block whose causal attention covers the WHOLE sequence -- so every
// layer keeps K/V for every row, prompt included -- and closes with its
// own final norm, h' = final_norm(u + attn + ffn). The first four layers
// are shared: each embeds code 0 of the frame being decoded, the first
// also adds that row's INPUT frame's codes 2..9. Layer 4 + j embeds code
// j and is followed by codebook head j, which predicts code j + 1.
//
// The reference: vllm-omni's codebook_tower.py and sglang-omni's
// decoder_layers.py (they agree). Its attention: 16 query heads over 2
// K/V heads of 256, q/k zero-centred RMSNorm, GPT-J (interleaved) rotary
// on the first 64 channels at base 1e7, a per-head sigmoid output gate --
// the row of wqkv is [q | gate | k | v]. SwiGLU 6144 wide.
//
// Rows are run in TWO ways through the same kernels. Teacher-forced
// (encode_rows): every row's next frame is known -- a prompt, or a
// golden -- and the codes come from it. Sampled (encode_step): one row,
// and after each per-codebook layer its head is sampled on the GPU into
// the frame the next layer reads, so a whole frame is one command buffer.

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

class MetalDrama3Tower {
public:
  struct Config {
    int   dim          = 2048;
    int   n_head       = 16;
    int   n_kv         = 2;
    int   head_dim     = 256;
    int   ffn          = 6144;
    float eps          = 1e-6f;
    int   num_codebooks = 10;
    int   codebook_size = 4096;
    int   shared_layers = 4;
    float rope_base    = 1.0e7f;
    int   rotary_dim   = 64;
    // Codebooks the backbone sees beside code 0 (code 1); the first layer
    // adds the input frame's codes from here to num_codebooks - 1.
    int   backbone_codebooks = 1;
    int   sem0 = 248099;          // semantic_start_token_id
    int   sem1 = 252194;          // semantic_end_token_id
    int layers() const { return shared_layers + num_codebooks - 1; }
    int qkv_width() const {
      return 2 * n_head * head_dim + 2 * n_kv * head_dim;
    }
  };

  // The draw parameters of codes 1..9 (the reference draws them with the
  // request's settings).
  struct Draw {
    float    temperature = 1.0f;
    int      top_k       = 30;
    float    top_p       = 0.9f;
    std::uint32_t seed   = 0;
    std::uint32_t frame  = 0;     // noise stream of this frame
  };

  static std::unique_ptr<MetalDrama3Tower> load(
      const std::shared_ptr<WeightSet>& ws, metal_compute::MetalCompute* mc,
      const Config& cfg, std::string* err = nullptr);
  ~MetalDrama3Tower();

  const Config& config() const { return _cfg; }

  // Make room for `rows` positions of K/V (grows, never shrinks; a grow
  // keeps what is cached). False on allocation failure.
  bool reserve(int rows);
  int  capacity() const { return _cap; }

  // Teacher-forced rows [pos0, pos0 + n): h [n, dim] the backbone's final-
  // normed hidden for each row, frames [n, 11] the rows' input frames,
  // labels [n, 11] each row's NEXT frame. Writes every layer's K/V for the
  // rows; with `head_logits` ([9, n, codebook_size], compute dtype, head-
  // major) also every row's nine codebook-head logits.
  bool encode_rows(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& h,
                   const metal_compute::SharedBuffer& frames,
                   const metal_compute::SharedBuffer& labels, int n, int pos0,
                   const metal_compute::SharedBuffer* head_logits = nullptr);

  // One sampled frame at `pos`: h [1, dim]; in_frame the frame the backbone
  // read there; out_frame the frame being decoded, whose [0] and [1] (main
  // token, code 0) the main sampler has written -- codes 1..9 land in it.
  bool encode_step(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& h,
                   const metal_compute::SharedBuffer& in_frame,
                   const metal_compute::SharedBuffer& out_frame, int pos,
                   const Draw& draw);

  // y[M, N] = x[M, K] @ w[N, K]^T on the tower's GEMM / GEMV routes (bf16,
  // f32 accumulation) -- also the model's main head. Byte offsets.
  void encode_gemm(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer& x, std::size_t xoff,
                   const metal_compute::SharedBuffer& y, std::size_t yoff,
                   int K, int N, int M);
  // The codebook embedding table [10*codebook_size, dim] (the backbone's
  // code-1 rows read it too).
  const metal_compute::SharedBuffer& codebook_embeddings() const
  { return _ecb; }

  std::size_t resident_bytes() const;
  // K/V bytes reserved now (capacity, all layers).
  std::size_t kv_bytes() const;
  // What `rows` positions of K/V cost, from a config alone.
  static std::size_t kv_bytes_for(const Config& c, int rows);

private:
  MetalDrama3Tower() = default;

  struct Layer {
    metal_compute::SharedBuffer emb;        // [cbsize, dim] code table
    metal_compute::SharedBuffer enorm, hnorm, attn_norm, ffn_norm, fnorm;
    metal_compute::SharedBuffer qn, kn;     // [head_dim]
    metal_compute::SharedBuffer eh;         // [dim, 2*dim]
    metal_compute::SharedBuffer wqkv;       // [qkv_width, dim]
    metal_compute::SharedBuffer wo;         // [dim, n_head*head_dim]
    metal_compute::SharedBuffer w13;        // [2*ffn, dim] gate/up rows
                                            // interleaved
    metal_compute::SharedBuffer w2;         // [dim, ffn]
    metal_compute::SharedBuffer head;       // [cbsize, dim] (per-codebook)
    metal_compute::SharedBuffer kc, vc;     // [n_kv, cap, head_dim]
  };

  bool init_(const std::shared_ptr<WeightSet>& ws, std::string* err);
  bool ensure_scratch_(int n);
  // The shared body: rows [pos0, pos0+n); `labels` holds the codes each
  // layer embeds. Sampled when `draw` is set (n must be 1).
  bool encode_(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& h,
               const metal_compute::SharedBuffer& frames,
               const metal_compute::SharedBuffer& labels, int n, int pos0,
               const metal_compute::SharedBuffer* head_logits,
               const Draw* draw);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;
  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _ecb;        // [10*cbsize, dim] (layer 0 extra)
  metal_compute::SharedBuffer _rope;       // f32 [cap, rotary_dim] cos|sin
  int _cap = 0;
  bool _use_mma = false;

  // Scratch for up to _scratch_rows rows.
  int _scratch_rows = 0;
  metal_compute::SharedBuffer _cat, _u, _xn, _qkv, _qh, _att, _o, _ao, _r,
      _fn, _gu, _act, _dn, _h0, _h1, _logits;

  metal_compute::ComputeLibrary _lib_d3, _lib_dense, _lib_mma, _lib_sdpa,
      _lib_elt;
  metal_compute::ComputeFunction _fn_mix, _fn_add_norm, _fn_qk_rope,
      _fn_gate_out, _fn_gemv, _fn_gemm, _fn_mma, _fn_mma_deep, _fn_sdpa_mb,
      _fn_sdpa_mma, _fn_swiglu1, _fn_swiglu_il, _fn_sample_code;
  metal_compute::SharedBuffer _zero_eid;   // one 0 (single-slab gather)
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DRAMA3_METAL_DRAMA3_TOWER_H
