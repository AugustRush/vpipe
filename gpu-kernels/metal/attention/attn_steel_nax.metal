// attn_steel_nax.metal -- MLX's MATRIX-CORE (NAX) steel flash-attention,
// vendored for vpipe's metal-compute path (T=half, head_dim 64). This is the
// kernel MLX itself dispatches on M5 (sdpa_full_self_attention_nax, bq=64,
// bk=32, wm=4): QK^T and P*V run on the hardware matrix units via MPP
// matmul2d (nax.h), with the online softmax done entirely on the register-
// resident MMATile (row_reduce / row_bin_op -- no threadgroup round-trip).
// It is the tuned register-blocked flash; vpipe's hand-rolled tg-staged
// sdpa_full_mma2_d64 lost to the (ALU) steel kernel, whereas THIS uses the
// matrix cores stock-mlx can't, so it should beat the no-NAX reference.
//
// Buffer/param contract (identical to attn_steel, i.e. MLX steel `attention`):
//   0:Q 1:K 2:V 3:O (half)  4:AttnParams*  (5:AttnMaskParams 6:mask 7:sinks
//   function-constant-gated off). Func constants: align_Q(200), align_K(201),
//   has_mask(300)=0, do_causal(301)=0, has_sinks(302)=0. scale in AttnParams
//   is plain 1/sqrt(D). Grid: threadgroups (NQ=ceil(qL/64), H, B), tg
//   (32, wm=4, wn=1).  M5-only (matmul2d): #if __HAVE_TENSOR__, stub else.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

using namespace metal;

#ifndef M_LOG2E_F
#define M_LOG2E_F 1.44269504088896340736f
#endif

template <typename U>
struct Limits {
  static constexpr constant U max = metal::numeric_limits<U>::max();
  static constexpr constant U min = metal::numeric_limits<U>::min();
  static constexpr constant U finite_max = metal::numeric_limits<U>::max();
  static constexpr constant U finite_min = metal::numeric_limits<U>::min();
};
template <>
struct Limits<float> {
  static constexpr constant float max =
      metal::numeric_limits<float>::infinity();
  static constexpr constant float min =
      -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max =
      metal::numeric_limits<float>::max();
  static constexpr constant float finite_min =
      -metal::numeric_limits<float>::max();
};
template <>
struct Limits<half> {
  static constexpr constant half max = metal::numeric_limits<half>::infinity();
  static constexpr constant half min =
      -metal::numeric_limits<half>::infinity();
  static constexpr constant half finite_max =
      metal::numeric_limits<half>::max();
  static constexpr constant half finite_min =
      -metal::numeric_limits<half>::max();
};
template <>
struct Limits<bfloat> {
  static constexpr constant bfloat max =
      metal::numeric_limits<bfloat>::infinity();
  static constexpr constant bfloat min =
      -metal::numeric_limits<bfloat>::infinity();
  static constexpr constant bfloat finite_max =
      metal::numeric_limits<bfloat>::max();
  static constexpr constant bfloat finite_min =
      -metal::numeric_limits<bfloat>::max();
};

#if defined(__HAVE_TENSOR__)

#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

// head_dim 64 (Qwen3-VL vision tower / Qwen3-ASR audio encoder). bq=64,
// bk=32, wm=4, wn=1 -- the MLX M5 default for the NAX attention.
template [[host_name("attn_steel_nax_h_bd64")]] [[kernel]] decltype(attention_nax<
                                                                    half,
                                                                    64,
                                                                    32,
                                                                    64,
                                                                    4,
                                                                    1,
                                                                    half,
                                                                    float>)
attention_nax<half, 64, 32, 64, 4, 1, half, float>;

// head_dim 128 (Krea-2 MMDiT joint attention -- GQA 48q/12kv). bq=64 (the NAX
// static_assert needs BQ >= kNWarps*kU = 4*16 = 64, a multiple of 64), bk=32,
// bd=128 -> TD = BD/kU = 8. Same AttnParams + func-const contract as bd64; only
// the head dim (and thus the register-resident O tile) grows.
template [[host_name("attn_steel_nax_h_bd128")]] [[kernel]] decltype(attention_nax<
                                                                     half,
                                                                     64,
                                                                     32,
                                                                     128,
                                                                     4,
                                                                     1,
                                                                     half,
                                                                     float>)
attention_nax<half, 64, 32, 128, 4, 1, half, float>;

// bf16 twin of the head_dim-64 NAX attention, for the MiniMax-H3 video VAE:
// its ViT half is 64-wide per head and the whole VAE runs bf16, so neither
// the f16 bd64 entry above nor the bf16 bd128 entry below fits it. Adding the
// instantiation is the whole of the change -- the kernel is dtype-generic and
// accumulates in f32 either way.
template [[host_name("attn_steel_nax_h_bd64_bf16")]] [[kernel]] decltype(attention_nax<
                                                                    bfloat,
                                                                    64,
                                                                    32,
                                                                    64,
                                                                    4,
                                                                    1,
                                                                    bfloat,
                                                                    float>)
attention_nax<bfloat, 64, 32, 64, 4, 1, bfloat, float>;

// bf16 twin of the head_dim-128 NAX (M5 matrix-core) attention, for the FLUX.2
// DiT: its residual stream overflows f16 range, so the whole DiT runs bf16 and
// its joint attention Q/K/V are bf16. Same kernel, T=bfloat (accumulation f32).
// Restores the M5 nax speedup for flux2 (the non-nax attn_steel_h_bd128_bf16 is
// the M4/fallback path).
template [[host_name("attn_steel_nax_h_bd128_bf16")]] [[kernel]] decltype(attention_nax<
                                                                     bfloat,
                                                                     64,
                                                                     32,
                                                                     128,
                                                                     4,
                                                                     1,
                                                                     bfloat,
                                                                     float>)
attention_nax<bfloat, 64, 32, 128, 4, 1, bfloat, float>;

// THE SURVIVING LONG-SEQUENCE VARIANT at bd 128. MEASURED on an M5 Pro and a
// 10-core M5 Air, 56x128 / 32x128 / 48q-12kv x128, 9k-100k rows, every arm
// bit-identical: BQ 256 lost on every machine, shape and length (0.85-1.11x),
// and the head-dim split (#3842) reached only 1.02-1.08x over BQ 64 where this
// tile reaches 1.21x -- dominated, so neither is built. The split stays at bd
// 256, where it wins and this tile is not the lever.
//
// WIDER QUERY TILE at bd 128, bf16 (BQ 128 = WM 8 x kU 16, TQ still 1). Each
// simdgroup reads K/V straight from device, so a threadgroup covering 128
// query rows walks one K strip with 8 simdgroups at once -- the K/V re-read
// per query tile is what caps long-sequence attention (51 MB of K/V per head at
// 100k rows, streamed once per tile). Instantiated to be measured against
// BQ 64; nothing binds it.
template [[host_name("attn_steel_nax_h_bd128_bq128_bf16")]] [[kernel]] decltype(attention_nax<
                                                                     bfloat,
                                                                     128,
                                                                     32,
                                                                     128,
                                                                     8,
                                                                     1,
                                                                     bfloat,
                                                                     float>)
attention_nax<bfloat, 128, 32, 128, 8, 1, bfloat, float>;


// head_dim 256, bf16: the plain kernel and MLX's head-dim SPLIT variant
// (#3842), which halves each simdgroup's accumulator set by splitting D across
// WN = 2 simdgroups. bd 256 ONLY, on measurement: on the M5 Pro, bidirectional
// bf16, the split was 0.78-1.02x the plain kernel at bd 128 (H3 DiT 56x128,
// FLUX.2-9B 32x128, Krea-2 48q/12kv x128, 2.3k-19k rows) and 0.67-0.76x at bd 64
// (H3 video VAE 32x64) -- the accumulator set is already small there. Instantiated to be measured against each other at
// Qwen3.5 / Gemma-4 prefill shapes -- nothing binds them yet.
template [[host_name("attn_steel_nax_h_bd256_bf16")]] [[kernel]] decltype(attention_nax<
                                                                     bfloat,
                                                                     64,
                                                                     32,
                                                                     256,
                                                                     4,
                                                                     1,
                                                                     bfloat,
                                                                     float>)
attention_nax<bfloat, 64, 32, 256, 4, 1, bfloat, float>;

template [[host_name("attn_steel_nax_dsplit_h_bd256_bf16")]] [[kernel]] decltype(attention_nax_dsplit<
                                                                     bfloat,
                                                                     64,
                                                                     32,
                                                                     256,
                                                                     4,
                                                                     2,
                                                                     bfloat,
                                                                     float>)
attention_nax_dsplit<bfloat, 64, 32, 256, 4, 2, bfloat, float>;

// f16 twins, for the f16 LLM prefill (Qwen3.5 full-attention layers).
template [[host_name("attn_steel_nax_h_bd256")]] [[kernel]] decltype(attention_nax<
                                                                half,
                                                                64,
                                                                32,
                                                                256,
                                                                4,
                                                                1,
                                                                half,
                                                                float>)
attention_nax<half, 64, 32, 256, 4, 1, half, float>;

template [[host_name("attn_steel_nax_dsplit_h_bd256")]] [[kernel]] decltype(attention_nax_dsplit<
                                                                half,
                                                                64,
                                                                32,
                                                                256,
                                                                4,
                                                                2,
                                                                half,
                                                                float>)
attention_nax_dsplit<half, 64, 32, 256, 4, 2, half, float>;

// head_dim 512 (Gemma-4's global layers): MLX #4487's FOUR-way split, each
// simdgroup a 128-wide slice of D, 32-row query tiles over WM = 2 row groups
// (256 threads). MEASURED on the M5 at 16 q / 1 kv heads, causal f16, against
// the matmul2d sdpa_causal_mma2_d512 it replaces: 3.9x at 1k rows, 4.7x at
// 4k, 5.1x at 8k; 8 q / 2 kv 4.7x at 4k and 3.1x at 16k; rel-L2 5e-4.
template [[host_name("attn_steel_nax_dsplit_h_bd512")]] [[kernel]]
decltype(attention_nax_dsplit<half, 32, 32, 512, 2, 4, half, float>)
attention_nax_dsplit<half, 32, 32, 512, 2, 4, half, float>;
template [[host_name("attn_steel_nax_dsplit_h_bd512_bf16")]] [[kernel]]
decltype(attention_nax_dsplit<bfloat, 32, 32, 512, 2, 4, bfloat, float>)
attention_nax_dsplit<bfloat, 32, 32, 512, 2, 4, bfloat, float>;




#else
// Tensor ops unavailable for this target: a stub so the metallib still builds.
// The loader never binds this on a non-tensor (pre-M5) GPU.
kernel void attn_steel_nax_h_bd64(device half* O [[buffer(3)]],
                                  uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (half)0; } }
kernel void attn_steel_nax_h_bd64_bf16(device bfloat* O [[buffer(3)]],
                                       uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (bfloat)0; } }
kernel void attn_steel_nax_h_bd128(device half* O [[buffer(3)]],
                                   uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (half)0; } }
kernel void attn_steel_nax_h_bd128_bf16(device bfloat* O [[buffer(3)]],
                                        uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (bfloat)0; } }
kernel void attn_steel_nax_h_bd128_bq128_bf16(device bfloat* O [[buffer(3)]],
                                              uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (bfloat)0; } }
kernel void attn_steel_nax_h_bd256_bf16(device bfloat* O [[buffer(3)]],
                                        uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (bfloat)0; } }
kernel void attn_steel_nax_dsplit_h_bd256_bf16(device bfloat* O [[buffer(3)]],
                                               uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (bfloat)0; } }
kernel void attn_steel_nax_h_bd256(device half* O [[buffer(3)]],
                                   uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (half)0; } }
kernel void attn_steel_nax_dsplit_h_bd256(device half* O [[buffer(3)]],
                                          uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (half)0; } }
kernel void attn_steel_nax_dsplit_h_bd512(device half* O [[buffer(3)]],
                                          uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (half)0; } }
kernel void attn_steel_nax_dsplit_h_bd512_bf16(device bfloat* O [[buffer(3)]],
                                               uint t [[thread_position_in_grid]])
{ if (t == 0) { O[0] = (bfloat)0; } }
#endif
