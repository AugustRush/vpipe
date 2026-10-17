#ifndef VPIPE_GENERATIVE_MODELS_SHARED_SIGNED_AFFINE_H
#define VPIPE_GENERATIVE_MODELS_SHARED_SIGNED_AFFINE_H

#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// TOOLKIT, header-only: compiled into each plugin, not exported.

namespace vpipe {
namespace genai {

// THE SIGNED-CODE LAYOUT of an MLX affine weight, converted IN PLACE on
// the CPU -- the layout I8GemmContext::gemm_signed reads (see there for
// why it exists and what it is worth). Codes q -> q ^ 0x80 per byte (w8)
// or q ^ 0x8 per nibble (w4), which two's complement reads as q - 128 /
// q - 8; zero points b -> b' = b + 128 s / 8 s, rounded once to the
// element type. Lossless but for that one rounding.
//
// POSITION-INDEPENDENT: every code byte flips the same way and each zero
// point depends only on the scale beside it, so the conversion commutes
// with any row permutation or slice a loader applies, and runs over
// whole buffers without knowing the weight's shape. It matches
// affine_to_signed_* byte for byte: 128 s and 8 s are exact, so b + off*s
// rounds once either way.
//
// The buffers must be the CALLER'S: a refilled slot, a read() it owns, a
// buffer it built. Never one a WeightSet handed out from its cache --
// those are shared and immutable (see the WeightSet contract). Returns
// false, touching nothing, when the sizes do not describe one weight.
namespace signed_affine_detail {

inline float
bf16_to_f32(std::uint16_t h) noexcept
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline std::uint16_t
f32_to_bf16(float f) noexcept   // round to nearest even, as the GPU does
{
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (std::uint16_t)(u >> 16);
}

}  // namespace signed_affine_detail

// The codes alone: q ^ 0x80 per byte (w8), q ^ 0x8 per nibble (w4).
inline bool
to_signed_codes(metal_compute::SharedBuffer& codes, int bits) noexcept
{
  if ((bits != 4 && bits != 8) || codes.empty()) { return false; }
  const std::uint64_t flip =
      bits == 8 ? 0x8080808080808080ull : 0x8888888888888888ull;
  auto* c = static_cast<std::uint8_t*>(codes.contents());
  const std::size_t n = codes.byte_size();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    std::uint64_t w = 0;
    std::memcpy(&w, c + i, 8);
    w ^= flip;
    std::memcpy(c + i, &w, 8);
  }
  for (; i < n; ++i) { c[i] = (std::uint8_t)(c[i] ^ (std::uint8_t)flip); }
  return true;
}

// The zero points alone: b -> b + 128 s (w8) / 8 s (w4), in place in
// `zb`, which holds b on entry and is the same size as `scales`.
inline bool
to_signed_zero_points(const metal_compute::SharedBuffer& scales,
                      metal_compute::SharedBuffer& zb, int bits,
                      bool bf16) noexcept
{
  namespace d = signed_affine_detail;
  if ((bits != 4 && bits != 8) || scales.empty() ||
      zb.byte_size() != scales.byte_size() || (zb.byte_size() % 2) != 0) {
    return false;
  }
  const std::size_t groups = zb.byte_size() / 2;
  const float off = bits == 8 ? 128.0f : 8.0f;
  if (bf16) {
    const auto* s = static_cast<const std::uint16_t*>(scales.contents());
    auto* b = static_cast<std::uint16_t*>(zb.contents());
    for (std::size_t g = 0; g < groups; ++g) {
      b[g] = d::f32_to_bf16(d::bf16_to_f32(b[g]) +
                            off * d::bf16_to_f32(s[g]));
    }
  } else {
    const auto* s = static_cast<const _Float16*>(scales.contents());
    auto* b = static_cast<_Float16*>(zb.contents());
    for (std::size_t g = 0; g < groups; ++g) {
      b[g] = (_Float16)((float)b[g] + off * (float)s[g]);
    }
  }
  return true;
}

// Both halves of one weight, checked to describe the same weight first
// (64 codes per zero point), so a refusal touches nothing.
inline bool
to_signed_affine(metal_compute::SharedBuffer& codes,
                 const metal_compute::SharedBuffer& scales,
                 metal_compute::SharedBuffer& zb, int bits,
                 bool bf16) noexcept
{
  if ((bits != 4 && bits != 8) || codes.empty() || scales.empty() ||
      zb.byte_size() != scales.byte_size() || (zb.byte_size() % 2) != 0 ||
      codes.byte_size() != (zb.byte_size() / 2) * (std::size_t)(8 * bits)) {
    return false;
  }
  return to_signed_codes(codes, bits) &&
         to_signed_zero_points(scales, zb, bits, bf16);
}

}  // namespace genai
}  // namespace vpipe

#endif
