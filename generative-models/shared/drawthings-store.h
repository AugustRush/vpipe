#ifndef VPIPE_GENAI_DRAWTHINGS_STORE_H
#define VPIPE_GENAI_DRAWTHINGS_STORE_H

// A Draw Things checkpoint (`.ckpt`): s4nnc's TRAILER STORE. Read-only.
//
// The file is an SQLite database whose one table, `tensors(name, type,
// format, datatype, dim, data)`, is the table of contents; the bulk of
// the bytes lies AFTER the database pages, from a base offset stored
// big-endian in header bytes 60..63 (SQLite's user_version). A row's
// `type` packs the codec in bits 32..55, 0x10000000 << 32 when the
// payload is external (then `data` holds its offset and length from the
// base) and 0x01000000 << 32 when an i8x weight was stored Hadamard-256
// rotated. `dim` is twelve little-endian int32s, zero-padded.
//
// The codecs (libccv / s4nnc, verified against diffusers' Qwen-Image-2.1
// on Draw Things' own files):
//   raw    0         the element type as is (datatype 0x20000 f16,
//                    0x4000 f32)
//   i8x    0x8a1e9b  int8 codes [N, K] row-major, then at
//                    round_up(N K, 128) the N row scales in the element
//                    type: w = q s, q in [-127, 127], s = amax / 127
//   q8p    0x8a1e8b  a palette per block of `block` elements (the block
//                    size heads the payload, or the external `data`):
//                    256 entries in the element type, then the block's
//                    index bytes
//   ezm7   0x511     an f16 split in two: int32 zsize, the exponents
//                    raw-DEFLATE compressed, then one byte an element,
//                    sign << 7 | the top 7 mantissa bits (exact for a
//                    bf16 origin)
//
// Names are Draw Things' own (`__dit__[t-x_q-3-0]`); map_to_diffusers()
// presents a recognised model under the names its loader reads (the
// diffusers ones for Qwen-Image-2.1, MiniMax's own for MiniMax-H3),
// which is what MetalLlamaWeights does with a `.ckpt`.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai::drawthings {

inline constexpr std::uint32_t kCodecRaw  = 0;
inline constexpr std::uint32_t kCodecEzm7 = 0x511;
inline constexpr std::uint32_t kCodecQ8p  = 0x8a1e8b;
inline constexpr std::uint32_t kCodecI8x  = 0x8a1e9b;

struct Tensor {
  std::string name;
  std::uint32_t codec = 0;
  bool external = false;
  bool hadamard = false;          // an i8x weight stored rotated
  int elt = 2;                    // element bytes: 2 f16, 4 f32
  std::vector<std::int64_t> dims;
  // An external payload's ABSOLUTE file offset and length; the q8p
  // block size where the row stores it.
  std::uint64_t offset = 0, length = 0;
  std::uint32_t block = 0;
  // An inline payload's bytes (the row's `data`).
  std::vector<std::uint8_t> payload;

  std::int64_t count() const;
  // The last dimension, and the product of the others.
  std::int64_t cols() const { return dims.empty() ? 0 : dims.back(); }
  std::int64_t rows() const { return cols() > 0 ? count() / cols() : 0; }
};

// An SQLite database, by its 16-byte magic. Cheap; says nothing about
// whether it is a Draw Things store (read_index does).
bool looks_like(int fd);

// The table of contents. False, with a reason, for a database with no
// `tensors` table, an unknown codec or datatype, or a payload outside
// the file.
bool read_index(const std::string& path, std::uint64_t file_size,
                std::vector<Tensor>* out, std::string* err);

// Where an i8x payload's scales start, from the payload's start.
inline std::uint64_t
i8x_scale_offset(const Tensor& t)
{
  return ((std::uint64_t)t.count() + 127) & ~(std::uint64_t)127;
}

// Any codec, decoded to f32 (count() floats). `file` is the whole file
// mapped (external payloads are read from it).
bool decode_f32(const Tensor& t, const std::uint8_t* file,
                std::size_t file_size, float* out, std::string* err);

// A model tensor built from the store's: the ROWS of `parts` gathered
// in the model's order, then `fn` applied. A vector's rows are its
// elements.
struct Mapped {
  // kLog / kNegLog undo what Draw Things stores in place of a log-space
  // parameter: e^x (a BigVGAN snake's alpha) and 1 / (e^x + eps) (its
  // beta), so the model's x is ln(stored) and -ln(stored).
  enum class Fn { kNone, kMinusOne, kLog, kNegLog };
  // How the parts' rows make the model's: stacked in order, or q|k|v
  // interleaved PER HEAD -- [h0 q | h0 k | h0 v | h1 q ...], head_dim
  // rows each, which is how MiniMax-H3's released qkv_proj lies.
  enum class Rows { kConcat, kPerHead };
  // One source tensor. `rope`: Draw Things stores a rotary head's first
  // `rot` rows as interleaved pairs (its row 2i + j is the model's row
  // j * rot/2 + i), and this puts them back.
  // `row0` / `nrows` take a run of the part's rows (in the model's row
  // unit) instead of all of them -- one store tensor that the model
  // keeps as several (a fused bias it splits); nrows < 0 is to the end.
  // A rope part's interleave is undone on the part's own row index.
  struct Part {
    int index = -1;
    bool rope = false;
    std::int64_t row0 = 0;
    std::int64_t nrows = -1;
  };
  std::string name;               // e.g. "transformer_blocks.3.attn.to_q"
  std::string leaf;               // "weight" or "bias"
  std::vector<Part> parts;
  Fn fn = Fn::kNone;
  Rows rows = Rows::kConcat;
  int head_dim = 0;               // kPerHead and rope
  int rot = 0;                    // rope
  // The model's shape: [rows, ...], or [rows] for a vector. A row is
  // everything past the first dimension, whatever dimensions the store
  // wrote the parts with (a conv's [O, I, 1, k] is O rows of I k).
  std::vector<std::int64_t> shape;
  // Present it as F32 whatever the store's element type, for a loader
  // that reads only F32 (the audio VAE's).
  bool f32 = false;
};

// The store tensor and row that output row `r` of `m` comes from.
struct SourceRow {
  int part = -1;
  std::int64_t row = 0;
};
SourceRow source_row(const Mapped& m, const std::vector<Tensor>& ts,
                     std::int64_t r);

// `m` decoded to f32, [rows, cols] (fn applied).
bool gather_f32(const Mapped& m, const std::vector<Tensor>& ts,
                const std::uint8_t* file, std::size_t file_size,
                float* out, std::string* err);

// Whether every part of `m` is an external, unrotated i8x weight of one
// width -- so the model's tensor is itself an i8x weight, its rows the
// parts' rows (each carries its own scale). A row is everything past the
// first dimension, so a conv stored as an i8x matrix is one too.
bool i8x_whole(const Mapped& m, const std::vector<Tensor>& ts);
// ...and then its codes [rows, row elements] int8 and its row scales (the
// parts' element type), either may be null.
bool gather_i8x(const Mapped& m, const std::vector<Tensor>& ts,
                const std::uint8_t* file, std::size_t file_size,
                std::int8_t* codes, std::uint8_t* scales, std::string* err);

// Recognise the model and map EVERY tensor of the store; false (with
// the family empty) when it is not one this reader knows or a tensor
// does not place. `family` names what was recognised: a DiT's
// generate-image family tag ("qwen-image-21", "minimax-h3"), or a
// component -- "qwen3-vl" (a text encoder and its vision tower),
// "qwen-image-21-vae", "minimax-h3-vae" (video and audio, both halves).
bool map_to_diffusers(const std::vector<Tensor>& ts,
                      std::vector<Mapped>* out, std::string* family,
                      std::string* err);

}  // namespace vpipe::genai::drawthings

#endif
