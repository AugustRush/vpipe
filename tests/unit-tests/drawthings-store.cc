// A Draw Things checkpoint (`.ckpt`), read without Draw Things.
//
// THE STORES HERE ARE BUILT BY HAND in the shape s4nnc writes them: an
// SQLite `tensors` table whose user_version names where the bulk starts,
// external payloads after the database at that offset, each codec
// encoded as libccv encodes it (i8x row scales padded to 128 bytes of
// codes, q8p palettes per block with a short last block, ezm7's
// exponents raw-DEFLATE compressed). That keeps the default suite free of
// the app while exercising the paths a real file takes.
//
// The real file is the last test, checked tensor by tensor against the
// diffusers checkpoint it was converted from.
//
// Env:
//   VPIPE_DRAWTHINGS_QI21_CKPT          Draw Things' qwen_image_2.1_i8x.ckpt
//   VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH  the diffusers root it is checked
//                                       against (its transformer/)
//   VPIPE_DRAWTHINGS_H3_CKPT            Draw Things' minimax_h3_fl2va_i8x.ckpt
//   VPIPE_H3_DIT_SHARDS                 a directory of the released DiT's
//                                       shards (any subset) to check it by
//   VPIPE_DRAWTHINGS_QWEN3VL_CKPT       a Qwen3-VL text encoder store
//                                       (qwen_3_vl_8b_instruct_f16 / _i8x,
//                                       qwen_3_vl_32b_50_i8x), checked
//                                       against VPIPE_QWEN3VL_RELEASE (the
//                                       release's text_encoder directory)
//   VPIPE_DRAWTHINGS_QI21_VAE_CKPT      qwen_image_2.1_vae_f16.ckpt, checked
//                                       against VPIPE_QI21_VAE_REFERENCE --
//                                       madebyollin's texture-fix VAE, which
//                                       is what Draw Things ships
//   VPIPE_DRAWTHINGS_H3_VAE_CKPT        minimax_h3_vae_f16.ckpt, against
//                                       VPIPE_H3_VAE_RELEASE (a partition,
//                                       holding video_vae/ and audio_vae/)
//
// The VAE tests also DECODE (and encode) through the model loaded from
// each, which is the check that the presented names, shapes and dtypes
// are the ones the loader reads, not merely the right values.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/shared/drawthings-store.h"

#include <compression.h>
#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace vpipe::genai;

namespace {

namespace fs = std::filesystem;

std::uint16_t
f16_bits_(float f)
{
  __fp16 h = (__fp16)f;
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

float
f16_val_(float f)
{
  return (float)(__fp16)f;
}

template <class T>
void
put_(std::vector<std::uint8_t>& b, T v)
{
  const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
  b.insert(b.end(), p, p + sizeof v);
}

// One store, written the way s4nnc lays it out.
struct Builder {
  struct Row {
    std::string name;
    std::uint32_t codec = 0;
    bool external = false, hadamard = false, f32 = false;
    std::vector<std::int32_t> dims;
    std::vector<std::uint8_t> payload;   // inline data, or the trailer's
    std::uint32_t block = 0;             // q8p
  };
  std::vector<Row> rows;
  // What each diffusers tensor should read back as: f32 values, and the
  // dtype the reader presents.
  std::map<std::string, std::vector<float>> want;

  static std::string
  dit_(const std::string& key, int layer, int leaf = 0)
  {
    return "__dit__[t-" + key + "-" + std::to_string(layer) + "-" +
           std::to_string(leaf) + "]";
  }

  void
  raw16_(const std::string& name, std::vector<std::int32_t> dims,
         std::vector<float>& v)
  {
    Row r{name, drawthings::kCodecRaw, true, false, false, dims, {}, 0};
    for (float& x : v) {
      x = f16_val_(x);
      put_(r.payload, f16_bits_(x));
    }
    rows.push_back(std::move(r));
  }
  static std::string
  at_(const std::string& group, const std::string& key, int layer,
      int leaf = 0)
  {
    return "__" + group + "__[t-" + key + "-" + std::to_string(layer) + "-" +
           std::to_string(leaf) + "]";
  }
  // Raw f16 kept in the row itself, as Draw Things keeps norms and biases.
  void
  raw16_inline_(const std::string& name, std::vector<std::int32_t> dims,
                std::vector<float>& v)
  {
    Row r{name, drawthings::kCodecRaw, false, false, false, dims, {}, 0};
    for (float& x : v) {
      x = f16_val_(x);
      put_(r.payload, f16_bits_(x));
    }
    rows.push_back(std::move(r));
  }
  void
  raw32_inline_(const std::string& name, std::vector<std::int32_t> dims,
                const std::vector<float>& v)
  {
    Row r{name, drawthings::kCodecRaw, false, false, true, dims, {}, 0};
    for (float x : v) { put_(r.payload, x); }
    rows.push_back(std::move(r));
  }
  // libccv's row-wise 8-bit: s = amax / 127, q = round(v / s) clamped.
  // Returns the codes and the f16 scales; `deq` the dequantized values.
  void
  i8x_(const std::string& name, int N, int K, const std::vector<float>& v,
       std::vector<std::int8_t>* codes, std::vector<std::uint16_t>* sc,
       std::vector<float>* deq, bool hadamard = false)
  {
    Row r{name, drawthings::kCodecI8x, true, hadamard, false, {N, K}, {}, 0};
    codes->assign((std::size_t)N * K, 0);
    sc->assign((std::size_t)N, 0);
    deq->assign((std::size_t)N * K, 0.0f);
    for (int n = 0; n < N; ++n) {
      float am = 0.0f;
      for (int k = 0; k < K; ++k) {
        am = std::max(am, std::fabs(v[(std::size_t)n * K + k]));
      }
      const float s = am / 127.0f;
      (*sc)[(std::size_t)n] = f16_bits_(s);
      const float s16 = f16_val_(s);
      for (int k = 0; k < K; ++k) {
        const std::size_t i = (std::size_t)n * K + k;
        const int q = std::clamp((int)std::lrint(v[i] / s), -127, 127);
        (*codes)[i] = (std::int8_t)q;
        (*deq)[i] = f16_val_((float)q * s16);
      }
    }
    for (std::int8_t c : *codes) { r.payload.push_back((std::uint8_t)c); }
    r.payload.resize(((std::size_t)N * K + 127) & ~(std::size_t)127, 0);
    for (std::uint16_t s : *sc) { put_(r.payload, s); }
    rows.push_back(std::move(r));
  }
  // A palette per block of `block` elements: the f16 values of `v` must
  // take at most 256 distinct values a block.
  void
  q8p_(const std::string& name, std::vector<std::int32_t> dims,
       std::vector<float>& v, std::uint32_t block)
  {
    Row r{name, drawthings::kCodecQ8p, true, false, false, dims, {}, block};
    for (std::size_t b0 = 0; b0 < v.size(); b0 += block) {
      const std::size_t m = std::min<std::size_t>(block, v.size() - b0);
      std::vector<std::uint16_t> pal;
      std::vector<std::uint8_t> ix(block, 0);
      for (std::size_t i = 0; i < m; ++i) {
        v[b0 + i] = f16_val_(v[b0 + i]);
        const std::uint16_t h = f16_bits_(v[b0 + i]);
        auto it = std::find(pal.begin(), pal.end(), h);
        if (it == pal.end()) {
          pal.push_back(h);
          it = pal.end() - 1;
        }
        ix[i] = (std::uint8_t)(it - pal.begin());
      }
      pal.resize(256, 0);
      for (std::uint16_t h : pal) { put_(r.payload, h); }
      r.payload.insert(r.payload.end(), ix.begin(), ix.end());
    }
    rows.push_back(std::move(r));
  }
  // ezm7: values with their low three mantissa bits clear (a bf16
  // origin), exponents deflated, then sign << 7 | top 7 mantissa bits.
  void
  ezm7_(const std::string& name, std::vector<std::int32_t> dims,
        std::vector<float>& v)
  {
    Row r{name, drawthings::kCodecEzm7, false, false, false, dims, {}, 0};
    std::vector<std::uint8_t> ex(v.size()), fw(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      std::uint16_t h = f16_bits_(v[i]) & 0xfff8;
      __fp16 hf;
      std::memcpy(&hf, &h, 2);
      v[i] = (float)hf;
      ex[i] = (std::uint8_t)((h >> 10) & 0x1f);
      fw[i] = (std::uint8_t)(((h >> 15) << 7) | ((h >> 3) & 0x7f));
    }
    std::vector<std::uint8_t> z(v.size() + 1024);
    const std::size_t zs = compression_encode_buffer(
        z.data(), z.size(), ex.data(), ex.size(), nullptr, COMPRESSION_ZLIB);
    put_(r.payload, (std::int32_t)zs);
    r.payload.insert(r.payload.end(), z.begin(), z.begin() + (long)zs);
    r.payload.insert(r.payload.end(), fw.begin(), fw.end());
    rows.push_back(std::move(r));
  }

  bool
  write(const std::string& path) const
  {
    fs::remove(path);
    const std::uint32_t base = 1u << 20;
    sqlite3* db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) { return false; }
    bool ok = sqlite3_exec(db,
                           "CREATE TABLE tensors (name TEXT, type INTEGER, "
                           "format INTEGER, datatype INTEGER, dim BLOB, "
                           "data BLOB, PRIMARY KEY (name));"
                           "PRAGMA user_version = 1048576;",
                           nullptr, nullptr, nullptr) == SQLITE_OK;
    sqlite3_stmt* st = nullptr;
    ok = ok && sqlite3_prepare_v2(db,
                                  "INSERT INTO tensors VALUES "
                                  "(?, ?, 0, ?, ?, ?)",
                                  -1, &st, nullptr) == SQLITE_OK;
    std::vector<std::uint8_t> trailer;
    for (const Row& r : rows) {
      if (!ok) { break; }
      const std::uint32_t ident = r.codec | (r.external ? 0x10000000u : 0) |
                                  (r.hadamard ? 0x01000000u : 0);
      std::vector<std::uint8_t> dim(48, 0);
      std::memcpy(dim.data(), r.dims.data(), r.dims.size() * 4);
      std::vector<std::uint8_t> data;
      if (r.external) {
        trailer.resize((trailer.size() + 63) & ~(std::size_t)63, 0);
        if (r.codec == drawthings::kCodecQ8p) {
          put_(data, r.block);
          put_(data, (std::uint32_t)0);
        }
        put_(data, (std::uint64_t)trailer.size());
        put_(data, (std::uint64_t)r.payload.size());
        trailer.insert(trailer.end(), r.payload.begin(), r.payload.end());
      } else {
        if (r.codec == drawthings::kCodecQ8p) { put_(data, r.block); }
        data.insert(data.end(), r.payload.begin(), r.payload.end());
      }
      sqlite3_reset(st);
      sqlite3_bind_text(st, 1, r.name.c_str(), -1, SQLITE_TRANSIENT);
      sqlite3_bind_int64(st, 2, (sqlite3_int64)((std::uint64_t)ident << 32));
      sqlite3_bind_int64(st, 3, r.f32 ? 0x4000 : 0x20000);
      sqlite3_bind_blob(st, 4, dim.data(), (int)dim.size(), SQLITE_TRANSIENT);
      sqlite3_bind_blob(st, 5, data.data(), (int)data.size(),
                        SQLITE_TRANSIENT);
      ok = sqlite3_step(st) == SQLITE_DONE;
    }
    if (st != nullptr) { sqlite3_finalize(st); }
    sqlite3_close(db);
    if (!ok || fs::file_size(path) > base) { return false; }
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (f == nullptr) { return false; }
    std::fseek(f, 0, SEEK_END);
    std::vector<std::uint8_t> pad(base - (std::size_t)std::ftell(f), 0);
    ok = std::fwrite(pad.data(), 1, pad.size(), f) == pad.size() &&
         std::fwrite(trailer.data(), 1, trailer.size(), f) == trailer.size();
    std::fclose(f);
    return ok;
  }
};

std::vector<float>
rnd_(std::size_t n, std::mt19937& rng, float s)
{
  std::normal_distribution<float> nd(0.0f, s);
  std::vector<float> v(n);
  for (float& x : v) { x = nd(rng); }
  return v;
}

std::string
tmp_(const char* name)
{
  return (fs::temp_directory_path() /
          (std::string("vpipe-dt-") + std::to_string(::getpid()) + "-" +
           name)).string();
}

// A tensor as the reader presents it, widened to f32.
bool
read_f32_(const MetalLlamaWeights& w, const std::string& name,
          std::vector<float>* out)
{
  const auto* ti = w.info(name);
  if (ti == nullptr) { return false; }
  std::vector<std::uint8_t> b(ti->nbytes);
  if (!w.read_into(name, b.data(), b.size())) { return false; }
  const std::size_t es = ti->dtype == "F32" ? 4 : 2;
  out->resize(b.size() / es);
  for (std::size_t i = 0; i < out->size(); ++i) {
    if (es == 4) {
      std::memcpy(&(*out)[i], b.data() + i * 4, 4);
    } else if (ti->dtype == "BF16") {
      std::uint32_t u = (std::uint32_t)b[i * 2] |
                        ((std::uint32_t)b[i * 2 + 1] << 8);
      u <<= 16;
      std::memcpy(&(*out)[i], &u, 4);
    } else {
      __fp16 h;
      std::memcpy(&h, b.data() + i * 2, 2);
      (*out)[i] = (float)h;
    }
  }
  return true;
}

}  // namespace

// A Qwen-Image-2.1-shaped store with every codec the reader decodes,
// opened as a checkpoint and read back under the diffusers names.
TEST(drawthings_store, a_store_reads_under_diffusers_names)
{
  const int H = 64, HD = 32, FF = 3 * H, IN = 16, OUT = 16, L = 2;
  std::mt19937 rng(17);
  Builder b;
  std::map<std::string, std::vector<std::int8_t>> codes;
  std::map<std::string, std::vector<std::uint16_t>> scales;
  for (int l = 0; l < L; ++l) {
    const std::string p = "transformer_blocks." + std::to_string(l) + ".";
    struct P { const char* key; const char* name; int n, k; };
    for (const P& q : {P{"x_q", "attn.to_q", H, H},
                       P{"x_k", "attn.to_k", H, H},
                       P{"x_v", "attn.to_v", H, H},
                       P{"x_o", "attn.to_out.0", H, H},
                       P{"ffn_gate_proj", "img_mlp.gate_layer", FF, H},
                       P{"ffn_up_proj", "img_mlp.proj", FF, H},
                       P{"ffn_down_proj", "img_mlp.out", H, FF}}) {
      const auto v = rnd_((std::size_t)q.n * q.k, rng, 0.05f);
      std::vector<float> deq;
      const std::string stem = p + q.name;
      b.i8x_(Builder::dit_(q.key, l), q.n, q.k, v, &codes[stem],
             &scales[stem], &deq);
      b.want[stem + ".weight"] = deq;
    }
    for (const char* nk : {"x_norm_q", "x_norm_k"}) {
      auto v = rnd_(HD, rng, 1.0f);
      b.ezm7_(Builder::dit_(nk, l), {1, 1, 1, HD}, v);
      b.want[p + (nk[7] == 'q' ? "attn.norm_q" : "attn.norm_k") +
             ".weight"] = v;
    }
  }
  {
    auto v = rnd_((std::size_t)H * IN, rng, 0.1f);
    b.raw16_(Builder::dit_("x_embedder", 0), {H, IN}, v);
    b.want["img_in.weight"] = v;
  }
  for (const char* k : {"context_embedder_0", "context_embedder_1",
                        "t_embedder_1"}) {
    auto v = rnd_((std::size_t)H * H, rng, 0.1f);
    b.raw16_(Builder::dit_(k, 0), {H, H}, v);
    b.want[std::string(k) == "context_embedder_0" ? "txt_in.in_layer.weight"
           : std::string(k) == "context_embedder_1"
               ? "txt_in.out_layer.weight"
               : "time_text_embed.timestep_embedder.linear_2.weight"] = v;
  }
  {
    auto v = rnd_((std::size_t)H * 256, rng, 0.1f);
    b.raw16_(Builder::dit_("t_embedder_0", 0), {H, 256}, v);
    b.want["time_text_embed.timestep_embedder.linear_1.weight"] = v;
    auto o = rnd_((std::size_t)OUT * H, rng, 0.1f);
    b.raw16_(Builder::dit_("linear", 0), {OUT, H}, o);
    b.want["proj_out.weight"] = o;
  }
  {
    // The text norm, +1 folded and f32, inline.
    const auto v = rnd_(H, rng, 0.1f);
    std::vector<float> folded(v);
    for (float& x : folded) { x += 1.0f; }
    b.raw32_inline_(Builder::dit_("context_norm", 0), {1, H}, folded);
    std::vector<float> w(H);
    for (int i = 0; i < H; ++i) {
      w[(std::size_t)i] = folded[(std::size_t)i] - 1.0f;
    }
    b.want["txt_in.text_norm.weight"] = w;
  }
  {
    // Palettes over 40 distinct values, blocks of 1000: four of a
    // [64, 64] chunk, the last one short.
    std::vector<float> mod;
    auto pal = rnd_(40, rng, 0.05f);
    std::uniform_int_distribution<int> pick(0, 39);
    for (int c = 0; c < 4; ++c) {
      std::vector<float> v((std::size_t)H * H);
      for (float& x : v) { x = pal[(std::size_t)pick(rng)]; }
      b.q8p_(Builder::dit_("x_ada_ln_" + std::to_string(c), 0), {H, H}, v,
             1000);
      mod.insert(mod.end(), v.begin(), v.end());
    }
    b.want["modulation.1.weight"] = mod;
    std::vector<float> v((std::size_t)H * H);
    for (float& x : v) { x = pal[(std::size_t)pick(rng)]; }
    b.q8p_(Builder::dit_("ada_ln_0", 0), {H, H}, v, 16384);
    b.want["norm_out.linear.weight"] = v;
  }
  const std::string path = tmp_("qi21.ckpt");
  ASSERT_TRUE(b.write(path));
  auto w = MetalLlamaWeights::open_model(path);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  std::size_t bad = 0, worst_n = 0;
  for (const auto& [name, v] : b.want) {
    std::vector<float> got;
    if (!read_f32_(*w, name, &got) || got.size() != v.size()) {
      std::printf("[drawthings] %s: missing or %zu values (want %zu)\n",
                  name.c_str(), got.size(), v.size());
      ++bad;
      continue;
    }
    std::size_t n = 0;
    for (std::size_t i = 0; i < v.size(); ++i) { n += got[i] != v[i]; }
    if (n != 0) {
      std::printf("[drawthings] %s: %zu of %zu values differ\n",
                  name.c_str(), n, v.size());
      ++bad;
      worst_n = std::max(worst_n, n);
    }
  }
  EXPECT_TRUE(bad == 0);
  // An i8x weight's codes and scales, where they lie.
  std::size_t cbad = 0;
  for (const auto& [stem, c] : codes) {
    const auto* ci = w->info(stem + ".i8x");
    const auto* si = w->info(stem + ".i8x_scales");
    if (ci == nullptr || si == nullptr || ci->dtype != "I8" ||
        si->dtype != "F16" || ci->nbytes != c.size() ||
        si->nbytes != scales[stem].size() * 2) {
      ++cbad;
      continue;
    }
    std::vector<std::uint8_t> cb(ci->nbytes), sb(si->nbytes);
    if (!w->read_into(stem + ".i8x", cb.data(), cb.size()) ||
        !w->read_into(stem + ".i8x_scales", sb.data(), sb.size()) ||
        std::memcmp(cb.data(), c.data(), cb.size()) != 0 ||
        std::memcmp(sb.data(), scales[stem].data(), sb.size()) != 0) {
      ++cbad;
    }
  }
  std::printf("[drawthings] %zu tensors read back, %zu differ; %zu i8x "
              "weights' codes and scales, %zu differ\n", b.want.size(), bad,
              codes.size(), cbad);
  EXPECT_TRUE(cbad == 0);
  // The model reads its dimensions off the tensors.
  MetalQwenImage21Transformer::Config c;
  EXPECT_TRUE(MetalQwenImage21Transformer::is_single_file_dit(path));
  ASSERT_TRUE(MetalQwenImage21Transformer::Config::read_dims(path, &c));
  EXPECT_TRUE(c.hidden == H && c.head_dim == HD && c.n_heads == H / HD &&
              c.n_layers == L && c.in_channels == IN &&
              c.out_channels == OUT && c.txt_dim == H && c.mlp_ratio == 3);
  fs::remove(path);
}

// What the reader refuses rather than misreads: a codec it does not
// decode, and an i8x weight stored Hadamard-rotated (Draw Things' own
// ordering, which nothing here undoes).
TEST(drawthings_store, an_unknown_layout_is_refused)
{
  std::mt19937 rng(3);
  for (int which = 0; which < 2; ++which) {
    Builder b;
    std::vector<std::int8_t> c;
    std::vector<std::uint16_t> s;
    std::vector<float> d;
    b.i8x_(Builder::dit_("x_q", 0), 8, 64, rnd_(512, rng, 0.1f), &c, &s, &d,
           /*hadamard=*/which == 1);
    if (which == 0) { b.rows.back().codec = 0x123456; }
    const std::string path = tmp_(which == 0 ? "codec.ckpt" : "h256.ckpt");
    ASSERT_TRUE(b.write(path));
    std::vector<drawthings::Tensor> ts;
    std::string err;
    EXPECT_TRUE(!drawthings::read_index(path, fs::file_size(path), &ts,
                                        &err));
    std::printf("[drawthings] refused: %s\n", err.c_str());
    EXPECT_TRUE(!MetalLlamaWeights::open_model(path).has_value());
    fs::remove(path);
  }
}

// THE REAL FILE: Draw Things' Qwen-Image-2.1 i8x checkpoint against the
// diffusers transformer, tensor by tensor, for the first and last block
// and every top-level tensor. The bars are the codecs' own: i8x rows
// at the precision of an int8 per row (~1e-2), palettes of 256 values,
// and the lossless codecs at an f16 rounding of the bf16 original.
TEST(drawthings_store, qwen_image_21_checkpoint_matches_diffusers)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_QI21_CKPT");
  const char* root = std::getenv("VPIPE_QWEN_IMAGE21_TEST_MODEL_PATH");
  if (ck == nullptr || root == nullptr || *ck == '\0' || *root == '\0') {
    return;
  }
  auto dt = MetalLlamaWeights::open_model(ck);
  auto df = MetalLlamaWeights::open_model(
      (fs::path(root) / "transformer").string());
  ASSERT_TRUE(dt.has_value() && df.has_value());
  if (!dt.has_value() || !df.has_value()) { return; }
  std::vector<std::string> names;
  for (const std::string& n : df->tensor_names()) {
    if (n.rfind("transformer_blocks.", 0) != 0 ||
        n.rfind("transformer_blocks.0.", 0) == 0 ||
        n.rfind("transformer_blocks.31.", 0) == 0) {
      names.push_back(n);
    }
  }
  std::sort(names.begin(), names.end());
  int bad = 0, seen = 0;
  for (const std::string& n : names) {
    std::vector<float> a, r;
    if (!read_f32_(*dt, n, &a) || !read_f32_(*df, n, &r) ||
        a.size() != r.size()) {
      std::printf("[drawthings] %s: not presented (or a different size)\n",
                  n.c_str());
      ++bad;
      continue;
    }
    double e2 = 0, r2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      e2 += ((double)a[i] - r[i]) * ((double)a[i] - r[i]);
      r2 += (double)r[i] * r[i];
    }
    const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
    const bool i8 = dt->info(n.substr(0, n.size() - 7) + ".i8x") != nullptr;
    const bool pal = n == "modulation.1.weight" ||
                     n == "norm_out.linear.weight";
    const double bar = i8 ? 3e-2 : pal ? 2e-2 : 2e-3;
    std::printf("[drawthings] %-52s rel %.3e%s\n", n.c_str(), rel,
                i8 ? " (i8x)" : pal ? " (palette)" : "");
    bad += rel > bar ? 1 : 0;
    ++seen;
  }
  std::printf("[drawthings] %d tensors compared, %d over their bar\n", seen,
              bad);
  EXPECT_TRUE(seen >= 25 && bad == 0);
}

// A MiniMax-H3-shaped store (two 128-wide heads over a 64-wide hidden),
// read back under MiniMax's own names. Draw Things splits what the
// release fuses, so this is where the reader rebuilds: q|k|v gathered
// PER HEAD, q and k (and their norms) with each head's first 96 rows put
// back from the interleaved RoPE order, fc1 as gate|up, the 18 AdaLN
// chunks modality-major, the final AdaLN as shift|scale -- and an i8x
// weight gathered that way is still served as codes and row scales.
// Every expectation is computed here, independently of the reader.
TEST(drawthings_store, an_h3_store_reads_under_the_release_names)
{
  const int H = 64, NH = 2, HD = 128, I = NH * HD, FF = 96, TD = 32;
  std::mt19937 rng(23);
  Builder b;
  struct I8 { std::vector<std::int8_t> c; std::vector<std::uint16_t> s;
              std::vector<float> d; int n, k; };
  std::map<std::string, I8> i8;
  auto i8x = [&](const std::string& key, int L, int n, int k) {
    I8& w = i8[key + "-" + std::to_string(L)];
    w.n = n;
    w.k = k;
    b.i8x_(Builder::dit_(key, L), n, k, rnd_((std::size_t)n * k, rng, 0.05f),
           &w.c, &w.s, &w.d);
    return &w;
  };
  auto raw = [&](const std::string& key, int L, int leaf,
                 std::vector<std::int32_t> dims, std::size_t n) {
    auto v = rnd_(n, rng, 0.1f);
    b.raw16_(Builder::dit_(key, L, leaf), dims, v);
    return v;
  };
  auto ezm = [&](const std::string& key, int L, int leaf,
                 std::vector<std::int32_t> dims, std::size_t n) {
    auto v = rnd_(n, rng, 1.0f);
    b.ezm7_(Builder::dit_(key, L, leaf), dims, v);
    return v;
  };
  // The model row j*48 + i of a rotary head is stored at 2i + j.
  auto unrope = [&](int d) { return d < 96 ? 2 * (d % 48) + d / 48 : d; };
  std::map<std::string, std::vector<float>> adaln_w, adaln_b;
  std::vector<float> nq, nk;
  for (int L = 0; L < 1; ++L) {
    for (const char* k : {"q", "k", "v"}) { i8x(k, L, I, H); }
    i8x("o", L, H, I);
    i8x("gate", L, FF, H);
    i8x("up", L, FF, H);
    i8x("down", L, H, FF);
    ezm("norm1", L, 0, {1, 1, H}, H);
    ezm("norm2", L, 0, {1, 1, H}, H);
    nq = ezm("norm_q", L, 0, {1, 1, 1, HD}, HD);
    nk = ezm("norm_k", L, 0, {1, 1, 1, HD}, HD);
    for (int c = 0; c < 6; ++c) {
      for (int m = 0; m < 3; ++m) {
        const std::string k =
            "adaln_" + std::to_string(c) + "_" + std::to_string(m);
        auto pal = rnd_(30, rng, 0.05f);
        std::uniform_int_distribution<int> pick(0, 29);
        std::vector<float> v((std::size_t)H * TD);
        for (float& x : v) { x = pal[(std::size_t)pick(rng)]; }
        b.q8p_(Builder::dit_(k, L), {H, TD}, v, 700);
        adaln_w[k] = v;
        adaln_b[k] = ezm(k, L, 1, {H}, H);
      }
    }
  }
  for (const char* k : {"refiner_q", "refiner_k", "refiner_v"}) {
    raw(k, 0, 0, {I, H}, (std::size_t)I * H);
  }
  raw("refiner_o", 0, 0, {H, I}, (std::size_t)H * I);
  raw("refiner_gate", 0, 0, {FF, H}, (std::size_t)FF * H);
  raw("refiner_up", 0, 0, {FF, H}, (std::size_t)FF * H);
  raw("refiner_down", 0, 0, {H, FF}, (std::size_t)H * FF);
  for (const char* k : {"refiner_norm1", "refiner_norm2"}) {
    raw(k, 0, 0, {1, 1, H}, H);
  }
  for (const char* k : {"refiner_norm_q", "refiner_norm_k"}) {
    raw(k, 0, 0, {1, 1, 1, HD}, HD);
  }
  raw("refiner_final_norm", 0, 0, {1, 1, H}, H);
  raw("context_embedder", 0, 0, {H, 80}, (std::size_t)H * 80);
  raw("context_embedder", 0, 1, {H}, H);
  raw("time_embedder_linear_1", 0, 0, {H, 256}, (std::size_t)H * 256);
  raw("time_embedder_linear_1", 0, 1, {H}, H);
  raw("time_embedder_linear_2", 0, 0, {TD, H}, (std::size_t)TD * H);
  raw("time_embedder_linear_2", 0, 1, {TD}, TD);
  raw("proj_in", 0, 0, {H, 96}, (std::size_t)H * 96);
  raw("proj_in", 0, 1, {H}, H);
  raw("audio_proj_in", 0, 0, {H, 32}, (std::size_t)H * 32);
  raw("audio_proj_in", 0, 1, {H}, H);
  raw("proj_out", 0, 0, {96, H}, (std::size_t)96 * H);
  raw("proj_out", 0, 1, {96}, 96);
  raw("audio_proj_out", 0, 0, {32, H}, (std::size_t)32 * H);
  raw("audio_proj_out", 0, 1, {32}, 32);
  ezm("norm_out", 0, 0, {1, 1, H}, H);
  auto shift_w = raw("norm_out_shift", 0, 0, {H, TD}, (std::size_t)H * TD);
  auto shift_b = raw("norm_out_shift", 0, 1, {H}, H);
  auto scale_w = raw("norm_out_scale", 0, 0, {H, TD}, (std::size_t)H * TD);
  auto scale_b = raw("norm_out_scale", 0, 1, {H}, H);
  const std::string path = tmp_("h3.ckpt");
  ASSERT_TRUE(b.write(path));
  auto w = MetalLlamaWeights::open_model(path);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  std::size_t bad = 0;
  auto same = [&](const std::string& name, const std::vector<float>& want) {
    std::vector<float> got;
    if (!read_f32_(*w, name, &got) || got.size() != want.size()) {
      std::printf("[drawthings] %s: missing or %zu values (want %zu)\n",
                  name.c_str(), got.size(), want.size());
      ++bad;
      return;
    }
    std::size_t n = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      n += got[i] != f16_val_(want[i]);
    }
    if (n != 0) {
      std::printf("[drawthings] %s: %zu of %zu differ\n", name.c_str(), n,
                  want.size());
      ++bad;
    }
  };
  // q|k|v per head, q and k un-roped, as dequantized values and as the
  // gathered codes and scales.
  std::vector<float> qkv((std::size_t)3 * I * H);
  std::vector<std::int8_t> qkv_c((std::size_t)3 * I * H);
  std::vector<std::uint16_t> qkv_s((std::size_t)3 * I);
  for (int r = 0; r < 3 * I; ++r) {
    const int h = r / (3 * HD), pt = (r / HD) % 3, d = r % HD;
    const I8& p = i8[std::string(pt == 0 ? "q" : pt == 1 ? "k" : "v") + "-0"];
    const int sr = h * HD + (pt < 2 ? unrope(d) : d);
    for (int k = 0; k < H; ++k) {
      qkv[(std::size_t)r * H + k] = p.d[(std::size_t)sr * H + k];
      qkv_c[(std::size_t)r * H + k] = p.c[(std::size_t)sr * H + k];
    }
    qkv_s[(std::size_t)r] = p.s[(std::size_t)sr];
  }
  same("blocks.0.attn.qkv_proj.weight", qkv);
  {
    const auto* ci = w->info("blocks.0.attn.qkv_proj.i8x");
    const auto* si = w->info("blocks.0.attn.qkv_proj.i8x_scales");
    std::vector<std::uint8_t> cb(qkv_c.size()), sb(qkv_s.size() * 2);
    const bool ok = ci != nullptr && si != nullptr && ci->dtype == "I8" &&
                    w->read_into("blocks.0.attn.qkv_proj.i8x", cb.data(),
                                 cb.size()) &&
                    w->read_into("blocks.0.attn.qkv_proj.i8x_scales",
                                 sb.data(), sb.size()) &&
                    std::memcmp(cb.data(), qkv_c.data(), cb.size()) == 0 &&
                    std::memcmp(sb.data(), qkv_s.data(), sb.size()) == 0;
    std::printf("[drawthings] qkv_proj codes and scales gathered: %s\n",
                ok ? "identical" : "DIFFERENT");
    EXPECT_TRUE(ok);
  }
  std::vector<float> uq(HD), uk(HD);
  for (int d = 0; d < HD; ++d) {
    uq[(std::size_t)d] = nq[(std::size_t)unrope(d)];
    uk[(std::size_t)d] = nk[(std::size_t)unrope(d)];
  }
  same("blocks.0.attn.q_norm.weight", uq);
  same("blocks.0.attn.k_norm.weight", uk);
  same("blocks.0.attn.out_proj.weight", i8["o-0"].d);
  std::vector<float> fc1 = i8["gate-0"].d;
  fc1.insert(fc1.end(), i8["up-0"].d.begin(), i8["up-0"].d.end());
  same("blocks.0.mlp.fc1.weight", fc1);
  same("blocks.0.mlp.fc2.weight", i8["down-0"].d);
  std::vector<float> aw, ab;
  for (int m = 0; m < 3; ++m) {
    for (int c = 0; c < 6; ++c) {
      const std::string k =
          "adaln_" + std::to_string(c) + "_" + std::to_string(m);
      aw.insert(aw.end(), adaln_w[k].begin(), adaln_w[k].end());
      ab.insert(ab.end(), adaln_b[k].begin(), adaln_b[k].end());
    }
  }
  same("blocks.0.adaln_proj.linear.weight", aw);
  same("blocks.0.adaln_proj.linear.bias", ab);
  std::vector<float> fw = shift_w, fb = shift_b;
  fw.insert(fw.end(), scale_w.begin(), scale_w.end());
  fb.insert(fb.end(), scale_b.begin(), scale_b.end());
  same("final_layer.adaln_proj.linear.weight", fw);
  same("final_layer.adaln_proj.linear.bias", fb);
  std::printf("[drawthings] H3 store: %zu tensors differ\n", bad);
  EXPECT_TRUE(bad == 0);
  // The model reads its dimensions off the tensors, qkv per head.
  MetalMiniMaxH3Transformer::Config c;
  std::string cerr;
  const bool cok = MetalMiniMaxH3Transformer::config_from_json(path, c, &cerr);
  std::printf("[drawthings] H3 config: %s (hidden %d heads %d ffn %d "
              "layers %d refiner %d per-head %d)\n",
              cok ? "read" : cerr.c_str(), c.hidden, c.n_heads, c.ffn,
              c.n_layers, c.n_refiner, (int)c.qkv_per_head);
  EXPECT_TRUE(cok && c.hidden == H && c.n_heads == NH && c.ffn == FF &&
              c.n_layers == 1 && c.n_refiner == 1 && c.qkv_per_head &&
              c.time_dim == TD && c.video_channels == 24);
  fs::remove(path);
}

// THE REAL H3 FILE: Draw Things' minimax_h3_fl2va_i8x.ckpt against the
// released transformer's shards (any subset: VPIPE_H3_DIT_SHARDS names a
// directory of them), every tensor the shards hold, at each codec's own
// bar -- which is what catches a wrong gather, since a misplaced head or
// an un-roped row is an O(1) error and the codecs' are ~1e-2.
TEST(drawthings_store, minimax_h3_checkpoint_matches_the_release)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_H3_CKPT");
  const char* sh = std::getenv("VPIPE_H3_DIT_SHARDS");
  if (ck == nullptr || sh == nullptr || *ck == '\0' || *sh == '\0') {
    return;
  }
  auto dt = MetalLlamaWeights::open_model(ck);
  ASSERT_TRUE(dt.has_value());
  if (!dt.has_value()) { return; }
  int bad = 0, seen = 0;
  std::vector<fs::path> files;
  for (const auto& e : fs::directory_iterator(sh)) {
    if (e.path().extension() == ".safetensors") { files.push_back(e.path()); }
  }
  std::sort(files.begin(), files.end());
  for (const fs::path& f : files) {
    auto ref = MetalLlamaWeights::open(f.string());
    if (!ref.has_value()) { continue; }
    std::vector<std::string> names = ref->tensor_names();
    std::sort(names.begin(), names.end());
    for (const std::string& n : names) {
      if (n == "rope.inv_freq") {
        // No such tensor in the store: the loader computes it in f32 from
        // the base, which has to give the release's table bit for bit.
        std::vector<float> r;
        bool exact = read_f32_(*ref, n, &r) && r.size() == 16;
        for (int i = 0; exact && i < 16; ++i) {
          const float v = 1.0f / std::pow(10000.0f, (float)(2 * i) / 32.0f);
          exact = v == r[(std::size_t)i];
        }
        std::printf("[drawthings] rope.inv_freq computed from base 10000: "
                    "%s\n", exact ? "bit-identical" : "DIFFERENT");
        EXPECT_TRUE(exact);
        continue;
      }
      std::vector<float> a, r;
      if (!read_f32_(*dt, n, &a) || !read_f32_(*ref, n, &r) ||
          a.size() != r.size()) {
        std::printf("[drawthings] %s: not presented (or a different "
                    "size)\n", n.c_str());
        ++bad;
        continue;
      }
      double e2 = 0, r2 = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        e2 += ((double)a[i] - r[i]) * ((double)a[i] - r[i]);
        r2 += (double)r[i] * r[i];
      }
      const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
      const std::string stem = n.size() > 7 && n.compare(n.size() - 7, 7,
                                                          ".weight") == 0
                                   ? n.substr(0, n.size() - 7) : n;
      const bool i8 = dt->info(stem + ".i8x") != nullptr;
      const bool pal = n.find("adaln_proj") != std::string::npos &&
                       n.rfind("blocks.", 0) == 0 &&
                       n.find(".weight") != std::string::npos;
      const double bar = i8 ? 4e-2 : pal ? 2e-2 : 4e-3;
      if (rel > bar || n.rfind("blocks.", 0) != 0 ||
          n.find(".0.") != std::string::npos) {
        std::printf("[drawthings] %-48s rel %.3e%s%s\n", n.c_str(), rel,
                    i8 ? " (i8x)" : pal ? " (palette)" : "",
                    rel > bar ? "  OVER" : "");
      }
      bad += rel > bar ? 1 : 0;
      ++seen;
    }
  }
  std::printf("[drawthings] %d tensors compared, %d over their bar\n", seen,
              bad);
  EXPECT_TRUE(seen > 0 && bad == 0);
}

namespace {

// `name` presented with `want`'s values (already f16 where the store
// held f16) and, when given, its shape and dtype; else a line and a count.
void
same_(const MetalLlamaWeights& w, const std::string& name,
      const std::vector<float>& want, std::size_t* bad,
      std::vector<std::int64_t> shape = {}, const char* dtype = nullptr,
      double tol = 0.0)
{
  std::vector<float> got;
  const auto* ti = w.info(name);
  if (ti == nullptr || !read_f32_(w, name, &got) ||
      got.size() != want.size()) {
    std::printf("[drawthings] %s: missing or %zu values (want %zu)\n",
                name.c_str(), got.size(), want.size());
    ++*bad;
    return;
  }
  if ((!shape.empty() && ti->shape != shape) ||
      (dtype != nullptr && ti->dtype != dtype)) {
    std::string s;
    for (auto d : ti->shape) { s += std::to_string(d) + " "; }
    std::printf("[drawthings] %s: shape [ %s] dtype %s\n", name.c_str(),
                s.c_str(), ti->dtype.c_str());
    ++*bad;
    return;
  }
  std::size_t n = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = std::fabs((double)got[i] - want[i]);
    n += d > tol * std::max(1.0, std::fabs((double)want[i]));
  }
  if (n != 0) {
    std::printf("[drawthings] %s: %zu of %zu differ\n", name.c_str(), n,
                want.size());
    ++*bad;
  }
}

// The rows of `v` ([rows, cols]) in the model's order, a rotary head of
// `D` rows stored with its first `rot` interleaved: model row j rot/2 + i
// at 2i + j.
std::vector<float>
unrope_rows_(const std::vector<float>& v, int rows, int cols, int D, int rot)
{
  std::vector<float> o(v.size());
  for (int r = 0; r < rows; ++r) {
    const int h = r / D, d = r % D, half = rot / 2;
    const int sr = h * D + (d < rot ? 2 * (d % half) + d / half : d);
    std::copy(v.begin() + (long)sr * cols, v.begin() + (long)(sr + 1) * cols,
              o.begin() + (long)r * cols);
  }
  return o;
}

std::vector<float>
cat_(std::vector<float> a, const std::vector<float>& b)
{
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

}  // namespace

// A Qwen3-VL store (Draw Things' text encoder for Qwen-Image-2.1 and
// MiniMax-H3) read under transformers' names: q / k rows, norms and
// biases un-interleaved over the WHOLE head, the vision qkv fused, the
// patch embedding a Conv3d again, the MLP's mid-key layer read.
TEST(drawthings_store, a_qwen3_vl_store_reads_under_the_release_names)
{
  const int H = 32, D = 16, NQ = 2, NKV = 1, FF = 48, V = 40;
  const int VH = 64, VD = VH / 16, P = 2, VFF = 96, M = 4 * VH;
  std::mt19937 rng(31);
  Builder b;
  const std::string T = "text_model", G = "vision_model";
  std::map<std::string, std::vector<float>> v;
  auto raw = [&](const std::string& g, const std::string& key, int L,
                 int leaf, std::vector<std::int32_t> dims) {
    std::size_t c = 1;
    for (auto d : dims) { c *= (std::size_t)d; }
    auto x = rnd_(c, rng, 0.1f);
    const std::string n = Builder::at_(g, key, L, leaf);
    if (leaf == 0 && dims.size() >= 2 && dims[0] > 1) {
      b.raw16_(n, dims, x);
    } else {
      b.raw16_inline_(n, dims, x);
    }
    v[n] = x;
    return x;
  };
  raw(T, "tok_embeddings", 0, 0, {V, H});
  raw(T, "input_layernorm", 0, 0, {1, H});
  raw(T, "post_attention_layernorm", 0, 0, {1, H});
  // q as i8x (the 32B's codec): a gathered weight keeps each row's scale.
  std::vector<std::int8_t> qc;
  std::vector<std::uint16_t> qs;
  std::vector<float> qd;
  b.i8x_(Builder::at_(T, "q_proj", 0), NQ * D, H,
         rnd_((std::size_t)NQ * D * H, rng, 0.05f), &qc, &qs, &qd);
  raw(T, "k_proj", 0, 0, {NKV * D, H});
  raw(T, "v_proj", 0, 0, {NKV * D, H});
  raw(T, "out_proj", 0, 0, {H, NQ * D});
  raw(T, "norm_q", 0, 0, {1, 1, 1, D});
  raw(T, "norm_k", 0, 0, {1, 1, 1, D});
  for (const char* x : {"gate", "up"}) {
    raw(T, std::string("mlp-0-mlp_") + x + "_proj", 0, 0, {FF, H});
  }
  raw(T, "mlp-0-mlp_down_proj", 0, 0, {H, FF});
  raw(G, "conv_in", 0, 0, {VH, 6 * P * P});
  raw(G, "conv_in", 0, 1, {VH});
  raw(G, "pos_embed", 0, 0, {16, VH});
  for (int leaf = 0; leaf < 2; ++leaf) {
    auto dims = [&](std::vector<std::int32_t> w) {
      return leaf == 0 ? w : std::vector<std::int32_t>{w[0]};
    };
    for (const char* k : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      raw(G, k, 0, leaf, dims({VH, VH}));
    }
    raw(G, "mlp_fc1", 0, leaf, dims({VFF, VH}));
    raw(G, "mlp_fc2", 0, leaf, dims({VH, VFF}));
    raw(G, "norm1", 0, leaf, {1, VH});
    raw(G, "norm2", 0, leaf, {1, VH});
    raw(G, "norm_out", 0, leaf, {1, VH});
    raw(G, "merger_mlp_0", 0, leaf, dims({M, M}));
    raw(G, "merger_mlp_1", 0, leaf, dims({H, M}));
    raw(G, "deepstack_0_norm", 0, leaf, {1, M});
    raw(G, "deepstack_0_mlp_0", 0, leaf, dims({M, M}));
    raw(G, "deepstack_0_mlp_1", 0, leaf, dims({H, M}));
  }
  const std::string path = tmp_("qwen3vl.ckpt");
  ASSERT_TRUE(b.write(path));
  auto w = MetalLlamaWeights::open_model(path);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) {
    fs::remove(path);
    return;
  }
  std::size_t bad = 0;
  auto at = [&](const std::string& g, const std::string& k, int leaf = 0) {
    return v[Builder::at_(g, k, 0, leaf)];
  };
  const std::string l = "model.language_model.", p = l + "layers.0.";
  same_(*w, l + "embed_tokens.weight", at(T, "tok_embeddings"), &bad,
        {V, H});
  same_(*w, p + "input_layernorm.weight", at(T, "input_layernorm"), &bad,
        {H});
  same_(*w, p + "self_attn.q_proj.weight",
        unrope_rows_(qd, NQ * D, H, D, D), &bad, {NQ * D, H});
  {
    // The codes and scales, gathered with their rows.
    std::vector<float> c(qc.begin(), qc.end()), sv(qs.size());
    for (std::size_t i = 0; i < qs.size(); ++i) {
      __fp16 h;
      std::memcpy(&h, &qs[i], 2);
      sv[i] = (float)h;
    }
    std::vector<std::uint8_t> cb(qc.size());
    const bool got = w->read_into(p + "self_attn.q_proj.i8x", cb.data(),
                                  cb.size());
    const auto want = unrope_rows_(c, NQ * D, H, D, D);
    std::size_t n = 0;
    for (std::size_t i = 0; got && i < cb.size(); ++i) {
      n += (float)(std::int8_t)cb[i] != want[i];
    }
    EXPECT_TRUE(got && n == 0);
    same_(*w, p + "self_attn.q_proj.i8x_scales",
          unrope_rows_(sv, NQ * D, 1, D, D), &bad, {NQ * D});
  }
  same_(*w, p + "self_attn.k_proj.weight",
        unrope_rows_(at(T, "k_proj"), NKV * D, H, D, D), &bad);
  same_(*w, p + "self_attn.v_proj.weight", at(T, "v_proj"), &bad);
  same_(*w, p + "self_attn.o_proj.weight", at(T, "out_proj"), &bad);
  same_(*w, p + "self_attn.q_norm.weight",
        unrope_rows_(at(T, "norm_q"), D, 1, D, D), &bad, {D});
  same_(*w, p + "self_attn.k_norm.weight",
        unrope_rows_(at(T, "norm_k"), D, 1, D, D), &bad, {D});
  same_(*w, p + "mlp.gate_proj.weight", at(T, "mlp-0-mlp_gate_proj"), &bad);
  same_(*w, p + "mlp.down_proj.weight", at(T, "mlp-0-mlp_down_proj"), &bad,
        {H, FF});
  const std::string vt = "model.visual.";
  same_(*w, vt + "patch_embed.proj.weight", at(G, "conv_in"), &bad,
        {VH, 3, 2, P, P});
  same_(*w, vt + "patch_embed.proj.bias", at(G, "conv_in", 1), &bad, {VH});
  same_(*w, vt + "pos_embed.weight", at(G, "pos_embed"), &bad, {16, VH});
  same_(*w, vt + "blocks.0.attn.qkv.weight",
        cat_(cat_(unrope_rows_(at(G, "q_proj"), VH, VH, VD, VD),
                  unrope_rows_(at(G, "k_proj"), VH, VH, VD, VD)),
             at(G, "v_proj")),
        &bad, {3 * VH, VH});
  same_(*w, vt + "blocks.0.attn.qkv.bias",
        cat_(cat_(unrope_rows_(at(G, "q_proj", 1), VH, 1, VD, VD),
                  unrope_rows_(at(G, "k_proj", 1), VH, 1, VD, VD)),
             at(G, "v_proj", 1)),
        &bad, {3 * VH});
  same_(*w, vt + "blocks.0.attn.proj.bias", at(G, "out_proj", 1), &bad);
  same_(*w, vt + "blocks.0.mlp.linear_fc1.weight", at(G, "mlp_fc1"), &bad);
  same_(*w, vt + "blocks.0.norm2.bias", at(G, "norm2", 1), &bad, {VH});
  same_(*w, vt + "merger.norm.weight", at(G, "norm_out"), &bad, {VH});
  same_(*w, vt + "merger.linear_fc2.weight", at(G, "merger_mlp_1"), &bad,
        {H, M});
  same_(*w, vt + "deepstack_merger_list.0.norm.bias",
        at(G, "deepstack_0_norm", 1), &bad, {M});
  same_(*w, vt + "deepstack_merger_list.0.linear_fc1.weight",
        at(G, "deepstack_0_mlp_0"), &bad, {M, M});
  // No head and no final norm: Draw Things stores neither.
  EXPECT_FALSE(w->has("lm_head.weight") || w->has(l + "norm.weight"));
  std::printf("[drawthings] Qwen3-VL store: %zu tensors differ\n", bad);
  EXPECT_TRUE(bad == 0);
  fs::remove(path);
}

// MiniMax-H3's VAE store, video and audio in one file, read under the
// released names: the video decoder's per-head qkv (64-row heads, 48
// rotary) and gate | up, the encoder's widening blocks' shortcuts, the
// norms whose dims carry a trailing byte count; the audio VAE in F32
// with its folded convs as plain weights, the decoder snakes back in log
// space and the head's fused bias split.
TEST(drawthings_store, an_h3_vae_store_reads_under_the_release_names)
{
  std::mt19937 rng(41);
  Builder b;
  std::map<std::string, std::vector<float>> v;
  auto put = [&](const std::string& g, const std::string& key, int L,
                 int leaf, std::vector<std::int32_t> dims, float lo = 0.0f) {
    std::size_t c = 1;
    for (std::size_t i = 0; i < dims.size() && dims[i] != 0; ++i) {
      c *= (std::size_t)dims[i];
    }
    auto x = rnd_(c, rng, 0.1f);
    // A strictly positive parameter (Draw Things' e^alpha, 1 / alpha).
    if (lo > 0.0f) {
      for (float& e : x) { e = lo + std::fabs(e); }
    }
    const std::string n = Builder::at_(g, key, L, leaf);
    if (leaf == 0 && dims.size() >= 2 && dims[0] > 1 && dims[1] > 1) {
      b.raw16_(n, dims, x);
    } else {
      b.raw16_inline_(n, dims, x);
    }
    v[n] = x;
  };
  auto at = [&](const std::string& g, const std::string& k, int L = 0,
                int leaf = 0) { return v[Builder::at_(g, k, L, leaf)]; };
  // The video encoder: two levels of two blocks, 8 then 16 wide.
  const std::string VE = "video_encoder", VDc = "video_decoder";
  put(VE, "conv_in", 0, 0, {8, 3, 3, 3, 3});
  put(VE, "conv_in", 0, 1, {8});
  for (int r = 0; r < 4; ++r) {
    const int ci = r == 2 ? 8 : (r < 2 ? 8 : 16), co = r < 2 ? 8 : 16;
    put(VE, "conv1", r, 0, {co, ci, 3, 3, 3});
    put(VE, "conv1", r, 1, {co});
    put(VE, "conv2", r, 0, {co, co, 3, 3, 3});
    put(VE, "conv2", r, 1, {co});
    for (int leaf = 0; leaf < 2; ++leaf) {
      put(VE, "norm1", r, leaf, {1, ci, 1, 1});
      put(VE, "norm2", r, leaf, {1, co, 1, 1});
    }
  }
  put(VE, "nin_shortcut", 0, 0, {16, 8, 1, 1, 1});
  put(VE, "nin_shortcut", 0, 1, {16});
  put(VE, "downsample_0", 0, 0, {8, 8, 3, 3, 3});
  put(VE, "downsample_0", 0, 1, {8});
  put(VE, "norm_out", 0, 0, {1, 16, 1, 1});
  put(VE, "norm_out", 0, 1, {1, 16, 1, 1});
  put(VE, "conv_out", 0, 0, {6, 16, 3, 3, 3});
  put(VE, "conv_out", 0, 1, {6});
  put(VE, "quant_conv", 0, 0, {6, 6, 1, 1, 1});
  put(VE, "quant_conv", 0, 1, {6});
  // The video decoder: one block of two 64-row heads.
  const int W = 128, HD = 64, FF = 32;
  put(VDc, "post_quant_conv", 0, 0, {3, 3});
  put(VDc, "post_quant_conv", 0, 1, {3});
  put(VDc, "decoder_proj_in", 0, 0, {W, 3});
  put(VDc, "decoder_proj_in", 0, 1, {W});
  put(VDc, "decoder_proj_out", 0, 0, {12, W});
  put(VDc, "decoder_proj_out", 0, 1, {12});
  // [d0, d1, d2, 0, bytes]: the byte count is not a dimension.
  put(VDc, "decoder_norm_out", 0, 0, {1, 1, W, 0, W * 2});
  put(VDc, "decoder_norm_out", 0, 1, {1, 1, W, 0, W * 2});
  put(VDc, "register_tokens", 0, 0, {1, 2, W, 0, 2 * W * 2});
  for (const char* k : {"norm1", "norm2", "scale1", "scale2"}) {
    put(VDc, k, 0, 0, {1, 1, W, 0, W * 2});
  }
  for (int leaf = 0; leaf < 2; ++leaf) {
    auto dims = [&](std::vector<std::int32_t> d) {
      return leaf == 0 ? d : std::vector<std::int32_t>{d[0]};
    };
    for (const char* k : {"to_q", "to_k", "to_v", "to_out"}) {
      put(VDc, k, 0, leaf, dims({W, W}));
    }
    put(VDc, "ff_gate", 0, leaf, dims({FF, W}));
    put(VDc, "ff_up", 0, leaf, dims({FF, W}));
    put(VDc, "ff_down", 0, leaf, dims({W, FF}));
  }
  // The audio decoder: one upsample stage, one resblock of one dilation.
  const std::string AD = "audio_decoder", AE = "audio_encoder";
  put(AD, "dec_in_proj", 0, 0, {8, 4, 1, 1});
  put(AD, "dec_in_proj", 0, 1, {8});
  put(AD, "audio_conv_pre", 0, 0, {8, 8, 1, 7});
  put(AD, "audio_conv_pre", 0, 1, {8});
  put(AD, "audio_up_0", 0, 0, {8, 4, 1, 4});
  put(AD, "audio_up_0", 0, 1, {4});
  auto snake = [&](const std::string& k, int c) {
    put(AD, k + "_snake_alpha", 0, 0, {1, c, 1, 1}, 0.5f);
    put(AD, k + "_snake_beta", 0, 0, {1, c, 1, 1}, 0.5f);
    put(AD, k + "_upsample", 0, 0, {1, 1, 1, 12});
    put(AD, k + "_downsample", 0, 0, {1, 1, 1, 12});
  };
  for (int K = 0; K < 2; ++K) {
    const std::string jk = "0_" + std::to_string(K);
    put(AD, "audio_amp_0_c" + jk, 0, 0, {4, 4, 1, 3});
    put(AD, "audio_amp_0_c" + jk, 0, 1, {4});
    snake("audio_amp_0_a" + jk, 4);
  }
  snake("audio_post", 4);
  put(AD, "audio_conv_post", 0, 0, {1, 4, 1, 7});
  // The audio encoder: one stage, 4 -> 8, and the posterior head.
  auto snake1 = [&](const std::string& k, int c) {
    put(AE, k + "_alpha", 0, 0, {1, c, 1, 1}, 0.2f);
    put(AE, k + "_reciprocal", 0, 0, {1, c, 1, 1}, 0.2f);
  };
  put(AE, "encoder_input", 0, 0, {4, 1, 1, 7});
  put(AE, "encoder_input", 0, 1, {4});
  put(AE, "encoder_block_0_res0_conv1", 0, 0, {4, 4, 1, 7});
  put(AE, "encoder_block_0_res0_conv1", 0, 1, {4});
  put(AE, "encoder_block_0_res0_conv2", 0, 0, {4, 4, 1, 1});
  put(AE, "encoder_block_0_res0_conv2", 0, 1, {4});
  snake1("encoder_block_0_res0_snake1", 4);
  snake1("encoder_block_0_res0_snake2", 4);
  snake1("encoder_block_0_snake", 4);
  put(AE, "encoder_block_0_downsample", 0, 0, {8, 4, 1, 4});
  put(AE, "encoder_block_0_downsample", 0, 1, {8});
  snake1("encoder_final_snake", 8);
  put(AE, "encoder_final_conv", 0, 0, {8, 8, 1, 3});
  put(AE, "encoder_final_conv", 0, 1, {8});
  for (int leaf = 0; leaf < 2; ++leaf) {
    auto dims = [&](std::vector<std::int32_t> d) {
      return leaf == 0 ? d : std::vector<std::int32_t>{d[0]};
    };
    put(AE, "norm1", 0, leaf, {1, 1, 8});
    put(AE, "norm2", 0, leaf, {1, 1, 4});
    put(AE, "norm3", 0, leaf, {1, 1, 8});
    put(AE, "mlp_norm", 0, leaf, {1, 1, 4});
    put(AE, "qkv", 0, leaf, dims({12, 4}));
    put(AE, "attention_output", 0, leaf, dims({4, 4}));
    put(AE, "input_projection", 0, leaf, dims({4, 8}));
    put(AE, "mlp_gate", 0, leaf, dims({6, 4}));
    put(AE, "mlp_up", 0, leaf, dims({6, 4}));
    put(AE, "mlp_down", 0, leaf, dims({4, 6}));
    put(AE, "mean_proj", 0, leaf, dims({4, 4}));
    put(AE, "logs_proj", 0, leaf, dims({4, 4}));
  }
  const std::string path = tmp_("h3vae.ckpt");
  ASSERT_TRUE(b.write(path));
  auto w = MetalLlamaWeights::open_model(path);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) {
    fs::remove(path);
    return;
  }
  std::size_t bad = 0;
  // Video, encoder.
  same_(*w, "encoder.conv_in.weight", at(VE, "conv_in"), &bad,
        {8, 3, 3, 3, 3}, "F16");
  same_(*w, "encoder.down.0.block.1.conv2.weight", at(VE, "conv2", 1), &bad,
        {8, 8, 3, 3, 3});
  same_(*w, "encoder.down.1.block.0.norm1.bias", at(VE, "norm1", 2, 1), &bad,
        {8});
  same_(*w, "encoder.down.1.block.1.conv1.weight", at(VE, "conv1", 3), &bad);
  same_(*w, "encoder.down.1.block.0.nin_shortcut.weight",
        at(VE, "nin_shortcut"), &bad, {16, 8, 1, 1, 1});
  EXPECT_FALSE(w->has("encoder.down.0.block.0.nin_shortcut.weight"));
  same_(*w, "encoder.down.0.downsample.conv.weight", at(VE, "downsample_0"),
        &bad);
  same_(*w, "encoder.norm_out.bias", at(VE, "norm_out", 0, 1), &bad, {16});
  same_(*w, "quant_conv.weight", at(VE, "quant_conv"), &bad,
        {6, 6, 1, 1, 1});
  // Video, decoder.
  same_(*w, "post_quant_conv.weight", at(VDc, "post_quant_conv"), &bad,
        {3, 3, 1, 1, 1});
  same_(*w, "decoder.x_embedder.weight", at(VDc, "decoder_proj_in"), &bad,
        {W, 3});
  same_(*w, "decoder.norm_out.weight", at(VDc, "decoder_norm_out"), &bad,
        {W});
  same_(*w, "decoder.register_tokens", at(VDc, "register_tokens"), &bad,
        {1, 2, W});
  const std::string tb = "decoder.transformer_blocks.0.";
  same_(*w, tb + "scale1", at(VDc, "scale1"), &bad, {W});
  same_(*w, tb + "norm2.weight", at(VDc, "norm2"), &bad, {W});
  for (int leaf = 0; leaf < 2; ++leaf) {
    const int cols = leaf == 0 ? W : 1;
    std::vector<float> want((std::size_t)3 * W * cols);
    for (int r = 0; r < 3 * W; ++r) {
      const int h = r / (3 * HD), pt = (r / HD) % 3, d = r % HD;
      const auto& src = at(VDc, pt == 0 ? "to_q" : pt == 1 ? "to_k" : "to_v",
                           0, leaf);
      const int sr = h * HD + (pt < 2 && d < 48 ? 2 * (d % 24) + d / 24 : d);
      std::copy(src.begin() + (long)sr * cols,
                src.begin() + (long)(sr + 1) * cols,
                want.begin() + (long)r * cols);
    }
    same_(*w, tb + "attn.to_qkv." + (leaf == 0 ? "weight" : "bias"), want,
          &bad);
    same_(*w, tb + "ff.w1." + (leaf == 0 ? "weight" : "bias"),
          cat_(at(VDc, "ff_gate", 0, leaf), at(VDc, "ff_up", 0, leaf)), &bad);
  }
  same_(*w, tb + "ff.w2.weight", at(VDc, "ff_down"), &bad, {W, FF});
  // Audio, decoder: F32, the folded convs as plain weights.
  same_(*w, "dec_in_proj.weight", at(AD, "dec_in_proj"), &bad, {8, 4, 1},
        "F32");
  same_(*w, "decoder.ups.0.0.weight", at(AD, "audio_up_0"), &bad, {8, 4, 4},
        "F32");
  same_(*w, "decoder.resblocks.0.convs2.0.weight", at(AD, "audio_amp_0_c0_1"),
        &bad, {4, 4, 3}, "F32");
  same_(*w, "decoder.conv_post.weight", at(AD, "audio_conv_post"), &bad,
        {1, 4, 7}, "F32");
  EXPECT_FALSE(w->has("decoder.conv_post.bias"));
  auto logs = [&](std::vector<float> x, float sign) {
    for (float& e : x) { e = sign * std::log(e); }
    return x;
  };
  const std::string act = "decoder.resblocks.0.activations.";
  same_(*w, act + "1.act.alpha",
        logs(at(AD, "audio_amp_0_a0_1_snake_alpha"), 1.0f), &bad, {4}, "F32",
        1e-6);
  same_(*w, act + "0.act.beta",
        logs(at(AD, "audio_amp_0_a0_0_snake_beta"), -1.0f), &bad, {4},
        "F32", 1e-6);
  same_(*w, act + "0.upsample.filter",
        at(AD, "audio_amp_0_a0_0_upsample"), &bad, {1, 1, 12}, "F32");
  same_(*w, "decoder.activation_post.downsample.lowpass.filter",
        at(AD, "audio_post_downsample"), &bad, {1, 1, 12}, "F32");
  // Audio, encoder.
  same_(*w, "encoder.block.0.weight", at(AE, "encoder_input"), &bad,
        {4, 1, 7}, "F32");
  const std::string eb = "encoder.block.1.block.";
  same_(*w, eb + "0.block.0.alpha", at(AE, "encoder_block_0_res0_snake1_alpha"),
        &bad, {1, 4, 1}, "F32");
  same_(*w, eb + "0.block.3.weight", at(AE, "encoder_block_0_res0_conv2"),
        &bad, {4, 4, 1}, "F32");
  same_(*w, eb + "3.alpha", at(AE, "encoder_block_0_snake_alpha"), &bad,
        {1, 4, 1});
  same_(*w, eb + "4.bias", at(AE, "encoder_block_0_downsample", 0, 1), &bad,
        {8});
  same_(*w, "encoder.block.2.alpha", at(AE, "encoder_final_snake_alpha"),
        &bad, {1, 8, 1});
  same_(*w, "encoder.block.3.weight", at(AE, "encoder_final_conv"), &bad,
        {8, 8, 3});
  const auto qb = at(AE, "qkv", 0, 1);
  same_(*w, "pre_block.attn.q_bias", {qb.begin(), qb.begin() + 4}, &bad,
        {4}, "F32");
  same_(*w, "pre_block.attn.zero_k_bias", {qb.begin() + 4, qb.begin() + 8},
        &bad, {4});
  same_(*w, "pre_block.attn.v_bias", {qb.begin() + 8, qb.end()}, &bad, {4});
  same_(*w, "pre_block.attn.qkv.weight", at(AE, "qkv"), &bad, {12, 4});
  same_(*w, "pre_block.mlp.w1.weight", at(AE, "mlp_up"), &bad, {6, 4});
  same_(*w, "pre_block.norm2.bias", at(AE, "norm2", 0, 1), &bad, {4});
  same_(*w, "mean_proj.weight", at(AE, "mean_proj"), &bad, {4, 4, 1});
  for (const auto& n : w->tensor_names()) {
    if (n.find("reciprocal") != std::string::npos) {
      std::printf("[drawthings] %s presented\n", n.c_str());
      ++bad;
    }
  }
  std::printf("[drawthings] H3 VAE store: %zu tensors differ\n", bad);
  EXPECT_TRUE(bad == 0);
  fs::remove(path);
}

// Qwen-Image-2.1's VAE store: Draw Things' name is diffusers' with the
// dots as underscores, which only the diffusers vocabulary can undo
// (conv_in, down_blocks, to_qkv hold underscores of their own); norms
// are gammas, stored [1, C, 1, 1].
TEST(drawthings_store, a_qwen_image21_vae_store_reads_under_diffusers_names)
{
  std::mt19937 rng(43);
  Builder b;
  std::map<std::string, std::vector<float>> v;
  auto put = [&](const std::string& g, const std::string& key, int leaf,
                 std::vector<std::int32_t> dims) {
    std::size_t c = 1;
    for (auto d : dims) { c *= (std::size_t)d; }
    auto x = rnd_(c, rng, 0.1f);
    const std::string n = Builder::at_(g, key, 0, leaf);
    if (leaf == 0 && dims.size() == 4 && dims[0] > 1) {
      b.raw16_(n, dims, x);
    } else {
      b.raw16_inline_(n, dims, x);
    }
    v[n] = x;
  };
  put("decoder", "decoder_conv_in", 0, {16, 4, 3, 3});
  put("decoder", "decoder_conv_in", 1, {16});
  put("decoder", "post_quant_conv", 0, {4, 4, 1, 1});
  put("decoder", "post_quant_conv", 1, {4});
  put("decoder", "decoder_mid_block_attentions_0_norm", 0, {1, 16, 1, 1});
  put("decoder", "decoder_mid_block_attentions_0_to_qkv", 0, {48, 16, 1, 1});
  put("decoder", "decoder_mid_block_attentions_0_to_qkv", 1, {48});
  put("decoder", "decoder_up_blocks_0_resnets_2_norm1", 0, {1, 16, 1, 1});
  put("decoder", "decoder_up_blocks_3_upsampler_resample_1", 0, {8, 16, 3, 3});
  put("decoder", "decoder_norm_out", 0, {1, 16, 1, 1});
  put("encoder", "quant_conv", 0, {8, 8, 1, 1});
  put("encoder", "encoder_conv_in", 0, {8, 4, 3, 3});
  put("encoder", "encoder_down_blocks_1_resnets_0_conv_shortcut", 0,
      {16, 8, 1, 1});
  put("encoder", "encoder_down_blocks_1_resnets_0_conv_shortcut", 1, {16});
  put("encoder", "encoder_down_blocks_0_downsampler_resample_1", 1, {8});
  const std::string path = tmp_("qi21vae.ckpt");
  ASSERT_TRUE(b.write(path));
  auto w = MetalLlamaWeights::open_model(path);
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) {
    fs::remove(path);
    return;
  }
  auto at = [&](const std::string& g, const std::string& k, int leaf = 0) {
    return v[Builder::at_(g, k, 0, leaf)];
  };
  std::size_t bad = 0;
  same_(*w, "decoder.conv_in.weight", at("decoder", "decoder_conv_in"), &bad,
        {16, 4, 3, 3}, "F16");
  same_(*w, "decoder.conv_in.bias", at("decoder", "decoder_conv_in", 1),
        &bad, {16});
  same_(*w, "post_quant_conv.weight", at("decoder", "post_quant_conv"), &bad,
        {4, 4, 1, 1});
  same_(*w, "decoder.mid_block.attentions.0.norm.gamma",
        at("decoder", "decoder_mid_block_attentions_0_norm"), &bad,
        {16, 1, 1});
  same_(*w, "decoder.mid_block.attentions.0.to_qkv.bias",
        at("decoder", "decoder_mid_block_attentions_0_to_qkv", 1), &bad,
        {48});
  same_(*w, "decoder.up_blocks.0.resnets.2.norm1.gamma",
        at("decoder", "decoder_up_blocks_0_resnets_2_norm1"), &bad,
        {16, 1, 1, 1});
  same_(*w, "decoder.up_blocks.3.upsampler.resample.1.weight",
        at("decoder", "decoder_up_blocks_3_upsampler_resample_1"), &bad,
        {8, 16, 3, 3});
  same_(*w, "decoder.norm_out.gamma", at("decoder", "decoder_norm_out"),
        &bad, {16, 1, 1, 1});
  same_(*w, "quant_conv.weight", at("encoder", "quant_conv"), &bad,
        {8, 8, 1, 1});
  same_(*w, "encoder.down_blocks.1.resnets.0.conv_shortcut.bias",
        at("encoder", "encoder_down_blocks_1_resnets_0_conv_shortcut", 1),
        &bad, {16});
  same_(*w, "encoder.down_blocks.0.downsampler.resample.1.bias",
        at("encoder", "encoder_down_blocks_0_downsampler_resample_1", 1),
        &bad, {8});
  std::printf("[drawthings] Qwen-Image-2.1 VAE store: %zu tensors differ\n",
              bad);
  EXPECT_TRUE(bad == 0);
  fs::remove(path);
  // A word outside the vocabulary is refused, not guessed at.
  b.rows.clear();
  put("decoder", "decoder_conv_in", 0, {16, 4, 3, 3});
  put("decoder", "decoder_up_blocks_0_time_conv", 0, {16, 16, 1, 1});
  ASSERT_TRUE(b.write(path));
  EXPECT_FALSE(MetalLlamaWeights::open_model(path).has_value());
  fs::remove(path);
}

namespace {

// Every tensor of the release `ref` against the store `dt`'s presentation
// of it, at `bar` rel-L2 (`i8_bar` where the store holds it as i8x). A
// weight-normed `<c>.weight_g` / `.weight_v` pair is compared FOLDED
// against the store's `<c>.weight`. `absent(name)` names what the store
// is known not to carry; `consumed`, when set, maps both sides to what
// the model actually reads before they are compared. Adds what it
// compared to `explained`. Returns the failures.
using Consumed = std::function<void(const std::string&, std::vector<float>*)>;
int
against_release_(const MetalLlamaWeights& dt, const MetalLlamaWeights& ref,
                 double bar, double i8_bar,
                 const std::function<bool(const std::string&)>& absent,
                 int* seen_out, std::set<std::string>* explained,
                 const Consumed& consumed = {})
{
  int bad = 0, seen = 0, skipped = 0;
  double worst = 0, worst_i8 = 0;
  std::string worst_n;
  std::vector<std::string> names = ref.tensor_names();
  std::sort(names.begin(), names.end());
  std::set<std::string> folded;
  for (const std::string& n : names) {
    const bool g = n.size() > 9 && n.compare(n.size() - 9, 9, ".weight_g") == 0;
    const bool v = n.size() > 9 && n.compare(n.size() - 9, 9, ".weight_v") == 0;
    if (g) { continue; }
    std::vector<float> r, a;
    std::string dn = n;
    if (v) {
      // w = g v / ||v|| over every axis but the first.
      const std::string stem = n.substr(0, n.size() - 9);
      std::vector<float> gv;
      if (!read_f32_(ref, n, &r) || !read_f32_(ref, stem + ".weight_g", &gv) ||
          gv.empty()) {
        ++bad;
        continue;
      }
      const std::size_t per = r.size() / gv.size();
      for (std::size_t o = 0; o < gv.size(); ++o) {
        double q = 0;
        for (std::size_t j = 0; j < per; ++j) {
          q += (double)r[o * per + j] * r[o * per + j];
        }
        const double sc = q > 0 ? gv[o] / std::sqrt(q) : 0.0;
        for (std::size_t j = 0; j < per; ++j) {
          r[o * per + j] = (float)(r[o * per + j] * sc);
        }
      }
      dn = stem + ".weight";
      folded.insert(dn);
    } else if (!read_f32_(ref, n, &r)) {
      ++bad;
      continue;
    }
    if (!dt.has(dn)) {
      if (absent(n)) {
        ++skipped;
      } else {
        std::printf("[drawthings] %s: not in the store\n", n.c_str());
        ++bad;
      }
      continue;
    }
    if (!read_f32_(dt, dn, &a) || a.size() != r.size()) {
      std::printf("[drawthings] %s: %zu values against %zu\n", n.c_str(),
                  a.size(), r.size());
      ++bad;
      continue;
    }
    if (consumed) {
      consumed(n, &a);
      consumed(n, &r);
    }
    double e2 = 0, r2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      e2 += ((double)a[i] - r[i]) * ((double)a[i] - r[i]);
      r2 += (double)r[i] * r[i];
    }
    // An all-zero tensor (a frozen k bias) must read back all zero.
    const double rel = r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2);
    const std::string stem = dn.size() > 7 && dn.compare(dn.size() - 7, 7,
                                                         ".weight") == 0
                                 ? dn.substr(0, dn.size() - 7) : dn;
    const bool i8 = dt.has(stem + ".i8x");
    const double b = i8 ? i8_bar : bar;
    if (rel > b) {
      std::printf("[drawthings] %-56s rel %.3e%s  OVER\n", n.c_str(), rel,
                  i8 ? " (i8x)" : "");
      ++bad;
    }
    if (i8) {
      worst_i8 = std::max(worst_i8, rel);
    } else if (rel > worst) {
      worst = rel;
      worst_n = n;
    }
    ++seen;
  }
  explained->insert(names.begin(), names.end());
  explained->insert(folded.begin(), folded.end());
  std::printf("[drawthings] %d tensors compared (%d known absent): worst "
              "%.3e (%s), worst i8x %.3e; %d failures\n",
              seen, skipped, worst, worst_n.c_str(), worst_i8, bad);
  if (seen_out != nullptr) { *seen_out = seen; }
  return bad;
}

// The other way: every name the store presents must be a release name
// (or a fold of one), or an i8x weight's codes and scales.
int
strangers_(const MetalLlamaWeights& dt, const std::set<std::string>& known)
{
  int bad = 0;
  for (const std::string& n : dt.tensor_names()) {
    const bool i8 = (n.size() > 4 && n.compare(n.size() - 4, 4, ".i8x") == 0) ||
                    n.find(".i8x_scales") != std::string::npos;
    if (!i8 && known.count(n) == 0) {
      std::printf("[drawthings] %s: presented, not a release name\n",
                  n.c_str());
      ++bad;
    }
  }
  return bad;
}

}  // namespace

// THE REAL TEXT ENCODERS. Draw Things stores no head and no final norm,
// and the 32B only the 50 layers MiniMax-H3 taps.
TEST(drawthings_store, a_qwen3_vl_checkpoint_matches_the_release)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_QWEN3VL_CKPT");
  const char* rd = std::getenv("VPIPE_QWEN3VL_RELEASE");
  if (ck == nullptr || rd == nullptr || *ck == '\0' || *rd == '\0') {
    return;
  }
  auto dt = MetalLlamaWeights::open_model(ck);
  auto ref = MetalLlamaWeights::open_model(rd);
  ASSERT_TRUE(dt.has_value() && ref.has_value());
  if (!dt.has_value() || !ref.has_value()) { return; }
  int layers = 0;
  while (dt->has("model.language_model.layers." + std::to_string(layers) +
                 ".self_attn.q_proj.weight")) {
    ++layers;
  }
  std::printf("[drawthings] Qwen3-VL store: %d language layers\n", layers);
  auto absent = [&](const std::string& n) {
    if (n == "lm_head.weight" || n == "model.language_model.norm.weight") {
      return true;
    }
    const std::string p = "model.language_model.layers.";
    return n.rfind(p, 0) == 0 && std::atoi(n.c_str() + p.size()) >= layers;
  };
  int seen = 0;
  std::set<std::string> known;
  const int bad = against_release_(*dt, *ref, 2e-3, 4e-2, absent, &seen,
                                   &known) +
                  strangers_(*dt, known);
  EXPECT_TRUE(layers > 0 && seen > 0 && bad == 0);
}

TEST(drawthings_store, a_qwen_image21_vae_checkpoint_matches_texture_fix)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_QI21_VAE_CKPT");
  const char* rd = std::getenv("VPIPE_QI21_VAE_REFERENCE");
  if (ck == nullptr || rd == nullptr || *ck == '\0' || *rd == '\0') {
    return;
  }
  auto dt = MetalLlamaWeights::open_model(ck);
  auto ref = MetalLlamaWeights::open_model(rd);
  ASSERT_TRUE(dt.has_value() && ref.has_value());
  if (!dt.has_value() || !ref.has_value()) { return; }
  // The temporal convs an image never runs.
  auto absent = [](const std::string& n) {
    return n.find(".time_conv.") != std::string::npos;
  };
  int seen = 0;
  std::set<std::string> known;
  const int bad = against_release_(*dt, *ref, 1e-3, 1e-3, absent, &seen,
                                   &known) +
                  strangers_(*dt, known);
  EXPECT_TRUE(seen > 0 && bad == 0);
}

TEST(drawthings_store, a_minimax_h3_vae_checkpoint_matches_the_release)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_H3_VAE_CKPT");
  const char* rd = std::getenv("VPIPE_H3_VAE_RELEASE");
  if (ck == nullptr || rd == nullptr || *ck == '\0' || *rd == '\0') {
    return;
  }
  auto dt = MetalLlamaWeights::open_model(ck);
  ASSERT_TRUE(dt.has_value());
  if (!dt.has_value()) { return; }
  int bad = 0, seen = 0;
  std::set<std::string> known;
  // An all-zero mask the decoder never reads.
  auto absent = [](const std::string& n) { return n == "decoder.mask_token"; };
  // The audio decoder's snakes are read as e^alpha and 1 / (e^beta +
  // eps), which is what Draw Things stores (in f16); their logs carry
  // that rounding as an ABSOLUTE error, a few e-3 relative to an alpha
  // near zero, which the model never sees.
  auto consumed = [](const std::string& n, std::vector<float>* v) {
    auto ends = [&](const char* x) {
      const std::size_t k = std::strlen(x);
      return n.size() > k && n.compare(n.size() - k, k, x) == 0;
    };
    if (ends(".act.alpha")) {
      for (float& e : *v) { e = std::exp(e); }
    } else if (ends(".act.beta")) {
      for (float& e : *v) { e = 1.0f / (std::exp(e) + 1e-9f); }
    }
  };
  for (const char* part : {"video_vae/source", "audio_vae"}) {
    auto ref = MetalLlamaWeights::open_model((fs::path(rd) / part).string());
    ASSERT_TRUE(ref.has_value());
    if (!ref.has_value()) { return; }
    int s = 0;
    std::printf("[drawthings] %s:\n", part);
    bad += against_release_(*dt, *ref, 1e-3, 1e-3, absent, &s, &known,
                            consumed);
    seen += s;
  }
  bad += strangers_(*dt, known);
  EXPECT_TRUE(seen > 0 && bad == 0);
}

namespace {

using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

// Uniform [-1, 1) in 16-bit floats: bf16 when `bf16`, else f16.
SharedBuffer
noise16_(MetalCompute* mc, std::size_t n, std::uint32_t seed, bool bf16)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  auto* d = static_cast<std::uint16_t*>(b.contents());
  std::uint32_t s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const float v = (float)(s >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
    if (bf16) {
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      d[i] = (std::uint16_t)(u >> 16);
    } else {
      d[i] = f16_bits_(v);
    }
  }
  return b;
}

// rel-L2 of two 16-bit buffers (bf16 or f16) of one size; -1 when they
// cannot be compared.
double
rel16_(const SharedBuffer& a, const SharedBuffer& b, bool bf16)
{
  if (a.empty() || a.byte_size() != b.byte_size()) { return -1.0; }
  const auto* pa = static_cast<const std::uint16_t*>(a.contents());
  const auto* pb = static_cast<const std::uint16_t*>(b.contents());
  auto f = [&](std::uint16_t h) {
    if (bf16) {
      const std::uint32_t u = (std::uint32_t)h << 16;
      float x;
      std::memcpy(&x, &u, 4);
      return x;
    }
    __fp16 x;
    std::memcpy(&x, &h, 2);
    return (float)x;
  };
  double e2 = 0, r2 = 0;
  for (std::size_t i = 0; i < a.byte_size() / 2; ++i) {
    const double x = f(pa[i]), y = f(pb[i]);
    if (!std::isfinite(x) || !std::isfinite(y)) { return -1.0; }
    e2 += (x - y) * (x - y);
    r2 += y * y;
  }
  return std::sqrt(e2 / std::max(r2, 1e-30));
}

double
rel_f32_(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.empty() || a.size() != b.size()) { return -1.0; }
  double e2 = 0, r2 = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i])) { return -1.0; }
    e2 += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
    r2 += (double)b[i] * b[i];
  }
  return std::sqrt(e2 / std::max(r2, 1e-30));
}

}  // namespace

// THE QWEN-IMAGE-2.1 VAE FROM THE STORE IS THE TEXTURE-FIX VAE, through
// the model: Draw Things' f16 is the round-to-nearest f16 of the F32
// release, which is what the loader narrows the F32 file to, so a decode
// and an encode must match TO THE BIT -- anything less is a name, shape
// or gamma the loader read differently.
TEST(drawthings_store, a_qwen_image21_vae_checkpoint_decodes_as_texture_fix)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_QI21_VAE_CKPT");
  const char* rd = std::getenv("VPIPE_QI21_VAE_REFERENCE");
  if (ck == nullptr || rd == nullptr || *ck == '\0' || *rd == '\0') {
    return;
  }
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::ifstream f(fs::path(rd) / "config.json");
  const std::string js((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  MetalKrea2Vae::Config cfg;
  MetalKrea2Vae::config_from_json(vpipe::FlexData::from_json(js), &cfg);
  fs::path ref = rd;
  for (const auto& e : fs::directory_iterator(rd)) {
    if (e.path().extension() == ".safetensors") { ref = e.path(); }
  }
  auto a = MetalKrea2Vae::load(ck, mc, cfg, /*with_encoder=*/true);
  auto b = MetalKrea2Vae::load(ref.string(), mc, cfg, /*with_encoder=*/true);
  ASSERT_TRUE(a != nullptr && b != nullptr);
  if (a == nullptr || b == nullptr) { return; }
  const int h8 = 8, w8 = 8, S = cfg.spatial_factor();
  const SharedBuffer z = noise16_(mc, (std::size_t)cfg.z_dim * h8 * w8, 5,
                                  false);
  std::string err;
  const SharedBuffer da = a->decode(z, h8, w8, &err);
  const SharedBuffer db = b->decode(z, h8, w8, &err);
  const bool dsame = !da.empty() && da.byte_size() == db.byte_size() &&
                     std::memcmp(da.contents(), db.contents(),
                                 da.byte_size()) == 0;
  std::printf("[drawthings] QI21 VAE decode %dx%d: rel %.3e, identical %d\n",
              h8 * S, w8 * S, rel16_(da, db, false), (int)dsame);
  EXPECT_TRUE(dsame);
  const SharedBuffer x =
      noise16_(mc, (std::size_t)cfg.in_channels * h8 * S * w8 * S, 9, false);
  const SharedBuffer ea = a->encode(x, h8 * S, w8 * S);
  const SharedBuffer eb = b->encode(x, h8 * S, w8 * S);
  const bool esame = !ea.empty() && ea.byte_size() == eb.byte_size() &&
                     std::memcmp(ea.contents(), eb.contents(),
                                 ea.byte_size()) == 0;
  std::printf("[drawthings] QI21 VAE encode: rel %.3e, identical %d\n",
              rel16_(ea, eb, false), (int)esame);
  EXPECT_TRUE(esame);
}

// MiniMax-H3's VAEs from the store, through the models, against the
// release. Not bit-identical, and not expected to be: Draw Things
// narrowed the video decoder's linears round-toward-zero (about one bf16
// weight in sixteen then rounds the other way), and folded the audio
// convs' weight norm in its own arithmetic. MEASURED on the M5 Ultra,
// random latents: video decode 9.5e-3 (a 36-layer bf16 ViT amplifies
// it), encode 2.7e-3; audio decode 4.4e-3, encode 6.1e-4. A misplaced
// head or an un-roped row is O(1).
TEST(drawthings_store, a_minimax_h3_vae_checkpoint_decodes_as_the_release)
{
  const char* ck = std::getenv("VPIPE_DRAWTHINGS_H3_VAE_CKPT");
  const char* rd = std::getenv("VPIPE_H3_VAE_RELEASE");
  if (ck == nullptr || rd == nullptr || *ck == '\0' || *rd == '\0') {
    return;
  }
  vpipe::Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto ws = WeightSet::open(ck, nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  {
    MetalMiniMaxH3VideoVae::Config c;
    ASSERT_TRUE(MetalMiniMaxH3VideoVae::config_from_json(rd, c, &err));
    auto a = MetalMiniMaxH3VideoVae::load(ws, mc, c);
    auto b = MetalMiniMaxH3VideoVae::load(rd, mc, c);
    ASSERT_TRUE(a != nullptr && b != nullptr);
    if (a == nullptr || b == nullptr) { return; }
    const int T = 1, h = 4, w = 4;
    const SharedBuffer z =
        noise16_(mc, (std::size_t)c.z_channels * T * h * w, 7, true);
    const double dr = rel16_(a->decode(z, T, h, w, &err),
                             b->decode(z, T, h, w, &err), true);
    const int ET = 5, EH = 64, EW = 64;
    const SharedBuffer x =
        noise16_(mc, (std::size_t)c.in_channels * ET * EH * EW, 11, true);
    const double er = rel16_(a->encode(x, ET, EH, EW, &err),
                             b->encode(x, ET, EH, EW, &err), true);
    std::printf("[drawthings] H3 video VAE: decode rel %.3e, encode rel "
                "%.3e\n", dr, er);
    EXPECT_TRUE(dr >= 0 && dr < 2e-2);
    EXPECT_TRUE(er >= 0 && er < 1e-2);
  }
  MetalMiniMaxH3AudioVae::Config c;
  ASSERT_TRUE(MetalMiniMaxH3AudioVae::config_from_json(rd, c, &err));
  auto a = MetalMiniMaxH3AudioVae::load(ws, mc, c);
  auto b = MetalMiniMaxH3AudioVae::load(rd, mc, c);
  ASSERT_TRUE(a != nullptr && b != nullptr);
  if (a == nullptr || b == nullptr) { return; }
  const int stereo = 2, frames = 12;
  std::vector<float> z((std::size_t)stereo * c.latent_channels * frames);
  std::uint32_t s = 3;
  for (float& v : z) {
    s = s * 1664525u + 1013904223u;
    v = (float)(s >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
  }
  std::vector<float> pa, pb, la, lb;
  ASSERT_TRUE(a->decode(z.data(), stereo, frames, &pa, &err));
  ASSERT_TRUE(b->decode(z.data(), stereo, frames, &pb, &err));
  const int samples = b->decoded_samples(frames);
  const bool enc = pb.size() == (std::size_t)stereo * samples &&
                   a->encode(pb.data(), stereo, samples, &la, nullptr, &err) &&
                   b->encode(pb.data(), stereo, samples, &lb, nullptr, &err);
  const double dr = rel_f32_(pa, pb), er = enc ? rel_f32_(la, lb) : -1.0;
  std::printf("[drawthings] H3 audio VAE: decode rel %.3e, encode rel "
              "%.3e\n", dr, er);
  EXPECT_TRUE(dr >= 0 && dr < 1e-2);
  EXPECT_TRUE(er >= 0 && er < 1e-2);
}
