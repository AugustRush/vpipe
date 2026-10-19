#include "generative-models/shared/drawthings-store.h"

#include <compression.h>
#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <tuple>

namespace vpipe::genai::drawthings {

namespace {

constexpr std::uint32_t kExternal = 0x10000000u;
constexpr std::uint32_t kHadamard = 0x01000000u;
constexpr std::int64_t kDtypeF16 = 0x20000;
constexpr std::int64_t kDtypeF32 = 0x4000;

float
f16_(const std::uint8_t* p)
{
  __fp16 h;
  std::memcpy(&h, p, 2);
  return (float)h;
}

float
elt_(const std::uint8_t* p, int elt)
{
  if (elt == 2) { return f16_(p); }
  float f;
  std::memcpy(&f, p, 4);
  return f;
}

// SQLite's URI form, so `immutable=1` can say the file will not change
// under the reader (no locks, no journal probing -- the app that wrote it
// may still have it open). '%', '?' and '#' are the characters a path
// can hold that a URI cannot.
std::string
uri_(const std::string& path)
{
  std::string u = "file:";
  for (char c : path) {
    if (c == '%' || c == '?' || c == '#') {
      char b[4];
      std::snprintf(b, sizeof b, "%%%02X", (unsigned char)c);
      u += b;
    } else {
      u += c;
    }
  }
  return u + "?immutable=1";
}

}  // namespace

std::int64_t
Tensor::count() const
{
  if (dims.empty()) { return 0; }
  std::int64_t n = 1;
  for (std::int64_t d : dims) { n *= d; }
  return n;
}

bool
looks_like(int fd)
{
  char m[16] = {};
  return ::pread(fd, m, 16, 0) == 16 &&
         std::memcmp(m, "SQLite format 3", 16) == 0;
}

bool
read_index(const std::string& path, std::uint64_t file_size,
           std::vector<Tensor>* out, std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = why; }
    return false;
  };
  out->clear();
  // The trailer's base: big-endian at 60..63 (SQLite's user_version).
  std::uint64_t base = 0;
  {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { return fail("cannot open " + path); }
    std::uint8_t h[64];
    const bool ok = std::fread(h, 1, 64, f) == 64;
    std::fclose(f);
    if (!ok) { return fail("short header"); }
    base = ((std::uint64_t)h[60] << 24) | ((std::uint64_t)h[61] << 16) |
           ((std::uint64_t)h[62] << 8) | (std::uint64_t)h[63];
  }
  sqlite3* raw = nullptr;
  if (sqlite3_open_v2(uri_(path).c_str(), &raw,
                      SQLITE_OPEN_READONLY | SQLITE_OPEN_URI,
                      nullptr) != SQLITE_OK) {
    if (raw != nullptr) { sqlite3_close(raw); }
    return fail("not an SQLite database");
  }
  std::unique_ptr<sqlite3, int (*)(sqlite3*)> db(raw, sqlite3_close);
  sqlite3_stmt* rs = nullptr;
  if (sqlite3_prepare_v2(db.get(),
                         "SELECT name, type, datatype, dim, data "
                         "FROM tensors ORDER BY rowid",
                         -1, &rs, nullptr) != SQLITE_OK) {
    return fail("no `tensors` table: not a Draw Things checkpoint");
  }
  std::unique_ptr<sqlite3_stmt, int (*)(sqlite3_stmt*)> st(rs,
                                                           sqlite3_finalize);
  int rc;
  while ((rc = sqlite3_step(st.get())) == SQLITE_ROW) {
    Tensor t;
    const auto* nm = sqlite3_column_text(st.get(), 0);
    t.name = nm != nullptr ? reinterpret_cast<const char*>(nm) : "";
    const std::uint64_t type =
        (std::uint64_t)sqlite3_column_int64(st.get(), 1);
    const std::uint32_t ident = (std::uint32_t)(type >> 32);
    t.codec = ident & 0x00ffffffu;
    t.external = (ident & kExternal) != 0;
    t.hadamard = (ident & kHadamard) != 0;
    const std::int64_t dt = sqlite3_column_int64(st.get(), 2);
    if (dt == kDtypeF16) {
      t.elt = 2;
    } else if (dt == kDtypeF32) {
      t.elt = 4;
    } else {
      return fail(t.name + ": datatype " + std::to_string(dt) +
                  " is not f16 or f32");
    }
    if (t.codec != kCodecRaw && t.codec != kCodecI8x &&
        t.codec != kCodecQ8p && t.codec != kCodecEzm7) {
      char b[16];
      std::snprintf(b, sizeof b, "0x%x", t.codec);
      return fail(t.name + ": codec " + b + " is not one this reads");
    }
    if (t.codec == kCodecEzm7 && t.elt != 2) {
      return fail(t.name + ": ezm7 over f32");
    }
    // Draw Things' rotation is its own Hadamard ordering (a Kronecker
    // power of the regular H4), which nothing here undoes yet.
    if (t.codec == kCodecI8x && t.hadamard) {
      return fail(t.name + ": a Hadamard-256 i8x weight (not read yet)");
    }
    const auto* dim = static_cast<const std::uint8_t*>(
        sqlite3_column_blob(st.get(), 3));
    const int dimn = sqlite3_column_bytes(st.get(), 3);
    // Up to the first zero: past it some writers leave the payload's
    // byte count ([1, 1, 2048, 0, 4096] is a 2048-vector), which is not
    // a dimension.
    for (int i = 0; dim != nullptr && i + 4 <= dimn && i < 48; i += 4) {
      std::int32_t d;
      std::memcpy(&d, dim + i, 4);
      if (d == 0) { break; }
      t.dims.push_back(d);
    }
    if (t.dims.empty() || t.count() <= 0) {
      return fail(t.name + ": no shape");
    }
    const auto* data = static_cast<const std::uint8_t*>(
        sqlite3_column_blob(st.get(), 4));
    const int datan = sqlite3_column_bytes(st.get(), 4);
    if (t.external) {
      // (u64 offset, u64 length) from the base; a palette row heads
      // them with its block size (u32, padded to 8).
      const int at = t.codec == kCodecQ8p ? 8 : 0;
      if (data == nullptr || datan < at + 16) {
        return fail(t.name + ": external row without its range");
      }
      if (t.codec == kCodecQ8p) { std::memcpy(&t.block, data, 4); }
      std::uint64_t off, len;
      std::memcpy(&off, data + at, 8);
      std::memcpy(&len, data + at + 8, 8);
      t.offset = base + off;
      t.length = len;
      if (t.offset + t.length > file_size) {
        return fail(t.name + ": payload past the end of the file");
      }
    } else {
      if (data == nullptr || datan <= 0) {
        return fail(t.name + ": empty inline payload");
      }
      t.payload.assign(data, data + datan);
      if (t.codec == kCodecQ8p) {
        if (datan < 4) { return fail(t.name + ": short palette payload"); }
        std::memcpy(&t.block, data, 4);
        t.payload.erase(t.payload.begin(), t.payload.begin() + 4);
      }
      t.length = t.payload.size();
    }
    out->push_back(std::move(t));
  }
  if (rc != SQLITE_DONE) {
    return fail(std::string("reading the index: ") +
                sqlite3_errmsg(db.get()));
  }
  if (out->empty()) { return fail("an empty store"); }
  return true;
}

bool
decode_f32(const Tensor& t, const std::uint8_t* file, std::size_t file_size,
           float* out, std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = t.name + ": " + why; }
    return false;
  };
  const std::uint8_t* p = nullptr;
  std::uint64_t n = t.length;
  if (t.external) {
    if (file == nullptr || t.offset + t.length > file_size) {
      return fail("payload outside the mapping");
    }
    p = file + t.offset;
  } else {
    p = t.payload.data();
  }
  const std::uint64_t cnt = (std::uint64_t)t.count();
  if (t.codec == kCodecRaw) {
    if (n < cnt * t.elt) { return fail("short raw payload"); }
    for (std::uint64_t i = 0; i < cnt; ++i) {
      out[i] = elt_(p + i * t.elt, t.elt);
    }
    return true;
  }
  if (t.codec == kCodecI8x) {
    const std::uint64_t N = (std::uint64_t)t.rows();
    const std::uint64_t K = (std::uint64_t)t.cols();
    const std::uint64_t so = i8x_scale_offset(t);
    if (n < so + N * t.elt) { return fail("short i8x payload"); }
    for (std::uint64_t r = 0; r < N; ++r) {
      const float s = elt_(p + so + r * t.elt, t.elt);
      const auto* q = reinterpret_cast<const std::int8_t*>(p + r * K);
      for (std::uint64_t k = 0; k < K; ++k) {
        out[r * K + k] = (float)q[k] * s;
      }
    }
    return true;
  }
  if (t.codec == kCodecQ8p) {
    const std::uint64_t bs = t.block;
    if (bs == 0) { return fail("palette block size 0"); }
    const std::uint64_t pal = 256ull * t.elt;
    for (std::uint64_t b = 0, at = 0; b * bs < cnt; ++b) {
      const std::uint64_t m = std::min(bs, cnt - b * bs);
      if (at + pal + m > n) { return fail("short palette payload"); }
      const std::uint8_t* pv = p + at;
      const std::uint8_t* ix = p + at + pal;
      for (std::uint64_t i = 0; i < m; ++i) {
        out[b * bs + i] = elt_(pv + (std::uint64_t)ix[i] * t.elt, t.elt);
      }
      at += pal + bs;
    }
    return true;
  }
  if (t.codec == kCodecEzm7) {
    std::int32_t zs = 0;
    if (n < 4) { return fail("short ezm7 payload"); }
    std::memcpy(&zs, p, 4);
    if (zs <= 0 || 4 + (std::uint64_t)zs + cnt > n) {
      return fail("short ezm7 payload");
    }
    std::vector<std::uint8_t> ex(cnt);
    const std::size_t got = compression_decode_buffer(
        ex.data(), ex.size(), p + 4, (std::size_t)zs, nullptr,
        COMPRESSION_ZLIB);
    if (got != cnt) { return fail("ezm7 exponents did not inflate whole"); }
    const std::uint8_t* fw = p + 4 + zs;
    for (std::uint64_t i = 0; i < cnt; ++i) {
      const std::uint16_t h = (std::uint16_t)(((fw[i] >> 7) << 15) |
                                              ((std::uint16_t)ex[i] << 10) |
                                              ((fw[i] & 0x7f) << 3));
      out[i] = f16_(reinterpret_cast<const std::uint8_t*>(&h));
    }
    return true;
  }
  return fail("unknown codec");
}

namespace {

// The elements of one row of `m`: everything past its first dimension
// (a conv's [O, I, kh, kw] is O rows of I * kh * kw), 1 for a vector.
std::int64_t
row_elems_(const Mapped& m)
{
  std::int64_t n = 1;
  for (std::size_t d = 1; d < m.shape.size(); ++d) { n *= m.shape[d]; }
  return n;
}

// The rows and columns a part lends to `m`: rows of `m`'s width, so a
// norm's [1, 1, 1, 128] is 128 rows of one and a conv's [O, I, 3, 3] O
// rows of I * 9, whatever dimensions the store wrote it with.
void
part_dims_(const Mapped& m, const Tensor& t, std::int64_t* rows,
           std::int64_t* cols)
{
  *cols = row_elems_(m);
  *rows = *cols > 0 ? t.count() / *cols : 0;
}

}  // namespace

SourceRow
source_row(const Mapped& m, const std::vector<Tensor>& ts, std::int64_t r)
{
  SourceRow s;
  if (m.rows == Mapped::Rows::kPerHead) {
    const std::int64_t P = (std::int64_t)m.parts.size(), D = m.head_dim;
    const std::int64_t h = r / (P * D);
    s.part = (int)((r / D) % P);
    s.row = h * D + r % D;
  } else {
    for (std::size_t i = 0; i < m.parts.size(); ++i) {
      const Mapped::Part& p = m.parts[i];
      std::int64_t rows = 0, cols = 0;
      part_dims_(m, ts[(std::size_t)p.index], &rows, &cols);
      const std::int64_t n = p.nrows >= 0 ? p.nrows : rows - p.row0;
      if (r < n) {
        s.part = (int)i;
        s.row = p.row0 + r;
        break;
      }
      r -= n;
    }
  }
  if (s.part >= 0 && m.parts[(std::size_t)s.part].rope && m.rot > 0) {
    // The model's row j * rot/2 + i of a head is stored at 2i + j.
    const std::int64_t D = m.head_dim, half = m.rot / 2;
    const std::int64_t h = s.row / D, d = s.row % D;
    if (d < m.rot) { s.row = h * D + 2 * (d % half) + d / half; }
  }
  return s;
}

bool
gather_f32(const Mapped& m, const std::vector<Tensor>& ts,
           const std::uint8_t* file, std::size_t file_size, float* out,
           std::string* err)
{
  std::vector<std::vector<float>> dec(m.parts.size());
  for (std::size_t i = 0; i < m.parts.size(); ++i) {
    const Tensor& t = ts[(std::size_t)m.parts[i].index];
    dec[i].resize((std::size_t)t.count());
    if (!decode_f32(t, file, file_size, dec[i].data(), err)) {
      return false;
    }
  }
  const std::int64_t rows = m.shape[0];
  const std::int64_t cols = row_elems_(m);
  for (std::int64_t r = 0; r < rows; ++r) {
    const SourceRow s = source_row(m, ts, r);
    if (s.part < 0) {
      if (err != nullptr) { *err = m.name + ": a row with no source"; }
      return false;
    }
    const std::vector<float>& src = dec[(std::size_t)s.part];
    if ((std::size_t)((s.row + 1) * cols) > src.size()) {
      if (err != nullptr) { *err = m.name + ": a row past its source"; }
      return false;
    }
    float* o = out + r * cols;
    std::memcpy(o, src.data() + s.row * cols, (std::size_t)cols * 4);
    if (m.fn == Mapped::Fn::kMinusOne) {
      for (std::int64_t c = 0; c < cols; ++c) { o[c] -= 1.0f; }
    } else if (m.fn == Mapped::Fn::kLog) {
      for (std::int64_t c = 0; c < cols; ++c) { o[c] = std::log(o[c]); }
    } else if (m.fn == Mapped::Fn::kNegLog) {
      for (std::int64_t c = 0; c < cols; ++c) { o[c] = -std::log(o[c]); }
    }
  }
  return true;
}

bool
i8x_whole(const Mapped& m, const std::vector<Tensor>& ts)
{
  if (m.shape.size() < 2 || m.fn != Mapped::Fn::kNone || m.f32 ||
      m.parts.empty()) {
    return false;
  }
  const Tensor& t0 = ts[(std::size_t)m.parts[0].index];
  for (const Mapped::Part& p : m.parts) {
    const Tensor& t = ts[(std::size_t)p.index];
    if (t.codec != kCodecI8x || !t.external || t.hadamard ||
        t.cols() != t0.cols() || t.elt != t0.elt) {
      return false;
    }
  }
  return true;
}

bool
gather_i8x(const Mapped& m, const std::vector<Tensor>& ts,
           const std::uint8_t* file, std::size_t file_size,
           std::int8_t* codes, std::uint8_t* scales, std::string* err)
{
  if (!i8x_whole(m, ts) || file == nullptr) {
    if (err != nullptr) { *err = m.name + ": not an i8x weight whole"; }
    return false;
  }
  const std::int64_t rows = m.shape[0], K = row_elems_(m);
  const int elt = ts[(std::size_t)m.parts[0].index].elt;
  for (std::int64_t r = 0; r < rows; ++r) {
    const SourceRow s = source_row(m, ts, r);
    const Tensor& t = ts[(std::size_t)m.parts[(std::size_t)s.part].index];
    if (s.part < 0 || s.row >= t.rows() ||
        t.offset + t.length > file_size) {
      if (err != nullptr) { *err = m.name + ": a row past its source"; }
      return false;
    }
    const std::uint8_t* p = file + t.offset;
    if (codes != nullptr) {
      std::memcpy(codes + r * K, p + s.row * K, (std::size_t)K);
    }
    if (scales != nullptr) {
      std::memcpy(scales + r * elt, p + i8x_scale_offset(t) + s.row * elt,
                  (std::size_t)elt);
    }
  }
  return true;
}

namespace {

// `__<group>__[t-<key>-<layer>-<0 weight | 1 bias>]`, or false. The
// key may itself carry dashes (a text model's `mlp-<L>-mlp_gate_proj`).
bool
parse_any_(const std::string& n, std::string* group, std::string* key,
           int* layer, int* leaf)
{
  if (n.size() < 8 || n.compare(0, 2, "__") != 0 || n.back() != ']') {
    return false;
  }
  const auto ge = n.find("__[t-", 2);
  if (ge == std::string::npos || ge == 2) { return false; }
  *group = n.substr(2, ge - 2);
  const std::size_t at = ge + 5;
  const std::string in = n.substr(at, n.size() - at - 1);
  const auto b = in.rfind('-');
  if (b == std::string::npos || b == 0) { return false; }
  const auto a = in.rfind('-', b - 1);
  if (a == std::string::npos || a == 0) { return false; }
  const std::string ls = in.substr(a + 1, b - a - 1), ws = in.substr(b + 1);
  if (ls.empty() || ws.size() != 1 || (ws[0] != '0' && ws[0] != '1') ||
      ls.find_first_not_of("0123456789") != std::string::npos) {
    return false;
  }
  *key = in.substr(0, a);
  *layer = std::atoi(ls.c_str());
  *leaf = ws[0] - '0';
  return true;
}

// A DiT's `__dit__[t-<key>-<layer>-<leaf>]`, or false.
bool
parse_name_(const std::string& n, std::string* key, int* layer, int* leaf)
{
  std::string group;
  return parse_any_(n, &group, key, layer, leaf) && group == "dit";
}

std::vector<std::int64_t>
shape_(const Tensor& t)
{
  if (t.rows() == 1) { return {t.cols()}; }
  return {t.rows(), t.cols()};
}

std::string
fill_(const char* pat, int layer)
{
  std::string s = pat;
  const auto at = s.find("{}");
  if (at != std::string::npos) { s.replace(at, 2, std::to_string(layer)); }
  return s;
}

// Every tensor of the store by (key, layer, leaf), for the maps that
// build one model tensor out of several.
struct Index {
  std::map<std::tuple<std::string, int, int>, int> at;
  int
  find(const std::string& key, int layer, int leaf) const
  {
    auto it = at.find({key, layer, leaf});
    return it == at.end() ? -1 : it->second;
  }
};

bool
index_(const std::vector<Tensor>& ts, Index* ix, std::string* err)
{
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string key;
    int layer = 0, leaf = 0;
    if (!parse_name_(ts[i].name, &key, &layer, &leaf)) {
      *err = ts[i].name + ": not a Draw Things DiT name";
      return false;
    }
    ix->at[{key, layer, leaf}] = (int)i;
  }
  return true;
}

// Qwen-Image-2.1 (Draw Things' QwenImage2_1.swift). Verified tensor by
// tensor against the diffusers checkpoint: the one difference beyond
// names is the text norm, stored with its zero-centred +1 folded in.
struct Rule {
  const char* key;
  const char* name;     // "{}" is the block index
};
constexpr Rule kQwenImage21[] = {
  {"x_q", "transformer_blocks.{}.attn.to_q"},
  {"x_k", "transformer_blocks.{}.attn.to_k"},
  {"x_v", "transformer_blocks.{}.attn.to_v"},
  {"x_o", "transformer_blocks.{}.attn.to_out.0"},
  {"x_norm_q", "transformer_blocks.{}.attn.norm_q"},
  {"x_norm_k", "transformer_blocks.{}.attn.norm_k"},
  {"ffn_gate_proj", "transformer_blocks.{}.img_mlp.gate_layer"},
  {"ffn_up_proj", "transformer_blocks.{}.img_mlp.proj"},
  {"ffn_down_proj", "transformer_blocks.{}.img_mlp.out"},
  {"x_embedder", "img_in"},
  {"context_embedder_0", "txt_in.in_layer"},
  {"context_embedder_1", "txt_in.out_layer"},
  {"context_norm", "txt_in.text_norm"},
  {"t_embedder_0", "time_text_embed.timestep_embedder.linear_1"},
  {"t_embedder_1", "time_text_embed.timestep_embedder.linear_2"},
  {"ada_ln_0", "norm_out.linear"},
  {"linear", "proj_out"},
};

bool
map_qwen_image21_(const std::vector<Tensor>& ts, std::vector<Mapped>* out,
                  std::string* err)
{
  // The ONE shared modulation, stored as its four row chunks.
  int mod[4] = {-1, -1, -1, -1};
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string key;
    int layer = 0, leaf = 0;
    if (!parse_name_(ts[i].name, &key, &layer, &leaf)) {
      *err = ts[i].name + ": not a Draw Things DiT name";
      return false;
    }
    if (key.size() == 10 && key.compare(0, 9, "x_ada_ln_") == 0 &&
        key[9] >= '0' && key[9] <= '3' && leaf == 0) {
      mod[key[9] - '0'] = (int)i;
      continue;
    }
    const Rule* r = nullptr;
    for (const Rule& c : kQwenImage21) {
      if (key == c.key) { r = &c; break; }
    }
    if (r == nullptr) {
      *err = ts[i].name + ": not a Qwen-Image-2.1 tensor";
      return false;
    }
    Mapped m;
    m.name = fill_(r->name, layer);
    m.leaf = leaf == 0 ? "weight" : "bias";
    m.parts = {{(int)i, false}};
    m.shape = shape_(ts[i]);
    if (key == "context_norm") { m.fn = Mapped::Fn::kMinusOne; }
    out->push_back(std::move(m));
  }
  Mapped m;
  m.name = "modulation.1";
  m.leaf = "weight";
  std::int64_t rows = 0;
  for (int c = 0; c < 4; ++c) {
    if (mod[c] < 0 || ts[(std::size_t)mod[c]].cols() !=
                          ts[(std::size_t)mod[0]].cols()) {
      *err = "the modulation's four chunks do not line up";
      return false;
    }
    m.parts.push_back({mod[c], false});
    rows += ts[(std::size_t)mod[c]].rows();
  }
  m.shape = {rows, ts[(std::size_t)mod[0]].cols()};
  out->push_back(std::move(m));
  return true;
}

// MiniMax-H3 (Draw Things' MiniMaxH3.swift), presented under MiniMax's
// OWN names -- the ones its loader reads, so a `.ckpt` loads as the
// released checkpoint does. Draw Things splits what the release fuses:
//
//   qkv_proj      q | k | v, interleaved back PER HEAD as released; q
//                 and k (and their norms) have each head's first 96
//                 rows in Draw Things' interleaved RoPE order
//   mlp.fc1       gate | up
//   adaln_proj    18 chunks adaln_<c>_<m> of [hidden, time_dim], stacked
//                 modality-major ([m][c]) as the release stores them
//   final_layer.adaln_proj   shift | scale
//
// The token refiner's attention has no rotary half.
constexpr int kH3HeadDim = 128;
constexpr int kH3Rot = 96;

bool
map_minimax_h3_(const std::vector<Tensor>& ts, std::vector<Mapped>* out,
                std::string* err)
{
  Index ix;
  if (!index_(ts, &ix, err)) { return false; }
  std::vector<bool> used(ts.size(), false);
  bool ok = true;
  // One model tensor out of named store tensors (key, layer, leaf).
  auto put = [&](const std::string& name, const std::string& leaf,
                 std::vector<std::tuple<std::string, int, int, bool>> src,
                 Mapped::Rows rows = Mapped::Rows::kConcat) {
    if (!ok) { return; }
    Mapped m;
    m.name = name;
    m.leaf = leaf;
    m.rows = rows;
    m.head_dim = kH3HeadDim;
    m.rot = kH3Rot;
    for (const auto& [key, layer, lf, rope] : src) {
      const int i = ix.find(key, layer, lf);
      if (i < 0) {
        *err = "no __dit__[t-" + key + "-" + std::to_string(layer) + "-" +
               std::to_string(lf) + "] for " + name;
        ok = false;
        return;
      }
      m.parts.push_back({i, rope});
      used[(std::size_t)i] = true;
    }
    const Tensor& t0 = ts[(std::size_t)m.parts[0].index];
    const bool vec = t0.rows() == 1;
    std::int64_t rows_n = 0;
    for (const Mapped::Part& p : m.parts) {
      const Tensor& t = ts[(std::size_t)p.index];
      if ((t.rows() == 1) != vec || (!vec && t.cols() != t0.cols())) {
        *err = name + ": its pieces do not line up";
        ok = false;
        return;
      }
      rows_n += vec ? t.count() : t.rows();
    }
    m.shape = vec ? std::vector<std::int64_t>{rows_n}
                  : std::vector<std::int64_t>{rows_n, t0.cols()};
    if (rows == Mapped::Rows::kPerHead &&
        (vec || rows_n % ((std::int64_t)m.parts.size() * kH3HeadDim) != 0)) {
      *err = name + ": not whole heads";
      ok = false;
      return;
    }
    out->push_back(std::move(m));
  };
  auto one = [&](const std::string& name, const std::string& key,
                 int layer, bool bias) {
    put(name, "weight", {{key, layer, 0, false}});
    if (bias) { put(name, "bias", {{key, layer, 1, false}}); }
  };
  int blocks = 0, refiner = 0;
  while (ix.find("q", blocks, 0) >= 0) { ++blocks; }
  while (ix.find("refiner_q", refiner, 0) >= 0) { ++refiner; }
  for (int L = 0; L < blocks; ++L) {
    const std::string p = "blocks." + std::to_string(L) + ".";
    put(p + "attn.qkv_proj", "weight",
        {{"q", L, 0, true}, {"k", L, 0, true}, {"v", L, 0, false}},
        Mapped::Rows::kPerHead);
    one(p + "attn.out_proj", "o", L, false);
    put(p + "attn.q_norm", "weight", {{"norm_q", L, 0, true}});
    put(p + "attn.k_norm", "weight", {{"norm_k", L, 0, true}});
    put(p + "mlp.fc1", "weight", {{"gate", L, 0, false}, {"up", L, 0, false}});
    one(p + "mlp.fc2", "down", L, false);
    one(p + "norm1", "norm1", L, false);
    one(p + "norm2", "norm2", L, false);
    for (int leaf = 0; leaf < 2; ++leaf) {
      std::vector<std::tuple<std::string, int, int, bool>> src;
      for (int mo = 0; mo < 3; ++mo) {
        for (int c = 0; c < 6; ++c) {
          src.push_back({"adaln_" + std::to_string(c) + "_" +
                             std::to_string(mo),
                         L, leaf, false});
        }
      }
      put(p + "adaln_proj.linear", leaf == 0 ? "weight" : "bias", src);
    }
  }
  for (int L = 0; L < refiner; ++L) {
    const std::string p = "token_refiner.blocks." + std::to_string(L) + ".";
    put(p + "attn.qkv_proj", "weight",
        {{"refiner_q", L, 0, false}, {"refiner_k", L, 0, false},
         {"refiner_v", L, 0, false}},
        Mapped::Rows::kPerHead);
    one(p + "attn.out_proj", "refiner_o", L, false);
    one(p + "attn.q_norm", "refiner_norm_q", L, false);
    one(p + "attn.k_norm", "refiner_norm_k", L, false);
    put(p + "mlp.fc1", "weight",
        {{"refiner_gate", L, 0, false}, {"refiner_up", L, 0, false}});
    one(p + "mlp.fc2", "refiner_down", L, false);
    one(p + "norm1", "refiner_norm1", L, false);
    one(p + "norm2", "refiner_norm2", L, false);
  }
  one("token_refiner.final_norm", "refiner_final_norm", 0, false);
  one("condition_proj", "context_embedder", 0, true);
  one("time_embedder.proj_in", "time_embedder_linear_1", 0, true);
  one("time_embedder.proj_out", "time_embedder_linear_2", 0, true);
  one("video_patch_proj", "proj_in", 0, true);
  one("audio_patch_proj", "audio_proj_in", 0, true);
  one("final_layer.video_out", "proj_out", 0, true);
  one("final_layer.audio_out", "audio_proj_out", 0, true);
  one("final_layer.norm", "norm_out", 0, false);
  for (int leaf = 0; leaf < 2; ++leaf) {
    put("final_layer.adaln_proj.linear", leaf == 0 ? "weight" : "bias",
        {{"norm_out_shift", 0, leaf, false},
         {"norm_out_scale", 0, leaf, false}});
  }
  if (!ok) { return false; }
  for (std::size_t i = 0; i < ts.size(); ++i) {
    if (!used[i]) {
      *err = ts[i].name + ": not a MiniMax-H3 tensor";
      return false;
    }
  }
  return blocks > 0;
}

// Every tensor of a component store by (group, key, layer, leaf). The
// component maps share it; `used` marks what a map placed, so a tensor
// nothing placed is reported rather than dropped.
struct GIndex {
  std::map<std::tuple<std::string, std::string, int, int>, int> at;
  std::vector<bool> used;
  int
  find(const std::string& g, const std::string& key, int layer,
       int leaf) const
  {
    auto it = at.find({g, key, layer, leaf});
    return it == at.end() ? -1 : it->second;
  }
};

bool
gindex_(const std::vector<Tensor>& ts, GIndex* ix, std::string* err)
{
  ix->used.assign(ts.size(), false);
  for (std::size_t i = 0; i < ts.size(); ++i) {
    std::string group, key;
    int layer = 0, leaf = 0;
    if (!parse_any_(ts[i].name, &group, &key, &layer, &leaf)) {
      *err = ts[i].name + ": not a Draw Things tensor name";
      return false;
    }
    ix->at[{group, key, layer, leaf}] = (int)i;
  }
  return true;
}

// One model tensor from store tensors; a missing part fails the map.
struct Placer {
  const std::vector<Tensor>& ts;
  GIndex& ix;
  std::vector<Mapped>* out;
  std::string* err;
  bool ok = true;

  // Each source: (group, key, layer, leaf, rope). `shape` is the model's
  // (empty: the store's, a 1-row tensor read as a vector).
  struct Src {
    std::string group, key;
    int layer = 0, leaf = 0;
    bool rope = false;
    std::int64_t row0 = 0, nrows = -1;
  };
  void
  put(const std::string& name, const std::string& leaf,
      const std::vector<Src>& src, std::vector<std::int64_t> shape = {},
      Mapped::Fn fn = Mapped::Fn::kNone, int head_dim = 0, bool f32 = false)
  {
    if (!ok) { return; }
    Mapped m;
    m.name = name;
    m.leaf = leaf;
    m.fn = fn;
    m.head_dim = head_dim;
    m.rot = head_dim;
    m.f32 = f32;
    std::int64_t count = 0;
    for (const Src& s : src) {
      const int i = ix.find(s.group, s.key, s.layer, s.leaf);
      if (i < 0) {
        *err = "no __" + s.group + "__[t-" + s.key + "-" +
               std::to_string(s.layer) + "-" + std::to_string(s.leaf) +
               "] for " + name + "." + leaf;
        ok = false;
        return;
      }
      ix.used[(std::size_t)i] = true;
      m.parts.push_back({i, s.rope, s.row0, s.nrows});
      count += ts[(std::size_t)i].count();
    }
    if (shape.empty()) {
      const Tensor& t0 = ts[(std::size_t)m.parts[0].index];
      if (m.parts.size() == 1) {
        // The store's own dimensions; a vector when at most one of them
        // is past 1 (a norm's [1, 1, 1, 128]).
        int big = 0;
        for (std::int64_t d : t0.dims) { big += d > 1 ? 1 : 0; }
        shape = big <= 1 ? std::vector<std::int64_t>{t0.count()} : t0.dims;
      } else {
        // Stacked rows of one width.
        std::int64_t rows = 0;
        for (const Mapped::Part& p : m.parts) {
          const Tensor& t = ts[(std::size_t)p.index];
          if (t.rows() > 1 && t.cols() != t0.cols()) {
            *err = name + ": its pieces do not line up";
            ok = false;
            return;
          }
          rows += t.rows() > 1 ? t.rows() : t.count();
        }
        shape = t0.rows() > 1 ? std::vector<std::int64_t>{rows, t0.cols()}
                              : std::vector<std::int64_t>{rows};
      }
    }
    std::int64_t want = 1;
    for (std::int64_t d : shape) { want *= d; }
    bool sliced = false;
    for (const Mapped::Part& p : m.parts) {
      sliced = sliced || p.row0 != 0 || p.nrows >= 0;
    }
    if (!sliced && want != count) {
      *err = name + "." + leaf + ": " + std::to_string(count) +
             " stored elements for " + std::to_string(want);
      ok = false;
      return;
    }
    m.shape = std::move(shape);
    out->push_back(std::move(m));
  }
};

bool
unplaced_(const std::vector<Tensor>& ts, const GIndex& ix, const char* what,
          std::string* err)
{
  for (std::size_t i = 0; i < ts.size(); ++i) {
    if (!ix.used[i]) {
      *err = ts[i].name + ": not a " + std::string(what) + " tensor";
      return true;
    }
  }
  return false;
}

// Qwen3-VL (Draw Things' Qwen3VL.swift), as transformers' Qwen3VLForCon-
// ditionalGeneration names it: the language model under
// `model.language_model.`, the vision tower under `model.visual.`.
// Verified tensor by tensor against Qwen-Image-2.1's text encoder (the
// 8B; f16 and i8x) and MiniMax-H3's (the 32B, kept to its 50 tapped
// layers). Draw Things stores no head and no final norm -- neither
// consumer reads them -- and puts every q / k row (and q / k norm and
// bias) of a head in interleaved RoPE order over the WHOLE head, which
// is undone here. The fused vision qkv is q | k | v stacked.
bool
map_qwen3_vl_(const std::vector<Tensor>& ts, std::vector<Mapped>* out,
              std::string* err)
{
  GIndex ix;
  if (!gindex_(ts, &ix, err)) { return false; }
  Placer pl{ts, ix, out, err};
  using Src = Placer::Src;
  const std::string T = "text_model", V = "vision_model";
  const std::string lm = "model.language_model.", vt = "model.visual.";
  // The text head width is the q norm's: one head's worth.
  const int nq = ix.find(T, "norm_q", 0, 0);
  if (nq < 0) {
    *err = "no __text_model__[t-norm_q-0-0]";
    return false;
  }
  const int D = (int)ts[(std::size_t)nq].count();
  int layers = 0;
  while (ix.find(T, "q_proj", layers, 0) >= 0) { ++layers; }
  pl.put(lm + "embed_tokens", "weight", {Src{T, "tok_embeddings", 0, 0}});
  for (int L = 0; L < layers && pl.ok; ++L) {
    const std::string p = lm + "layers." + std::to_string(L) + ".";
    pl.put(p + "input_layernorm", "weight", {Src{T, "input_layernorm", L}});
    pl.put(p + "post_attention_layernorm", "weight",
           {Src{T, "post_attention_layernorm", L}});
    pl.put(p + "self_attn.q_proj", "weight", {Src{T, "q_proj", L, 0, true}},
           {}, Mapped::Fn::kNone, D);
    pl.put(p + "self_attn.k_proj", "weight", {Src{T, "k_proj", L, 0, true}},
           {}, Mapped::Fn::kNone, D);
    pl.put(p + "self_attn.v_proj", "weight", {Src{T, "v_proj", L}});
    pl.put(p + "self_attn.o_proj", "weight", {Src{T, "out_proj", L}});
    pl.put(p + "self_attn.q_norm", "weight", {Src{T, "norm_q", L, 0, true}},
           {D}, Mapped::Fn::kNone, D);
    pl.put(p + "self_attn.k_norm", "weight", {Src{T, "norm_k", L, 0, true}},
           {D}, Mapped::Fn::kNone, D);
    // The MLP's key carries the layer in its middle:
    // `mlp-<L>-mlp_<x>_proj-0-0`.
    for (const char* x : {"gate", "up", "down"}) {
      pl.put(p + "mlp." + x + "_proj", "weight",
             {Src{T, "mlp-" + std::to_string(L) + "-mlp_" + x + "_proj", 0}});
    }
  }
  if (layers == 0 || !pl.ok) {
    if (pl.ok) { *err = "no text model layers"; }
    return false;
  }
  // The vision tower, when the store has one.
  if (ix.find(V, "conv_in", 0, 0) >= 0) {
    // A Conv3d over (C 3, T 2, P, P) patches, stored flattened [hidden,
    // C T P P] in the same element order.
    const Tensor& ci = ts[(std::size_t)ix.find(V, "conv_in", 0, 0)];
    const std::int64_t hid = ci.rows();
    const auto P = (std::int64_t)std::lround(std::sqrt(ci.cols() / 6.0));
    if (P * P * 6 != ci.cols()) {
      *err = ci.name + ": not a (3, 2, P, P) patch embedding";
      return false;
    }
    // Every Qwen3-VL tower has 16 heads.
    const int VD = (int)(hid / 16);
    pl.put(vt + "patch_embed.proj", "weight", {Src{V, "conv_in", 0, 0}},
           {hid, 3, 2, P, P});
    pl.put(vt + "patch_embed.proj", "bias", {Src{V, "conv_in", 0, 1}}, {hid});
    pl.put(vt + "pos_embed", "weight", {Src{V, "pos_embed", 0, 0}});
    int blocks = 0;
    while (ix.find(V, "q_proj", blocks, 0) >= 0) { ++blocks; }
    for (int B = 0; B < blocks && pl.ok; ++B) {
      const std::string p = vt + "blocks." + std::to_string(B) + ".";
      for (int leaf = 0; leaf < 2; ++leaf) {
        const char* lf = leaf == 0 ? "weight" : "bias";
        pl.put(p + "attn.qkv", lf,
               {Src{V, "q_proj", B, leaf, true},
                Src{V, "k_proj", B, leaf, true}, Src{V, "v_proj", B, leaf}},
               {}, Mapped::Fn::kNone, VD);
        pl.put(p + "attn.proj", lf, {Src{V, "out_proj", B, leaf}});
        pl.put(p + "mlp.linear_fc1", lf, {Src{V, "mlp_fc1", B, leaf}});
        pl.put(p + "mlp.linear_fc2", lf, {Src{V, "mlp_fc2", B, leaf}});
        pl.put(p + "norm1", lf, {Src{V, "norm1", B, leaf}});
        pl.put(p + "norm2", lf, {Src{V, "norm2", B, leaf}});
      }
    }
    for (int leaf = 0; leaf < 2; ++leaf) {
      const char* lf = leaf == 0 ? "weight" : "bias";
      pl.put(vt + "merger.norm", lf, {Src{V, "norm_out", 0, leaf}});
      pl.put(vt + "merger.linear_fc1", lf, {Src{V, "merger_mlp_0", 0, leaf}});
      pl.put(vt + "merger.linear_fc2", lf, {Src{V, "merger_mlp_1", 0, leaf}});
      for (int k = 0; ix.find(V, "deepstack_" + std::to_string(k) + "_norm",
                              0, 0) >= 0; ++k) {
        const std::string ds = "deepstack_" + std::to_string(k) + "_";
        const std::string q =
            vt + "deepstack_merger_list." + std::to_string(k) + ".";
        pl.put(q + "norm", lf, {Src{V, ds + "norm", 0, leaf}});
        pl.put(q + "linear_fc1", lf, {Src{V, ds + "mlp_0", 0, leaf}});
        pl.put(q + "linear_fc2", lf, {Src{V, ds + "mlp_1", 0, leaf}});
      }
    }
  }
  if (!pl.ok) { return false; }
  return !unplaced_(ts, ix, "Qwen3-VL", err);
}

// MiniMax-H3's audio VAE (Draw Things' MiniMaxH3 audio encoder and
// BigVGAN decoder), under the released names and in the released F32,
// which is what its loader reads. Verified tensor by tensor against
// FL2VA/audio_vae. Draw Things stores every weight-normed convolution
// FOLDED (g v / ||v|| over all but dim 0), presented as the plain
// `<conv>.weight` the loader also takes; the decoder snakes' log-space
// alpha and beta as e^alpha and 1 / (e^beta + eps), taken back to the
// logs here; the encoder snakes' alpha as is, beside a redundant
// reciprocal that nothing reads; and the head's q | 0 | v biases fused
// into one, split back.
bool
map_h3_audio_(const std::vector<Tensor>& ts, GIndex& ix,
              std::vector<Mapped>* out, std::string* err)
{
  Placer pl{ts, ix, out, err};
  using Src = Placer::Src;
  using Fn = Mapped::Fn;
  const std::string D = "audio_decoder", E = "audio_encoder";
  // A [d0, d1, 1, k] conv as the released [d0, d1, k].
  auto conv = [&](const std::string& name, const std::string& g,
                  const std::string& key, bool bias = true) {
    const int i = ix.find(g, key, 0, 0);
    if (i < 0) {
      pl.put(name, "weight", {Src{g, key, 0, 0}});  // reports it missing
      return;
    }
    const Tensor& t = ts[(std::size_t)i];
    std::vector<std::int64_t> sh;
    for (std::size_t d = 0; d < t.dims.size(); ++d) {
      if (d == 2 && t.dims.size() == 4 && t.dims[d] == 1) { continue; }
      sh.push_back(t.dims[d]);
    }
    if (sh.size() == 2) { sh.push_back(1); }       // a [O, I] 1x1 conv
    pl.put(name, "weight", {Src{g, key, 0, 0}}, sh, Fn::kNone, 0, true);
    // Its own length: a transposed conv's dim 0 is its INPUT width.
    if (bias && ix.find(g, key, 0, 1) >= 0) {
      pl.put(name, "bias", {Src{g, key, 0, 1}}, {}, Fn::kNone, 0, true);
    }
  };
  auto vec = [&](const std::string& name, const std::string& leaf,
                 const std::string& g, const std::string& key, int sleaf,
                 std::vector<std::int64_t> sh = {}, Fn fn = Fn::kNone) {
    const int i = ix.find(g, key, 0, sleaf);
    if (sh.empty() && i >= 0) { sh = {ts[(std::size_t)i].count()}; }
    pl.put(name, leaf, {Src{g, key, 0, sleaf}}, sh, fn, 0, true);
  };
  if (ix.find(D, "dec_in_proj", 0, 0) >= 0) {
    conv("dec_in_proj", D, "dec_in_proj");
    conv("decoder.conv_pre", D, "audio_conv_pre");
    conv("decoder.conv_post", D, "audio_conv_post", false);
    auto snake = [&](const std::string& base, const std::string& key) {
      vec(base + ".act", "alpha", D, key + "_snake_alpha", 0, {}, Fn::kLog);
      vec(base + ".act", "beta", D, key + "_snake_beta", 0, {}, Fn::kNegLog);
      const int u = ix.find(D, key + "_upsample", 0, 0);
      const std::int64_t taps = u >= 0 ? ts[(std::size_t)u].count() : 0;
      vec(base + ".upsample", "filter", D, key + "_upsample", 0,
          {1, 1, taps});
      vec(base + ".downsample.lowpass", "filter", D, key + "_downsample", 0,
          {1, 1, taps});
    };
    snake("decoder.activation_post", "audio_post");
    for (int i = 0; ix.find(D, "audio_up_" + std::to_string(i), 0, 0) >= 0;
         ++i) {
      conv("decoder.ups." + std::to_string(i) + ".0", D,
           "audio_up_" + std::to_string(i));
    }
    // audio_amp_<I>_c<J>_<K>: resblock I's convs<K+1>.<J>, and
    // audio_amp_<I>_a<J>_<K> the activation in front of it, (2J + K).
    for (int I = 0; ix.find(D, "audio_amp_" + std::to_string(I) + "_c0_0",
                            0, 0) >= 0; ++I) {
      const std::string a = "audio_amp_" + std::to_string(I) + "_";
      const std::string r = "decoder.resblocks." + std::to_string(I) + ".";
      for (int J = 0; ix.find(D, a + "c" + std::to_string(J) + "_0", 0, 0) >=
                      0; ++J) {
        for (int K = 0; K < 2; ++K) {
          const std::string jk = std::to_string(J) + "_" + std::to_string(K);
          conv(r + "convs" + std::to_string(K + 1) + "." + std::to_string(J),
               D, a + "c" + jk);
          snake(r + "activations." + std::to_string(2 * J + K), a + "a" + jk);
        }
      }
    }
  }
  if (ix.find(E, "encoder_input", 0, 0) >= 0) {
    conv("encoder.block.0", E, "encoder_input");
    // The encoder snakes' one parameter is linear, [1, C, 1] as released;
    // Draw Things' `_reciprocal` beside it is 1 / alpha, which the loader
    // derives itself.
    auto snake1 = [&](const std::string& name, const std::string& key) {
      const int i = ix.find(E, key + "_alpha", 0, 0);
      const std::int64_t c = i >= 0 ? ts[(std::size_t)i].count() : 0;
      vec(name, "alpha", E, key + "_alpha", 0, {1, c, 1});
      const int rc = ix.find(E, key + "_reciprocal", 0, 0);
      if (rc >= 0) { ix.used[(std::size_t)rc] = true; }
    };
    int stages = 0;
    for (; ix.find(E, "encoder_block_" + std::to_string(stages) +
                          "_downsample", 0, 0) >= 0; ++stages) {
      const std::string b = "encoder_block_" + std::to_string(stages) + "_";
      const std::string r = "encoder.block." + std::to_string(stages + 1) +
                            ".block.";
      for (int J = 0; ix.find(E, b + "res" + std::to_string(J) + "_conv1",
                              0, 0) >= 0; ++J) {
        const std::string u = b + "res" + std::to_string(J) + "_";
        const std::string q = r + std::to_string(J) + ".block.";
        snake1(q + "0", u + "snake1");
        conv(q + "1", E, u + "conv1");
        snake1(q + "2", u + "snake2");
        conv(q + "3", E, u + "conv2");
      }
      snake1(r + "3", b + "snake");
      conv(r + "4", E, b + "downsample");
    }
    snake1("encoder.block." + std::to_string(stages + 1),
           "encoder_final_snake");
    conv("encoder.block." + std::to_string(stages + 2), E,
         "encoder_final_conv");
    // The posterior head.
    for (int leaf = 0; leaf < 2; ++leaf) {
      const char* lf = leaf == 0 ? "weight" : "bias";
      vec("pre_block.norm1", lf, E, "norm1", leaf);
      vec("pre_block.norm2", lf, E, "norm2", leaf);
      vec("pre_block.norm3", lf, E, "norm3", leaf);
      vec("pre_block.mlp.norm", lf, E, "mlp_norm", leaf);
    }
    auto lin = [&](const std::string& name, const std::string& key) {
      pl.put(name, "weight", {Src{E, key, 0, 0}}, {}, Fn::kNone, 0, true);
      pl.put(name, "bias", {Src{E, key, 0, 1}}, {}, Fn::kNone, 0, true);
    };
    pl.put("pre_block.attn.qkv", "weight", {Src{E, "qkv", 0, 0}}, {},
           Fn::kNone, 0, true);
    {
      const int qb = ix.find(E, "qkv", 0, 1);
      const std::int64_t n = qb >= 0 ? ts[(std::size_t)qb].count() / 3 : 0;
      const char* leaves[3] = {"q_bias", "zero_k_bias", "v_bias"};
      for (int p = 0; p < 3; ++p) {
        Src s{E, "qkv", 0, 1};
        s.row0 = p * n;
        s.nrows = n;
        pl.put("pre_block.attn", leaves[p], {s}, {n}, Fn::kNone, 0, true);
      }
    }
    lin("pre_block.attn.proj", "attention_output");
    lin("pre_block.proj", "input_projection");
    lin("pre_block.mlp.w0", "mlp_gate");
    lin("pre_block.mlp.w1", "mlp_up");
    lin("pre_block.mlp.w2", "mlp_down");
    conv("mean_proj", E, "mean_proj");
    conv("logs_proj", E, "logs_proj");
  }
  return pl.ok;
}

// MiniMax-H3's video VAE (Draw Things' MiniMaxH3 video encoder and ViT
// decoder), under the released names. Verified element for element
// against FL2VA/video_vae (the decoder's linears were narrowed
// round-toward-zero, so they sit within 1 f16 ulp of the release, not
// on its round-to-nearest bytes). The encoder is a straight rename; the
// decoder splits what the release fuses: to_qkv grouped PER HEAD (64
// rows, the first 48 of q and k rotary and stored interleaved), and
// ff.w1 = gate | up.
constexpr int kH3VaeHeadDim = 64;
constexpr int kH3VaeRot = 48;
// The released encoder's layers_per_block; the store has no config.
constexpr int kH3VaeEncBlocks = 2;

bool
map_h3_video_(const std::vector<Tensor>& ts, GIndex& ix,
              std::vector<Mapped>* out, std::string* err)
{
  Placer pl{ts, ix, out, err};
  using Src = Placer::Src;
  const std::string D = "video_decoder", E = "video_encoder";
  auto wb = [&](const std::string& name, const std::string& g,
                const std::string& key, int layer,
                std::vector<std::int64_t> sh = {}) {
    pl.put(name, "weight", {Src{g, key, layer, 0}}, sh);
    if (ix.find(g, key, layer, 1) >= 0) {
      pl.put(name, "bias", {Src{g, key, layer, 1}});
    }
  };
  if (ix.find(E, "conv_in", 0, 0) >= 0) {
    wb("encoder.conv_in", E, "conv_in", 0);
    wb("encoder.norm_out", E, "norm_out", 0);
    wb("encoder.conv_out", E, "conv_out", 0);
    wb("quant_conv", E, "quant_conv", 0);
    // Block r is level r / 2's block r % 2; the k-th block that widens
    // (its conv1 [O, I, ...] with I != O) carries nin_shortcut-k.
    int nin = 0;
    for (int r = 0; ix.find(E, "conv1", r, 0) >= 0 && pl.ok; ++r) {
      const std::string b = "encoder.down." +
                            std::to_string(r / kH3VaeEncBlocks) + ".block." +
                            std::to_string(r % kH3VaeEncBlocks) + ".";
      for (const char* x : {"conv1", "conv2", "norm1", "norm2"}) {
        wb(b + x, E, x, r);
      }
      const Tensor& c1 = ts[(std::size_t)ix.find(E, "conv1", r, 0)];
      if (c1.dims.size() >= 2 && c1.dims[0] != c1.dims[1]) {
        wb(b + "nin_shortcut", E, "nin_shortcut", nin++);
      }
    }
    for (int k = 0; ix.find(E, "downsample_" + std::to_string(k), 0, 0) >= 0;
         ++k) {
      wb("encoder.down." + std::to_string(k) + ".downsample.conv", E,
         "downsample_" + std::to_string(k), 0);
    }
  }
  if (ix.find(D, "decoder_proj_in", 0, 0) >= 0) {
    {
      const int i = ix.find(D, "post_quant_conv", 0, 0);
      const std::int64_t z = i >= 0 ? ts[(std::size_t)i].rows() : 0;
      wb("post_quant_conv", D, "post_quant_conv", 0, {z, z, 1, 1, 1});
    }
    wb("decoder.x_embedder", D, "decoder_proj_in", 0);
    wb("decoder.proj_out", D, "decoder_proj_out", 0);
    wb("decoder.norm_out", D, "decoder_norm_out", 0);
    pl.put("decoder", "register_tokens", {Src{D, "register_tokens", 0, 0}});
    for (int i = 0; ix.find(D, "to_q", i, 0) >= 0 && pl.ok; ++i) {
      const std::string b = "decoder.transformer_blocks." +
                            std::to_string(i);
      pl.put(b + ".norm1", "weight", {Src{D, "norm1", i, 0}});
      pl.put(b + ".norm2", "weight", {Src{D, "norm2", i, 0}});
      pl.put(b, "scale1", {Src{D, "scale1", i, 0}});
      pl.put(b, "scale2", {Src{D, "scale2", i, 0}});
      for (int leaf = 0; leaf < 2; ++leaf) {
        const char* lf = leaf == 0 ? "weight" : "bias";
        pl.put(b + ".attn.to_qkv", lf,
               {Src{D, "to_q", i, leaf, true}, Src{D, "to_k", i, leaf, true},
                Src{D, "to_v", i, leaf}});
        if (pl.ok) {
          Mapped& m = out->back();
          m.rows = Mapped::Rows::kPerHead;
          m.head_dim = kH3VaeHeadDim;
          m.rot = kH3VaeRot;
        }
        pl.put(b + ".attn.to_out", lf, {Src{D, "to_out", i, leaf}});
        pl.put(b + ".ff.w1", lf,
               {Src{D, "ff_gate", i, leaf}, Src{D, "ff_up", i, leaf}});
        pl.put(b + ".ff.w2", lf, {Src{D, "ff_down", i, leaf}});
      }
    }
  }
  return pl.ok;
}

// Qwen-Image-2.1's VAE (AutoencoderKLQwenImage21; Draw Things'
// QwenImage2_1 VAE), under the diffusers names. Draw Things' name is the
// diffusers one with every '.' an '_', in group `encoder` or `decoder`
// (quant_conv rides the encoder's, post_quant_conv the decoder's); the
// leaf of a norm is its `gamma`, stored [1, C, 1, 1]. Nothing else
// differs: no time_conv (an image never runs it), the convs [O, I, kh,
// kw] as released, to_qkv fused q | k | v as released. Verified element
// for element -- against the checkpoint Draw Things actually ships,
// which is madebyollin's texture-fix fine-tune (the same model class
// and config; its weights scaled by powers of two that its RMS norms
// cancel, and its own decoder), not Qwen's release.
bool
map_qwen_image21_vae_(const std::vector<Tensor>& ts,
                      std::vector<Mapped>* out, std::string* err)
{
  GIndex ix;
  if (!gindex_(ts, &ix, err)) { return false; }
  Placer pl{ts, ix, out, err};
  using Src = Placer::Src;
  // The diffusers tokens that hold an '_' of their own.
  static const char* const kJoined[] = {
      "conv_in",   "conv_out",  "norm_out",      "down_blocks", "up_blocks",
      "mid_block", "to_qkv",    "conv_shortcut", "quant_conv",
      "post_quant_conv",
  };
  static const char* const kWords[] = {
      "encoder", "decoder", "resnets",  "attentions", "conv1",  "conv2",
      "norm1",   "norm2",   "norm",     "proj",       "downsampler",
      "upsampler", "resample",
  };
  for (const auto& [k, i] : ix.at) {
    const auto& [g, key, layer, leaf] = k;
    if (!pl.ok) { break; }
    if ((g != "encoder" && g != "decoder") || layer != 0) {
      *err = ts[(std::size_t)i].name + ": not a Qwen-Image-2.1 VAE tensor";
      return false;
    }
    // Rejoin the key's '_'-split words into diffusers' dotted name.
    std::vector<std::string> w;
    for (std::size_t a = 0; a <= key.size();) {
      const std::size_t b = std::min(key.find('_', a), key.size());
      w.push_back(key.substr(a, b - a));
      a = b + 1;
    }
    std::string name;
    for (std::size_t j = 0; j < w.size();) {
      // The longest joined token that starts here (post_quant_conv over
      // quant_conv), else the word alone.
      std::size_t take = 1;
      std::string tok = w[j];
      for (const char* jn : kJoined) {
        const std::size_t n =
            1 + (std::size_t)std::count(jn, jn + std::strlen(jn), '_');
        if (n <= take || j + n > w.size()) { continue; }
        std::string t = w[j];
        for (std::size_t q = 1; q < n; ++q) { t += "_" + w[j + q]; }
        if (t == jn) {
          take = n;
          tok = t;
        }
      }
      const bool known =
          take > 1 ||
          tok.find_first_not_of("0123456789") == std::string::npos ||
          std::any_of(std::begin(kWords), std::end(kWords),
                      [&](const char* x) { return tok == x; }) ||
          std::any_of(std::begin(kJoined), std::end(kJoined),
                      [&](const char* x) { return tok == x; });
      if (!known) {
        *err = ts[(std::size_t)i].name + ": '" + tok +
               "' is not a Qwen-Image-2.1 VAE name";
        return false;
      }
      name += (name.empty() ? "" : ".") + tok;
      j += take;
    }
    const Tensor& t = ts[(std::size_t)i];
    const std::string last = name.substr(name.rfind('.') + 1);
    const bool norm = last == "norm1" || last == "norm2" ||
                      last == "norm_out" || last == "norm";
    if (norm && leaf == 0) {
      // [C, 1, 1, 1], or [C, 1, 1] for the attention's.
      std::vector<std::int64_t> sh{t.count(), 1, 1};
      if (last != "norm") { sh.push_back(1); }
      pl.put(name, "gamma", {Src{g, key, 0, 0}}, sh);
    } else {
      pl.put(name, leaf == 0 ? "weight" : "bias", {Src{g, key, 0, leaf}});
    }
  }
  if (!pl.ok) { return false; }
  return !unplaced_(ts, ix, "Qwen-Image-2.1 VAE", err);
}

// MiniMax-H3's VAEs, video and audio: Draw Things ships both in one
// store, and their released names never collide, so one set presents
// both and each loader finds its own.
bool
map_minimax_h3_vae_(const std::vector<Tensor>& ts, std::vector<Mapped>* out,
                    std::string* err)
{
  GIndex ix;
  if (!gindex_(ts, &ix, err)) { return false; }
  if (!map_h3_audio_(ts, ix, out, err)) { return false; }
  if (!map_h3_video_(ts, ix, out, err)) { return false; }
  return !unplaced_(ts, ix, "MiniMax-H3 VAE", err);
}

}  // namespace

bool
map_to_diffusers(const std::vector<Tensor>& ts, std::vector<Mapped>* out,
                 std::string* family, std::string* err)
{
  out->clear();
  family->clear();
  auto has = [&](const char* n) {
    return std::any_of(ts.begin(), ts.end(),
                       [&](const Tensor& t) { return t.name == n; });
  };
  std::string e;
  bool ok = false;
  // A component store is told by its groups (a DiT's is `dit` alone).
  std::set<std::string> groups;
  for (const Tensor& t : ts) {
    std::string g, key;
    int layer = 0, leaf = 0;
    if (parse_any_(t.name, &g, &key, &layer, &leaf)) { groups.insert(g); }
  }
  auto any = [&](std::initializer_list<const char*> gs) {
    for (const char* g : gs) {
      if (groups.count(g) != 0) { return true; }
    }
    return false;
  };
  if (any({"text_model"})) {
    ok = map_qwen3_vl_(ts, out, &e);
    if (ok) { *family = "qwen3-vl"; }
  } else if (any({"audio_decoder", "audio_encoder", "video_decoder",
                  "video_encoder"})) {
    ok = map_minimax_h3_vae_(ts, out, &e);
    if (ok) { *family = "minimax-h3-vae"; }
  } else if (any({"encoder", "decoder"}) &&
             (has("__decoder__[t-decoder_conv_in-0-0]") ||
              has("__encoder__[t-encoder_conv_in-0-0]"))) {
    ok = map_qwen_image21_vae_(ts, out, &e);
    if (ok) { *family = "qwen-image-21-vae"; }
  } else if (has("__dit__[t-x_ada_ln_0-0-0]") &&
             has("__dit__[t-context_norm-0-0]") &&
      has("__dit__[t-ffn_gate_proj-0-0]")) {
    ok = map_qwen_image21_(ts, out, &e);
    if (ok) { *family = "qwen-image-21"; }
  } else if (has("__dit__[t-adaln_0_0-0-0]") &&
             has("__dit__[t-refiner_q-0-0]") &&
             has("__dit__[t-context_embedder-0-0]")) {
    ok = map_minimax_h3_(ts, out, &e);
    if (ok) { *family = "minimax-h3"; }
  } else {
    e = "not a model this reader knows (Qwen-Image-2.1 and MiniMax-H3, "
        "their VAEs, and Qwen3-VL are)";
  }
  if (ok) { return true; }
  out->clear();
  if (err != nullptr) { *err = e; }
  return false;
}

}  // namespace vpipe::genai::drawthings
