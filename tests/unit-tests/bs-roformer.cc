// BS-RoFormer (generative-models/bs-roformer/) against MSST's reference,
// and the three shared pieces it stands on: the MSST YAML reader, the
// complex STFT and the streamed overlap-add.
//
// Goldens come from tools/dump_bs_roformer_golden.py, MSST's own network
// in fp32:
//
//   VPIPE_BS_ROFORMER_TINY_GOLDEN  --tiny: a random-weight model saved
//       as a torch .ckpt + MSST YAML beside its outputs. Exercises the
//       load path, every layout, two stems, and both demix paths in a
//       few seconds.
//   VPIPE_BS_ROFORMER_GOLDEN       a real checkpoint (meta.json names
//       it) at its own chunk size -- 1536 frames for the Resurrection
//       vocals model, so the GEMMs clear matmul2d's 64-row floor and the
//       attention runs its real lengths -- plus a 40 s demix.
//
// What the pieces check that a model run would not:
//   * the YAML reader takes PyYAML's own spellings (an indentless
//     sequence, !!python/tuple) and MSST's hand-written ones, and
//     refuses anchors and unknown tags rather than misreading them;
//   * the config REFUSES the variants this port does not implement --
//     each changes the forward, and a load that read past one would run
//     a different network over the same tensors;
//   * the stream: an identity separator must give the input back
//     exactly through the padding, the fades and the stitch, and the
//     output must not depend on how the input was cut into beats.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/perf-flops.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-separate-stage.h"
#include "stages/model-detect.h"
#include "generative-models/bs-roformer/bs-roformer-config.h"
#include "generative-models/bs-roformer/metal-bs-roformer.h"
#include "generative-models/shared/overlap-demix.h"
#include "generative-models/shared/stft.h"
#include "generative-models/shared/yaml-subset.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { return {}; }
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> v((std::size_t)n / 4);
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

FlexData
read_meta_(const std::string& dir)
{
  std::ifstream f(dir + "/meta.json");
  if (!f) { return FlexData::make_null(); }
  std::stringstream ss;
  ss << f.rdbuf();
  return FlexData::from_json(ss.str());
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1e9; }
  return rel_l2_(a.data(), b.data(), a.size());
}

const char* const kMsstYaml = R"(audio:
  chunk_size: 4032
  dim_f: 1024
  dim_t: 1536 # don't work (use in model)
  num_channels: 2
  sample_rate: 44100
  min_mean_abs: 0.000

model:
  dim: 64
  depth: 2
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  linear_transformer_depth: 0
  freqs_per_bands: !!python/tuple
    - 8
    - 8
    - 16
    - 32
    - 65
  dim_head: 64
  heads: 2
  attn_dropout: 0.
  flash_attn: true
  stft_n_fft: 256
  stft_hop_length: 64
  stft_win_length: 256
  stft_normalized: false
  mask_estimator_depth: 2
  mlp_expansion_factor: 2
  skip_connection: False # Enable skip connection

training:
  instruments: ['vocals', "other"]
  target_instrument: vocals
  lr: 1.0e-5
  augmentation_type: simple1

inference:
  batch_size: 2
  num_overlap: 4
  normalize: false
)";

}  // namespace

// ---------------------------------------------------------------------
// The YAML subset.

TEST(bs_roformer, yaml_subset_reads_msst_and_pyyaml_spellings)
{
  std::string err;
  const FlexData d = yaml_subset::parse(kMsstYaml, &err);
  ASSERT_TRUE(d.is_object());
  if (!d.is_object()) {
    std::fprintf(stderr, "  %s\n", err.c_str());
    return;
  }
  const FlexData m = d.as_object().at("model");
  const FlexData fpb = m.as_object().at("freqs_per_bands");
  ASSERT_TRUE(fpb.is_array() && fpb.as_array().size() == 5);
  EXPECT_TRUE(fpb.as_array().at(4).as_int() == 65);
  EXPECT_TRUE(m.as_object().at("attn_dropout").is_real());
  EXPECT_TRUE(m.as_object().at("skip_connection").is_bool() &&
              !m.as_object().at("skip_connection").as_bool());
  const FlexData a = d.as_object().at("audio");
  // The comment after a value with an apostrophe in it is a comment.
  EXPECT_TRUE(a.as_object().at("dim_t").as_int() == 1536);
  EXPECT_TRUE(a.as_object().at("min_mean_abs").is_real());
  const FlexData t = d.as_object().at("training");
  const FlexData ins = t.as_object().at("instruments");
  ASSERT_TRUE(ins.is_array() && ins.as_array().size() == 2);
  EXPECT_TRUE(ins.as_array().at(1).as_string() == "other");
  EXPECT_TRUE(std::fabs(t.as_object().at("lr").as_real() - 1e-5) < 1e-12);
  EXPECT_TRUE(t.as_object().at("augmentation_type").as_string() ==
              "simple1");

  // PyYAML's own spelling: the sequence at its key's indentation, a
  // null, a mapping inside a sequence item.
  const FlexData p = yaml_subset::parse(
      "a:\n- 1\n- 2\nb: null\nc:\n  - x: 1\n    y: [2, 3]\n  - z\n", &err);
  ASSERT_TRUE(p.is_object());
  if (p.is_object()) {
    EXPECT_TRUE(p.as_object().at("a").as_array().size() == 2);
    EXPECT_TRUE(p.as_object().at("b").is_null());
    const FlexData c = p.as_object().at("c");
    ASSERT_TRUE(c.is_array() && c.as_array().size() == 2);
    EXPECT_TRUE(c.as_array().at(0).as_object().at("y").as_array().size() ==
                2);
    EXPECT_TRUE(c.as_array().at(1).as_string() == "z");
  }

  // Refused, each by name.
  for (const char* bad : {"a: &x 1\nb: *x\n", "a: |\n  text\n",
                          "a: !!python/object:os.system x\n",
                          "a: [1, 2\n", "a: 1\na: 2\n"}) {
    std::string e;
    const FlexData v = yaml_subset::parse(bad, &e);
    EXPECT_TRUE(v.is_null());
    EXPECT_TRUE(!e.empty());
  }
}

TEST(bs_roformer, config_reads_msst_and_refuses_other_networks)
{
  std::string err;
  const FlexData d = yaml_subset::parse(kMsstYaml, &err);
  BsRoformerConfig c;
  ASSERT_TRUE(bs_roformer_config_from_yaml(d, &c, &err));
  EXPECT_TRUE(c.dim == 64 && c.depth == 2 && c.stereo && c.heads == 2);
  EXPECT_TRUE(c.bands() == 5 && c.bins() == 129);
  EXPECT_TRUE(c.frame_features() == 129 * 2 * 2);
  EXPECT_TRUE(c.frames() == 4032 / 64 + 1);
  EXPECT_TRUE(c.num_overlap == 4 && c.chunk_size == 4032);
  EXPECT_TRUE(c.stems.size() == 1 && c.stems[0] == "vocals");
  EXPECT_TRUE(c.zero_dc);
  EXPECT_TRUE(looks_like_bs_roformer_yaml(d));

  auto with = [&](const std::string& from, const std::string& to) {
    std::string y = kMsstYaml;
    const std::size_t at = y.find(from);
    if (at != std::string::npos) { y.replace(at, from.size(), to); }
    return yaml_subset::parse(y, &err);
  };
  for (const auto& [from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"linear_transformer_depth: 0", "linear_transformer_depth: 1"},
           {"skip_connection: False", "use_pope: true"},
           {"skip_connection: False", "num_residual_streams: 4"},
           {"skip_connection: False", "use_value_residual_learning: true"},
           {"    - 65", "    - 64"},
           {"freqs_per_bands", "num_bands: 60\n  bands_unused"}}) {
    std::string e;
    BsRoformerConfig x;
    EXPECT_TRUE(!bs_roformer_config_from_yaml(with(from, to), &x, &e));
    EXPECT_TRUE(!e.empty());
  }
  EXPECT_TRUE(!looks_like_bs_roformer_yaml(
      with("freqs_per_bands", "num_bands: 60\n  bands_unused")));
}

// ---------------------------------------------------------------------
// The STFT, against torch.stft / torch.istft.

TEST(bs_roformer, stft_matches_torch)
{
  const char* gd = std::getenv("VPIPE_BS_ROFORMER_TINY_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string g = gd;
  const std::vector<float> chunk = read_f32_(g + "/chunk.f32");
  const std::vector<float> ref = read_f32_(g + "/stft.f32");
  const std::vector<float> rt = read_f32_(g + "/istft_roundtrip.f32");
  ASSERT_TRUE(!chunk.empty() && !ref.empty() && !rt.empty());
  if (chunk.empty() || ref.empty() || rt.empty()) { return; }
  stft::Params p;
  p.n_fft = 256;
  p.hop = 64;
  const stft::Stft st(p);
  const int C = (int)chunk.size() / 2, F = st.bins(), T = st.frames(C);
  ASSERT_TRUE(ref.size() == (std::size_t)2 * F * T * 2);
  // The golden is torch's [channel, bin, frame, re/im].
  std::vector<float> got(ref.size());
  for (int ch = 0; ch < 2; ++ch) {
    ASSERT_TRUE(st.forward(chunk.data() + (std::size_t)ch * C, C,
                           got.data() + (std::size_t)ch * F * T * 2,
                           stft::Layout{2, (std::ptrdiff_t)T * 2}));
  }
  const double e = rel_l2_(got, ref);
  std::printf("  stft rel %.3g\n", e);
  EXPECT_TRUE(e < 1e-6);
  std::vector<float> back((std::size_t)2 * C);
  for (int ch = 0; ch < 2; ++ch) {
    ASSERT_TRUE(st.inverse(got.data() + (std::size_t)ch * F * T * 2,
                           stft::Layout{2, (std::ptrdiff_t)T * 2}, T, C,
                           back.data() + (std::size_t)ch * C));
  }
  const double r = rel_l2_(back, rt);
  const double id = rel_l2_(back, chunk);
  std::printf("  istft rel to torch %.3g, to the input %.3g\n", r, id);
  EXPECT_TRUE(r < 1e-6);
  EXPECT_TRUE(id < 1e-6);
}

// ---------------------------------------------------------------------
// The stream.

namespace {

// Feed `x` (planar [ch][n]) in pieces of `piece` samples, collect the
// output. `fwd` is the separator.
std::vector<float>
demix_(OverlapDemix::Params p, const std::vector<float>& x, int n,
       std::int64_t piece, const OverlapDemix::Forward& fwd,
       std::vector<float>* mix_out = nullptr, std::int64_t* chunks = nullptr)
{
  OverlapDemix d(p);
  std::vector<std::vector<float>> parts, mparts;
  std::vector<float> s, m;
  auto drain = [&]() {
    if (d.take(&s, &m) > 0) {
      parts.push_back(s);
      mparts.push_back(m);
    }
  };
  for (std::int64_t at = 0; at < n; at += piece) {
    const std::int64_t k = std::min<std::int64_t>(piece, n - at);
    std::vector<float> pl((std::size_t)p.channels * k);
    for (int c = 0; c < p.channels; ++c) {
      std::copy(x.begin() + (std::ptrdiff_t)c * n + at,
                x.begin() + (std::ptrdiff_t)c * n + at + k,
                pl.begin() + (std::ptrdiff_t)c * k);
    }
    if (!d.push(pl.data(), k, k, fwd)) { return {}; }
    drain();
  }
  if (!d.finish(fwd)) { return {}; }
  drain();
  // Reassemble planar [stems][ch][n] from the pieces.
  const std::size_t sc = (std::size_t)p.stems * p.channels;
  std::vector<float> out(sc * (std::size_t)n);
  std::vector<float> mix((std::size_t)p.channels * n);
  std::size_t at = 0;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const std::size_t len = parts[i].size() / sc;
    for (std::size_t k = 0; k < sc; ++k) {
      std::copy(parts[i].begin() + (std::ptrdiff_t)(k * len),
                parts[i].begin() + (std::ptrdiff_t)((k + 1) * len),
                out.begin() + (std::ptrdiff_t)(k * n + at));
    }
    for (int c = 0; c < p.channels; ++c) {
      std::copy(mparts[i].begin() + (std::ptrdiff_t)(c * len),
                mparts[i].begin() + (std::ptrdiff_t)((c + 1) * len),
                mix.begin() + (std::ptrdiff_t)(c * n + at));
    }
    at += len;
  }
  if (at != (std::size_t)n) { return {}; }
  if (mix_out != nullptr) { *mix_out = mix; }
  if (chunks != nullptr) { *chunks = d.chunks(); }
  return out;
}

}  // namespace

TEST(bs_roformer, overlap_demix_is_exact_and_cut_independent)
{
  // An identity separator, twice (stem 1 negated): the stitch must give
  // the input back through the padding, every fade and the division.
  const int C = 1000, ch = 2;
  auto ident = [&](const float* in, float* out) {
    for (int k = 0; k < ch * C; ++k) {
      out[k] = in[k];
      out[ch * C + k] = -in[k];
    }
    return true;
  };
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 0.3f);
  for (const int overlap : {1, 2, 4}) {
    for (const int n : {300, 600, 1000, 1499, 2400, 5003}) {
      std::vector<float> x((std::size_t)ch * n);
      for (float& v : x) { v = nd(rng); }
      OverlapDemix::Params p{C, overlap, ch, 2};
      std::vector<float> mix;
      const std::vector<float> whole = demix_(p, x, n, n, ident, &mix);
      ASSERT_TRUE(whole.size() == (std::size_t)2 * ch * n);
      if (whole.empty()) { continue; }
      double worst = 0.0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        worst = std::max(worst, (double)std::fabs(whole[i] - x[i]));
        worst = std::max(worst, (double)std::fabs(whole[x.size() + i] +
                                                  x[i]));
      }
      // At one chunk per step a window's zero end is the ONLY weight on
      // its sample, which MSST's rules then output as 0; with overlap
      // every sample has a positive weight and the input comes back.
      if (overlap >= 2) {
        EXPECT_TRUE(worst < 1e-5);
        if (worst >= 1e-5) {
          std::printf("  overlap %d n %d: worst %.3g\n", overlap, n, worst);
        }
      }
      EXPECT_TRUE(mix == x);
      // The same signal in ragged pieces: the same bytes.
      for (const std::int64_t piece : {1, 97, 333, 1000, 1001}) {
        const std::vector<float> cut = demix_(p, x, n, piece, ident);
        EXPECT_TRUE(cut == whole);
      }
    }
  }
}

// ---------------------------------------------------------------------
// The model.

namespace {

struct Loaded {
  std::shared_ptr<WeightSet> ws;
  std::unique_ptr<MetalBsRoformer> model;
};

Loaded
load_(MetalCompute* mc, const std::string& ckpt, const std::string& yaml)
{
  Loaded l;
  BsRoformerConfig cfg;
  std::string err;
  if (!bs_roformer_config_from_file(yaml, &cfg, &err)) {
    std::fprintf(stderr, "  config: %s\n", err.c_str());
    return l;
  }
  l.ws = WeightSet::open(ckpt, nullptr);
  if (l.ws == nullptr) {
    std::fprintf(stderr, "  cannot open %s\n", ckpt.c_str());
    return l;
  }
  l.model = MetalBsRoformer::load(l.ws, mc, cfg, &err);
  if (l.model == nullptr) { std::fprintf(stderr, "  load: %s\n", err.c_str()); }
  return l;
}

// One chunk against the golden's intermediates: whether every tap is
// within `tap_tol`, and the output's rel-L2 (1e9 on failure).
struct ChunkCheck {
  bool   taps_ok = false;
  double out = 1e9;
};

ChunkCheck
check_chunk_(MetalBsRoformer& m, const std::string& g, double tap_tol)
{
  ChunkCheck r0;
  const BsRoformerConfig& c = m.config();
  const std::vector<float> chunk = read_f32_(g + "/chunk.f32");
  const std::vector<float> ref = read_f32_(g + "/chunk_out.f32");
  if (chunk.size() != (std::size_t)c.channels() * c.chunk_size) {
    return r0;
  }
  std::vector<float> out((std::size_t)c.num_stems * c.channels() *
                         c.chunk_size);
  MetalBsRoformer::Taps taps;
  std::string err;
  if (!m.forward(chunk.data(), out.data(), &err, &taps)) {
    std::fprintf(stderr, "  forward: %s\n", err.c_str());
    return r0;
  }
  // The golden's spectrum is torch's [channel, bin, frame, re/im]; the
  // model's is [frame, (bin, channel, re/im)].
  const std::vector<float> rs = read_f32_(g + "/stft.f32");
  const int T = c.frames(), F = c.bins(), S = c.channels();
  std::vector<float> spec_ref(taps.spec.size());
  for (int s = 0; s < S; ++s) {
    for (int f = 0; f < F; ++f) {
      for (int t = 0; t < T; ++t) {
        for (int z = 0; z < 2; ++z) {
          spec_ref[(std::size_t)t * c.frame_features() + (f * S + s) * 2 +
                   z] = rs[(((std::size_t)s * F + f) * T + t) * 2 + z];
        }
      }
    }
  }
  struct Row { const char* name; const std::vector<float>* got; double tol; };
  const Row rows[] = {
      {"stft", &taps.spec, 1e-5},
      {"band_split", &taps.band_split, tap_tol},
      {"layer0_time", &taps.layer0_time, tap_tol},
      {"layer0_freq", &taps.layer0_freq, tap_tol},
      {"final", &taps.final_norm, tap_tol},
      {"mask", &taps.mask, tap_tol},
  };
  bool ok = true;
  for (const Row& r : rows) {
    const std::vector<float> want = std::string(r.name) == "stft"
                                        ? spec_ref
                                        : read_f32_(g + "/" + r.name + ".f32");
    const double e = rel_l2_(*r.got, want);
    std::printf("  %-12s rel %.3g\n", r.name, e);
    ok = ok && e < r.tol;
  }
  r0.taps_ok = ok;
  r0.out = rel_l2_(out, ref);
  std::printf("  %-12s rel %.3g\n", "chunk_out", r0.out);
  return r0;
}

}  // namespace

TEST(bs_roformer, tiny_model_matches_reference)
{
  const char* gd = std::getenv("VPIPE_BS_ROFORMER_TINY_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string g = gd;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Loaded l = load_(mc, g + "/tiny.ckpt", g + "/tiny.yaml");
  ASSERT_TRUE(l.model != nullptr);
  if (l.model == nullptr) { return; }
  MetalBsRoformer& m = *l.model;
  const ChunkCheck cc = check_chunk_(m, g, 5e-3);
  EXPECT_TRUE(cc.taps_ok);
  EXPECT_TRUE(cc.out < 5e-3);

  // The whole clip through the stream, then a clip shorter than two
  // borders (demix's other path: no edge padding).
  const BsRoformerConfig& c = m.config();
  auto fwd = [&](const float* in, float* out) {
    std::string err;
    return m.forward(in, out, &err);
  };
  OverlapDemix::Params p{c.chunk_size, c.num_overlap, c.channels(),
                         c.num_stems};
  for (const char* which : {"demix", "demix_short"}) {
    const std::vector<float> ref = read_f32_(g + "/" + which + ".f32");
    const std::vector<float> mix = read_f32_(g + "/mix.f32");
    const int n = (int)(ref.size() / ((std::size_t)c.num_stems * 2));
    std::vector<float> x((std::size_t)2 * n);
    for (int ch = 0; ch < 2; ++ch) {
      std::copy(mix.begin() + (std::ptrdiff_t)ch * (mix.size() / 2),
                mix.begin() + (std::ptrdiff_t)ch * (mix.size() / 2) + n,
                x.begin() + (std::ptrdiff_t)ch * n);
    }
    std::int64_t chunks = 0;
    const std::vector<float> got =
        demix_(p, x, n, 1500, fwd, nullptr, &chunks);
    const double e = rel_l2_(got, ref);
    std::printf("  %-12s rel %.3g (%lld chunks)\n", which, e,
                (long long)chunks);
    EXPECT_TRUE(e < 5e-3);
  }
}

TEST(bs_roformer, chunk_matches_reference)
{
  const char* gd = std::getenv("VPIPE_BS_ROFORMER_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string g = gd;
  const FlexData meta = read_meta_(g);
  ASSERT_TRUE(meta.is_object());
  if (!meta.is_object()) { return; }
  const std::string ckpt(meta.as_object().at("ckpt").as_string());
  const std::string yaml(meta.as_object().at("config").as_string());
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Loaded l = load_(mc, ckpt, yaml);
  ASSERT_TRUE(l.model != nullptr);
  if (l.model == nullptr) { return; }
  MetalBsRoformer& m = *l.model;
  // The matrix-core routes ENGAGE where the GPU has them: an unwired
  // tier passes every numeric bar.
  EXPECT_TRUE(m.matrix_core_gemm() == (mc->supports_matrix_cores() &&
              std::getenv("VPIPE_BS_ROFORMER_NO_MMA2") == nullptr));
  EXPECT_TRUE(m.nax_attention() == (mc->supports_matrix_cores() &&
              std::getenv("VPIPE_BS_ROFORMER_NO_NAX_ATTN") == nullptr));
  const double half = meta.as_object().at("ref_half").as_object()
                          .at("rel").as_real();
  std::printf("  the reference's own half-precision error: %.3g\n", half);
  const ChunkCheck cc = check_chunk_(m, g, 2e-2);
  EXPECT_TRUE(cc.taps_ok);
  EXPECT_TRUE(cc.out < 1e-2);
  const auto& t = m.last_timing();
  std::printf("  timing: stft %.1f ms, gpu %.1f ms, istft %.1f ms\n",
              t.stft_ms, t.gpu_ms, t.istft_ms);
}

TEST(bs_roformer, demix_matches_reference)
{
  const char* gd = std::getenv("VPIPE_BS_ROFORMER_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string g = gd;
  const FlexData meta = read_meta_(g);
  if (!meta.is_object()) { return; }
  const std::string ckpt(meta.as_object().at("ckpt").as_string());
  const std::string yaml(meta.as_object().at("config").as_string());
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Loaded l = load_(mc, ckpt, yaml);
  ASSERT_TRUE(l.model != nullptr);
  if (l.model == nullptr) { return; }
  MetalBsRoformer& m = *l.model;
  const BsRoformerConfig& c = m.config();
  const std::vector<float> mix = read_f32_(g + "/mix.f32");
  const std::vector<float> ref = read_f32_(g + "/demix.f32");
  const int n = (int)(mix.size() / 2);
  ASSERT_TRUE(ref.size() == (std::size_t)c.num_stems * 2 * n);
  auto fwd = [&](const float* in, float* out) {
    std::string err;
    const bool ok = m.forward(in, out, &err);
    if (!ok) { std::fprintf(stderr, "  forward: %s\n", err.c_str()); }
    return ok;
  };
  OverlapDemix::Params p{c.chunk_size, c.num_overlap, c.channels(),
                         c.num_stems};
  const auto t0 = std::chrono::steady_clock::now();
  std::int64_t chunks = 0;
  const std::vector<float> got = demix_(p, mix, n, n, fwd, nullptr, &chunks);
  const double s = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
  const double e = rel_l2_(got, ref);
  // Signal-to-distortion of ours against the reference's estimate.
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size() && i < ref.size(); ++i) {
    num += (double)ref[i] * ref[i];
    den += ((double)got[i] - ref[i]) * ((double)got[i] - ref[i]);
  }
  std::printf("  demix %.1f s of audio in %.2f s (%lld chunks): rel %.3g, "
              "SDR vs reference %.1f dB\n", (double)n / c.sample_rate, s,
              (long long)chunks, e, 10.0 * std::log10(num / den));
  EXPECT_TRUE(e < 1e-2);
  // Cut into one-second beats: the same bytes.
  const std::vector<float> cut =
      demix_(p, mix, n, c.sample_rate, fwd, nullptr, nullptr);
  EXPECT_TRUE(cut == got);
}

TEST(bs_roformer, counts_its_flops)
{
  const char* gd = std::getenv("VPIPE_BS_ROFORMER_TINY_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string g = gd;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Loaded l = load_(mc, g + "/tiny.ckpt", g + "/tiny.yaml");
  ASSERT_TRUE(l.model != nullptr);
  if (l.model == nullptr) { return; }
  MetalBsRoformer& m = *l.model;
  const BsRoformerConfig& c = m.config();
  std::vector<float> in((std::size_t)c.channels() * c.chunk_size, 0.01f);
  std::vector<float> out((std::size_t)c.num_stems * c.channels() *
                         c.chunk_size);
  namespace pf = vpipe::perf_flops;
  const std::uint64_t f0 = m.flops().flushed();
  const std::uint64_t g0 = pf::available() ? pf::totals(pf::kGpu).flops : 0;
  std::string err;
  ASSERT_TRUE(m.forward(in.data(), out.data(), &err));
  const std::uint64_t f = m.flops().flushed() - f0;
  const std::uint64_t want = MetalBsRoformer::forward_flops(c);
  EXPECT_TRUE(m.flops().pending() == 0);
  EXPECT_TRUE(f == want);
  if (pf::available()) {
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0 < f; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_TRUE(pf::totals(pf::kGpu).flops - g0 == f);
  }
  // The real geometry, derived: what a Resurrection vocals chunk counts.
  BsRoformerConfig r = c;
  r.dim = 256; r.depth = 12; r.heads = 8; r.time_depth = 1;
  r.freq_depth = 1; r.num_stems = 1; r.n_fft = 2048; r.hop = 512;
  r.win_length = 2048; r.chunk_size = 785920; r.mlp_expansion = 4;
  r.freqs_per_bands.assign(24, 2);
  r.freqs_per_bands.insert(r.freqs_per_bands.end(), 12, 4);
  r.freqs_per_bands.insert(r.freqs_per_bands.end(), 8, 12);
  r.freqs_per_bands.insert(r.freqs_per_bands.end(), 8, 24);
  r.freqs_per_bands.insert(r.freqs_per_bands.end(), 8, 48);
  r.freqs_per_bands.push_back(128);
  r.freqs_per_bands.push_back(129);
  std::printf("  tiny chunk %.4f GFLOP; a 1536-frame dim-256 chunk "
              "%.4f TFLOP\n", f / 1e9,
              MetalBsRoformer::forward_flops(r) / 1e12);
}

// ---------------------------------------------------------------------
// The stage.

namespace {

class PcmSource : public TypedStage<PcmSource> {
public:
  static constexpr const char* kTypeName = "ut-bsr-source";
  using TypedStage::TypedStage;
  std::vector<TensorBeat> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    auto p = make_payload<TensorBeatPayload>(TensorBeat(beats[next++]));
    co_await ctx.write(0, std::move(p));
    co_return;
  }
};

class PcmSink : public TypedStage<PcmSink> {
public:
  static constexpr const char* kTypeName = "ut-bsr-sink";
  using TypedStage::TypedStage;
  int beats = 0;
  std::vector<float> ch[2];
  std::uint64_t first_ts = ~0ull;
  Job
  process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    const auto* tb = dynamic_cast<const TensorBeatPayload*>(in.get());
    if (tb == nullptr || tb->shape.size() != 2) { co_return; }
    ++beats;
    const int n = (int)tb->shape[1];
    const float* d = tb->as_f32();
    for (int c = 0; c < 2 && c < (int)tb->shape[0]; ++c) {
      ch[c].insert(ch[c].end(), d + (std::size_t)c * n,
                   d + (std::size_t)(c + 1) * n);
    }
    if (first_ts == ~0ull && tb->sideband.is_object()) {
      FlexData sb = tb->sideband;
      auto o = sb.as_object();
      if (o.contains("timestamp_us")) {
        first_ts = o.at("timestamp_us").as_uint();
      }
    }
    co_return;
  }
};

}  // namespace

TEST(bs_roformer, audio_separate_stage_streams_the_reference)
{
  const char* gd = std::getenv("VPIPE_BS_ROFORMER_TINY_GOLDEN");
  if (gd == nullptr || *gd == '\0') { return; }
  const std::string g = gd;
  // The golden's directory holds a .ckpt beside an MSST YAML and no
  // config.json: what model-register sees for any MSST download.
  EXPECT_TRUE(detect_model_dir(g, "").model_type == "bs-roformer");

  const std::vector<float> mix = read_f32_(g + "/mix.f32");
  const std::vector<float> ref = read_f32_(g + "/demix.f32");
  ASSERT_TRUE(!mix.empty() && ref.size() == 2 * mix.size());
  if (mix.empty() || ref.size() != 2 * mix.size()) { return; }
  const int n = (int)(mix.size() / 2);

  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto src_u = std::make_unique<PcmSource>(&sess, "src",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  // Ragged beats, as audio-to-pcm's would be.
  for (int at = 0, k = 0; at < n; ++k) {
    const int len = std::min(n - at, 1700 + 911 * (k % 3));
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {2, len};
    tb.resize_contiguous((std::size_t)2 * len);
    for (int c = 0; c < 2; ++c) {
      std::copy(mix.begin() + (std::ptrdiff_t)c * n + at,
                mix.begin() + (std::ptrdiff_t)c * n + at + len,
                tb.as_f32() + (std::size_t)c * len);
    }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("sample_rate", FlexData::make_int(44100));
    sb.as_object().insert_or_assign(
        "timestamp_us",
        FlexData::make_uint(5000000ull + (std::uint64_t)at * 1000000ull /
                                             44100ull));
    tb.sideband = std::move(sb);
    src_u->beats.push_back(std::move(tb));
    at += len;
  }
  auto* src = static_cast<PcmSource*>(pl->insert_stage(std::move(src_u)));
  src->allocate_oports(1);
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("hf_dir", FlexData::make_string(g));
  // The tiny model has two stems; take the second.
  cfg.as_object().insert_or_assign("stem", FlexData::make_string("other"));
  auto st_u = std::make_unique<AudioSeparateStage>(
      &sess, "sep", std::vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* st =
      static_cast<AudioSeparateStage*>(pl->insert_stage(std::move(st_u)));
  auto* stem = static_cast<PcmSink*>(pl->insert_stage(
      std::make_unique<PcmSink>(&sess, "stem", std::vector<InEdge>{{st, 0}},
                                FlexData::make_object())));
  auto* resid = static_cast<PcmSink*>(pl->insert_stage(
      std::make_unique<PcmSink>(&sess, "resid", std::vector<InEdge>{{st, 1}},
                                FlexData::make_object())));
  PipelineRuntime rt(pl.get(), &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(st->stem_name() == "other");
  EXPECT_TRUE(st->samples_emitted() == (std::uint64_t)n);
  ASSERT_TRUE(stem->ch[0].size() == (std::size_t)n &&
              stem->ch[1].size() == (std::size_t)n &&
              resid->ch[0].size() == (std::size_t)n);
  if (stem->ch[0].size() != (std::size_t)n ||
      resid->ch[0].size() != (std::size_t)n) {
    return;
  }
  // The reference's stem 1 is planar [ch][n] after stem 0's [2][n].
  std::vector<float> got((std::size_t)2 * n), want((std::size_t)2 * n);
  for (int c = 0; c < 2; ++c) {
    std::copy(stem->ch[c].begin(), stem->ch[c].end(),
              got.begin() + (std::ptrdiff_t)c * n);
    std::copy(ref.begin() + (std::ptrdiff_t)(2 + c) * n,
              ref.begin() + (std::ptrdiff_t)(3 + c) * n,
              want.begin() + (std::ptrdiff_t)c * n);
  }
  const double e = rel_l2_(got, want);
  std::printf("  stage stem rel %.3g over %d beats out\n", e, stem->beats);
  EXPECT_TRUE(e < 5e-3);
  double worst = 0.0;
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < n; ++i) {
      const double m = mix[(std::size_t)c * n + i];
      worst = std::max(worst, std::fabs(stem->ch[c][(std::size_t)i] +
                                        resid->ch[c][(std::size_t)i] - m));
    }
  }
  EXPECT_TRUE(worst < 1e-6);
  EXPECT_TRUE(stem->beats == resid->beats && stem->beats > 1);
  EXPECT_TRUE(stem->first_ts == 5000000ull);
}
