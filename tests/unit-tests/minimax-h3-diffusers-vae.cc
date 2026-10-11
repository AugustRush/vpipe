// MiniMax-H3's VAEs in the DIFFUSERS layout, and the VAE class table.
//
// The MiniMaxAI/MiniMax-H3 repository carries a diffusers pipeline at its
// root (`vae/` = AutoencoderKLMiniMaxH3, `audio_vae/` =
// AutoencoderKLMiniMaxH3Audio) beside the released `FL2VA/` and `Ref2VA/`
// partitions. Pointed at that root, vae-decode used to open `vae/` as a
// Qwen-Image VAE -- no class table knew the name and the last arm was a
// default -- and went inert claiming its config "carries 24/24 of the 16
// latents_mean / latents_std entries the un-whiten needs".
//
// Three things are held here:
//   1. a checkout with BOTH layouts opens the released VAEs (the copy
//      every other H3 stage shares), and one with only the diffusers
//      pipeline opens that;
//   2. a VAE class nothing built in reads is refused BY NAME, by
//      vae-decode, vae-encode and audio-vae-decode alike;
//   3. the diffusers VAEs ARE the released ones: renamed at open, the
//      split attention and swapped feed-forward reassembled, their
//      configs read -- the same Config, and a bit-identical decode and
//      encode (needs real weights).
//
// 1 and 2 and the translation need no weights: a few tiny safetensors
// carry the names and shapes. 3 needs VPIPE_MINIMAX_H3_DIFFUSERS_ROOT, a
// checkout holding the root `vae/` and `audio_vae/` AND a partition.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-vae-decode-stage.h"
#include "stages/model-detect.h"
#include "stages/vae-decode-stage.h"
#include "stages/vae-encode-stage.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

fs::path
scratch_()
{
  return fs::temp_directory_path() / "vpipe-h3-diffusers-vae-test";
}

// A VALID safetensors of F32 zeros, one tensor per (name, shape).
void
write_st_(const fs::path& p,
          const std::vector<std::pair<std::string,
                                      std::vector<std::int64_t>>>& ts)
{
  FlexData hdr = FlexData::make_object();
  std::int64_t off = 0;
  {
    auto o = hdr.as_object();
    for (const auto& [name, shape] : ts) {
      std::int64_t n = 1;
      FlexData sh = FlexData::make_array();
      for (std::int64_t d : shape) {
        sh.as_array().push_back(FlexData::make_int(d));
        n *= d;
      }
      FlexData t = FlexData::make_object();
      auto to = t.as_object();
      to.insert_or_assign("dtype", FlexData::make_string("F32"));
      to.insert_or_assign("shape", std::move(sh));
      FlexData offs = FlexData::make_array();
      offs.as_array().push_back(FlexData::make_int(off));
      offs.as_array().push_back(FlexData::make_int(off + 4 * n));
      to.insert_or_assign("data_offsets", std::move(offs));
      o.insert_or_assign(name, std::move(t));
      off += 4 * n;
    }
  }
  const std::string json = hdr.to_json(false);
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  const std::uint64_t n = json.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(json.data(), (std::streamsize)json.size());
  const std::vector<char> zeros((std::size_t)off, 0);
  f.write(zeros.data(), (std::streamsize)zeros.size());
}

void
write_text_(const fs::path& p, const std::string& text)
{
  fs::create_directories(p.parent_path());
  std::ofstream(p) << text;
}

std::string
class_json_(const std::string& cls)
{
  return "{\"_class_name\": \"" + cls + "\"}";
}

// The diffusers video VAE config as published, with the statistics
// replaced by 24 placeholders (the values are not under test here).
std::string
diffusers_video_config_()
{
  std::string mean = "[", stdv = "[";
  for (int i = 0; i < 24; ++i) {
    mean += (i > 0 ? ", " : "") + std::string("0.0");
    stdv += (i > 0 ? ", " : "") + std::string("1.0");
  }
  mean += "]";
  stdv += "]";
  return "{\"_class_name\": \"AutoencoderKLMiniMaxH3\", "
         "\"in_channels\": 3, \"out_channels\": 3, \"latent_channels\": 24, "
         "\"block_out_channels\": [128, 256, 256, 512, 512, 1024], "
         "\"layers_per_block\": 2, "
         "\"spatial_downsample_factors\": [2, 2, 2, 2, 1, 1], "
         "\"temporal_downsample_factors\": [1, 2, 2, 1, 1, 1], "
         "\"norm_num_groups\": 32, \"norm_eps\": 1e-06, "
         "\"spatial_padding_mode\": \"reflect\", "
         "\"decoder_num_layers\": 36, \"decoder_num_attention_heads\": 32, "
         "\"decoder_attention_head_dim\": 64, "
         "\"decoder_num_register_tokens\": 4, \"decoder_ffn_mult\": 4, "
         "\"decoder_rope_theta\": 100.0, \"decoder_rope_dim_ratio\": 0.75, "
         "\"decoder_norm_eps\": 1e-05, \"clip_length\": 17, "
         "\"token_drop\": 3, \"latents_mean\": " + mean +
         ", \"latents_std\": " + stdv + "}";
}

// One tensor that is not a checkpoint of anything.
void
placeholder_(const fs::path& p)
{
  write_st_(p, {{"probe.weight", {1}}});
}

// A diffusers-layout video VAE in miniature: every KIND of name the
// translation handles, at toy widths (2 heads of 4, one block).
std::vector<std::pair<std::string, std::vector<std::int64_t>>>
mini_vae_()
{
  const std::string b = "decoder.transformer_blocks.0.";
  return {
      {"decoder.proj_in.weight", {8, 4}},
      {"decoder.proj_in.bias", {8}},
      {"decoder.register_tokens", {1, 4, 8}},
      {"decoder.proj_out.weight", {12, 8}},
      {"decoder.proj_out.bias", {12}},
      {"decoder.norm_out.weight", {8}},
      {"decoder.norm_out.bias", {8}},
      {b + "attn.to_q.weight", {8, 8}},
      {b + "attn.to_q.bias", {8}},
      {b + "attn.to_k.weight", {8, 8}},
      {b + "attn.to_k.bias", {8}},
      {b + "attn.to_v.weight", {8, 8}},
      {b + "attn.to_v.bias", {8}},
      {b + "attn.to_out.0.weight", {8, 8}},
      {b + "attn.to_out.0.bias", {8}},
      {b + "ff.net.0.proj.weight", {32, 8}},
      {b + "ff.net.0.proj.bias", {32}},
      {b + "ff.net.2.weight", {8, 16}},
      {b + "ff.net.2.bias", {8}},
      {b + "norm1.weight", {8}},
      {b + "norm2.weight", {8}},
      {b + "scale1", {8}},
      {b + "scale2", {8}},
      {"encoder.conv_in.weight", {4, 3, 3, 3, 3}},
      {"encoder.conv_in.bias", {4}},
      {"encoder.down_blocks.0.resnets.0.conv1.weight", {4, 4, 3, 3, 3}},
      {"encoder.down_blocks.0.resnets.0.norm1.weight", {4}},
      {"encoder.down_blocks.0.resnets.1.conv_shortcut.weight",
       {8, 4, 1, 1, 1}},
      {"encoder.down_blocks.0.downsamplers.0.conv.weight", {4, 4, 3, 3, 3}},
      {"encoder.norm_out.weight", {8}},
      {"encoder.conv_out.weight", {4, 8, 3, 3, 3}},
      {"quant_conv.weight", {4, 4, 1, 1, 1}},
      {"post_quant_conv.weight", {2, 2, 1, 1, 1}},
  };
}

// Records every error and warning a stage reports.
struct CapturingUi : public UiDelegateIntf {
  std::vector<std::string> lines;
  void error(const VpipeFormat& f) override { lines.push_back(f()); }
  void warn (const VpipeFormat& f) override { lines.push_back(f()); }
  void info (const VpipeFormat&) override {}
  UiInputStatus getline(const VpipeFormat&, std::string&,
                        const std::function<bool()>&) override
  {
    return UiInputStatus::Eof;
  }
  std::unique_ptr<UiTextStream> open_text_stream() override
  {
    return std::make_unique<NullUiTextStream>();
  }
  bool contains(std::string_view needle) const
  {
    for (const std::string& l : lines) {
      if (l.find(needle) != std::string::npos) { return true; }
    }
    return false;
  }
};

// Sends `payload` (if any), then ends.
class SourceOne : public TypedStage<SourceOne> {
public:
  static constexpr const char* kTypeName = "ut-h3dvae-source";
  SourceOne(const SessionContextIntf* s, std::string id,
            std::vector<InEdge> ip, FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  {
    allocate_oports(1);
  }
  std::unique_ptr<BeatPayloadIntf> payload;
  Job
  process(RuntimeContext& ctx) override
  {
    if (payload) { co_await ctx.write(0, std::move(payload)); }
    ctx.signal_done();
    co_return;
  }
};

class Sink : public TypedStage<Sink> {
public:
  static constexpr const char* kTypeName = "ut-h3dvae-sink";
  using TypedStage::TypedStage;
  Job
  process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
  }
};

// Runs source -> `Stage`(hf_dir = root) -> sink and returns what the
// stage reported.
template <typename Stage>
std::vector<std::string>
run_stage_(const std::string& root, std::unique_ptr<BeatPayloadIntf> beat)
{
  Session sess;
  auto  ui_owned = std::make_unique<CapturingUi>();
  auto* ui       = ui_owned.get();
  sess.set_ui_delegate(std::move(ui_owned));
  if (sess.metal_compute() == nullptr) { return {"<no gpu>"}; }
  auto pl = std::make_unique<Pipeline>("p", &sess);
  auto srcu = std::make_unique<SourceOne>(&sess, "src",
                                          std::vector<InEdge>{},
                                          FlexData::make_object());
  srcu->payload = std::move(beat);
  auto* src = pl->insert_stage(std::move(srcu));
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  auto* st = pl->insert_stage(std::make_unique<Stage>(
      &sess, "vae", std::vector<InEdge>{{src, 0}}, std::move(cfg)));
  pl->insert_stage(std::make_unique<Sink>(
      &sess, "sink", std::vector<InEdge>{{st, 0}}, FlexData::make_object()));
  {
    PipelineRuntime rt(pl.get(), &sess);
    rt.launch();
    rt.wait_idle();
    rt.stop();
  }
  return ui->lines;
}

bool
has_(const std::vector<std::string>& lines, std::string_view needle)
{
  for (const std::string& l : lines) {
    if (l.find(needle) != std::string::npos) { return true; }
  }
  return false;
}

}  // namespace

// The renames apply to a whole checkpoint or not at all, and the two
// tensors stored in pieces come back in the model's row order: to_qkv
// one HEAD of q, k, v at a time (the fused tensor is grouped per head),
// w1 with its halves exchanged.
TEST(minimax_h3_diffusers_vae, the_names_translate_whole_or_not_at_all)
{
  const fs::path dir = scratch_() / "names";
  std::error_code ec;
  fs::remove_all(dir, ec);
  const fs::path file = dir / "vae.safetensors";
  write_st_(file, mini_vae_());
  auto w = MetalLlamaWeights::open_model(file.string());
  ASSERT_TRUE(w.has_value());
  if (!w.has_value()) { return; }
  const std::string b = "decoder.transformer_blocks.0.";
  for (const char* n :
       {"decoder.x_embedder.weight", "decoder.x_embedder.bias",
        "decoder.register_tokens", "decoder.proj_out.weight",
        "decoder.transformer_blocks.0.attn.to_qkv.q.weight",
        "decoder.transformer_blocks.0.attn.to_qkv.v.bias",
        "decoder.transformer_blocks.0.attn.to_out.weight",
        "decoder.transformer_blocks.0.ff.w1.ug.weight",
        "decoder.transformer_blocks.0.ff.w2.bias",
        "decoder.transformer_blocks.0.scale2",
        "encoder.down.0.block.0.conv1.weight",
        "encoder.down.0.block.0.norm1.weight",
        "encoder.down.0.block.1.nin_shortcut.weight",
        "encoder.down.0.downsample.conv.weight", "post_quant_conv.weight"}) {
    if (!w->has(n)) { std::printf("[h3_dvae] missing %s\n", n); }
    EXPECT_TRUE(w->has(n));
  }
  EXPECT_FALSE(w->has("decoder.proj_in.weight"));
  EXPECT_FALSE(w->has(b + "attn.to_q.weight"));
  EXPECT_FALSE(w->has(b + "ff.net.0.proj.weight"));

  // Two heads of 4 rows: q0 k0 v0 q1 k1 v1.
  const auto qkv = minimax_h3::vae_assembled_parts(*w, b + "attn.to_qkv", 4);
  ASSERT_TRUE(qkv.size() == 6);
  if (qkv.size() == 6) {
    const char* order[] = {".q", ".k", ".v", ".q", ".k", ".v"};
    for (int i = 0; i < 6; ++i) {
      EXPECT_TRUE(qkv[i].layer == b + "attn.to_qkv" + order[i]);
      EXPECT_TRUE(qkv[i].row0 == (i < 3 ? 0 : 4) && qkv[i].rows == 4);
    }
  }
  // gate | up from up | gate.
  const auto w1 = minimax_h3::vae_assembled_parts(*w, b + "ff.w1", 4);
  ASSERT_TRUE(w1.size() == 2);
  if (w1.size() == 2) {
    EXPECT_TRUE(w1[0].row0 == 16 && w1[0].rows == 16);
    EXPECT_TRUE(w1[1].row0 == 0 && w1[1].rows == 16);
  }
  // A whole tensor needs no assembly.
  EXPECT_TRUE(minimax_h3::vae_assembled_parts(*w, "decoder.x_embedder", 4)
                  .empty());

  // One name it cannot place, and nothing is renamed: the file keeps
  // its own names and fails to load as what it is.
  auto more = mini_vae_();
  more.push_back({b + "attn.gate.weight", {8, 8}});
  const fs::path odd = dir / "odd.safetensors";
  write_st_(odd, more);
  auto w2 = MetalLlamaWeights::open_model(odd.string());
  ASSERT_TRUE(w2.has_value());
  if (w2.has_value()) {
    EXPECT_TRUE(w2->has("decoder.proj_in.weight"));
    EXPECT_FALSE(w2->has("decoder.x_embedder.weight"));
  }
  fs::remove_all(dir, ec);
}

// Both layouts in one checkout: the RELEASED VAEs answer, from every
// resolver -- the stages' generic one and both H3 classes' own -- so
// vae-decode, vae-encode, video-ref-encoder and audio-vae-decode open
// the same copy. With only the diffusers pipeline, that answers.
TEST(minimax_h3_diffusers_vae, both_layouts_open_the_released_vaes)
{
  const fs::path root = scratch_() / "checkout";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_text_(root / "vae" / "config.json", diffusers_video_config_());
  placeholder_(root / "vae" / "diffusion_pytorch_model.safetensors");
  write_text_(root / "audio_vae" / "config.json",
              class_json_(minimax_h3::kDiffusersAudioVaeClass));
  placeholder_(root / "audio_vae" / "diffusion_pytorch_model.safetensors");
  const fs::path part = root / "FL2VA";
  write_text_(part / "video_vae" / "config.json",
              class_json_("MiniMaxH3VideoVAE"));
  placeholder_(part / "video_vae" / "source" / "model.safetensors");
  write_text_(part / "audio_vae" / "config.json",
              class_json_("MiniMaxH3AudioVAE"));
  placeholder_(part / "audio_vae" / "model.safetensors");

  const std::string r = root.string();
  EXPECT_TRUE(resolve_vae_dir(r) == (part / "video_vae").string());
  EXPECT_TRUE(MetalMiniMaxH3VideoVae::resolve_vae_dir(r) ==
              (part / "video_vae" / "source").string());
  EXPECT_TRUE(MetalMiniMaxH3AudioVae::resolve_vae_dir(r) ==
              (part / "audio_vae").string());

  // The diffusers pipeline alone.
  fs::remove_all(part, ec);
  const std::string vae = (root / "vae").string();
  const std::string avae = (root / "audio_vae").string();
  EXPECT_TRUE(resolve_vae_dir(r) == vae);
  EXPECT_TRUE(MetalMiniMaxH3VideoVae::resolve_vae_dir(r) == vae);
  EXPECT_TRUE(MetalMiniMaxH3VideoVae::resolve_vae_dir(vae) == vae);
  EXPECT_TRUE(MetalMiniMaxH3AudioVae::resolve_vae_dir(r) == avae);
  EXPECT_TRUE(MetalMiniMaxH3AudioVae::resolve_vae_dir(avae) == avae);
  EXPECT_TRUE(builtin_vae_family(vae) == "minimax-h3");

  // And its config reads as the released net's geometry.
  MetalMiniMaxH3VideoVae::Config c;
  std::string err;
  const bool ok = MetalMiniMaxH3VideoVae::config_from_json(r, c, &err);
  if (!ok) { std::printf("[h3_dvae] config: %s\n", err.c_str()); }
  EXPECT_TRUE(ok);
  EXPECT_TRUE(c.z_channels == 24 && c.base_ch == 128 && c.res_blocks == 2);
  EXPECT_TRUE((c.ch_mult == std::vector<int>{1, 2, 2, 4, 4, 8}));
  EXPECT_TRUE(c.patch == 16 && c.patch_t == 4);
  EXPECT_TRUE(c.n_layers == 36 && c.n_heads == 32 && c.head_dim == 64);
  EXPECT_TRUE(c.dim == 2048 && c.ffn_inner == 8192 && c.n_register == 4);
  EXPECT_TRUE(c.clip_length == 17 && c.token_drop == 3);
  EXPECT_TRUE(c.latents_mean.size() == 24 && c.latents_std.size() == 24);
  fs::remove_all(root, ec);
}

// Every class the table knows maps to its decoder; one it does not is
// refused BY NAME rather than opened as a Qwen-Image VAE. A VAE with no
// class at all -- a standalone checkpoint, its config borrowed -- keeps
// the Qwen-Image reading it always had.
TEST(minimax_h3_diffusers_vae, a_vae_class_nothing_reads_is_refused_by_name)
{
  const fs::path dir = scratch_() / "classes";
  std::error_code ec;
  fs::remove_all(dir, ec);
  const std::pair<const char*, const char*> known[] = {
      {"AutoencoderKLQwenImage", "krea2"},
      {"AutoencoderKLQwenImage21", "krea2"},
      {"AutoencoderKLQwenImage2D", "krea2"},
      {"AutoencoderKLFlux2", "flux2"},
      {"AutoencoderKL", "flux2"},
      {"AutoencoderKLWan", "wan"},
      {"MiniMaxH3VideoVAE", "minimax-h3"},
      {"AutoencoderKLMiniMaxH3", "minimax-h3"},
  };
  for (const auto& [cls, fam] : known) {
    write_text_(dir / "config.json", class_json_(cls));
    std::string unknown = "x";
    EXPECT_TRUE(builtin_vae_family(dir.string(), &unknown) == fam);
    EXPECT_TRUE(unknown.empty());
  }
  write_text_(dir / "config.json", class_json_("AutoencoderKLSomethingNew"));
  std::string unknown;
  EXPECT_TRUE(builtin_vae_family(dir.string(), &unknown).empty());
  EXPECT_TRUE(unknown == "AutoencoderKLSomethingNew");
  EXPECT_TRUE(builtin_vae_classes().find("AutoencoderKLMiniMaxH3") !=
              std::string::npos);
  write_text_(dir / "config.json", "{\"z_dim\": 16}");
  EXPECT_TRUE(builtin_vae_family(dir.string()) == "krea2");
  fs::remove(dir / "config.json", ec);
  EXPECT_TRUE(builtin_vae_family(dir.string()) == "krea2");
  fs::remove_all(dir, ec);
}

// The same refusal from the stages, which is what a user sees. Each
// stage names the class; none reports the misleading statistics count
// a Qwen-Image reading produced.
TEST(minimax_h3_diffusers_vae, the_stages_name_the_class_they_refuse)
{
  const fs::path root = scratch_() / "stage-refusal";
  std::error_code ec;
  fs::remove_all(root, ec);
  write_text_(root / "vae" / "config.json",
              class_json_("AutoencoderKLSomethingNew"));
  placeholder_(root / "vae" / "diffusion_pytorch_model.safetensors");
  write_text_(root / "audio_vae" / "config.json",
              class_json_("AutoencoderKLSomethingNewAudio"));
  placeholder_(root / "audio_vae" / "diffusion_pytorch_model.safetensors");

  const auto dec = run_stage_<VaeDecodeStage>(root.string(), nullptr);
  if (has_(dec, "<no gpu>")) { fs::remove_all(root, ec); return; }
  for (const auto& l : dec) {
    std::printf("[h3_dvae] decode: %s\n", l.c_str());
  }
  EXPECT_TRUE(has_(dec, "names the VAE class 'AutoencoderKLSomethingNew'"));
  EXPECT_FALSE(has_(dec, "the un-whiten needs"));

  // vae-encode loads on its first picture, so send one.
  auto img = std::make_unique<TensorBeatPayload>();
  img->dtype = TensorBeat::DType::U8;
  img->shape = {3, 16, 16};
  img->resize_contiguous(3 * 16 * 16);
  const auto enc = run_stage_<VaeEncodeStage>(root.string(), std::move(img));
  for (const auto& l : enc) {
    std::printf("[h3_dvae] encode: %s\n", l.c_str());
  }
  EXPECT_TRUE(has_(enc, "names the VAE class 'AutoencoderKLSomethingNew'"));

  const auto aud = run_stage_<AudioVaeDecodeStage>(root.string(), nullptr);
  for (const auto& l : aud) {
    std::printf("[h3_dvae] audio: %s\n", l.c_str());
  }
  // The audio resolver does not take an unknown class as an audio VAE
  // directory, so the stage says it holds none -- the case that message
  // was written for. Named, when it is the directory it was handed.
  EXPECT_TRUE(has_(aud, "holds no audio VAE"));
  const auto aud2 =
      run_stage_<AudioVaeDecodeStage>((root / "audio_vae").string(), nullptr);
  for (const auto& l : aud2) {
    std::printf("[h3_dvae] audio dir: %s\n", l.c_str());
  }
  EXPECT_TRUE(has_(aud2, "'AutoencoderKLSomethingNewAudio'"));

  // The report this started from: the diffusers H3 VAE named by the
  // checkout's root is now the H3 family's to load (and fails here on
  // the placeholder's missing tensors, as the H3 VAE).
  write_text_(root / "vae" / "config.json", diffusers_video_config_());
  const auto h3 = run_stage_<VaeDecodeStage>(root.string(), nullptr);
  for (const auto& l : h3) { std::printf("[h3_dvae] h3: %s\n", l.c_str()); }
  EXPECT_TRUE(has_(h3, "MiniMax-H3 video VAE"));
  EXPECT_FALSE(has_(h3, "the un-whiten needs"));
  fs::remove_all(root, ec);
}

namespace {

// Deterministic values in [-1, 1) as bf16, so equal outputs mean equal
// arithmetic rather than equal zeros.
SharedBuffer
noise_bf16_(MetalCompute* mc, std::size_t n, std::uint32_t seed)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  auto* d = static_cast<std::uint16_t*>(b.contents());
  std::uint32_t s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const float v = (float)(s >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    d[i] = (std::uint16_t)(u >> 16);
  }
  return b;
}

bool
same_bytes_(const SharedBuffer& a, const SharedBuffer& b)
{
  return !a.empty() && a.byte_size() == b.byte_size() &&
         std::memcmp(a.contents(), b.contents(), a.byte_size()) == 0;
}

}  // namespace

// THE DIFFUSERS VAEs ARE THE RELEASED ONES, end to end: the same Config,
// and a decode and an encode that match to the BIT. Bit-identical is the
// bar because nothing may differ -- the same f32 weights, renamed and
// reassembled, rounded to bf16 the same way. A wrong head grouping or an
// unswapped feed-forward still decodes a plausible-looking picture.
TEST(minimax_h3_diffusers_vae, the_diffusers_vaes_are_the_released_ones)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_DIFFUSERS_ROOT");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string vae = std::string(root) + "/vae";
  const std::string avae = std::string(root) + "/audio_vae";
  // The checkout resolves to its released copy; the diffusers one is
  // named directly.
  const std::string released = MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  std::printf("[h3_dvae] released %s\n", released.c_str());
  ASSERT_TRUE(released != vae && released.find("/source") !=
                                     std::string::npos);
  EXPECT_TRUE(resolve_vae_dir(root) != vae);

  MetalMiniMaxH3VideoVae::Config a, b;
  std::string err;
  const bool nok = MetalMiniMaxH3VideoVae::config_from_json(root, a, &err);
  ASSERT_TRUE(nok);
  const bool dok = MetalMiniMaxH3VideoVae::config_from_json(vae, b, &err);
  if (!dok) { std::printf("[h3_dvae] %s\n", err.c_str()); }
  ASSERT_TRUE(dok);
  if (!nok || !dok) { return; }
  EXPECT_TRUE(a.z_channels == b.z_channels && a.out_channels == b.out_channels);
  EXPECT_TRUE(a.dim == b.dim && a.n_heads == b.n_heads &&
              a.head_dim == b.head_dim && a.n_layers == b.n_layers);
  EXPECT_TRUE(a.ffn_inner == b.ffn_inner && a.patch == b.patch &&
              a.patch_t == b.patch_t && a.n_register == b.n_register);
  EXPECT_TRUE(a.rope_theta == b.rope_theta &&
              a.rope_dim_ratio == b.rope_dim_ratio &&
              a.norm_eps == b.norm_eps);
  EXPECT_TRUE(a.in_channels == b.in_channels && a.base_ch == b.base_ch &&
              a.res_blocks == b.res_blocks && a.gn_groups == b.gn_groups &&
              a.enc_norm_eps == b.enc_norm_eps);
  EXPECT_TRUE(a.ch_mult == b.ch_mult && a.space_down == b.space_down &&
              a.time_down == b.time_down);
  EXPECT_TRUE(a.clip_length == b.clip_length &&
              a.token_drop == b.token_drop && a.tile_size == b.tile_size &&
              a.tile_overlap_min == b.tile_overlap_min);
  EXPECT_TRUE(a.latents_mean == b.latents_mean &&
              a.latents_std == b.latents_std && !a.latents_mean.empty());

  {
    auto ra = MetalMiniMaxH3VideoVae::load(root, mc, a);
    auto rb = MetalMiniMaxH3VideoVae::load(vae, mc, b);
    ASSERT_TRUE(ra != nullptr && rb != nullptr);
    if (ra == nullptr || rb == nullptr) { return; }
    const int T = 1, h = 4, w = 4;
    const SharedBuffer z =
        noise_bf16_(mc, (std::size_t)a.z_channels * T * h * w, 7);
    const SharedBuffer da = ra->decode(z, T, h, w, &err);
    const SharedBuffer db = rb->decode(z, T, h, w, &err);
    std::printf("[h3_dvae] video decode %zu bytes, identical %d\n",
                da.byte_size(), (int)same_bytes_(da, db));
    EXPECT_TRUE(same_bytes_(da, db));
    const int ET = 5, EH = 64, EW = 64;
    const SharedBuffer x =
        noise_bf16_(mc, (std::size_t)a.in_channels * ET * EH * EW, 11);
    const SharedBuffer ea = ra->encode(x, ET, EH, EW, &err);
    const SharedBuffer eb = rb->encode(x, ET, EH, EW, &err);
    std::printf("[h3_dvae] video encode %zu bytes, identical %d\n",
                ea.byte_size(), (int)same_bytes_(ea, eb));
    EXPECT_TRUE(same_bytes_(ea, eb));
  }

  MetalMiniMaxH3AudioVae::Config aa, ab;
  const bool rok = MetalMiniMaxH3AudioVae::config_from_json(root, aa, &err);
  ASSERT_TRUE(rok);
  const bool aok = MetalMiniMaxH3AudioVae::config_from_json(avae, ab, &err);
  if (!aok) { std::printf("[h3_dvae] %s\n", err.c_str()); }
  ASSERT_TRUE(aok);
  if (!rok || !aok) { return; }
  EXPECT_TRUE(aa.latent_channels == ab.latent_channels &&
              aa.latent_dim == ab.latent_dim &&
              aa.decoder_dim == ab.decoder_dim &&
              aa.stereo_channels == ab.stereo_channels &&
              aa.sample_rate == ab.sample_rate);
  EXPECT_TRUE(aa.up_rates == ab.up_rates && aa.down_rates == ab.down_rates &&
              aa.res_kernels == ab.res_kernels &&
              aa.res_dilations == ab.res_dilations &&
              aa.encoder_dim == ab.encoder_dim &&
              aa.enc_attn_heads == ab.enc_attn_heads);
  EXPECT_TRUE(aa.latents_mean == ab.latents_mean &&
              aa.latents_std == ab.latents_std && !aa.latents_mean.empty());
  auto pa = MetalMiniMaxH3AudioVae::load(root, mc, aa);
  auto pb = MetalMiniMaxH3AudioVae::load(avae, mc, ab);
  ASSERT_TRUE(pa != nullptr && pb != nullptr);
  if (pa == nullptr || pb == nullptr) { return; }
  const int stereo = 2, frames = 12;
  std::vector<float> z((std::size_t)stereo * aa.latent_channels * frames);
  std::uint32_t s = 3;
  for (float& v : z) {
    s = s * 1664525u + 1013904223u;
    v = (float)(s >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
  }
  std::vector<float> pcm_a, pcm_b;
  const bool da = pa->decode(z.data(), stereo, frames, &pcm_a, &err);
  const bool db = pb->decode(z.data(), stereo, frames, &pcm_b, &err);
  ASSERT_TRUE(da && db);
  if (!da || !db) { return; }
  std::printf("[h3_dvae] audio decode %zu samples, identical %d\n",
              pcm_a.size(), (int)(pcm_a == pcm_b));
  EXPECT_TRUE(!pcm_a.empty() && pcm_a == pcm_b);
  std::vector<float> lat_a, lat_b;
  // What the decode produced, planar [stereo, samples].
  const int samples = pa->decoded_samples(frames);
  ASSERT_TRUE(pcm_a.size() == (std::size_t)stereo * samples);
  if (pcm_a.size() != (std::size_t)stereo * samples) { return; }
  ASSERT_TRUE(pa->encode(pcm_a.data(), stereo, samples, &lat_a, nullptr,
                         &err));
  ASSERT_TRUE(pb->encode(pcm_a.data(), stereo, samples, &lat_b, nullptr,
                         &err));
  std::printf("[h3_dvae] audio encode %zu values, identical %d\n",
              lat_a.size(), (int)(lat_a == lat_b));
  EXPECT_TRUE(!lat_a.empty() && lat_a == lat_b);
}
