#include "generative-models/bs-roformer/bs-roformer-config.h"

#include "generative-models/shared/yaml-subset.h"

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <string>

namespace vpipe::genai {

namespace {

// The section `name` of the document, or null.
FlexData
section_(const FlexData& doc, const char* name)
{
  if (!doc.is_object()) { return FlexData::make_null(); }
  const auto o = doc.as_object();
  return o.contains(name) ? o.at(name) : FlexData::make_null();
}

bool
has_(const FlexData& sec, const char* key)
{
  return sec.is_object() && sec.as_object().contains(key) &&
         !sec.as_object().at(key).is_null();
}

FlexData
get_(const FlexData& sec, const char* key)
{
  return has_(sec, key) ? sec.as_object().at(key) : FlexData::make_null();
}

// An integer key. A YAML float with an integral value is accepted (a
// hand-edited `8.0`); anything else is an error naming the key.
bool
int_(const FlexData& sec, const char* key, int* out, bool required,
     std::string* err)
{
  if (!has_(sec, key)) {
    if (required && err != nullptr) {
      *err = std::string("`") + key + "` is missing";
    }
    return !required;
  }
  const FlexData v = get_(sec, key);
  if (v.is_int() || v.is_uint()) {
    *out = (int)v.as_int();
    return true;
  }
  if (v.is_real() && v.as_real() == (double)(long long)v.as_real()) {
    *out = (int)v.as_real();
    return true;
  }
  if (err != nullptr) { *err = std::string("`") + key + "` is not an integer"; }
  return false;
}

bool
bool_(const FlexData& sec, const char* key, bool* out, std::string* err)
{
  if (!has_(sec, key)) { return true; }
  const FlexData v = get_(sec, key);
  if (!v.is_bool()) {
    if (err != nullptr) { *err = std::string("`") + key + "` is not a bool"; }
    return false;
  }
  *out = v.as_bool();
  return true;
}

}  // namespace

bool
looks_like_bs_roformer_yaml(const FlexData& yaml)
{
  const FlexData m = section_(yaml, "model");
  return has_(m, "freqs_per_bands") && has_(m, "dim") &&
         has_(m, "depth") && !has_(m, "num_bands");
}

bool
bs_roformer_config_from_yaml(const FlexData& yaml, BsRoformerConfig* c,
                             std::string* err)
{
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return false;
  };
  if (c == nullptr) { return fail("no config to fill"); }
  *c = BsRoformerConfig{};
  const FlexData m = section_(yaml, "model");
  if (!m.is_object()) { return fail("the YAML has no `model:` section"); }
  if (has_(m, "num_bands") && !has_(m, "freqs_per_bands")) {
    return fail("this is a Mel-Band RoFormer config (`num_bands`), not a "
                "BS-RoFormer one");
  }

  std::string e;
  auto need = [&](const char* k, int* v) {
    if (!int_(m, k, v, true, &e)) { return fail("model." + e); }
    return true;
  };
  auto opt = [&](const char* k, int* v) {
    if (!int_(m, k, v, false, &e)) { return fail("model." + e); }
    return true;
  };
  if (!need("dim", &c->dim) || !need("depth", &c->depth)) { return false; }
  if (!opt("num_stems", &c->num_stems) ||
      !opt("time_transformer_depth", &c->time_depth) ||
      !opt("freq_transformer_depth", &c->freq_depth) ||
      !opt("dim_head", &c->dim_head) || !opt("heads", &c->heads) ||
      !opt("stft_n_fft", &c->n_fft) ||
      !opt("stft_hop_length", &c->hop) ||
      !opt("mask_estimator_depth", &c->mask_depth) ||
      !opt("mlp_expansion_factor", &c->mlp_expansion)) {
    return false;
  }
  c->win_length = c->n_fft;
  if (!opt("stft_win_length", &c->win_length)) { return false; }
  if (!bool_(m, "stereo", &c->stereo, &e) ||
      !bool_(m, "stft_normalized", &c->stft_normalized, &e) ||
      !bool_(m, "skip_connection", &c->skip_connection, &e) ||
      !bool_(m, "zero_dc", &c->zero_dc, &e)) {
    return fail("model." + e);
  }

  // The variants this port does not implement. Each changes the
  // forward, so each is a refusal rather than a key to ignore.
  {
    int lin = 0;
    if (!opt("linear_transformer_depth", &lin)) { return false; }
    if (lin != 0) {
      return fail("model.linear_transformer_depth " + std::to_string(lin) +
                  ": the linear-attention layers are not implemented");
    }
    bool pope = false;
    if (!bool_(m, "use_pope", &pope, &e)) { return fail("model." + e); }
    if (pope) { return fail("model.use_pope: PoPE is not implemented"); }
    int streams = 1;
    if (!opt("num_residual_streams", &streams)) { return false; }
    if (streams != 1) {
      return fail("model.num_residual_streams " + std::to_string(streams) +
                  ": hyper-connections are not implemented");
    }
    bool vres = false;
    if (!bool_(m, "use_value_residual_learning", &vres, &e)) {
      return fail("model." + e);
    }
    if (vres) {
      return fail("model.use_value_residual_learning: value residuals are "
                  "not implemented");
    }
    if (has_(m, "freq_range")) {
      return fail("model.freq_range: a sliced frequency range is not "
                  "implemented");
    }
    if (has_(m, "stft_window_fn")) {
      return fail("model.stft_window_fn: only the Hann window is "
                  "implemented");
    }
  }

  {
    const FlexData b = get_(m, "freqs_per_bands");
    if (!b.is_array() || b.as_array().size() < 2) {
      return fail("model.freqs_per_bands must list at least two bands");
    }
    for (const FlexData v : b.as_array()) {
      if (!(v.is_int() || v.is_uint()) || v.as_int() <= 0) {
        return fail("model.freqs_per_bands holds a non-positive or "
                    "non-integer entry");
      }
      c->freqs_per_bands.push_back((int)v.as_int());
    }
  }

  // Geometry the network depends on.
  if (c->dim <= 0 || c->depth <= 0 || c->heads <= 0 || c->dim_head <= 0 ||
      c->num_stems <= 0 || c->time_depth < 0 || c->freq_depth < 0 ||
      c->mask_depth <= 0 || c->mlp_expansion <= 0) {
    return fail("model: dim, depth, heads, dim_head, num_stems, "
                "mask_estimator_depth and mlp_expansion_factor must be "
                "positive");
  }
  if (c->n_fft < 2 || c->hop <= 0 || c->win_length <= 0 ||
      c->win_length > c->n_fft) {
    return fail("model: stft_n_fft / stft_hop_length / stft_win_length "
                "describe no transform");
  }
  const int sum = std::accumulate(c->freqs_per_bands.begin(),
                                  c->freqs_per_bands.end(), 0);
  if (sum != c->bins()) {
    return fail("model.freqs_per_bands sums to " + std::to_string(sum) +
                ", but stft_n_fft " + std::to_string(c->n_fft) + " has " +
                std::to_string(c->bins()) + " bins");
  }

  // ---- audio ----
  const FlexData a = section_(yaml, "audio");
  const FlexData inf = section_(yaml, "inference");
  if (!int_(a, "sample_rate", &c->sample_rate, false, &e)) {
    return fail("audio." + e);
  }
  // inference.chunk_size wins where both exist, as MSST's demix() reads.
  if (has_(inf, "chunk_size")) {
    if (!int_(inf, "chunk_size", &c->chunk_size, true, &e)) {
      return fail("inference." + e);
    }
  } else if (!int_(a, "chunk_size", &c->chunk_size, true, &e)) {
    return fail("audio." + e);
  }
  if (c->sample_rate <= 0 || c->chunk_size <= c->n_fft) {
    return fail("audio: sample_rate must be positive and chunk_size longer "
                "than one STFT window");
  }
  {
    int ch = c->channels();
    if (!int_(a, "num_channels", &ch, false, &e)) {
      return fail("audio." + e);
    }
    if (ch != c->channels()) {
      return fail("audio.num_channels " + std::to_string(ch) +
                  " disagrees with model.stereo");
    }
  }
  if (!int_(inf, "num_overlap", &c->num_overlap, false, &e)) {
    return fail("inference." + e);
  }
  if (c->num_overlap < 1 || c->chunk_size / c->num_overlap <= 0) {
    return fail("inference.num_overlap must be at least 1");
  }
  if (!bool_(inf, "normalize", &c->normalize, &e)) {
    return fail("inference." + e);
  }

  // ---- training: the stems' names ----
  const FlexData t = section_(yaml, "training");
  // Bound to locals: as_array() is a VIEW into its FlexData.
  const FlexData target = get_(t, "target_instrument");
  const FlexData instruments = get_(t, "instruments");
  if (target.is_string() && !target.as_string().empty()) {
    c->stems.emplace_back(target.as_string());
  } else if (instruments.is_array()) {
    for (const FlexData v : instruments.as_array()) {
      c->stems.emplace_back(v.as_string());
    }
  }
  if ((int)c->stems.size() != c->num_stems) {
    // A config whose names do not line up with the heads still runs;
    // the stems are then known by their index.
    c->stems.clear();
    for (int s = 0; s < c->num_stems; ++s) {
      c->stems.push_back("stem" + std::to_string(s));
    }
  }
  return true;
}

namespace {

namespace fs = std::filesystem;

std::string
lower_(std::string s)
{
  for (char& ch : s) {
    if (ch >= 'A' && ch <= 'Z') { ch = (char)(ch - 'A' + 'a'); }
  }
  return s;
}

bool
is_ckpt_(const fs::path& p)
{
  const std::string e = lower_(p.extension().string());
  return e == ".ckpt" || e == ".pth" || e == ".pt" || e == ".safetensors";
}

bool
is_yaml_(const fs::path& p)
{
  const std::string e = lower_(p.extension().string());
  return e == ".yaml" || e == ".yml";
}

// The regular files directly in `dir` that `pick` accepts, sorted.
std::vector<fs::path>
files_in_(const fs::path& dir, bool (*pick)(const fs::path&))
{
  std::vector<fs::path> out;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (it->is_regular_file(ec) && pick(it->path())) {
      out.push_back(it->path());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string
names_(const std::vector<fs::path>& v)
{
  std::string s;
  for (const fs::path& p : v) {
    if (!s.empty()) { s += ", "; }
    s += p.filename().string();
  }
  return s;
}

}  // namespace

bool
resolve_bs_roformer_files(const std::string& path,
                          const std::vector<std::string>& pinned,
                          const std::string& checkpoint,
                          const std::string& config,
                          BsRoformerFiles* out, std::string* err)
{
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return false;
  };
  if (out == nullptr || path.empty()) { return fail("no model path"); }
  std::error_code ec;
  const fs::path root(path);
  fs::path ckpt, dir;
  if (fs::is_regular_file(root, ec)) {
    ckpt = root;
    dir = root.parent_path();
  } else if (fs::is_directory(root, ec)) {
    dir = root;
    if (!checkpoint.empty()) {
      ckpt = fs::path(checkpoint).is_absolute() ? fs::path(checkpoint)
                                                : dir / checkpoint;
    } else {
      std::vector<fs::path> c;
      for (const std::string& f : pinned) {
        if (is_ckpt_(f)) { c.push_back(dir / f); }
      }
      if (c.empty()) { c = files_in_(dir, is_ckpt_); }
      if (c.empty()) { return fail(path + " holds no checkpoint"); }
      if (c.size() > 1) {
        return fail(path + " holds several checkpoints (" + names_(c) +
                    "); name one with `checkpoint`");
      }
      ckpt = c.front();
    }
  } else {
    return fail(path + " does not exist");
  }
  if (!fs::is_regular_file(ckpt, ec)) {
    return fail(ckpt.string() + " does not exist");
  }

  fs::path yaml;
  if (!config.empty()) {
    yaml = fs::path(config).is_absolute() ? fs::path(config) : dir / config;
  } else {
    std::vector<fs::path> c;
    for (const std::string& f : pinned) {
      if (is_yaml_(f)) { c.push_back(dir / f); }
    }
    if (c.size() == 1) {
      yaml = c.front();
    } else {
      const std::string stem = ckpt.stem().string();
      for (const char* ext : {".yaml", ".yml"}) {
        for (const std::string& n :
             {stem + "-Config", stem + "_config", stem + "-config",
              "config_" + stem, stem}) {
          if (yaml.empty() && fs::is_regular_file(dir / (n + ext), ec)) {
            yaml = dir / (n + ext);
          }
        }
      }
      if (yaml.empty()) {
        const std::vector<fs::path> all = files_in_(dir, is_yaml_);
        if (all.size() == 1) {
          yaml = all.front();
        } else if (all.empty()) {
          return fail("no YAML config beside " + ckpt.filename().string() +
                      "; name one with `config`");
        } else {
          return fail("several YAML configs beside " +
                      ckpt.filename().string() + " (" + names_(all) +
                      ") and none named for it; name one with `config`");
        }
      }
    }
  }
  if (!fs::is_regular_file(yaml, ec)) {
    return fail(yaml.string() + " does not exist");
  }
  out->ckpt = ckpt.string();
  out->yaml = yaml.string();
  return true;
}

bool
looks_like_bs_roformer_dir(const std::string& path)
{
  std::error_code ec;
  const fs::path root(path);
  const fs::path dir = fs::is_regular_file(root, ec) ? root.parent_path()
                                                     : root;
  if (!fs::is_directory(dir, ec)) { return false; }
  if (fs::is_regular_file(root, ec) ? !is_ckpt_(root)
                                    : files_in_(dir, is_ckpt_).empty()) {
    return false;
  }
  for (const fs::path& y : files_in_(dir, is_yaml_)) {
    std::string e;
    const FlexData doc = yaml_subset::parse_file(y.string(), &e);
    if (looks_like_bs_roformer_yaml(doc)) { return true; }
  }
  return false;
}

bool
bs_roformer_config_from_file(const std::string& path, BsRoformerConfig* c,
                             std::string* err)
{
  std::string e;
  const FlexData doc = yaml_subset::parse_file(path, &e);
  if (doc.is_null()) {
    if (err != nullptr) { *err = path + ": " + e; }
    return false;
  }
  if (!bs_roformer_config_from_yaml(doc, c, &e)) {
    if (err != nullptr) { *err = path + ": " + e; }
    return false;
  }
  return true;
}

}  // namespace vpipe::genai
