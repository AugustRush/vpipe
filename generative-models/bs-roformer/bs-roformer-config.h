#ifndef VPIPE_GENERATIVE_MODELS_BS_ROFORMER_BS_ROFORMER_CONFIG_H
#define VPIPE_GENERATIVE_MODELS_BS_ROFORMER_BS_ROFORMER_CONFIG_H

// The configuration of a BS-RoFormer source separator, read from the
// YAML its checkpoint ships with.
//
// THE FORMAT IS MSST'S (ZFTurbo/Music-Source-Separation-Training): four
// sections, of which `model` holds the constructor's keywords verbatim,
// `audio` the clip geometry, `inference` the overlap and `training` the
// stem names. Every BS-RoFormer published so far was trained with
// MSST's copy of the network, so that is the network implemented --
// NOT lucidrains/BS-RoFormer's current head, which has since grown
// hyper-connections, value residuals and PoPE that no released weights
// use. A config that asks for any of those, or for MSST's linear-
// attention or Mel-Band variants, is refused by name at load: each one
// changes the forward, and reading past it would run a different
// network over the same tensors.

#include "common/flex-data.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai {

struct BsRoformerConfig {
  // ---- model ----
  int  dim            = 0;
  int  depth          = 0;
  bool stereo         = false;
  int  num_stems      = 1;
  int  time_depth     = 2;     // time_transformer_depth
  int  freq_depth     = 2;     // freq_transformer_depth
  std::vector<int> freqs_per_bands;
  int  dim_head       = 64;
  int  heads          = 8;
  int  n_fft          = 2048;
  int  hop            = 512;
  int  win_length     = 2048;
  bool stft_normalized = false;
  int  mask_depth     = 2;     // mask_estimator_depth
  int  mlp_expansion  = 4;     // mlp_expansion_factor
  bool skip_connection = false;
  // MSST's `bs_roformer` zeroes the 0 Hz bin of every estimate before
  // the inverse transform; its `bs_roformer_experimental` does not.
  // Inaudible either way. Default: the plain class, the one MSST names
  // `bs_roformer` and runs these configs with.
  bool zero_dc        = true;

  // ---- audio / inference ----
  int  sample_rate    = 44100;
  int  chunk_size     = 0;     // samples per forward
  int  num_overlap    = 2;
  // MSST's `inference.normalize`: whole-track mean/std normalisation.
  // Recorded, not applied -- see the loader's note.
  bool normalize      = false;

  // ---- training ----
  // What the model's stems ARE, in output order: MSST's
  // prefer_target_instrument() -- [target_instrument] when one is set,
  // else `instruments`.
  std::vector<std::string> stems;

  int channels() const { return stereo ? 2 : 1; }
  int inner() const { return heads * dim_head; }
  int bands() const { return (int)freqs_per_bands.size(); }
  int bins() const { return n_fft / 2 + 1; }
  // Features of one frame: every bin of every channel, complex.
  int frame_features() const { return bins() * channels() * 2; }
  // Frames of one chunk (torch.stft, center=True).
  int frames() const { return chunk_size / hop + 1; }
  int ff_inner() const { return 4 * dim; }          // FeedForward mult 4
  int mask_hidden() const { return dim * mlp_expansion; }
};

// Read a parsed MSST YAML. False with `err` naming the key on anything
// missing, inconsistent or not implemented.
bool bs_roformer_config_from_yaml(const FlexData& yaml, BsRoformerConfig* c,
                                  std::string* err);

// The config of the YAML at `path`.
bool bs_roformer_config_from_file(const std::string& path,
                                  BsRoformerConfig* c, std::string* err);

// True when `yaml` is an MSST config for a BS-RoFormer -- the probe
// model detection uses. False for anything else, Mel-Band included.
bool looks_like_bs_roformer_yaml(const FlexData& yaml);

// A checkpoint and the YAML that describes it.
//
// MSST publishes them as loose pairs, and one directory often holds
// several -- pcunwa/BS-Roformer-Resurrection ships a vocals and an
// instrumental model side by side -- so the directory alone does not
// name a model. Resolved in this order, refusing rather than guessing
// whenever more than one candidate is left:
//
//   checkpoint  `path` itself when it is a file; else `checkpoint` (a
//               file name in the directory); else the one torch file
//               (.ckpt / .pth / .pt) or .safetensors among `pinned` (a
//               registry record's files); else the directory's only one.
//   config      `config` (relative to the checkpoint's directory unless
//               absolute); else the one pinned .yaml; else the YAML
//               NAMED for the checkpoint -- <stem>-Config.yaml,
//               <stem>_config.yaml, <stem>-config.yaml,
//               config_<stem>.yaml, <stem>.yaml (.yml too); else the
//               directory's only YAML.
struct BsRoformerFiles {
  std::string ckpt;
  std::string yaml;
};
bool resolve_bs_roformer_files(const std::string& path,
                               const std::vector<std::string>& pinned,
                               const std::string& checkpoint,
                               const std::string& config,
                               BsRoformerFiles* out, std::string* err);

// True when `path` (a directory or a checkpoint file) holds a torch
// checkpoint and an MSST YAML for a BS-RoFormer beside it. Cheap: it
// reads YAML files, never the checkpoint.
bool looks_like_bs_roformer_dir(const std::string& path);

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_BS_ROFORMER_BS_ROFORMER_CONFIG_H
