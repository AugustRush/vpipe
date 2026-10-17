#ifndef GENERATIVE_MODELS_FLASHVSR_FLASHVSR_FAMILY_H
#define GENERATIVE_MODELS_FLASHVSR_FLASHVSR_FAMILY_H

#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/video-model-registry.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace vpipe {
namespace genai {

// FlashVSR as a `generate-video` family, registered rather than added as
// a fourth branch of that stage's built-in dispatch.
//
// WHY REGISTERED, when it is in-tree and the built-in families are not.
// The registry is the newer path and the stage consults it first, so
// this costs the built-ins nothing. What decided it is the SOURCE CLIP:
// a family reached through the registry takes its extra input through
// the request's named-input lookup, which is the seam that exists for
// exactly this, where a built-in branch would invite one more typed
// field on a struct whose comment already lists six such fields as the
// reason not to add a seventh.
//
// THE NAME IT ASKS FOR is `source_video`, and it is the first name in
// the set -- see gen-input.h, which says the set is empty and that
// names are added, never renamed. It resolves to the bicubic-upscaled
// low-quality clip as planar u8 RGB [frames, 3, H, W], which is the
// shape temporal-stack builds and the reference ports already take. A
// request that does not carry it is REFUSED rather than run: this model
// conditions on nothing else, so without it there is noise and a
// constant prompt, and what comes out is a confident 4x upscale of
// nothing.
//
// WHAT IT IGNORES, and why none of it is an error. The conditioning
// port carries a text encoding this family never reads -- its own
// context is a constant that ships with the checkpoint -- so a graph
// still has to wire something there because the stage's first port
// drives the beat, and the cheapest honest thing to wire is a
// text-prompt source with an empty string. The negative port, the
// sampler and the scheduler are all inert: one step at a fixed
// timestep, guidance distilled away, no schedule to select.
//
// GEOMETRY IS NOT NEGOTIABLE HERE the way it is elsewhere. Output width
// and height must be multiples of 128 because the window partition
// needs the token grid to divide by 8, and frames must be 8k+1 because
// the denoise consumes 8 source frames per chunk. Both are reported and
// rounded up, per the contract's rule that a family adjusts rather than
// refuses so a graph can change families without being re-authored.
//
// THE SOURCE SETS THE SIZE. The output is 4x the source clip, so a
// graph that resamples its input has already chosen the output
// resolution; the stage's own width and height are read as a cap when
// they are set and ignored when they are not.
// WHERE FLASHVSR'S PARTS LIVE under one checkpoint root, in each of the
// layouts it arrives in. Every consumer -- the denoiser's load, the
// source encoder stage, claims(), the memory declarations -- goes through
// this, so the layouts are told apart in exactly one place.
//
// v1.1:
//   CONVERTED (tools/flashvsr_prepare_checkpoint.py): one index names the
//     denoiser, `lq_proj.*` and `posi_context`; the VAE is `vae/`.
//   PUBLISHED (the model page, as the catalogue fetches it): the denoiser
//     safetensors, `LQ_proj_in.ckpt` and `posi_prompt.pth` as torch
//     files, and `Wan2.1_VAE.pth` bare. Read in place -- torch files map
//     as shards (shared/torch-zip.h) -- so nothing is converted or copied.
//   Converted wins when both are present, which is what a directory the
//   converter ran over looks like: its index then names the same bytes.
//
// FlashVSR+ (ruchangyao/FlashVSR_plus): one repository, two variants
// side by side -- `turbo/` and `base/`, each a `dit.safetensors` and a
// `lq_proj_in.safetensors` -- sharing the root's `Wan2.1_VAE.pth` and the
// fixed context (`posi_prompt.pth`, byte-identical to v1.1's).
//   A VARIANT FOLDER is the model: its name, or the root config.json's
//     `variants` map, says which -- and it has to, because the two are the
//     same tensors and differ only in how the stack is driven (Turbo's
//     hybrid layers). The VAE and the context are found beside it or one
//     level up.
//   THE REPOSITORY ROOT is Turbo, the reference's default -- or Base when
//     only Base is there.
struct FlashVsrLayout {
  std::string denoiser;        // what the denoiser's WeightSet opens
  std::string source;          // ...the source projection's
  std::string source_prefix;   // tensor-name prefix inside `source`
  std::string context;         // ...the fixed context's
  std::string context_name;    // its tensor name
  std::string vae;             // the VAE, or empty when absent
  // The reference's TINY decoder, TCDecoder.ckpt -- the same file for both
  // releases, beside the VAE -- or empty when absent (FlashVsrTinyDecoder).
  std::string tiny_decoder;
  bool        published = false;
  // FlashVSR+, and which variant: "turbo" or "base", empty when nothing
  // says (refused at load rather than guessed).
  bool        plus = false;
  std::string variant;
};

bool resolve_flashvsr_layout(const std::string& root, FlashVsrLayout* out);

// How the layout's release drives the stack (see
// MetalFlashVsrTransformer::Schedule). A FlashVSR+ variant nothing names
// is sized as Base, whose windows are the larger.
MetalFlashVsrTransformer::Schedule
flashvsr_schedule(const FlashVsrLayout& layout);
// ...for a root: v1.1's when it is not a FlashVSR checkpoint at all.
MetalFlashVsrTransformer::Schedule
flashvsr_schedule(const std::string& root);

// ---- ONE SOURCE, ONE PASS ----------------------------------------------
//
// FlashVSR restores a video as a stream: a causal projection over the
// source, and a denoiser whose blocks attend a key/value window over the
// chunks before. Cutting the source into independent clips restarts all
// of it -- and the VAE decode after it -- at every cut, which is a visible
// change of texture there. A STREAM carries the state across instead:
// the source encoder keeps its projection's carries, the denoiser each
// block's window, the VAE decode its causal carries, and the result is
// what one pass over the whole source makes.
//
// The source encoder's rows say where in the stream they sit, beside the
// generic keys of common/beat-keys.h (sideband::kStreamContinues,
// kStreamFinal, kValidFrames, kColorRef), which the family forwards onto
// the latent for the decode.
namespace flashvsr_stream {
// Ints, on the rows: the stream's index of the first chunk they hold
// (0 = the opening one), and how many they hold.
inline constexpr std::string_view kChunk0 = "flashvsr_chunk0";
inline constexpr std::string_view kChunks = "flashvsr_chunks";
}  // namespace flashvsr_stream

// What a stream HOLDS between beats at this output geometry: every
// denoiser block's retained key/value windows, and the decode's carries
// -- the VAE's causal carries over the whole plane (a carried decode does
// not tile), or with `tiny` the tiny decoder's whole scratch, which it
// keeps between calls. 0 when the checkpoint cannot be read.
std::size_t flashvsr_stream_bytes(const std::string& root, int width,
                                  int height, double kv_ratio = 3.0,
                                  bool tiny = false);
// Whether this box can restore a stream at this geometry: the stream's
// carries, the denoiser at its streaming floor, the projection, and the
// decode's working set -- a whole-plane VAE decode, or with `tiny` the
// tiny decoder (when the checkpoint has one) -- with kStreamHeadroom to
// spare, against TOTAL RAM (VPIPE_RAM_LIMIT_MB), so every stage that
// asks answers alike. `need` gets the figure. True when RAM is unknown.
bool flashvsr_stream_fits(const std::string& root, int width, int height,
                          std::size_t* need = nullptr, bool tiny = false);

class FlashVsrVideoFamily final : public VideoModelFamily {
 public:
  std::string_view tag() const noexcept override { return "flashvsr"; }

  // Claims a directory holding the four files the checkpoint ships. The
  // check has to be narrow: the denoiser alone is Wan 2.1-T2V-1.3B and
  // would claim a plain Wan checkpoint, which would then be driven with
  // a source projection it does not have. So the projection is what is
  // looked for, not the denoiser.
  bool claims(const std::string& root,
              const std::string& model_type) const override;

  int  align_frames(const std::string& root, int frames) const override;
  void size_grid(const std::string& root, int* gh, int* gw) const override;

  std::vector<ResourceClaim>
  declare_resources(const std::string& root) const override;
  std::vector<StageHolding>
  declare_holdings(const std::string& root) const override;
  std::size_t latent_bytes(const std::string& root, int width, int height,
                           int frames) const override;
  // The key/value window and the activation scratch, which at full HD
  // are several times the checkpoint. The window follows `kv_ratio` from
  // the flashvsr-model-config beat when one is wired.
  std::size_t denoise_scratch_bytes(const std::string& root, int width,
                                    int height, int frames,
                                    const FlexData* model_config)
      const override;

  std::unique_ptr<VideoGenerator>
  load(const VideoModelCreateArgs& args) override;
};

// Register the family with the process-wide VideoModelRegistry. Called
// once from the stage library's registration point, the way the other
// in-tree registries are seeded.
void register_flashvsr_video_family();

}  // namespace genai
}  // namespace vpipe

#endif
