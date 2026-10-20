#ifndef VPIPE_GENERATIVE_MODELS_DRAMA3_DRAMA3_PROMPT_H
#define VPIPE_GENERATIVE_MODELS_DRAMA3_DRAMA3_PROMPT_H

// Fish Audio Drama 3's prompt, the reference-bracket-v1 layout (sglang-
// omni's fishaudio_drama3/prompt.py, vllm-omni's fish_omni/prompt.py):
//
//   <|im_start|>system\n
//   convert the provided text to speech[ reference to the following:]
//   \n\n[Reference 1]\nText: <|speaker:N|>transcript\nSpeech: <frames>
//   ...
//   <|im_end|>\n
//   <|im_start|>user\n<|speaker:0|> earlier<|im_end|>\n        (history:
//   <|im_start|>assistant\n<|voice|><frames><|im_end|>\n         0 or more)
//   <|im_start|>user\n<|speaker:0|> text<|im_end|>\n
//   <|im_start|>assistant\n<|voice|>
//
// One ROW per token, [id, 0 x 10], and one row per reference codec frame,
// [semantic_begin + c0, c0 .. c9]. Every literal is tokenized ON ITS OWN:
// joining them first changes the ids at the seams.
//
// The reference's tokenizer splits its ADDED tokens (<|im_start|>,
// <|speaker:3|>, <|semantic:17|>, <think>, ...) out of any text it is
// given; vpipe's Tokenizer encodes text as text and leaves specials to the
// caller. encode_spliced() does the reference's split: leftmost, longest
// match over every added token -- every one of Drama 3's is <...>-shaped,
// which load() checks.

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe::genai {

class Tokenizer;

namespace drama3 {

constexpr int kMaxSpeaker = 15;     // <|speaker:0|> .. <|speaker:15|>
// The reference's prompt bound (DRAMA3_MAX_PROMPT_ROWS): a longer one is
// refused, and the chunk driver drops history until it fits.
constexpr int kMaxPromptRows = 8192;
// Long text: chunks of about this many UTF-8 bytes with at most this many
// speakers each (DRAMA3_CHUNK_BYTES / _MAX_SPEAKERS).
constexpr int kChunkBytes = 300;
constexpr int kChunkMaxSpeakers = 3;

// A transcript and its codec frames: a reference voice, or an earlier take
// written into the prompt as a finished user / assistant turn.
struct Reference {
  std::string transcript;
  // [frames][10] codes, codebook-fastest (frame f's code k at f*10 + k).
  std::vector<std::int32_t> codes;
  int frames() const { return (int)(codes.size() / 10); }
};
using Take = Reference;

// The reference tokenizer's encode of `text`: added tokens split out
// (leftmost, longest), the rest through Tokenizer::encode.
std::vector<std::int32_t> encode_spliced(const Tokenizer& tok,
                                         std::string_view text);

// "<|speaker:0|> " + text unless a speaker tag already leads it (after
// Unicode white space).
std::string with_speaker_tag(std::string_view text);

// The speaker a text is spoken as: its leading tag's id (after Unicode
// white space), else 0.
int speaker_of(std::string_view text);

// Why `text` cannot be caller text, or "" when it can: empty after white
// space, a structural token the builder writes (<|im_start|>, <|im_end|>,
// <|voice|>, <|text|>, <|interleave|>, <|semantic:), or a speaker tag
// outside 0..15 / with a leading zero.
std::string validate_text(std::string_view text);

// The prompt rows, [rows][11] flattened: the references, then `history`
// (earlier takes, oldest first) as finished turns, then the open turn for
// `text`. Empty with `err` set when a text, a reference or a take does not
// validate, or when the prompt passes kMaxPromptRows (then `*too_long` is
// set, so a caller can retry with less history).
std::vector<std::int32_t> build_prompt(const Tokenizer& tok,
                                       std::string_view text,
                                       const std::vector<Reference>& refs,
                                       const std::vector<const Take*>& history,
                                       int semantic_begin, int codebook_size,
                                       std::string* err = nullptr,
                                       bool* too_long = nullptr);
inline std::vector<std::int32_t>
build_prompt(const Tokenizer& tok, std::string_view text,
             const std::vector<Reference>& refs, int semantic_begin,
             int codebook_size, std::string* err = nullptr)
{
  return build_prompt(tok, text, refs, {}, semantic_begin, codebook_size,
                      err);
}

// ---- long text (fish-speech fish_omni/chunking.py, the hosted server's
// rule) ------------------------------------------------------------------

// Chunks of about `max_bytes` UTF-8 bytes, cut at sentence ends, then (a
// sentence still too long) clause marks, then spaces, then on code-point
// boundaries, and packed back up to `max_bytes`. Never cut inside a special
// token, a paired tag (<whisper>..</whisper>) or a [bracketed] cue, and a
// '.' between digits is not a sentence end. Text of `max_bytes` or less
// comes back whole and unstripped; otherwise every chunk is stripped of
// Unicode white space, and empty ones dropped.
std::vector<std::string> split_by_length(std::string_view text,
                                         int max_bytes = kChunkBytes);

// (speaker digits, body) per speaker-tagged segment; text before the first
// tag is speaker "0". Segments whose body is only white space are dropped.
std::vector<std::pair<std::string, std::string>>
parse_speaker_segments(std::string_view text);

// The generation chunks of `text`, in order. Untagged text is split by
// length. Tagged text is split into speaker segments (a long one cut
// further, each piece keeping its tag as "<|speaker:N|> piece") and
// regrouped, joined by '\n': a chunk ends when the next segment would bring
// a speaker past `max_speakers` or the chunk past `max_bytes`, so A, B, A
// can share one. Empty for text that is only white space.
std::vector<std::string> split_text(std::string_view text,
                                    int max_bytes = kChunkBytes,
                                    int max_speakers = kChunkMaxSpeakers);

// The context the next chunk is generated with (Drama3History): the FIRST
// take stays pinned, plus the latest `recent` ones. candidates() lists the
// histories to try, longest first -- the oldest recent takes dropped one at
// a time, then the pinned one, ending with none.
class History {
public:
  explicit History(int recent = 1) : _keep(recent) {}
  void push(Take t);
  void clear() { _pinned.reset(); _recent.clear(); }
  bool empty() const { return !_pinned; }
  std::vector<std::vector<const Take*>> candidates() const;

private:
  int _keep;
  std::unique_ptr<Take> _pinned;
  std::vector<Take> _recent;
};

}  // namespace drama3
}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DRAMA3_DRAMA3_PROMPT_H
