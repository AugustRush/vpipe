#include "generative-models/drama3/drama3-prompt.h"

#include "generative-models/tokenizer.h"

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <format>
#include <span>

namespace vpipe::genai::drama3 {

namespace {

constexpr const char* kTtsInstruction = "convert the provided text to speech";
constexpr const char* kRefInstruction =
    "convert the provided text to speech reference to the following:";

// The longest added token is <|speaker_change|>-sized; 64 bytes covers
// every one with room.
constexpr std::size_t kMaxAddedLen = 64;

// One UTF-8 code point at s[i] (its length in `len`); U+FFFD on a bad byte.
char32_t
utf8_at(std::string_view s, std::size_t i, std::size_t* len)
{
  const auto b = (unsigned char)s[i];
  auto cont = [&](std::size_t k) -> int {
    if (i + k >= s.size()) { return -1; }
    const auto c = (unsigned char)s[i + k];
    return (c & 0xC0) == 0x80 ? (c & 0x3F) : -1;
  };
  if (b < 0x80) { *len = 1; return b; }
  if ((b & 0xE0) == 0xC0) {
    const int c1 = cont(1);
    if (c1 >= 0) { *len = 2; return ((b & 0x1F) << 6) | c1; }
  } else if ((b & 0xF0) == 0xE0) {
    const int c1 = cont(1), c2 = cont(2);
    if (c1 >= 0 && c2 >= 0) {
      *len = 3;
      return ((b & 0x0F) << 12) | (c1 << 6) | c2;
    }
  } else if ((b & 0xF8) == 0xF0) {
    const int c1 = cont(1), c2 = cont(2), c3 = cont(3);
    if (c1 >= 0 && c2 >= 0 && c3 >= 0) {
      *len = 4;
      return ((b & 0x07) << 18) | (c1 << 12) | (c2 << 6) | c3;
    }
  }
  *len = 1;
  return 0xFFFD;
}

// The Unicode White_Space property -- the reference strips this exact set
// (str.strip() alone would also take U+001C..U+001F).
bool
is_ws(char32_t c)
{
  return (c >= 0x9 && c <= 0xD) || c == 0x20 || c == 0x85 || c == 0xA0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
         c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

std::size_t
ws_prefix(std::string_view s)
{
  std::size_t i = 0;
  while (i < s.size()) {
    std::size_t n = 1;
    if (!is_ws(utf8_at(s, i, &n))) { break; }
    i += n;
  }
  return i;
}

bool
all_ws(std::string_view s)
{
  return ws_prefix(s) == s.size();
}

// A speaker tag at s[i]: its length, its id in *id; 0 when none (a tag
// needs 1-2 digits, no leading zero, <= kMaxSpeaker).
std::size_t
speaker_tag_at(std::string_view s, std::size_t i, int* id)
{
  static constexpr std::string_view kHead = "<|speaker:";
  if (s.substr(i, kHead.size()) != kHead) { return 0; }
  std::size_t j = i + kHead.size();
  const std::size_t d0 = j;
  while (j < s.size() && s[j] >= '0' && s[j] <= '9') { ++j; }
  if (j == d0 || s.substr(j, 2) != "|>") { return 0; }
  const std::string_view digits = s.substr(d0, j - d0);
  if (digits.size() > 2 || (digits.size() > 1 && digits[0] == '0')) {
    return 0;
  }
  const int v = std::stoi(std::string(digits));
  if (v > kMaxSpeaker) { return 0; }
  if (id != nullptr) { *id = v; }
  return j + 2 - i;
}

}  // namespace

std::vector<std::int32_t>
encode_spliced(const Tokenizer& tok, std::string_view text)
{
  std::vector<std::int32_t> out;
  std::size_t done = 0, i = 0;
  while (i < text.size()) {
    const std::size_t lt = text.find('<', i);
    if (lt == std::string_view::npos) { break; }
    // The longest added token starting here.
    std::size_t best = 0;
    std::int32_t best_id = -1;
    for (std::size_t j = lt + 1;
         j < text.size() && j - lt < kMaxAddedLen; ++j) {
      if (text[j] != '>') { continue; }
      const std::int32_t id =
          tok.special_token_id(text.substr(lt, j - lt + 1));
      if (id >= 0) { best = j - lt + 1; best_id = id; }
    }
    if (best_id < 0) { i = lt + 1; continue; }
    if (lt > done) {
      const auto ids = tok.encode(text.substr(done, lt - done));
      out.insert(out.end(), ids.begin(), ids.end());
    }
    out.push_back(best_id);
    done = i = lt + best;
  }
  if (done < text.size()) {
    const auto ids = tok.encode(text.substr(done));
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

std::string
with_speaker_tag(std::string_view text)
{
  const std::size_t w = ws_prefix(text);
  if (speaker_tag_at(text, w, nullptr) > 0) { return std::string(text); }
  return "<|speaker:0|> " + std::string(text);
}

int
speaker_of(std::string_view text)
{
  int id = 0;
  return speaker_tag_at(text, ws_prefix(text), &id) > 0 ? id : 0;
}

std::string
validate_text(std::string_view text)
{
  if (all_ws(text)) { return "the text is empty"; }
  for (const char* t : {"<|im_start|>", "<|im_end|>", "<|voice|>",
                        "<|text|>", "<|interleave|>", "<|semantic:"}) {
    if (text.find(t) != std::string_view::npos) {
      return std::format("{} is written by the prompt builder and must not "
                         "appear in the text", t);
    }
  }
  for (std::size_t p = text.find("<|speaker"); p != std::string_view::npos;
       p = text.find("<|speaker", p + 1)) {
    if (speaker_tag_at(text, p, nullptr) == 0) {
      return std::format("speaker tags are <|speaker:N|> with N in 0..{} "
                         "and no leading zero", kMaxSpeaker);
    }
  }
  return {};
}

std::vector<std::int32_t>
build_prompt(const Tokenizer& tok, std::string_view text,
             const std::vector<Reference>& refs,
             const std::vector<const Take*>& history, int semantic_begin,
             int codebook_size, std::string* err, bool* too_long)
{
  if (too_long != nullptr) { *too_long = false; }
  auto fail = [&](std::string m) -> std::vector<std::int32_t> {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  constexpr int W = 11;
  std::vector<std::int32_t> rows;
  auto lit = [&](std::string_view s) {
    for (std::int32_t id : encode_spliced(tok, s)) {
      rows.push_back(id);
      rows.insert(rows.end(), W - 1, 0);
    }
  };
  // Codec frames as rows: [semantic_begin + c0, c0 .. c9].
  auto frames = [&](const Reference& r, const std::string& where)
      -> std::string {
    if (r.frames() <= 0 || r.codes.size() % 10 != 0) {
      return std::format("{}: no codec frames", where);
    }
    for (std::int32_t c : r.codes) {
      if (c < 0 || c >= codebook_size) {
        return std::format("{}: a code outside [0, {})", where,
                           codebook_size);
      }
    }
    for (int f = 0; f < r.frames(); ++f) {
      const std::int32_t* c = r.codes.data() + (std::size_t)f * 10;
      rows.push_back(semantic_begin + c[0]);
      rows.insert(rows.end(), c, c + 10);
    }
    return {};
  };
  if (std::string e = validate_text(text); !e.empty()) { return fail(e); }
  lit("<|im_start|>system\n");
  lit(refs.empty() ? kTtsInstruction : kRefInstruction);
  for (std::size_t r = 0; r < refs.size(); ++r) {
    const Reference& ref = refs[r];
    const std::string where = std::format("reference {}", r + 1);
    if (std::string e = validate_text(ref.transcript); !e.empty()) {
      return fail(std::format("{}: {}", where, e));
    }
    // The tag must follow "Text: " directly, so one behind white space
    // moves in front of it.
    const std::string tagged = with_speaker_tag(ref.transcript);
    const std::size_t w = ws_prefix(tagged);
    const std::size_t tl = speaker_tag_at(tagged, w, nullptr);
    const std::string tr = tagged.substr(w, tl) + tagged.substr(0, w) +
                           tagged.substr(w + tl);
    lit(std::format("\n\n[Reference {}]\nText: {}\nSpeech: ", r + 1, tr));
    if (std::string e = frames(ref, where); !e.empty()) { return fail(e); }
  }
  lit("<|im_end|>\n");
  for (std::size_t h = 0; h < history.size(); ++h) {
    const Take& take = *history[h];
    const std::string where = std::format("history {}", h + 1);
    if (std::string e = validate_text(take.transcript); !e.empty()) {
      return fail(std::format("{}: {}", where, e));
    }
    lit("<|im_start|>user\n");
    lit(with_speaker_tag(take.transcript));
    lit("<|im_end|>\n");
    lit("<|im_start|>assistant\n<|voice|>");
    if (std::string e = frames(take, where); !e.empty()) { return fail(e); }
    lit("<|im_end|>\n");
  }
  lit("<|im_start|>user\n");
  lit(with_speaker_tag(text));
  lit("<|im_end|>\n");
  lit("<|im_start|>assistant\n<|voice|>");
  if ((int)(rows.size() / W) > kMaxPromptRows) {
    if (too_long != nullptr) { *too_long = true; }
    return fail(std::format("the prompt has {} rows; at most {}",
                            rows.size() / W, kMaxPromptRows));
  }
  return rows;
}

// ---- long text -------------------------------------------------------

namespace {

// Python's \w over a code point: str.isalnum() or '_' -- every letter and
// number category, no marks (CoreFoundation's alphanumerics are L*, M*,
// N*, so the marks come off).
bool
is_word(char32_t c)
{
  if (c < 0x80) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
  }
  static const CFCharacterSetRef alnum =
      CFCharacterSetGetPredefined(kCFCharacterSetAlphaNumeric);
  static const CFCharacterSetRef mark =
      CFCharacterSetGetPredefined(kCFCharacterSetNonBase);
  const auto u = static_cast<UTF32Char>(c);
  return CFCharacterSetIsLongCharacterMember(alnum, u) &&
         !CFCharacterSetIsLongCharacterMember(mark, u);
}

// str.isdigit() for the decimal rule (Nd).
bool
is_digit(char32_t c)
{
  if (c < 0x80) { return c >= '0' && c <= '9'; }
  static const CFCharacterSetRef digit =
      CFCharacterSetGetPredefined(kCFCharacterSetDecimalDigit);
  return CFCharacterSetIsLongCharacterMember(digit,
                                             static_cast<UTF32Char>(c));
}

bool
ascii_alpha(char c)
{
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// One code point of a piece: its value and its byte span.
struct Cp {
  char32_t c;
  std::size_t at, len;
};

std::vector<Cp>
code_points(std::string_view s)
{
  std::vector<Cp> out;
  for (std::size_t i = 0; i < s.size();) {
    std::size_t n = 1;
    const char32_t c = utf8_at(s, i, &n);
    out.push_back({c, i, n});
    i += n;
  }
  return out;
}

std::string_view
strip_ws(std::string_view s)
{
  const std::size_t a = ws_prefix(s);
  std::size_t end = a;
  for (std::size_t i = a; i < s.size();) {
    std::size_t n = 1;
    const char32_t c = utf8_at(s, i, &n);
    i += n;
    if (!is_ws(c)) { end = i; }
  }
  return s.substr(a, end - a);
}

// The length of PROTECTED_SPAN's match at t[i], 0 when none -- the three
// alternatives tried in order, as Python's re does:
//   <\|[^|>]*\|>  |  <([A-Za-z][\w-]*)\b[^>]*>.*?</\1>  |  \[[^\[\]]*\]
// (re.S). The second backtracks its name from the longest down, each with
// its word boundary, and takes the EARLIEST closing tag.
std::size_t
protected_at(std::string_view t, std::size_t i)
{
  const std::size_t n = t.size();
  if (t[i] == '<' && i + 1 < n && t[i + 1] == '|') {
    std::size_t j = i + 2;
    while (j < n && t[j] != '|' && t[j] != '>') { ++j; }
    return j + 1 < n && t[j] == '|' && t[j + 1] == '>' ? j + 2 - i : 0;
  }
  if (t[i] == '<' && i + 1 < n && ascii_alpha(t[i + 1])) {
    // Where the name may end, with whether its last character is a word
    // character.
    std::vector<std::pair<std::size_t, bool>> ends{{i + 2, true}};
    for (std::size_t j = i + 2; j < n;) {
      std::size_t l = 1;
      const char32_t c = utf8_at(t, j, &l);
      if (c != '-' && !is_word(c)) { break; }
      j += l;
      ends.push_back({j, c != '-'});
    }
    const std::size_t gt = t.find('>', i + 2);
    if (gt == std::string_view::npos) { return 0; }
    for (auto it = ends.rbegin(); it != ends.rend(); ++it) {
      const std::size_t e = it->first;
      std::size_t l = 1;
      const bool right = e < n && is_word(utf8_at(t, e, &l));
      if (it->second == right) { continue; }        // no \b here
      const std::string close =
          "</" + std::string(t.substr(i + 1, e - i - 1)) + ">";
      const std::size_t f = t.find(close, gt + 1);
      if (f != std::string_view::npos) { return f + close.size() - i; }
    }
    return 0;
  }
  if (t[i] == '[') {
    std::size_t j = i + 1;
    while (j < n && t[j] != '[' && t[j] != ']') { ++j; }
    return j < n && t[j] == ']' ? j + 1 - i : 0;
  }
  return 0;
}

using Piece = std::pair<std::string, bool>;     // (text, protected)

std::vector<Piece>
protected_pieces(std::string_view t)
{
  std::vector<Piece> out;
  std::size_t last = 0;
  for (std::size_t i = 0; i < t.size();) {
    const std::size_t m = protected_at(t, i);
    if (m == 0) {
      std::size_t l = 1;
      utf8_at(t, i, &l);
      i += l;
      continue;
    }
    if (i > last) { out.push_back({std::string(t.substr(last, i - last)),
                                   false}); }
    out.push_back({std::string(t.substr(i, m)), true});
    i += m;
    last = i;
  }
  if (last < t.size()) { out.push_back({std::string(t.substr(last)), false}); }
  return out;
}

bool
in_marks(char32_t c, std::span<const char32_t> marks)
{
  return std::find(marks.begin(), marks.end(), c) != marks.end();
}

// Split each free piece longer than `length` bytes after every mark,
// keeping the mark; a '.' between digits is not one.
std::vector<Piece>
break_at_marks(const std::vector<Piece>& pieces, std::size_t length,
               std::span<const char32_t> marks)
{
  std::vector<Piece> out;
  for (const auto& [text, prot] : pieces) {
    if (prot || text.size() <= length) {
      out.push_back({text, prot});
      continue;
    }
    const std::vector<Cp> cps = code_points(text);
    std::size_t start = 0;
    for (std::size_t k = 0; k < cps.size(); ++k) {
      if (!in_marks(cps[k].c, marks)) { continue; }
      if (cps[k].c == '.' && k > 0 && k + 1 < cps.size() &&
          is_digit(cps[k - 1].c) && is_digit(cps[k + 1].c)) {
        continue;
      }
      const std::size_t end = cps[k].at + cps[k].len;
      out.push_back({text.substr(start, end - start), false});
      start = end;
    }
    if (start < text.size()) { out.push_back({text.substr(start), false}); }
  }
  return out;
}

// Split the free pieces still longer than `length` bytes on code points.
std::vector<Piece>
break_by_length(const std::vector<Piece>& pieces, std::size_t length)
{
  std::vector<Piece> out;
  for (const auto& [text, prot] : pieces) {
    if (prot || text.size() <= length) {
      out.push_back({text, prot});
      continue;
    }
    std::string cur;
    for (const Cp& cp : code_points(text)) {
      if (!cur.empty() && cur.size() + cp.len > length) {
        out.push_back({cur, false});
        cur.clear();
      }
      cur.append(text, cp.at, cp.len);
    }
    out.push_back({cur, false});
  }
  return out;
}

// The speaker tags of `t`: (start, end, digits), leftmost first.
struct TagMatch {
  std::size_t start, end;
  std::string digits;
};

std::vector<TagMatch>
speaker_tags(std::string_view t)
{
  static constexpr std::string_view kHead = "<|speaker:";
  std::vector<TagMatch> out;
  for (std::size_t p = t.find(kHead); p != std::string_view::npos;) {
    std::size_t j = p + kHead.size();
    const std::size_t d0 = j;
    while (j < t.size() && t[j] >= '0' && t[j] <= '9') { ++j; }
    if (j > d0 && t.substr(j, 2) == "|>") {
      out.push_back({p, j + 2, std::string(t.substr(d0, j - d0))});
      p = t.find(kHead, j + 2);
    } else {
      p = t.find(kHead, p + 1);
    }
  }
  return out;
}

}  // namespace

std::vector<std::string>
split_by_length(std::string_view text, int max_bytes)
{
  if (all_ws(text)) { return {}; }
  const std::size_t length = (std::size_t)std::max(0, max_bytes);
  if (text.size() <= length) { return {std::string(text)}; }
  // SENTENCE_ENDS, CLAUSE_MARKS, SPACES.
  static constexpr char32_t kSentenceEnds[] = {
      '.', '!', '?', ';', 0x3002, 0xFF01, 0xFF1F, 0xFF1B, 0x2026, '\n'};
  static constexpr char32_t kClauseMarks[] = {
      ',', ':', 0xFF0C, 0xFF1A, 0x3001};
  static constexpr char32_t kSpaces[] = {' '};
  std::vector<Piece> pieces =
      break_at_marks(protected_pieces(text), 0, kSentenceEnds);
  pieces = break_at_marks(pieces, length, kClauseMarks);
  pieces = break_at_marks(pieces, length, kSpaces);
  pieces = break_by_length(pieces, length);
  std::vector<std::string> chunks;
  std::string cur;
  for (const auto& [piece, prot] : pieces) {
    if (!cur.empty() && cur.size() + piece.size() > length) {
      chunks.push_back(std::move(cur));
      cur.clear();
    }
    cur += piece;
  }
  chunks.push_back(std::move(cur));
  std::vector<std::string> out;
  for (const std::string& c : chunks) {
    const std::string_view st = strip_ws(c);
    if (!st.empty()) { out.emplace_back(st); }
  }
  return out;
}

std::vector<std::pair<std::string, std::string>>
parse_speaker_segments(std::string_view text)
{
  const std::vector<TagMatch> tags = speaker_tags(text);
  std::vector<std::pair<std::string, std::string>> out;
  const std::string_view head =
      text.substr(0, tags.empty() ? text.size() : tags[0].start);
  if (!all_ws(head)) { out.push_back({"0", std::string(head)}); }
  for (std::size_t k = 0; k < tags.size(); ++k) {
    const std::size_t end =
        k + 1 < tags.size() ? tags[k + 1].start : text.size();
    const std::string_view body =
        text.substr(tags[k].end, end - tags[k].end);
    if (!all_ws(body)) { out.push_back({tags[k].digits, std::string(body)}); }
  }
  return out;
}

std::vector<std::string>
split_text(std::string_view text, int max_bytes, int max_speakers)
{
  if (speaker_tags(text).empty()) { return split_by_length(text, max_bytes); }
  std::vector<std::pair<std::string, std::string>> segments;
  for (const auto& [spk, body] : parse_speaker_segments(text)) {
    for (const std::string& piece : split_by_length(body, max_bytes)) {
      segments.push_back({spk, "<|speaker:" + spk + "|> " + piece});
    }
  }
  std::vector<std::string> chunks;
  std::vector<std::string> batch, speakers;
  std::size_t size = 0;
  auto join = [&]() {
    std::string c;
    for (std::size_t k = 0; k < batch.size(); ++k) {
      if (k > 0) { c += '\n'; }
      c += batch[k];
    }
    return c;
  };
  for (const auto& [spk, seg] : segments) {
    bool fresh = std::find(speakers.begin(), speakers.end(), spk) ==
                 speakers.end();
    const bool too_many = fresh && (int)speakers.size() >= max_speakers;
    const bool too_long = !batch.empty() &&
                          size + seg.size() > (std::size_t)max_bytes;
    if (too_many || too_long) {
      chunks.push_back(join());
      batch.clear();
      speakers.clear();
      size = 0;
      fresh = true;
    }
    batch.push_back(seg);
    if (fresh) { speakers.push_back(spk); }
    size += seg.size();
  }
  if (!batch.empty()) { chunks.push_back(join()); }
  return chunks;
}

void
History::push(Take t)
{
  if (!_pinned) {
    _pinned = std::make_unique<Take>(std::move(t));
    return;
  }
  _recent.push_back(std::move(t));
  while ((int)_recent.size() > _keep) { _recent.erase(_recent.begin()); }
}

std::vector<std::vector<const Take*>>
History::candidates() const
{
  std::vector<std::vector<const Take*>> out;
  for (std::size_t start = 0; start <= _recent.size(); ++start) {
    std::vector<const Take*> h;
    if (_pinned) { h.push_back(_pinned.get()); }
    for (std::size_t k = start; k < _recent.size(); ++k) {
      h.push_back(&_recent[k]);
    }
    out.push_back(std::move(h));
  }
  if (_pinned) { out.push_back({}); }
  return out;
}

}  // namespace vpipe::genai::drama3
