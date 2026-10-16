#include "generative-models/shared/yaml-subset.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace vpipe::genai::yaml_subset {

namespace {

struct Line {
  int         indent = 0;
  std::string text;          // comment stripped, trailing space trimmed
  int         number = 0;    // 1-based, for messages
};

bool
is_space_(char c)
{
  return c == ' ' || c == '\t';
}

std::string
trim_(std::string_view s)
{
  std::size_t a = 0, b = s.size();
  while (a < b && is_space_(s[a])) { ++a; }
  while (b > a && (is_space_(s[b - 1]) || s[b - 1] == '\r')) { --b; }
  return std::string(s.substr(a, b - a));
}

// The line with a trailing comment removed. `#` starts a comment at the
// start of the content or after whitespace, never inside quotes.
//
// A quote opens a quoted scalar only where a value can start -- the
// start of the line or after `: `, `- `, `[`, `{` or `,` -- so the
// apostrophe in a plain `don't` does not swallow the comment after it.
std::string
strip_comment_(std::string_view s)
{
  char q = 0;
  char prev = 0;                 // the last non-space character seen
  for (std::size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (q != 0) {
      if (q == '\'' && c == '\'' && i + 1 < s.size() && s[i + 1] == '\'') {
        ++i;                                    // '' inside '...'
      } else if (q == '"' && c == '\\') {
        ++i;
      } else if (c == q) {
        q = 0;
      }
      continue;
    }
    if ((c == '\'' || c == '"') &&
        (prev == 0 || prev == ':' || prev == '-' || prev == '[' ||
         prev == '{' || prev == ',')) {
      q = c;
    } else if (c == '#' && (i == 0 || is_space_(s[i - 1]))) {
      return std::string(s.substr(0, i));
    }
    if (!is_space_(c)) { prev = c; }
  }
  return std::string(s);
}

class Parser {
 public:
  Parser(std::vector<Line> lines, std::string* err)
    : _l(std::move(lines)), _err(err) {}

  FlexData document()
  {
    if (_l.empty()) { return FlexData::make_null(); }
    std::size_t i = 0;
    FlexData v = node_(i, _l[0].indent);
    if (!_failed && i < _l.size()) {
      fail_(_l[i], "content outside the document's top-level block");
    }
    return _failed ? FlexData::make_null() : v;
  }

  bool failed() const { return _failed; }

 private:
  std::vector<Line> _l;
  std::string*      _err;
  bool              _failed = false;

  void fail_(const Line& ln, const std::string& what)
  {
    if (_failed) { return; }
    _failed = true;
    if (_err != nullptr) {
      *_err = "line " + std::to_string(ln.number) + ": " + what;
    }
  }

  static bool is_item_(const std::string& t)
  {
    return t == "-" || (t.size() >= 2 && t[0] == '-' && t[1] == ' ');
  }

  // Position of the `:` that ends a mapping key in `t`, or npos: the
  // first one outside quotes and brackets that is followed by a space or
  // ends the line.
  static std::size_t key_colon_(const std::string& t)
  {
    char q = 0;
    int depth = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
      const char c = t[i];
      if (q != 0) {
        if (q == '\'' && c == '\'' && i + 1 < t.size() && t[i + 1] == '\'') {
          ++i;
        } else if (q == '"' && c == '\\') {
          ++i;
        } else if (c == q) {
          q = 0;
        }
        continue;
      }
      if ((c == '\'' || c == '"') && i == 0) {
        q = c;
      } else if (c == '[' || c == '{') {
        ++depth;
      } else if (c == ']' || c == '}') {
        --depth;
      } else if (c == ':' && depth == 0 &&
                 (i + 1 == t.size() || is_space_(t[i + 1]))) {
        return i;
      }
    }
    return std::string::npos;
  }

  FlexData node_(std::size_t& i, int indent)
  {
    if (i >= _l.size()) { return FlexData::make_null(); }
    const Line& ln = _l[i];
    if (is_item_(ln.text)) { return sequence_(i, indent); }
    if (key_colon_(ln.text) != std::string::npos) {
      return mapping_(i, indent);
    }
    // A bare scalar block: the whole document is one value.
    ++i;
    return inline_value_(ln.text, ln);
  }

  FlexData mapping_(std::size_t& i, int indent)
  {
    FlexData out = FlexData::make_object();
    while (!_failed && i < _l.size() && _l[i].indent == indent &&
           !is_item_(_l[i].text)) {
      const Line ln = _l[i];
      const std::size_t c = key_colon_(ln.text);
      if (c == std::string::npos) {
        fail_(ln, "expected `key: value` in a mapping, got '" + ln.text +
                      "'");
        break;
      }
      std::string key = trim_(std::string_view(ln.text).substr(0, c));
      if (!key.empty() && (key[0] == '\'' || key[0] == '"')) {
        std::size_t at = 0;
        FlexData k = quoted_(key, at, ln);
        if (at != key.size()) { fail_(ln, "junk after a quoted key"); }
        key = std::string(k.as_string());
      }
      const std::string rest =
          trim_(std::string_view(ln.text).substr(c + 1));
      ++i;
      FlexData v;
      std::string tag;
      std::string val = rest;
      if (!take_tag_(val, &tag, ln)) { break; }
      if (!val.empty()) {
        v = apply_tag_(inline_value_(val, ln), tag, ln);
      } else if (i < _l.size() && _l[i].indent > indent) {
        v = apply_tag_(node_(i, _l[i].indent), tag, ln);
      } else if (i < _l.size() && _l[i].indent == indent &&
                 is_item_(_l[i].text)) {
        // PyYAML's spelling: a sequence at its key's own indentation.
        v = apply_tag_(sequence_(i, indent), tag, ln);
      } else {
        v = apply_tag_(FlexData::make_null(), tag, ln);
      }
      auto o = out.as_object();
      if (!o.insert(key, std::move(v))) {
        fail_(ln, "duplicate key '" + key + "'");
      }
    }
    if (!_failed && i < _l.size() && _l[i].indent > indent) {
      fail_(_l[i], "unexpected indentation");
    }
    return out;
  }

  FlexData sequence_(std::size_t& i, int indent)
  {
    FlexData out = FlexData::make_array();
    while (!_failed && i < _l.size() && _l[i].indent == indent &&
           is_item_(_l[i].text)) {
      Line& ln = _l[i];
      std::string rest = trim_(std::string_view(ln.text).substr(1));
      std::string tag;
      if (!take_tag_(rest, &tag, ln)) { break; }
      FlexData v;
      if (rest.empty()) {
        ++i;
        if (i < _l.size() && _l[i].indent > indent) {
          v = node_(i, _l[i].indent);
        } else {
          v = FlexData::make_null();
        }
      } else if (key_colon_(rest) != std::string::npos ||
                 is_item_(rest)) {
        // `- key: value` (or `- - x`): the item is a block whose first
        // line starts where `rest` does. Rewrite this line in place and
        // parse the block from it; its continuation lines sit at the
        // same column.
        const std::size_t off = ln.text.find(rest);
        ln.indent += (int)off;
        ln.text = rest;
        v = node_(i, ln.indent);
      } else {
        const Line copy = ln;
        ++i;
        v = inline_value_(rest, copy);
      }
      out.as_array().push_back(apply_tag_(std::move(v), tag, ln));
    }
    return out;
  }

  // Strip a leading `!!tag` from `v`. False (after failing) for one this
  // reader refuses.
  bool take_tag_(std::string& v, std::string* tag, const Line& ln)
  {
    if (v.size() < 2 || v[0] != '!') { return true; }
    std::size_t e = 0;
    while (e < v.size() && !is_space_(v[e])) { ++e; }
    const std::string t = v.substr(0, e);
    static const char* const kOk[] = {
        "!!python/tuple", "!!python/list", "!!seq", "!!map", "!!str",
        "!!int", "!!float", "!!bool", "!!null"};
    bool ok = false;
    for (const char* k : kOk) { ok = ok || t == k; }
    if (!ok) {
      fail_(ln, "tag '" + t + "' is not one this reader takes");
      return false;
    }
    *tag = t;
    v = trim_(std::string_view(v).substr(e));
    return true;
  }

  FlexData apply_tag_(FlexData v, const std::string& tag, const Line& ln)
  {
    if (tag.empty()) { return v; }
    if (tag == "!!python/tuple" || tag == "!!python/list" ||
        tag == "!!seq") {
      if (v.is_null()) { return FlexData::make_array(); }
      if (!v.is_array()) { fail_(ln, tag + " on a non-sequence"); }
      return v;
    }
    if (tag == "!!map") {
      if (v.is_null()) { return FlexData::make_object(); }
      if (!v.is_object()) { fail_(ln, "!!map on a non-mapping"); }
      return v;
    }
    if (tag == "!!str") {
      if (v.is_string()) { return v; }
      if (v.is_null()) { return FlexData::make_string(""); }
      if (v.is_array() || v.is_object()) {
        fail_(ln, "!!str on a collection");
        return v;
      }
      return FlexData::make_string(scalar_text_(v));
    }
    if (tag == "!!int") {
      if (!v.is_int() && !v.is_uint()) { fail_(ln, "!!int on a non-int"); }
      return v;
    }
    if (tag == "!!float") {
      if (v.is_int() || v.is_uint()) {
        return FlexData::make_real(v.as_real());
      }
      if (!v.is_real()) { fail_(ln, "!!float on a non-number"); }
      return v;
    }
    if (tag == "!!bool") {
      if (!v.is_bool()) { fail_(ln, "!!bool on a non-bool"); }
      return v;
    }
    if (tag == "!!null") {
      if (!v.is_null()) { fail_(ln, "!!null on a value"); }
      return v;
    }
    return v;
  }

  static std::string scalar_text_(const FlexData& v)
  {
    if (v.is_bool()) { return v.as_bool() ? "true" : "false"; }
    if (v.is_int()) { return std::to_string(v.as_int()); }
    if (v.is_uint()) { return std::to_string(v.as_uint()); }
    if (v.is_real()) {
      std::ostringstream s;
      s << v.as_real();
      return s.str();
    }
    return std::string(v.as_string());
  }

  // A value on one line: a flow collection, a quoted string or a plain
  // scalar.
  FlexData inline_value_(const std::string& t, const Line& ln)
  {
    if (t.empty()) { return FlexData::make_null(); }
    if (t[0] == '&' || t[0] == '*') {
      fail_(ln, "anchors and aliases are not read");
      return FlexData::make_null();
    }
    if (t[0] == '|' || t[0] == '>') {
      fail_(ln, "block scalars (| and >) are not read");
      return FlexData::make_null();
    }
    std::size_t at = 0;
    FlexData v = flow_(t, at, ln, /*in_flow=*/false);
    while (at < t.size() && is_space_(t[at])) { ++at; }
    if (!_failed && at != t.size()) {
      fail_(ln, "junk after a value: '" + t.substr(at) + "'");
    }
    return v;
  }

  // `in_flow`: inside a [...] or {...}, where ',' ']' '}' end a plain
  // scalar. Outside one, a plain scalar is the rest of the line.
  FlexData flow_(const std::string& t, std::size_t& at, const Line& ln,
                 bool in_flow)
  {
    while (at < t.size() && is_space_(t[at])) { ++at; }
    if (at >= t.size()) { return FlexData::make_null(); }
    const char c = t[at];
    if (c == '[') {
      ++at;
      FlexData a = FlexData::make_array();
      skip_ws_(t, at);
      if (at < t.size() && t[at] == ']') { ++at; return a; }
      while (!_failed) {
        a.as_array().push_back(flow_(t, at, ln, true));
        skip_ws_(t, at);
        if (at < t.size() && t[at] == ',') {
          ++at;
          skip_ws_(t, at);
          if (at < t.size() && t[at] == ']') { ++at; break; }
          continue;
        }
        if (at < t.size() && t[at] == ']') { ++at; break; }
        fail_(ln, "a flow sequence must close on its own line");
        break;
      }
      return a;
    }
    if (c == '{') {
      ++at;
      FlexData o = FlexData::make_object();
      skip_ws_(t, at);
      if (at < t.size() && t[at] == '}') { ++at; return o; }
      while (!_failed) {
        FlexData k = flow_scalar_(t, at, ln, true, /*in_map_key=*/true);
        skip_ws_(t, at);
        if (at >= t.size() || t[at] != ':') {
          fail_(ln, "expected ':' in a flow mapping");
          break;
        }
        ++at;
        FlexData v = flow_(t, at, ln, true);
        if (!o.as_object().insert(scalar_text_(k), std::move(v))) {
          fail_(ln, "duplicate key '" + scalar_text_(k) + "'");
        }
        skip_ws_(t, at);
        if (at < t.size() && t[at] == ',') { ++at; continue; }
        if (at < t.size() && t[at] == '}') { ++at; break; }
        fail_(ln, "a flow mapping must close on its own line");
        break;
      }
      return o;
    }
    return flow_scalar_(t, at, ln, in_flow, false);
  }

  static void skip_ws_(const std::string& t, std::size_t& at)
  {
    while (at < t.size() && is_space_(t[at])) { ++at; }
  }

  FlexData flow_scalar_(const std::string& t, std::size_t& at,
                        const Line& ln, bool in_flow, bool in_map_key)
  {
    skip_ws_(t, at);
    if (at < t.size() && (t[at] == '\'' || t[at] == '"')) {
      return quoted_(t, at, ln);
    }
    // A plain scalar runs to the end of the line, or -- inside a flow
    // collection -- to the next ',', ']' or '}' (and ':' for a key).
    std::size_t e = at;
    while (e < t.size()) {
      const char c = t[e];
      if (in_flow && (c == ',' || c == ']' || c == '}')) { break; }
      if (in_map_key && c == ':') { break; }
      ++e;
    }
    std::string s = trim_(std::string_view(t).substr(at, e - at));
    at = e;
    return resolve_plain_(s);
  }

  FlexData quoted_(const std::string& t, std::size_t& at, const Line& ln)
  {
    const char q = t[at++];
    std::string s;
    while (at < t.size()) {
      const char c = t[at];
      if (q == '\'') {
        if (c == '\'') {
          if (at + 1 < t.size() && t[at + 1] == '\'') {
            s += '\'';
            at += 2;
            continue;
          }
          ++at;
          return FlexData::make_string(s);
        }
        s += c;
        ++at;
        continue;
      }
      if (c == '"') { ++at; return FlexData::make_string(s); }
      if (c == '\\' && at + 1 < t.size()) {
        const char e = t[at + 1];
        switch (e) {
          case 'n':  s += '\n'; break;
          case 't':  s += '\t'; break;
          case '"':  s += '"'; break;
          case '\\': s += '\\'; break;
          case '/':  s += '/'; break;
          case '0':  s += '\0'; break;
          default:
            fail_(ln, std::string("escape \\") + e + " is not read");
            return FlexData::make_null();
        }
        at += 2;
        continue;
      }
      s += c;
      ++at;
    }
    fail_(ln, "a quoted string must close on its own line");
    return FlexData::make_null();
  }

  // YAML 1.1's resolution of a plain scalar, as PyYAML applies it.
  static FlexData resolve_plain_(const std::string& s)
  {
    if (s.empty() || s == "~" || s == "null" || s == "Null" ||
        s == "NULL") {
      return FlexData::make_null();
    }
    static const char* const kTrue[] = {"true", "True", "TRUE", "yes",
                                        "Yes", "YES", "on", "On", "ON"};
    static const char* const kFalse[] = {"false", "False", "FALSE", "no",
                                         "No", "NO", "off", "Off", "OFF"};
    for (const char* k : kTrue) {
      if (s == k) { return FlexData::make_bool(true); }
    }
    for (const char* k : kFalse) {
      if (s == k) { return FlexData::make_bool(false); }
    }
    if (s == ".inf" || s == ".Inf" || s == ".INF" || s == "+.inf") {
      return FlexData::make_real(std::numeric_limits<double>::infinity());
    }
    if (s == "-.inf" || s == "-.Inf" || s == "-.INF") {
      return FlexData::make_real(-std::numeric_limits<double>::infinity());
    }
    if (s == ".nan" || s == ".NaN" || s == ".NAN") {
      return FlexData::make_real(std::numeric_limits<double>::quiet_NaN());
    }
    // An integer: optional sign, digits (underscores allowed, as YAML
    // 1.1 allows them).
    {
      std::string d;
      std::size_t k = 0;
      if (s[0] == '+' || s[0] == '-') { d += s[0]; k = 1; }
      bool digits = k < s.size();
      for (; k < s.size(); ++k) {
        if (s[k] == '_') { continue; }
        if (s[k] < '0' || s[k] > '9') { digits = false; break; }
        d += s[k];
      }
      if (digits) {
        errno = 0;
        char* end = nullptr;
        const long long v = std::strtoll(d.c_str(), &end, 10);
        if (errno == 0 && end != nullptr && *end == '\0') {
          return FlexData::make_int(v);
        }
      }
    }
    // A float needs a '.' or an exponent and nothing else around it.
    if (s.find_first_of(".eE") != std::string::npos) {
      bool ok = true;
      for (const char c : s) {
        ok = ok && ((c >= '0' && c <= '9') || c == '.' || c == 'e' ||
                    c == 'E' || c == '+' || c == '-' || c == '_');
      }
      if (ok) {
        std::string d;
        for (const char c : s) { if (c != '_') { d += c; } }
        errno = 0;
        char* end = nullptr;
        const double v = std::strtod(d.c_str(), &end);
        if (errno == 0 && end != nullptr && *end == '\0' &&
            end != d.c_str()) {
          return FlexData::make_real(v);
        }
      }
    }
    return FlexData::make_string(s);
  }
};

}  // namespace

FlexData
parse(std::string_view text, std::string* err)
{
  std::vector<Line> lines;
  int number = 0;
  bool started = false;
  std::size_t pos = 0;
  while (pos <= text.size()) {
    std::size_t e = text.find('\n', pos);
    if (e == std::string_view::npos) { e = text.size(); }
    const std::string_view raw = text.substr(pos, e - pos);
    pos = e + 1;
    ++number;
    std::size_t ind = 0;
    while (ind < raw.size() && raw[ind] == ' ') { ++ind; }
    if (ind < raw.size() && raw[ind] == '\t') {
      if (err != nullptr) {
        *err = "line " + std::to_string(number) +
               ": a tab in the indentation (YAML forbids it)";
      }
      return FlexData::make_null();
    }
    std::string t = trim_(strip_comment_(raw.substr(ind)));
    if (t.empty()) {
      if (e == text.size()) { break; }
      continue;
    }
    if (ind == 0 && (t == "---" || t.rfind("--- ", 0) == 0)) {
      if (started) {
        if (err != nullptr) {
          *err = "line " + std::to_string(number) +
                 ": a second document (only one is read)";
        }
        return FlexData::make_null();
      }
      if (t == "---") { continue; }
      t = trim_(std::string_view(t).substr(4));
    }
    if (ind == 0 && t == "...") { break; }
    if (ind == 0 && t[0] == '%') { continue; }        // %YAML directive
    started = true;
    lines.push_back(Line{(int)ind, std::move(t), number});
    if (e == text.size()) { break; }
  }
  Parser p(std::move(lines), err);
  FlexData v = p.document();
  return p.failed() ? FlexData::make_null() : v;
}

FlexData
parse_file(const std::string& path, std::string* err)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (err != nullptr) { *err = "cannot read '" + path + "'"; }
    return FlexData::make_null();
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str(), err);
}

}  // namespace vpipe::genai::yaml_subset
