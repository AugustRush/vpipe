#ifndef VPIPE_GENERATIVE_MODELS_SHARED_YAML_SUBSET_H
#define VPIPE_GENERATIVE_MODELS_SHARED_YAML_SUBSET_H

// A reader for the YAML that model training repos write their configs
// in -- MSST's audio-separation checkpoints ship one YAML per .ckpt and
// nothing else -- read into a FlexData tree. Not a YAML implementation.
//
// TAKEN, which is every shape those files use:
//   * block mappings (`key: value`, `key:` + an indented block), and
//     block sequences (`- item`), including PyYAML's own spelling where
//     a sequence sits at its key's indentation and `- key: value` items;
//   * flow sequences and mappings on one line (`['vocals', 'other']`);
//   * plain, single- and double-quoted scalars, resolved as YAML 1.1
//     does for null / bool / int / float, and `#` comments;
//   * the tags `!!python/tuple` (read as a sequence -- MSST's
//     `freqs_per_bands`), `!!python/list`, `!!seq`, `!!map`, `!!str`,
//     `!!int`, `!!float`, `!!bool`, `!!null`.
//
// REFUSED, each by name rather than misread: anchors and aliases, block
// scalars (`|`, `>`), multi-line flow collections, multiple documents,
// tabs in indentation, and any other tag (`!!python/object` would be
// code in PyYAML; here it is an error).

#include "common/flex-data.h"

#include <string>
#include <string_view>

namespace vpipe::genai::yaml_subset {

// The document as a FlexData; null with `err` set on anything outside
// the subset, naming the line.
FlexData parse(std::string_view text, std::string* err);

// parse() of a file's contents; null with `err` set when it cannot be
// read.
FlexData parse_file(const std::string& path, std::string* err);

}  // namespace vpipe::genai::yaml_subset

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_YAML_SUBSET_H
