// residency-reserve.h -- the residency reserve a streamed-hold test can
// afford on the box it runs on.
//
// The promotion arms ("promoted", "held by promotion", "no slots") only
// test something when residency actually KEEPS blocks, and admission
// leaves the reserve free on top of a block and its hysteresis. A fixed
// figure sized for a large box admits nothing on a 16 GB one -- 8 GB
// there kept 0 blocks, and every promotion check then compared a
// streamed run with itself. Capped at a sixteenth of RAM (1 GB on 16 GB,
// 4 on 64, the figure asked for on 256).
#pragma once

#include "stages/model-memory.h"

#include <algorithm>
#include <cstddef>

inline std::size_t
test_residency_reserve(std::size_t want)
{
  const std::size_t ram = vpipe::model_memory::phys_ram();
  return ram > 0 ? std::min(want, ram / 16) : want;
}
