// block-slots.cc -- the slot pair every streamed DiT refills
// (generative-models/shared/block-slots.h), without a model.
//
// A refill places BYTES into a slot whose layout -- buffer sizes, weight
// widths, derived products -- was decided by the block that BUILT it. A
// mixed pack (model-quantize `mixed`: whole blocks at w8, the rest at w4)
// breaks that assumption, and the model reads a block at the slot's width
// rather than its own: no single-width checkpoint can show it. Ops::
// same_layout lets a model say "this block is not laid out like the
// slot", and the slot is then BUILT for it. Here the "layout" is a toy
// width that changes with the block index, so the property is checked
// on its own: every block acquired carries its own width, with the read
// ahead on and off.

#include "minitest.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"
#include "generative-models/shared/block-slots.h"

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

using namespace vpipe;
using vpipe::genai::BlockSlots;
using vpipe::metal_compute::MetalCompute;
using vpipe::metal_compute::SharedBuffer;

namespace {

struct ToyBlock {
  int index = -1;     // the block whose bytes the slot holds
  int width = 0;      // its layout, decided when the slot is BUILT
  SharedBuffer buf;   // sized by the width, as a pack's codes are
};

// Two runs of w8 among w4, so both slots cross widths in both directions.
int
toy_width_(int i)
{
  return (i % 5 == 1 || i % 5 == 2) ? 8 : 4;
}

struct ToyRun {
  int mismatches = 0;
  int relaid = 0;
  int built = 0;
};

ToyRun
run_toy_(MetalCompute* mc, bool with_same_layout, bool prefetch, int n)
{
  ToyRun r;
  BlockSlots<ToyBlock> slots;
  BlockSlots<ToyBlock>::Ops ops;
  // No raw tensors: a "refill" is post_refill stamping the block's
  // index into whatever layout the slot already has -- exactly what a
  // byte-placing refill does to a block of another width.
  ops.each = [](int, ToyBlock&, const BlockSlots<ToyBlock>::TensorFn&) {};
  ops.rebuild_one = [](const std::string&, vpipe::genai::Placement) {
    return SharedBuffer{};
  };
  ops.build = [&r, mc](int i, ToyBlock& b) {
    ++r.built;
    b.index = i;
    b.width = toy_width_(i);
    b.buf = mc->make_shared_buffer((std::size_t)b.width * 4096);
    return !b.buf.empty();
  };
  ops.clone = [mc](const ToyBlock& s, ToyBlock& d, bool) {
    d.index = s.index;
    d.width = s.width;
    d.buf = mc->make_shared_buffer(s.buf.byte_size());
    return !d.buf.empty();
  };
  ops.bytes = [](const ToyBlock& b) { return b.buf.byte_size(); };
  ops.empty = [](const ToyBlock& b) { return b.buf.empty(); };
  ops.post_refill = [](int i, ToyBlock& b) {
    b.index = i;
    return true;
  };
  if (with_same_layout) {
    ops.same_layout = [](int i, const ToyBlock& slot) {
      return slot.width == toy_width_(i);
    };
  }
  slots.configure(mc, std::move(ops), "toy", nullptr);
  slots.begin_forward();
  for (int i = 0; i < n; ++i) {
    const ToyBlock* b = slots.acquire(i);
    if (b == nullptr || b->index != i || b->width != toy_width_(i) ||
        b->buf.byte_size() != (std::size_t)toy_width_(i) * 4096) {
      ++r.mismatches;
    }
    if (prefetch && i + 1 < n) { slots.prefetch(i + 1); }
  }
  slots.join();
  r.relaid = slots.relaid();
  return r;
}

}  // namespace

TEST(block_slots, a_block_of_another_layout_is_built_into_its_slot)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const int n = 20;
  for (bool prefetch : {false, true}) {
    const ToyRun with = run_toy_(mc, true, prefetch, n);
    const ToyRun without = run_toy_(mc, false, prefetch, n);
    std::printf("[block_slots] prefetch %s: same_layout -> %d wrong, %d "
                "relaid, %d builds; without -> %d wrong\n",
                prefetch ? "on" : "off", with.mismatches, with.relaid,
                with.built, without.mismatches);
    EXPECT_TRUE(with.mismatches == 0);
    // It engaged: slots crossed widths and were rebuilt for it.
    EXPECT_TRUE(with.relaid > 0);
    // And the property is a real one -- refilled by bytes alone, blocks
    // of the other width come out at the slot's.
    EXPECT_TRUE(without.mismatches > 0);
    EXPECT_TRUE(without.relaid == 0);
  }
}

// A STACK RESIDENCY KEPT WHOLE gives its slot pair back -- two blocks a
// preload never held, and on a roomy box the only bytes stream-then-
// retain holds beyond a preload -- and a block given back later rebuilds
// the pair at its next acquire. A stack with any block still streamed
// keeps it.
TEST(block_slots, a_saturated_stack_gives_its_slots_back)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  BlockSlots<ToyBlock> slots;
  BlockSlots<ToyBlock>::Ops ops;
  ops.each = [](int, ToyBlock&, const BlockSlots<ToyBlock>::TensorFn&) {};
  ops.rebuild_one = [](const std::string&, vpipe::genai::Placement) {
    return SharedBuffer{};
  };
  ops.build = [mc](int i, ToyBlock& b) {
    b.index = i;
    b.width = 4;
    b.buf = mc->make_shared_buffer(4 * 4096);
    return !b.buf.empty();
  };
  ops.clone = [mc](const ToyBlock& s, ToyBlock& d, bool) {
    d.index = s.index;
    d.width = s.width;
    d.buf = mc->make_shared_buffer(s.buf.byte_size());
    return !d.buf.empty();
  };
  ops.bytes = [](const ToyBlock& b) { return b.buf.byte_size(); };
  ops.empty = [](const ToyBlock& b) { return b.buf.empty(); };
  ops.post_refill = [](int i, ToyBlock& b) {
    b.index = i;
    return true;
  };
  slots.configure(mc, std::move(ops), "toy", nullptr);
  const int n = 6;
  std::vector<ToyBlock> resident((std::size_t)n);
  // Pass one: every block read through the slots and kept.
  slots.begin_forward();
  for (int i = 0; i < n; ++i) {
    const ToyBlock* b = slots.acquire(i);
    if (i + 1 < n) { slots.prefetch(i + 1); }
    if (b != nullptr) { (void)slots.promote_into(resident[(std::size_t)i]); }
  }
  slots.join();
  const std::size_t held = slots.held_bytes();
  // One block not kept: the pair stays.
  ToyBlock out = std::move(resident[2]);
  resident[2] = ToyBlock{};
  const std::size_t kept_one_short = slots.release_if_saturated(resident);
  const std::size_t still = slots.held_bytes();
  resident[2] = std::move(out);
  // Every block kept: the pair goes.
  const std::size_t freed = slots.release_if_saturated(resident);
  const std::size_t after = slots.held_bytes();
  // A block given back reads through a rebuilt pair.
  resident[4] = ToyBlock{};
  slots.begin_forward();
  const ToyBlock* again = slots.acquire(4);
  const bool rebuilt = again != nullptr && again->index == 4 &&
                       slots.held_bytes() > 0;
  std::printf("[block_slots] slots %zu B; one block short -> released %zu, "
              "held %zu; all kept -> released %zu, held %zu; after an "
              "eviction %s\n", held, kept_one_short, still, freed, after,
              rebuilt ? "rebuilt" : "NOT rebuilt");
  EXPECT_TRUE(held > 0 && kept_one_short == 0 && still == held);
  EXPECT_TRUE(freed == held && after == 0);
  EXPECT_TRUE(rebuilt);
}
