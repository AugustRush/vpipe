// The session's kernel-tuning answers (KernelTuningCache,
// generative-models/shared/tuned-answers.h): kept for the session, read
// by the next tuner to ask the same question -- a model loaded again --
// and by nobody else: another session, another question, or a session
// told not to keep them measures as before.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/kernel-sets/decode-gqa-attn-set.h"
#include "generative-models/shared/kernel-sets/prefill-gqa-attn-set.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/tuned-answers.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;
using vpipe::genai::TuneKey;
using vpipe::genai::find_tuned;
using vpipe::genai::store_tuned;
using vpipe::genai::tuning_cache;

namespace {

MetalCompute*
matrix_cores_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid() || !mc->supports_matrix_cores()) {
    std::printf("[kernel_tuning] no matrix cores here; not exercised\n");
    return nullptr;
  }
  return mc;
}

}  // namespace

// The service itself: on by default, off by the config or the env (the
// env winning either way), its answers integers under whole keys, and
// what MetalCompute's session reaches is the session's own.
TEST(kernel_tuning, the_session_keeps_answers_unless_told_not_to)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  Session sess;
  KernelTuningCache* c = sess.services()->kernel_tuning_cache();
  ASSERT_TRUE(c != nullptr);
  if (c == nullptr) { return; }
  EXPECT_TRUE(sess.kernel_tuning_cache() == c);
  if (MetalCompute* mc = sess.metal_compute(); mc != nullptr) {
    EXPECT_TRUE(tuning_cache(mc) == c);
  }

  const TuneKey k = TuneKey("test/1")("N", 3072)("route", "requant")
      ("bf16", true)("zero", 0);
  EXPECT_TRUE(k.str() == "test/1 N=3072 route=requant bf16=1 zero=0");
  std::vector<int> v;
  EXPECT_TRUE(!find_tuned(c, k, v));
  store_tuned(c, k, std::vector<int>{2, 0, -1});
  EXPECT_TRUE(find_tuned(c, k, v));
  EXPECT_TRUE(v == (std::vector<int>{2, 0, -1}));
  // One integer asked of three is not an answer to that question.
  int one = 7;
  EXPECT_TRUE(!find_tuned(c, k, one) && one == 7);
  store_tuned(c, TuneKey("test/1 single"), 5);
  EXPECT_TRUE(find_tuned(c, TuneKey("test/1 single"), one) && one == 5);
  // Text that does not read as integers is no answer: measured again,
  // it is filed over.
  c->store("test/1 bad", "2,x");
  EXPECT_TRUE(!find_tuned(c, TuneKey("test/1 bad"), v));
  // A key is compared whole.
  EXPECT_TRUE(!find_tuned(c, TuneKey("test/1"), v));
  const KernelTuningCache::Stats st = c->stats();
  EXPECT_TRUE(st.entries == 3);
  EXPECT_TRUE(st.hits >= 3 && st.misses >= 2);

  // Another session keeps its own.
  Session other;
  EXPECT_TRUE(other.kernel_tuning_cache() != nullptr);
  EXPECT_TRUE(other.kernel_tuning_cache() != c);
  EXPECT_TRUE(!find_tuned(other.kernel_tuning_cache(), k, v));

  Session off(R"({"kernel_tuning": {"cache": false}})");
  EXPECT_TRUE(off.kernel_tuning_cache() == nullptr);
  EXPECT_TRUE(off.services()->kernel_tuning_cache() == nullptr);
  if (MetalCompute* mc = off.metal_compute(); mc != nullptr) {
    EXPECT_TRUE(tuning_cache(mc) == nullptr);
  }
  ::setenv("VPIPE_TUNING_CACHE", "0", 1);
  Session env_off;
  EXPECT_TRUE(env_off.kernel_tuning_cache() == nullptr);
  ::setenv("VPIPE_TUNING_CACHE", "1", 1);
  Session env_on(R"({"kernel_tuning": {"cache": false}})");
  EXPECT_TRUE(env_on.kernel_tuning_cache() != nullptr);
  ::unsetenv("VPIPE_TUNING_CACHE");
}

// THE i8 SPLIT, which every picture model tunes at its second step: a
// second context in the session -- the model loaded again -- runs the
// first one's answer from its very first call and records nothing to
// measure, at a similar size too (the same row bucket). Another element
// type, another session, or a session keeping none measures again.
TEST(kernel_tuning, i8_split_answers_are_kept_for_the_session)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  Session sess;
  MetalCompute* mc = matrix_cores_(sess);
  if (mc == nullptr) { return; }
  const int N = 512, K = 2048;
  auto gemm = [&](MetalCompute* m, genai::I8GemmContext& ctx, int M) {
    SharedBuffer x = m->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer w = m->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer y = m->make_shared_buffer((std::size_t)M * N * 2);
    bool ok = false;
    {
      CommandStream st = m->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ok = ctx.gemm(e, x, 0, w, y, 0, M, N, K);
      }
      st.commit().wait();
    }
    return ok;
  };

  genai::I8GemmContext a(mc, /*want=*/true, /*bf16=*/false);
  if (!a.enabled() || !a.split_available()) {
    std::printf("[kernel_tuning] no i8 split kernels; not exercised\n");
    return;
  }
  ASSERT_TRUE(gemm(mc, a, 1536));
  EXPECT_TRUE(a.pending_for_test() == 1);
  a.tune_pending(mc);
  EXPECT_TRUE(a.measured_for_test() == 1);
  EXPECT_TRUE(a.tuned_for_test(1536, N, K));
  const int pick = a.plan_for_test(1536, N, K);

  // The model loaded again, at a similar size: 2048 rows share 1536's
  // bucket, so its answer.
  genai::I8GemmContext b(mc, /*want=*/true, /*bf16=*/false);
  EXPECT_TRUE(gemm(mc, b, 2048));
  EXPECT_TRUE(b.pending_for_test() == 0);
  EXPECT_TRUE(b.tuned_for_test(2048, N, K));
  EXPECT_TRUE(b.plan_for_test(2048, N, K) == pick);
  b.tune_pending(mc);
  EXPECT_TRUE(b.measured_for_test() == 0);
  // ...but not past it: 4096 rows is another bucket, another question.
  EXPECT_TRUE(gemm(mc, b, 4096));
  EXPECT_TRUE(b.pending_for_test() == 1);

  // Another element type is another question.
  genai::I8GemmContext c(mc, /*want=*/true, /*bf16=*/true);
  EXPECT_TRUE(gemm(mc, c, 1536));
  EXPECT_TRUE(c.pending_for_test() == 1);

  // Another session has none of it; one told to keep none never will.
  Session other;
  genai::I8GemmContext d(other.metal_compute(), true, false);
  EXPECT_TRUE(gemm(other.metal_compute(), d, 1536));
  EXPECT_TRUE(d.pending_for_test() == 1);
  Session off(R"({"kernel_tuning": {"cache": false}})");
  for (int i = 0; i < 2; ++i) {
    genai::I8GemmContext e(off.metal_compute(), true, false);
    EXPECT_TRUE(gemm(off.metal_compute(), e, 1536));
    EXPECT_TRUE(e.pending_for_test() == 1);
    e.tune_pending(off.metal_compute());
    EXPECT_TRUE(e.measured_for_test() == 1);
  }
  std::printf("[kernel_tuning] i8 split S=%d at %dx%dx%d, read by the next "
              "context\n", pick, 1536, N, K);
}

// THE DENSE SPLIT-K, which a DiT tunes through its own closure: a caller
// that names what the closure encodes has its answer kept, and the next
// instance asking the same never runs it. One that names nothing keeps
// its answer to itself, as before.
TEST(kernel_tuning, a_named_split_is_read_by_the_next_instance)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  Session sess;
  MetalCompute* mc = matrix_cores_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_dense = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  const int K = 12288, N = 512, M = 1024;
  int runs = 0;
  auto run = [&](ComputeEncoder&) { ++runs; };

  genai::MmaSplitK a;
  a.load(mc, lib_dense, lib_elt);
  if (!a.enabled) {
    std::printf("[kernel_tuning] no split-K kernels; not exercised\n");
    return;
  }
  const int pick = a.tune(mc, K, N, M, run, "kernel_tuning test");
  if (runs == 0) {
    std::printf("[kernel_tuning] one split candidate at %dx%dx%d; not "
                "exercised\n", M, N, K);
    return;
  }

  genai::MmaSplitK b;
  b.load(mc, lib_dense, lib_elt);
  runs = 0;
  EXPECT_TRUE(b.tune(mc, K, N, M, run, "kernel_tuning test") == pick);
  EXPECT_TRUE(runs == 0);
  // Another caller's closure is another question.
  genai::MmaSplitK c;
  c.load(mc, lib_dense, lib_elt);
  runs = 0;
  (void)c.tune(mc, K, N, M, run, "kernel_tuning other");
  EXPECT_TRUE(runs > 0);
  // Unnamed: this instance's alone, measured.
  genai::MmaSplitK d;
  d.load(mc, lib_dense, lib_elt);
  runs = 0;
  (void)d.tune(mc, K, N, M, run);
  EXPECT_TRUE(runs > 0);
  // ...and filed nowhere, so the next unnamed one measures too.
  genai::MmaSplitK e;
  e.load(mc, lib_dense, lib_elt);
  runs = 0;
  (void)e.tune(mc, K, N, M, run);
  EXPECT_TRUE(runs > 0);
}

// THE LM ATTENTION SETS, which a chat model tunes at load: loaded again in
// the session -- a helper unloaded for another model's memory -- they read
// the first load's members and splits and measure nothing.
TEST(kernel_tuning, the_attention_sets_read_the_session_s_picks)
{
  ::unsetenv("VPIPE_TUNING_CACHE");
  ::unsetenv("VPIPE_QWEN_GQA_AUTOTUNE");
  ::unsetenv("VPIPE_QWEN_PREFILL_AUTOTUNE");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ComputeLibrary lib_sdpa = mc->load_library("sdpa");
  ComputeLibrary lib_attn = mc->load_library("attn_steel");
  ComputeLibrary lib_mma = mc->load_library("sdpa_mma");
  const bool m5 = mc->supports_matrix_cores();
  const genai::DecodeGqaAttnSet::Dims dd{256, 16, 4};

  genai::DecodeGqaAttnSet d1, d2;
  if (!d1.load(lib_sdpa, false) || !d2.load(lib_sdpa, false)) {
    std::printf("[kernel_tuning] no decode set; not exercised\n");
    return;
  }
  genai::TuningReport r1, r2;
  d1.prepare(mc, dd, r1);
  d2.prepare(mc, dd, r2);
  std::printf("[kernel_tuning] decode: %s | again: %s\n",
              r1.summary().c_str(), r2.summary().c_str());
  EXPECT_TRUE(r1.summary().find("the session's") == std::string::npos);
  EXPECT_TRUE(r2.summary().find("the session's") != std::string::npos);
  for (int pos : {100, 8000, 30000}) {
    EXPECT_TRUE(d1.uses_vec2(pos) == d2.uses_vec2(pos));
    EXPECT_TRUE(d1.split_for(pos) == d2.split_for(pos));
  }

  genai::PrefillGqaAttnSet p1, p2;
  if (!p1.load(lib_sdpa, &lib_attn, &lib_mma, m5) ||
      !p2.load(lib_sdpa, &lib_attn, &lib_mma, m5)) {
    std::printf("[kernel_tuning] no prefill set; not exercised\n");
    return;
  }
  genai::TuningReport q1, q2;
  p1.prepare(mc, {256, 16, 4}, q1);
  p2.prepare(mc, {256, 16, 4}, q2);
  std::printf("[kernel_tuning] prefill: %s | again: %s\n",
              q1.summary().c_str(), q2.summary().c_str());
  EXPECT_TRUE(q2.summary().find("the session's") != std::string::npos);
  for (int n : {600, 2000, 3000, 7000}) {
    EXPECT_TRUE(std::string(p1.kernel_name(n)) ==
                std::string(p2.kernel_name(n)));
  }
}
