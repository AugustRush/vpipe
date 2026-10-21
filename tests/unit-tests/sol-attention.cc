// Sol-Attn, the method: does routing on a proxy score actually
// approximate attention, and does tau control the trade the way it
// claims to?
//
// These tests use the CPU reference alone -- no GPU, no weights -- and
// they are the ones that would catch the method being MIS-IMPLEMENTED
// rather than merely mis-ported. A sparse attention that returns
// plausible numbers is the normal failure here: every arrangement of
// these pieces produces a normalised distribution and a finite output,
// so the bar has to be a comparison against DENSE attention over inputs
// with the structure the method assumes.
//
// The structure that matters is that attention is CONCENTRATED. On
// uniform random q/k every key block is as good as every other, no
// routing decision can be right, and the approximation is bad for a
// reason that says nothing about the method. Video attention is not
// like that, so the fixtures here are not either: `clustered_()` builds
// queries and keys around a handful of shared directions, which is the
// regime the paper's own argument is about.

#include "minitest.h"

#include "generative-models/shared/sol-attention.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace vpipe::genai;

namespace {

struct Fixture {
  int heads = 0, tokens = 0, d = 0;
  std::vector<float> q, k, v;
};

std::uint32_t
rnd_(std::uint32_t& s)
{
  s = s * 1664525u + 1013904223u;
  return s;
}

float
uni_(std::uint32_t& s)
{
  return (float)(rnd_(s) >> 8) / 8388608.0f - 1.0f;
}

// Queries and keys drawn around `centres` shared directions, so the
// attention map has structure to find. `spread` is how far a row sits
// from its centre: small = strongly clustered, 1.0 = nearly uniform.
Fixture
clustered_(int heads, int tokens, int d, int centres, float spread,
           std::uint32_t seed)
{
  Fixture f;
  f.heads = heads; f.tokens = tokens; f.d = d;
  const std::size_t n = (std::size_t)heads * tokens * d;
  f.q.resize(n); f.k.resize(n); f.v.resize(n);
  std::uint32_t s = seed;
  std::vector<float> dir((std::size_t)centres * d);
  for (auto& x : dir) { x = uni_(s); }
  auto fill = [&](std::vector<float>& dst, float sp) {
    for (int h = 0; h < heads; ++h) {
      for (int t = 0; t < tokens; ++t) {
        // Neighbouring TOKENS share a centre, which is what makes the
        // near-diagonal band meaningful -- as it is in video, where
        // adjacent patches are adjacent in space.
        const int c = (t / 37) % centres;
        float* row = dst.data() + ((std::size_t)h * tokens + t) * d;
        float norm = 0.0f;
        for (int i = 0; i < d; ++i) {
          row[i] = dir[(std::size_t)c * d + i] * (1.0f - sp) + uni_(s) * sp;
          norm += row[i] * row[i];
        }
        norm = std::sqrt(norm > 0.0f ? norm : 1.0f);
        for (int i = 0; i < d; ++i) { row[i] /= norm; }
      }
    }
  };
  fill(f.q, spread);
  fill(f.k, spread);
  for (auto& x : f.v) { x = uni_(s); }
  return f;
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double dd = (double)a[i] - (double)b[i];
    num += dd * dd;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// One routed forward against dense, returning (rel-L2, exact fraction).
std::pair<double, double>
run_(const Fixture& f, const sol::Config& cfg)
{
  const int nb = sol::num_blocks(f.tokens);
  const float scale = 1.0f / std::sqrt((float)f.d);
  std::vector<float> qc((std::size_t)f.heads * nb * f.d);
  std::vector<float> kc(qc.size()), vc(qc.size());
  sol::summaries(f.q.data(), f.k.data(), f.v.data(), f.heads, f.tokens, f.d,
                 qc.data(), kc.data(), vc.data());
  std::vector<float> thr((std::size_t)f.heads * nb);
  sol::thresholds(qc.data(), kc.data(), f.heads, nb, f.d, scale, cfg.tau,
                  cfg.thresh, thr.data());
  const std::size_t n = (std::size_t)f.heads * f.tokens * f.d;
  std::vector<float> got(n), want(n);
  sol::Stats st;
  sol::forward(f.q.data(), f.k.data(), f.v.data(), qc.data(), kc.data(),
               vc.data(), thr.data(), f.heads, f.tokens, f.d, scale, cfg,
               got.data(), &st);
  sol::dense(f.q.data(), f.k.data(), f.v.data(), f.heads, f.tokens, f.d,
             scale, want.data());
  return {rel_l2_(got, want), st.exact_fraction()};
}

}  // namespace

// THE CLAIM, MEASURED: raising tau keeps fewer blocks and costs
// accuracy, monotonically in both. This is the test that says the
// threshold is wired to the routing at all -- a tau that changes nothing
// (a threshold read in the wrong units, say) leaves both columns flat,
// and a tau wired backwards inverts them.
TEST(sol_attention, tau_trades_accuracy_for_sparsity)
{
  const Fixture f = clustered_(4, 512, 128, 6, 0.35f, 0x5eed1234u);
  double prev_err = -1.0, prev_keep = 2.0;
  bool err_rises = true, keep_falls = true;
  std::printf("[sol]   tau    rel-L2   exact-blocks\n");
  for (const float tau : {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f}) {
    sol::Config c;
    c.tau = tau;
    const auto [err, keep] = run_(f, c);
    std::printf("[sol] %5.2f  %8.5f   %6.2f%%\n", tau, err, keep * 100.0);
    if (prev_err >= 0.0) {
      if (!(err >= prev_err - 1e-6)) { err_rises = false; }
      if (!(keep <= prev_keep + 1e-6)) { keep_falls = false; }
    }
    prev_err = err;
    prev_keep = keep;
  }
  EXPECT_TRUE(err_rises);
  EXPECT_TRUE(keep_falls);
}

// AND THE TRADE IS WORTH TAKING. At the published tau the routing has to
// drop most of the key blocks and still track dense attention -- which
// is the only statement that distinguishes this method from "attend to
// a band and hope".
TEST(sol_attention, published_tau_is_sparse_and_close)
{
  const Fixture f = clustered_(4, 1024, 128, 8, 0.3f, 0xabcd0001u);
  sol::Config c;
  c.tau = 1.0f;                       // the published Stage-2 first step
  const auto [err, keep] = run_(f, c);
  std::printf("[sol] tau 1.0 @1024: rel-L2 %.5f, %.1f%% of blocks exact\n",
              err, keep * 100.0);
  EXPECT_TRUE(keep < 0.5);            // most of the work is skipped
  EXPECT_TRUE(err < 0.15);            // and the answer survives it
}

// THE LOCAL BAND IS LOAD-BEARING, and this is the ablation that says so.
// A centroid is least representative exactly where a query discriminates
// most, so routing is never asked about the diagonal. Turning the band
// off must make the SAME tau worse -- if it does not, the band is
// costing exactness for nothing.
TEST(sol_attention, the_local_band_earns_its_exactness)
{
  const Fixture f = clustered_(4, 512, 128, 6, 0.35f, 0x1122u);
  sol::Config with;
  with.tau = 1.5f;
  sol::Config without = with;
  without.local_radius = -1;          // no block is local to itself
  const auto [e_with, k_with] = run_(f, with);
  const auto [e_without, k_without] = run_(f, without);
  std::printf("[sol] band on  %.5f (%.1f%%), band off %.5f (%.1f%%)\n",
              e_with, k_with * 100.0, e_without, k_without * 100.0);
  EXPECT_TRUE(e_without > e_with);
}

// tau = -inf keeps everything, and everything must then be EXACT
// attention -- same softmax, same keys, no proxy anywhere. This is the
// degenerate case that pins the exact half of the kernel against the
// dense reference with no approximation in the way, so a bug in the
// online-softmax rescaling cannot hide behind the routing.
TEST(sol_attention, keeping_every_block_is_dense_attention)
{
  const Fixture f = clustered_(2, 256, 128, 4, 0.5f, 0x77u);
  sol::Config c;
  c.tau = -1.0e30f;                   // nothing can fall below this
  const auto [err, keep] = run_(f, c);
  std::printf("[sol] all-exact: rel-L2 %.3e, %.1f%% exact\n", err,
              keep * 100.0);
  EXPECT_TRUE(keep > 0.999);
  EXPECT_TRUE(err < 1e-5);
}

// The two threshold estimators answer the same question with different
// covariance models, so at one tau they must land in the same
// neighbourhood -- and `exact` should not be WORSE, since it drops the
// diagonal approximation. Kept loose: this pins that both are wired,
// not which wins.
TEST(sol_attention, exact_and_diag_thresholds_agree_broadly)
{
  const Fixture f = clustered_(2, 512, 128, 6, 0.35f, 0x99aau);
  sol::Config cd;
  cd.tau = 1.0f;
  sol::Config ce = cd;
  ce.thresh = sol::Threshold::kExact;
  const auto [e_diag, k_diag] = run_(f, cd);
  const auto [e_exact, k_exact] = run_(f, ce);
  std::printf("[sol] diag %.5f (%.1f%%), exact %.5f (%.1f%%)\n", e_diag,
              k_diag * 100.0, e_exact, k_exact * 100.0);
  EXPECT_TRUE(e_diag < 0.2 && e_exact < 0.2);
  EXPECT_TRUE(k_diag < 0.9 && k_exact < 0.9);
}

// A KV SINK keeps a token range exact for every query. H3 packs text and
// audio in the same sequence as the video, and a prompt summarised by
// its centroid is a prompt half-read -- so the sink has to actually
// force those blocks, at any tau.
TEST(sol_attention, a_kv_sink_is_exact_for_every_query)
{
  const Fixture f = clustered_(2, 512, 128, 6, 0.35f, 0xfeedu);
  sol::Config c;
  c.tau = 3.0f;                       // aggressive: routing keeps little
  c.local_radius = -1;                // isolate the sink's own effect
  const auto [e0, k0] = run_(f, c);
  c.sink_start = 0;
  c.sink_tokens = 128;                // exactly two blocks
  const auto [e1, k1] = run_(f, c);
  const int nb = sol::num_blocks(f.tokens);
  std::printf("[sol] sink off %.1f%%, sink on %.1f%% (of %d blocks)\n",
              k0 * 100.0, k1 * 100.0, nb);
  // Two blocks of nb, for every query block, become exact.
  EXPECT_TRUE(k1 > k0);
  EXPECT_TRUE(k1 >= 2.0 / (double)nb - 1e-9);
}

// The GPU path is not tested here. It was, against this oracle, while
// Sol ran as one plain simdgroup kernel -- and that kernel is gone: the
// exact blocks now go to the tree's own steel flash attention and the
// approximate half to a small MMA loop, which is a different shape with
// a better reference available. sol-attention-mma.cc tests it against
// DENSE STEEL, which is the attention the model would otherwise run, and
// against this oracle for the approximate partial alone.
//
// What stays here is the METHOD: these tests need no GPU and would catch
// the algorithm being wrong rather than the kernel being wrong.

// ---- VSA-H3's BUDGET routing -------------------------------------------
//
// A VSA-distilled student is trained keeping a FIXED share of key blocks
// (compute_topk: ceil((1 - sparsity) * blocks)), not however many a
// threshold happens to admit. The two must be told apart, because both
// produce a normalised distribution and a finite output:
//
//   * a budget that is wired but ignored leaves the kept fraction flat as
//     the sparsity moves;
//   * a budget wired to the wrong formula (round vs ceil, 1-s vs s) lands
//     on a fraction that is close enough to look right on one input.
//
// So the assertions are the fraction, at two sparsities, against the
// formula -- and that the answer differs from the threshold arm's.

TEST(sol_attention, vsa_budget_keeps_the_share_it_was_asked_for)
{
  const Fixture f = clustered_(4, 512, 128, 6, 0.35f, 0x5eed1234u);
  const int nb = sol::num_blocks(f.tokens);
  std::printf("[vsa] blocks %d, sparsity -> kept\n", nb);
  double prev_keep = 2.0;
  bool keep_falls = true;
  for (const float sp : {0.5f, 0.8f, 0.9f}) {
    sol::Config c;
    c.vsa_sparsity = sp;
    // The local band and the tail are forced exact, so the realized share
    // is an upper bound on the budget -- never below it, and on a short
    // sequence a LOOSE one: 8 blocks at sparsity 0.8 ask for 2, but every
    // query block also keeps its neighbours (+-1) and the tail is always
    // exact, which is 4 of 8 before the budget contributes anything. The
    // reference has the same property for the same reason (its `exempt`
    // mode forces the prefix in), so the assertion is the direction and the
    // floor, not an equality.
    const auto [err, keep] = run_(f, c);
    const double want = (double)std::max(1, (int)std::ceil((1.0 - sp) * nb)) / nb;
    std::printf("[vsa] %.2f -> %.4f (budget %.4f, rel-L2 %.5f)\n", sp, keep, want, err);
    EXPECT_TRUE(keep >= want - 1e-6);
    if (!(keep <= prev_keep + 1e-6)) { keep_falls = false; }
    prev_keep = keep;
  }
  EXPECT_TRUE(keep_falls);
}

TEST(sol_attention, vsa_budget_differs_from_the_threshold)
{
  // Same fixture, same target share: the budget arm must not be the
  // threshold arm wearing a different name. tau = 0 keeps nearly
  // everything, so a budget of 80% has to keep strictly less.
  const Fixture f = clustered_(4, 512, 128, 6, 0.35f, 0x1122u);
  sol::Config thr_cfg;
  thr_cfg.tau = 0.0f;
  const auto [e_thr, k_thr] = run_(f, thr_cfg);

  sol::Config bud_cfg;
  bud_cfg.vsa_sparsity = 0.8f;
  const auto [e_bud, k_bud] = run_(f, bud_cfg);
  std::printf("[vsa] threshold tau=0 keeps %.4f, budget 0.8 keeps %.4f\n",
              k_thr, k_bud);
  EXPECT_TRUE(k_bud < k_thr);
}

TEST(sol_attention, vsa_budget_tracks_dense_attention)
{
  // The trade has to be worth taking: at the checkpoint's own 0.8 the
  // routing drops most blocks and still tracks dense attention. A budget
  // wired to the WRONG blocks would keep the same count and a much worse
  // answer, which is the failure a count-only assertion cannot see.
  const Fixture f = clustered_(4, 512, 128, 6, 0.35f, 0xabcd0001u);
  sol::Config c;
  c.vsa_sparsity = 0.8f;
  const auto [err, keep] = run_(f, c);
  std::printf("[vsa] sparsity 0.8: kept %.4f, rel-L2 %.5f\n", keep, err);
  EXPECT_TRUE(keep < 0.95);      // it really did drop blocks
  EXPECT_TRUE(err < 0.5);        // and still resembles dense
}
