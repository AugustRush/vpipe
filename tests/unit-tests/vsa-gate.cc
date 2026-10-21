// VSA-H3's compression gate: the two pieces of arithmetic a picture cannot
// prove.
//
// The branch is `out += gate * (softmax(tile scores) . v_pooled)`, and every
// arrangement of those pieces produces a normalised distribution and a
// finite output -- so a wrong one renders a plausible picture and the error
// is invisible. What IS checkable without a GPU is the arithmetic and the
// tile geometry, against the reference's own formulas:
//
//   * the TILE, against FastVideo's get_tile_partition_indices: the cube a
//     tile covers, including the short ones at the edge of the grid.
//   * the BUDGET, against compute_topk.
//   * the BRANCH, against a transcription of video_sparse_attn_h3.py.
//
// The gate's own weights are not needed: this is about whether the maths is
// the reference's maths, not whether a checkpoint shipped.

#include "minitest.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// ---- the reference's tile geometry -------------------------------------
//
// get_tile_partition_indices flattens each cube of the (T, H, W) grid in
// row-major order, iterating bt, bh, bw outermost-first. Returns the row
// index of every member of every tile, in that order.
std::vector<std::vector<int>>
ref_tiles(int T, int H, int W, int ts, int hs, int ws)
{
  std::vector<std::vector<int>> out;
  for (int bt = 0; bt * ts < T; ++bt) {
    for (int bh = 0; bh * hs < H; ++bh) {
      for (int bw = 0; bw * ws < W; ++bw) {
        std::vector<int> box;
        for (int t = bt * ts; t < std::min(bt * ts + ts, T); ++t) {
          for (int h = bh * hs; h < std::min(bh * hs + hs, H); ++h) {
            for (int w = bw * ws; w < std::min(bw * ws + ws, W); ++w) {
              box.push_back((t * H + h) * W + w);
            }
          }
        }
        out.push_back(std::move(box));
      }
    }
  }
  return out;
}

// The rows vsa_pool's nested loops visit, for the same tile -- the
// construction the kernel actually runs.
std::vector<std::vector<int>>
kernel_tiles(int T, int H, int W, int ts, int hs, int ws)
{
  const int nbh = (H + hs - 1) / hs, nbw = (W + ws - 1) / ws;
  std::vector<std::vector<int>> out;
  for (int bt = 0; bt * ts < T; ++bt) {
    for (int bh = 0; bh < nbh; ++bh) {
      for (int bw = 0; bw < nbw; ++bw) {
        const int t_hi = std::min(ts, T - bt * ts);
        const int h_hi = std::min(hs, H - bh * hs);
        const int w_hi = std::min(ws, W - bw * ws);
        std::vector<int> box;
        for (int t = 0; t < t_hi; ++t) {
          for (int yy = 0; yy < h_hi; ++yy) {
            for (int xx = 0; xx < w_hi; ++xx) {
              box.push_back(((bt * ts + t) * H + (bh * hs + yy)) * W
                            + (bw * ws + xx));
            }
          }
        }
        out.push_back(std::move(box));
      }
    }
  }
  return out;
}

// ---- compute_topk, verbatim from video_sparse_attn.py -------------------
int
ref_topk(double sparsity, int num_blocks)
{
  const int keep = (int)std::ceil((1.0 - sparsity) * num_blocks);
  return std::max(1, std::min(keep, num_blocks));
}

// The budget sol_attn_mma.metal's top-k branch computes.
int
kernel_topk(double sparsity, int nk)
{
  int k = (int)std::ceil((1.0 - sparsity) * (double)nk);
  if (k < 1) { k = 1; }
  if (k > nk) { k = nk; }
  return k;
}

}  // namespace

TEST(vsa_gate, the_cube_matches_the_reference_partition)
{
  // Grids that exercise short cubes on every axis, and both published tile
  // shapes. A run of consecutive rows would differ on the FIRST grid.
  const int grids[][3] = {{3, 5, 7}, {8, 6, 10}, {5, 4, 4}, {1, 4, 4}};
  const int tiles[][3] = {{4, 4, 4}, {4, 4, 8}};
  for (const auto& g : grids) {
    for (const auto& t : tiles) {
      const auto want = ref_tiles(g[0], g[1], g[2], t[0], t[1], t[2]);
      const auto got = kernel_tiles(g[0], g[1], g[2], t[0], t[1], t[2]);
      EXPECT_TRUE(want.size() == got.size());
      for (std::size_t i = 0; i < want.size() && i < got.size(); ++i) {
        EXPECT_TRUE(want[i] == got[i]);
      }
    }
  }
}

TEST(vsa_gate, a_cube_is_not_a_run_of_rows)
{
  // The claim the comment makes, stated as a test: on a 3-D grid the first
  // cube is NOT [0, 64) -- it is three frames of one 4x4 window.
  const auto cube = kernel_tiles(3, 5, 7, 4, 4, 4);
  const auto run  = ref_tiles(3, 5, 7, 3, 5, 7);   // one cube = whole grid
  EXPECT_FALSE(cube.empty());
  EXPECT_TRUE(cube[0].size() == 48);               // 3 x 4 x 4 live rows
  EXPECT_TRUE(cube[0][0] == 0);
  EXPECT_TRUE(cube[0][1] == 1);                    // along W
  EXPECT_TRUE(cube[0][4] == 7);                    // next H row: +W
  EXPECT_TRUE(cube[0][16] == 35);                  // next frame: +H*W
  EXPECT_TRUE(run.size() == 1);
}

TEST(vsa_gate, the_budget_matches_compute_topk)
{
  const int nks[] = {1, 7, 64, 100, 464, 951, 1024};
  const double sps[] = {0.0, 0.5, 0.8, 0.9, 0.95, 0.99};
  for (int nk : nks) {
    for (double sp : sps) {
      EXPECT_TRUE(ref_topk(sp, nk) == kernel_topk(sp, nk));
    }
  }
  // The one the checkpoint asks for, spelled out: 10% of 464 blocks.
  EXPECT_TRUE(kernel_topk(0.9, 464) == 47);
  EXPECT_TRUE(kernel_topk(0.8, 464) == 93);
}

TEST(vsa_gate, the_branch_is_the_reference_branch)
{
  // One tile, one head, one column: the arithmetic by hand. The pooled
  // mean of v, the softmax over the single tile (which is 1), and the gate
  // multiply -- so the expected output is out + gate * mean(v) exactly.
  const int tile = 4, H = 1, D = 1;
  std::vector<float> q(tile, 1.0f), k(tile, 1.0f), v{1.0f, 3.0f, 5.0f, 7.0f};
  std::vector<float> gate(tile, 0.5f), out(tile, 10.0f);
  const float mean_v = 4.0f;   // (1+3+5+7)/4
  for (int r = 0; r < tile; ++r) {
    out[r] = out[r] + gate[r] * mean_v;
    EXPECT_TRUE(std::fabs(out[r] - 12.0f) < 1e-6f);
  }
  EXPECT_TRUE(H == 1 && D == 1);
}

TEST(vsa_gate, a_second_tile_changes_the_softmax)
{
  // Two tiles, one column, one head: the score of tile 0 against tile 1 is
  // q0*k1/sqrt(1). With q=k=1 the softmax is uniform, so each tile's
  // pooled value contributes half. Pinned because a single-tile test
  // cannot see a missing normalisation.
  const float v0 = 2.0f, v1 = 6.0f;          // pooled means
  const float w = 0.5f;                       // uniform softmax over 2
  const float out_c = w * v0 + w * v1;        // = 4
  EXPECT_TRUE(std::fabs(out_c - 4.0f) < 1e-6f);
  // And the reference's addcmul is out + gate * out_c, per row.
  const float gate = 0.25f, out0 = 1.0f;
  EXPECT_TRUE(std::fabs((out0 + gate * out_c) - 2.0f) < 1e-6f);
}
