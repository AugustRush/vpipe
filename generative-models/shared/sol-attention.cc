#include "generative-models/shared/sol-attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace sol {

namespace {

// The softmax runs in base 2, so every score carries this factor and
// the threshold is built in the same units. Keeping it in one constant
// is what stops the threshold and the scores drifting apart.
constexpr float kLog2e = 1.44269504088896340736f;

}  // namespace

bool
parse_threshold(const std::string& name, Threshold* out)
{
  if (name == "diag")  { *out = Threshold::kDiag;  return true; }
  if (name == "exact") { *out = Threshold::kExact; return true; }
  return false;
}

const char*
threshold_name(Threshold t)
{
  return t == Threshold::kExact ? "exact" : "diag";
}

void
summaries(const float* q, const float* k, const float* v, int heads,
          int tokens, int d, float* qc, float* kc, float* vc)
{
  if (heads <= 0 || tokens <= 0 || d <= 0) { return; }
  const int nb = num_blocks(tokens);
  for (int h = 0; h < heads; ++h) {
    for (int b = 0; b < nb; ++b) {
      const int t0  = b * kBlock;
      const int len = std::min(kBlock, tokens - t0);
      float* qo = qc + ((std::size_t)h * nb + b) * d;
      float* ko = kc + ((std::size_t)h * nb + b) * d;
      float* vo = vc + ((std::size_t)h * nb + b) * d;
      for (int i = 0; i < d; ++i) { qo[i] = ko[i] = vo[i] = 0.0f; }
      for (int t = 0; t < len; ++t) {
        const std::size_t row = ((std::size_t)h * tokens + t0 + t) * d;
        for (int i = 0; i < d; ++i) {
          qo[i] += q[row + i];
          ko[i] += k[row + i];
          vo[i] += v[row + i];
        }
      }
      const float inv = 1.0f / (float)len;
      // q and k are CENTROIDS; v is a SUM, and that asymmetry is the
      // whole approximation -- exp(proxy) * sum_j v_j is what stands in
      // for sum_j exp(s_j) v_j when the block's logits are flat.
      for (int i = 0; i < d; ++i) { qo[i] *= inv; ko[i] *= inv; }
    }
  }
}

void
thresholds(const float* qc, const float* kc, int heads, int blocks, int d,
           float scale, float tau, Threshold t, float* out)
{
  if (heads <= 0 || blocks <= 0 || d <= 0) { return; }
  const float ls = scale * kLog2e;
  std::vector<float> mean((std::size_t)d);
  std::vector<float> var((std::size_t)d);
  std::vector<float> mom;                       // kExact: [d, d]
  if (t == Threshold::kExact) { mom.assign((std::size_t)d * d, 0.0f); }

  for (int h = 0; h < heads; ++h) {
    const float* kch = kc + (std::size_t)h * blocks * d;
    // The key centroids' own distribution over blocks: mean, and either
    // the per-channel variance (kDiag) or the full second moment
    // (kExact). This is per HEAD and reused by every query block.
    for (int i = 0; i < d; ++i) { mean[(std::size_t)i] = 0.0f; }
    for (int b = 0; b < blocks; ++b) {
      const float* row = kch + (std::size_t)b * d;
      for (int i = 0; i < d; ++i) { mean[(std::size_t)i] += row[i]; }
    }
    const float invb = 1.0f / (float)blocks;
    for (int i = 0; i < d; ++i) { mean[(std::size_t)i] *= invb; }

    if (t == Threshold::kExact) {
      std::fill(mom.begin(), mom.end(), 0.0f);
      for (int b = 0; b < blocks; ++b) {
        const float* row = kch + (std::size_t)b * d;
        for (int i = 0; i < d; ++i) {
          const float ri = row[i];
          float* mi = mom.data() + (std::size_t)i * d;
          for (int j = 0; j < d; ++j) { mi[j] += ri * row[j]; }
        }
      }
      for (float& m : mom) { m *= invb; }
    } else {
      for (int i = 0; i < d; ++i) { var[(std::size_t)i] = 0.0f; }
      for (int b = 0; b < blocks; ++b) {
        const float* row = kch + (std::size_t)b * d;
        for (int i = 0; i < d; ++i) {
          const float c = row[i] - mean[(std::size_t)i];
          var[(std::size_t)i] += c * c;
        }
      }
      for (int i = 0; i < d; ++i) { var[(std::size_t)i] *= invb; }
    }

    for (int b = 0; b < blocks; ++b) {
      const float* qb = qc + ((std::size_t)h * blocks + b) * d;
      float raw_mean = 0.0f;
      for (int i = 0; i < d; ++i) { raw_mean += qb[i] * mean[(std::size_t)i]; }
      float raw_var = 0.0f;
      if (t == Threshold::kExact) {
        // qbar^T M qbar - (qbar . mu)^2: the variance of the proxy score
        // across key blocks, with the centroids' full covariance.
        for (int i = 0; i < d; ++i) {
          const float* mi = mom.data() + (std::size_t)i * d;
          float p = 0.0f;
          for (int j = 0; j < d; ++j) { p += mi[j] * qb[j]; }
          raw_var += qb[i] * p;
        }
        raw_var -= raw_mean * raw_mean;
      } else {
        for (int i = 0; i < d; ++i) {
          raw_var += qb[i] * qb[i] * var[(std::size_t)i];
        }
      }
      const float m = raw_mean * ls;
      const float vr = std::max(raw_var, 0.0f) * ls * ls;
      out[(std::size_t)h * blocks + b] = m + tau * std::sqrt(vr + 1.0e-6f);
    }
  }
}

void
forward(const float* q, const float* k, const float* v, const float* qc,
        const float* kc, const float* vc, const float* thr, int heads,
        int tokens, int d, float scale, const Config& cfg, float* out,
        Stats* stats)
{
  if (heads <= 0 || tokens <= 0 || d <= 0) { return; }
  const int nb = num_blocks(tokens);
  const float ls = scale * kLog2e;

  // The sink, in BLOCKS and rounded OUTWARD: a block is exact if it
  // overlaps the range at all. Exactness is a property of the block the
  // kernel visits, so half a block cannot be exact.
  int sink_lo = 0, sink_hi = 0;
  if (cfg.sink_tokens > 0) {
    const int s0 = std::max(0, cfg.sink_start);
    const int s1 = std::min(tokens, s0 + cfg.sink_tokens);
    if (s1 > s0) {
      sink_lo = s0 / kBlock;
      sink_hi = (s1 + kBlock - 1) / kBlock;
    }
  }

  std::vector<float> acc((std::size_t)d);
  std::vector<float> proxy((std::size_t)nb);
  std::vector<char>  exact((std::size_t)nb);

  for (int h = 0; h < heads; ++h) {
    const float* qh  = q  + (std::size_t)h * tokens * d;
    const float* kh  = k  + (std::size_t)h * tokens * d;
    const float* vh  = v  + (std::size_t)h * tokens * d;
    const float* kch = kc + (std::size_t)h * nb * d;
    const float* vch = vc + (std::size_t)h * nb * d;
    float* oh = out + (std::size_t)h * tokens * d;

    for (int qb = 0; qb < nb; ++qb) {
      const int q0 = qb * kBlock;
      const int qlen = std::min(kBlock, tokens - q0);
      const float route = thr[(std::size_t)h * nb + qb];

      // ROUTE ONCE PER QUERY BLOCK, from the query CENTROID -- the same
      // quantity the threshold was built against, so the comparison is
      // between like and like. See the header on why this is not the
      // score tile.
      const float* qbar = qc + ((std::size_t)h * nb + qb) * d;
      for (int n = 0; n < nb; ++n) {
        const float* kr = kch + (std::size_t)n * d;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) { dot += qbar[i] * kr[i]; }
        proxy[(std::size_t)n] = dot * ls;
        const bool local = std::abs(qb - n) <= cfg.local_radius;
        const bool sink  = (n >= sink_lo && n < sink_hi);
        exact[(std::size_t)n] =
            (char)(proxy[(std::size_t)n] > route || local || sink);
      }
      // VSA-H3: a fixed BUDGET instead of the threshold. The scores are
      // already in `proxy`, so this ranks them and keeps the top
      // ceil((1 - sparsity) * nb), with the local band and the sink still
      // forced in -- the reference's `exempt` mode does the same for the
      // prefix, and keeping them costs nothing here because they are
      // already marked.
      if (cfg.vsa_sparsity > 0.0f && cfg.vsa_sparsity < 1.0f) {
        int keep = (int)std::ceil((1.0 - (double)cfg.vsa_sparsity) * nb);
        keep = std::max(1, std::min(keep, nb));
        std::vector<int> order((std::size_t)nb);
        for (int n = 0; n < nb; ++n) { order[(std::size_t)n] = n; }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
          return proxy[(std::size_t)a] > proxy[(std::size_t)b];
        });
        std::vector<char> picked((std::size_t)nb, 0);
        for (int i = 0; i < keep; ++i) { picked[(std::size_t)order[i]] = 1; }
        for (int n = 0; n < nb; ++n) {
          const bool local = std::abs(qb - n) <= cfg.local_radius;
          const bool sink  = (n >= sink_lo && n < sink_hi);
          // The tail block is always exact: its length is not a whole
          // block, which is the one place the pooled shortcut would not
          // hold. See sol_attn_mma.metal's header.
          const bool tail  = (n == nb - 1);
          exact[(std::size_t)n] =
              (char)(picked[(std::size_t)n] || local || sink || tail);
        }
      }
      if (stats != nullptr) {
        for (int n = 0; n < nb; ++n) {
          if (exact[(std::size_t)n]) { ++stats->exact_blocks; }
          else                       { ++stats->approx_blocks; }
          ++stats->total_blocks;
        }
      }

      for (int t = 0; t < qlen; ++t) {
        const float* qr = qh + (std::size_t)(q0 + t) * d;
        std::fill(acc.begin(), acc.end(), 0.0f);
        float m = -INFINITY, l = 0.0f;

        // APPROXIMATE blocks first, exactly as the reference orders
        // them: one proxy logit per block, standing in for all its keys.
        for (int n = 0; n < nb; ++n) {
          if (exact[(std::size_t)n]) { continue; }
          const float* kr = kch + (std::size_t)n * d;
          float dot = 0.0f;
          for (int i = 0; i < d; ++i) { dot += qr[i] * kr[i]; }
          const float s = dot * ls;
          const float mn = std::max(m, s);
          const float corr = (m == -INFINITY) ? 0.0f : std::exp2(m - mn);
          const float p = std::exp2(s - mn);
          const int len = std::min(kBlock, tokens - n * kBlock);
          l = l * corr + p * (float)len;
          const float* vr = vch + (std::size_t)n * d;
          for (int i = 0; i < d; ++i) { acc[(std::size_t)i] =
              acc[(std::size_t)i] * corr + p * vr[i]; }
          m = mn;
        }

        // ...then the exact ones, key by key.
        for (int n = 0; n < nb; ++n) {
          if (!exact[(std::size_t)n]) { continue; }
          const int t0 = n * kBlock;
          const int len = std::min(kBlock, tokens - t0);
          for (int j = 0; j < len; ++j) {
            const float* kr = kh + (std::size_t)(t0 + j) * d;
            float dot = 0.0f;
            for (int i = 0; i < d; ++i) { dot += qr[i] * kr[i]; }
            const float s = dot * ls;
            const float mn = std::max(m, s);
            const float corr = (m == -INFINITY) ? 0.0f : std::exp2(m - mn);
            const float p = std::exp2(s - mn);
            l = l * corr + p;
            const float* vr = vh + (std::size_t)(t0 + j) * d;
            for (int i = 0; i < d; ++i) { acc[(std::size_t)i] =
                acc[(std::size_t)i] * corr + p * vr[i]; }
            m = mn;
          }
        }

        float* orow = oh + (std::size_t)(q0 + t) * d;
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        for (int i = 0; i < d; ++i) { orow[i] = acc[(std::size_t)i] * inv; }
      }
    }
  }
}

void
dense(const float* q, const float* k, const float* v, int heads, int tokens,
      int d, float scale, float* out)
{
  if (heads <= 0 || tokens <= 0 || d <= 0) { return; }
  std::vector<float> acc((std::size_t)d);
  for (int h = 0; h < heads; ++h) {
    const float* qh = q + (std::size_t)h * tokens * d;
    const float* kh = k + (std::size_t)h * tokens * d;
    const float* vh = v + (std::size_t)h * tokens * d;
    float* oh = out + (std::size_t)h * tokens * d;
    for (int t = 0; t < tokens; ++t) {
      const float* qr = qh + (std::size_t)t * d;
      std::fill(acc.begin(), acc.end(), 0.0f);
      float m = -INFINITY, l = 0.0f;
      for (int j = 0; j < tokens; ++j) {
        const float* kr = kh + (std::size_t)j * d;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) { dot += qr[i] * kr[i]; }
        const float s = dot * scale;
        const float mn = std::max(m, s);
        const float corr = (m == -INFINITY) ? 0.0f : std::exp(m - mn);
        const float p = std::exp(s - mn);
        l = l * corr + p;
        const float* vr = vh + (std::size_t)j * d;
        for (int i = 0; i < d; ++i) { acc[(std::size_t)i] =
            acc[(std::size_t)i] * corr + p * vr[i]; }
        m = mn;
      }
      float* orow = oh + (std::size_t)t * d;
      const float inv = 1.0f / l;
      for (int i = 0; i < d; ++i) { orow[i] = acc[(std::size_t)i] * inv; }
    }
  }
}

}  // namespace sol
}  // namespace genai
}  // namespace vpipe
