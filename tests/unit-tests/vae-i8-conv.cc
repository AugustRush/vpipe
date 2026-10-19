// The int8 3x3 convolution a VAE's resnet convs take under i8_gemm
// (conv2d_hw_3x3_*_i8 and its quantizers in conv2d_mma.metal;
// MetalKrea2Vae::conv3x3_i8_): each activation channel at its own step,
// the steps folded into a per-call int8 copy of the weight, the scales
// and the bias applied in the conv's epilogue.

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
mc_of_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  return (mc != nullptr && mc->valid() && mc->supports_matrix_cores())
             ? mc : nullptr;
}

struct Kernels {
  ComputeLibrary conv, elt;
  ComputeFunction i8[2][3];     // [stride 1, 2][tile 64, 32, 16]
  ComputeFunction f16[2][3];
  ComputeFunction zero, amax, steps, quant, fold, bias_add, silu;

  explicit Kernels(MetalCompute* mc)
  {
    conv = mc->load_library("conv2d_mma");
    elt = mc->load_library("llm_elementwise");
    const char* sfx[3] = {"", "_c32", "_c16"};
    for (int s = 0; s < 2; ++s) {
      for (int t = 0; t < 3; ++t) {
        const std::string st = s == 0 ? "s1" : "s2";
        i8[s][t] = conv.function("conv2d_hw_3x3_" + st + sfx[t] + "_i8");
        f16[s][t] = conv.function("conv2d_hw_3x3_" + st + sfx[t] + "_f16");
      }
    }
    zero = conv.function("vae_i8_zero_u32");
    amax = conv.function("vae_i8_chan_amax_f16");
    steps = conv.function("vae_i8_steps_f16");
    quant = conv.function("vae_i8_quant_chan_f16");
    fold = conv.function("vae_i8_weight_fold_f16");
    bias_add = elt.function("bias_add_rows_f16");
    silu = elt.function("mul_sigmoid_f16");
  }
  bool valid() const
  {
    for (int s = 0; s < 2; ++s) {
      for (int t = 0; t < 3; ++t) {
        if (!i8[s][t].valid() || !f16[s][t].valid()) { return false; }
      }
    }
    return zero.valid() && amax.valid() && steps.valid() && quant.valid() &&
           fold.valid() &&
           bias_add.valid() && silu.valid();
  }
};

int
tile_of_(int cout)
{
  return cout % 64 == 0 ? 0 : cout % 32 == 0 ? 1 : cout % 16 == 0 ? 2 : -1;
}

// One conv's buffers, laid out as the VAE holds them: the activation
// [H, W, Cin] f16, the weight [Cout, 9*Cin] f16 ((ky, kx, ci) flattened)
// and its HWIO twin, the bias.
struct Conv {
  int H = 0, W = 0, cin = 0, cout = 0, stride = 1;
  SharedBuffer x, w, whwio, b, out, out16;
  SharedBuffer amax, wscale, wq, xq;
  std::vector<float> xh, wh, bh;   // the f16 values, as floats

  Conv(MetalCompute* mc, int h, int wd, int ci, int co, int s,
       unsigned seed, float outlier)
    : H(h), W(wd), cin(ci), cout(co), stride(s)
  {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    // Channels at their own scale, and a few ten times the rest: what a
    // channel step is for.
    std::vector<float> cs((std::size_t)ci);
    for (int c = 0; c < ci; ++c) {
      cs[(std::size_t)c] = (c % 17 == 3 ? outlier : 1.0f) *
                           (0.5f + (float)(c % 7) * 0.25f);
    }
    const std::size_t nx = (std::size_t)h * wd * ci;
    xh.resize(nx);
    for (std::size_t i = 0; i < nx; ++i) {
      xh[i] = (float)(_Float16)(g(rng) * cs[i % (std::size_t)ci]);
    }
    const std::size_t nw = (std::size_t)co * 9 * ci;
    wh.resize(nw);
    const float ws = 1.0f / std::sqrt(9.0f * (float)ci);
    for (auto& v : wh) { v = (float)(_Float16)(g(rng) * ws); }
    bh.resize((std::size_t)co);
    for (auto& v : bh) { v = (float)(_Float16)(g(rng) * 0.1f); }
    auto f16 = [&](const std::vector<float>& v) {
      SharedBuffer buf = mc->make_shared_buffer(v.size() * 2);
      auto* d = static_cast<_Float16*>(buf.contents());
      for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
      return buf;
    };
    x = f16(xh); w = f16(wh); b = f16(bh);
    std::vector<float> hw((std::size_t)9 * ci * co);
    for (int o = 0; o < co; ++o) {
      for (int k = 0; k < 9 * ci; ++k) {
        hw[(std::size_t)k * co + o] = wh[(std::size_t)o * 9 * ci + k];
      }
    }
    whwio = f16(hw);
    const std::size_t no = (std::size_t)(h / s) * (wd / s) * co;
    out = mc->make_shared_buffer(no * 2);
    out16 = mc->make_shared_buffer(no * 2);
    amax = mc->make_shared_buffer((std::size_t)ci * 4);
    wscale = mc->make_shared_buffer((std::size_t)co * 4);
    wq = mc->make_shared_buffer(nw);
    xq = mc->make_shared_buffer(nx);
  }

  // The int8 route, as MetalKrea2Vae::conv3x3_i8_ encodes it.
  void encode_i8(ComputeEncoder& enc, const Kernels& k, bool silu,
                 bool conv_only = false) const
  {
    const int rows = H * W, cin4 = cin / 4;
    const int sil = silu ? 1 : 0;
    if (!conv_only) {
      enc.set_function(k.zero);
      enc.set_buffer(0, amax); enc.set_constant(1, cin);
      enc.dispatch({(unsigned)cin, 1, 1}, {64, 1, 1});
      const int rp = 256;
      enc.set_function(k.amax);
      enc.set_buffer(0, x); enc.set_buffer(1, amax);
      enc.set_constant(2, cin); enc.set_constant(3, rows);
      enc.set_constant(4, rp); enc.set_constant(5, sil);
      enc.dispatch({(unsigned)(cin4 * ((rows + rp - 1) / rp)), 1, 1},
                   {256, 1, 1});
      enc.set_function(k.steps);
      enc.set_buffer(0, amax); enc.set_constant(1, cin);
      enc.set_constant(2, 0.25f);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
      enc.set_function(k.fold);
      enc.set_buffer(0, w); enc.set_buffer(1, amax);
      enc.set_buffer(2, wq); enc.set_buffer(3, wscale);
      enc.set_constant(4, cin); enc.set_constant(5, cout);
      enc.dispatch({(unsigned)cout * 256u, 1, 1}, {256, 1, 1});
      enc.set_function(k.quant);
      enc.set_buffer(0, x); enc.set_buffer(1, xq); enc.set_buffer(2, amax);
      const unsigned n4 = (unsigned)(rows * cin4);
      enc.set_constant(3, cin); enc.set_constant(4, sil);
      enc.set_constant(5, n4);
      enc.dispatch({n4, 1, 1}, {256, 1, 1});
    }
    const int t = tile_of_(cout);
    const int tc = t == 0 ? 64 : t == 1 ? 32 : 16;
    const int OH = H / stride, OW = W / stride;
    enc.set_function(k.i8[stride - 1][t]);
    enc.set_buffer(0, xq); enc.set_buffer(1, wq); enc.set_buffer(2, out);
    enc.set_constant(3, W); enc.set_constant(4, H);
    enc.set_constant(5, cin); enc.set_constant(6, cout);
    enc.set_buffer(7, wscale); enc.set_buffer(8, b);
    enc.set_constant(9, 1); enc.set_constant(10, 3);
    enc.dispatch({(unsigned)((OW / 8) * 128), (unsigned)(OH / 8),
                  (unsigned)(cout / tc)}, {128, 1, 1});
  }

  // The f16 route the int8 one replaces: SiLU in place (on a copy, so a
  // bench can repeat it), the hardware conv, the bias pass.
  void encode_f16(ComputeEncoder& enc, const Kernels& k, bool silu,
                  const SharedBuffer& xin) const
  {
    const int rows = H * W;
    if (silu) {
      const int n = rows * cin;
      enc.set_function(k.silu);
      enc.set_buffer(0, xin); enc.set_buffer(1, xin); enc.set_buffer(2, xin);
      enc.set_constant(3, n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
    const int t = tile_of_(cout);
    const int tc = t == 0 ? 64 : t == 1 ? 32 : 16;
    const int OH = H / stride, OW = W / stride;
    enc.set_function(k.f16[stride - 1][t]);
    enc.set_buffer(0, xin); enc.set_buffer(1, whwio);
    enc.set_buffer(2, out16);
    enc.set_constant(3, W); enc.set_constant(4, H);
    enc.set_constant(5, cin); enc.set_constant(6, cout);
    if (stride == 2) { enc.set_constant(7, 3); }
    enc.dispatch({(unsigned)((OW / 8) * 128), (unsigned)(OH / 8),
                  (unsigned)(cout / tc)}, {128, 1, 1});
    const std::size_t orow = (std::size_t)OH * OW;
    enc.set_function(k.bias_add);
    enc.set_buffer(0, out16); enc.set_buffer(1, b);
    enc.set_constant(2, cout);
    enc.set_constant(3, (unsigned)(orow * cout));
    enc.dispatch({(unsigned)cout, (unsigned)orow, 1}, {256, 1, 1});
  }

  // The CPU's f32 conv of the (SiLU'd) f16 values. Stride 2 pads
  // (0, 1, 0, 1) -- the downsample's -- as off_mode 3 reads it.
  std::vector<float> cpu(bool silu) const
  {
    const int OH = H / stride, OW = W / stride;
    std::vector<float> xs(xh);
    if (silu) {
      for (auto& v : xs) { v = v / (1.0f + std::exp(-v)); }
    }
    std::vector<float> y((std::size_t)OH * OW * cout);
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        for (int o = 0; o < cout; ++o) {
          double a = bh[(std::size_t)o];
          for (int ky = 0; ky < 3; ++ky) {
            const int iy = stride == 2 ? oy * 2 + ky : oy + ky - 1;
            if (iy < 0 || iy >= H) { continue; }
            for (int kx = 0; kx < 3; ++kx) {
              const int ix = stride == 2 ? ox * 2 + kx : ox + kx - 1;
              if (ix < 0 || ix >= W) { continue; }
              const float* px = &xs[((std::size_t)iy * W + ix) * cin];
              const float* pw =
                  &wh[((std::size_t)o * 9 + ky * 3 + kx) * cin];
              for (int c = 0; c < cin; ++c) { a += (double)px[c] * pw[c]; }
            }
          }
          y[((std::size_t)oy * OW + ox) * cout + o] = (float)a;
        }
      }
    }
    return y;
  }
};

double
rel_l2_(const SharedBuffer& b, const std::vector<float>& ref)
{
  const auto* p = static_cast<const _Float16*>(b.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = (double)p[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

// Every tile and both strides against the CPU's f32 conv, the SiLU on
// and off, with outlier channels ten times the rest: within an int8
// route's error, and the f16 hardware conv beside it as the yardstick.
TEST(vae_i8_conv, matches_a_cpu_conv)
{
  Session sess;
  MetalCompute* mc = mc_of_(sess);
  if (mc == nullptr) { return; }
  Kernels k(mc);
  ASSERT_TRUE(k.valid());
  struct Case { int hw, cin, cout, stride; bool silu; };
  const Case cases[] = {
      {16, 128, 64, 1, true},  {16, 128, 96, 1, false},
      {16, 144, 144, 1, true}, {32, 128, 64, 2, false},
      {32, 192, 48, 2, true},
  };
  unsigned seed = 11;
  for (const Case& c : cases) {
    Conv cv(mc, c.hw, c.hw, c.cin, c.cout, c.stride, seed++, 10.0f);
    const std::vector<float> ref = cv.cpu(c.silu);
    SharedBuffer xcopy = mc->make_shared_buffer(cv.xh.size() * 2);
    std::memcpy(xcopy.contents(), cv.x.contents(), cv.xh.size() * 2);
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      cv.encode_i8(enc, k, c.silu);
      cv.encode_f16(enc, k, c.silu, xcopy);
    }
    ASSERT_TRUE(st.commit().wait_ok(nullptr));
    const double r8 = rel_l2_(cv.out, ref);
    const double r16 = rel_l2_(cv.out16, ref);
    std::printf("[vae_i8_conv] %2dx%-2d %3d->%-3d s%d silu=%d: int8 rel-L2 "
                "%.2e, f16 %.2e\n", c.hw, c.hw, c.cin, c.cout, c.stride,
                (int)c.silu, r8, r16);
    // The bar is an int8 conv's on this data, whose wide channels --
    // one in seventeen, ten times the rest -- are what share a weight
    // step with the narrow ones (2.2-3.5e-2 measured). A real picture's
    // activations do far better: see qwen_image_21_vae.int8_against_f16.
    EXPECT_TRUE(r8 < 5e-2);
    EXPECT_TRUE(r16 < 2e-3);
  }
}

// THE RATE at Qwen-Image-2.1's decoder shapes at 1024^2, the route
// against the f16 one it replaces (SiLU, hardware conv, bias pass), and
// the int8 conv alone. VPIPE_VAE_I8_BENCH=1.
TEST(vae_i8_conv, rate_at_the_decoders_shapes)
{
  if (std::getenv("VPIPE_VAE_I8_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = mc_of_(sess);
  if (mc == nullptr) { return; }
  Kernels k(mc);
  ASSERT_TRUE(k.valid());
  struct Shape { int hw, cin, cout; };
  const Shape shapes[] = {
      {64, 1152, 1152}, {128, 1152, 1152}, {256, 1152, 576},
      {256, 576, 576},  {512, 576, 288},   {512, 288, 288},
      {1024, 288, 288}, {1024, 288, 144}, {1024, 144, 144},
  };
  for (const Shape& s : shapes) {
    Conv cv(mc, s.hw, s.hw, s.cin, s.cout, 1, 5, 4.0f);
    SharedBuffer xcopy = mc->make_shared_buffer(cv.xh.size() * 2);
    const int iters = s.hw >= 512 ? 4 : 10;
    auto time = [&](auto&& body) {
      double best = 0.0;
      for (int r = 0; r < 3; ++r) {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder enc = st.begin_compute();
          for (int i = 0; i < iters; ++i) { body(enc); }
        }
        const auto t0 = std::chrono::steady_clock::now();
        st.commit().wait();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / iters;
        if (best == 0.0 || ms < best) { best = ms; }
      }
      return best;
    };
    const double f16 = time([&](ComputeEncoder& e) {
      cv.encode_f16(e, k, true, xcopy);
    });
    const double i8 = time([&](ComputeEncoder& e) {
      cv.encode_i8(e, k, true);
    });
    const double conv = time([&](ComputeEncoder& e) {
      cv.encode_i8(e, k, true, /*conv_only=*/true);
    });
    const double gf = 2.0 * s.hw * s.hw * s.cout * 9.0 * s.cin / 1e9;
    std::printf("[vae_i8_conv] %4d^2 %4d->%-4d f16 %7.2f ms (%5.1f TF/s)  "
                "int8 %7.2f ms (%.2fx)  conv alone %7.2f ms (%5.1f TOP/s)\n",
                s.hw, s.cin, s.cout, f16, gf / f16, i8, f16 / i8, conv,
                gf / conv);
  }
}
