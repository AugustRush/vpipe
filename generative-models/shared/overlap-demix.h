#ifndef VPIPE_GENERATIVE_MODELS_SHARED_OVERLAP_DEMIX_H
#define VPIPE_GENERATIVE_MODELS_SHARED_OVERLAP_DEMIX_H

// Run a fixed-length separator over a signal of any length, the way
// MSST's demix() does -- as a STREAM.
//
// A source separator sees a fixed window (BS-RoFormer: 17.8 s) and is
// bidirectional inside it, so a long signal is cut into overlapping
// chunks and the estimates cross-faded back together. MSST's rules,
// which the checkpoints' published scores were measured with:
//
//   * step = chunk / num_overlap; border = chunk - step;
//   * a signal longer than 2 * border is reflect-padded by `border` at
//     both ends first (and the padding stripped at the end), so its
//     first and last samples are never at a chunk's edge;
//   * chunks start every `step`; a last chunk shorter than `chunk` is
//     reflect-padded on its right when it is over half a chunk long,
//     zero-padded otherwise;
//   * each estimate is weighted by a window that fades linearly in and
//     out over chunk / 10 samples -- except the first chunk's fade-in
//     and the last chunk's fade-out -- and the sum is divided by the
//     summed weights; a sample no window covers comes out 0.
//
// STREAMED, NOT APPROXIMATED. A sample is final once every chunk that
// covers it has run, which is as soon as the chunk starting after it
// is about to; push() runs each chunk the moment its samples are in,
// and the output grows as samples become final. The result is
// identical, sample for sample, to running the same rules over the
// whole signal at once -- the padding decision is made once more than
// 2 * border samples have arrived (or at the end, for a shorter one),
// and the end padding at finish(). Memory is bounded by a chunk and a
// step, whatever the length.
//
// The rules are restated with MSST's inference.batch_size at 1: MSST
// picks the fade flags once per BATCH, after the batch's last chunk, so
// at a larger batch a chunk can be given another chunk's window. At 1
// every chunk gets its own.

#include <cstdint>
#include <functional>
#include <vector>

namespace vpipe::genai {

class OverlapDemix {
 public:
  struct Params {
    int chunk       = 0;   // samples per forward
    int num_overlap = 2;
    int channels    = 2;
    int stems       = 1;
  };

  // One forward: `in` planar [channels][chunk] in, `out` planar
  // [stems][channels][chunk] out. False aborts the stream.
  using Forward = std::function<bool(const float* in, float* out)>;

  explicit OverlapDemix(const Params& p);

  bool valid() const noexcept { return _ok; }
  const Params& params() const noexcept { return _p; }
  int step() const noexcept { return _step; }
  int border() const noexcept { return _border; }

  // Append `n` samples, channel c's at planar[c * stride + i], and run
  // every chunk that is ready. False when a forward failed.
  bool push(const float* planar, std::int64_t n, std::int64_t stride,
            const Forward& fwd);

  // End of input: pad the end and run the rest. Everything becomes
  // final. False when a forward failed.
  bool finish(const Forward& fwd);

  // Samples final and not yet taken.
  std::int64_t available() const noexcept;

  // Take every final sample: the estimates as planar
  // [stems][channels][n] and the input over the same range as
  // [channels][n] (what a caller subtracts an estimate from). Returns n.
  std::int64_t take(std::vector<float>* stems, std::vector<float>* mix);

  // Forwards run so far.
  std::int64_t chunks() const noexcept { return _chunks; }

 private:
  bool run_ready_(const Forward& fwd);
  bool run_chunk_(const Forward& fwd, bool last);
  void decide_padding_(bool at_end);
  void append_p_(const float* planar, std::int64_t n, std::int64_t stride);
  void finalize_to_(std::int64_t p_end);

  Params _p;
  bool   _ok = false;
  int    _step = 0, _border = 0, _fade = 0;
  std::vector<float> _window;          // the chunk window, no flags

  // Input before the padding is decided, and the input's last
  // border + 1 samples (for the end padding), per channel.
  std::vector<std::vector<float>> _pending, _tail;
  std::int64_t _received = 0;          // input samples so far
  bool _decided = false, _padded = false, _eos = false;

  // The padded signal P from P index _base, per channel, and the
  // accumulators over the same range.
  std::int64_t _base = 0;
  std::vector<std::vector<float>> _sig;              // [channels]
  std::vector<std::vector<float>> _acc;              // [stems*channels]
  std::vector<float> _cnt;
  std::int64_t _p_len = 0;             // P samples known (from index 0)
  std::int64_t _next = 0;              // P index of the next chunk
  std::int64_t _final = 0;             // P positions below are final
  std::int64_t _taken = 0;             // P positions below were taken
  std::int64_t _chunks = 0;
  std::vector<float> _in, _out;        // one chunk's buffers
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_OVERLAP_DEMIX_H
