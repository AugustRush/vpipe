#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_VDN_BRANCH_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_VDN_BRANCH_H

// VDN-H3's hybrid attention as a component: the weights of one block's
// linear branch, the scratch it needs, and the dispatch sequence.
//
// WHAT THIS IS NOT. It is not a model -- it has no forward of its own
// and no q/k/v. VDN's branch shares the BACKBONE's projections, reading
// the raw pre-QK-norm, pre-RoPE q/k/v that H3's attention already
// computes, which is why its width is exactly H3's 56 x 128 and why one
// LoRA on attn.orig.to_{q,k,v} feeds both halves of the hybrid. So this
// takes those tensors as inputs and hands back the readout the caller
// adds to its own attention output, on VIDEO rows only:
//
//     out = orig.to_out(softmax_gate(x) * window_softmax(q, k, v))
//           + to_out_linear(branch_readout)          <- video rows only
//
// WEIGHTS COME FROM THE MANAGER, per the WeightSet contract: the loader
// takes a shared_ptr and keeps it for its own lifetime, because the
// buffers it hands out are refcounted aliases of the set's. The
// checkpoint is bf16 and every kernel here is fp32 -- A is the matrix
// the solve inverts and bf16 loses the symmetry it depends on -- so the
// reads go through derived(), which is the accessor for a KEPT
// TRANSFORM. The key names the dtype, since that is what changes the
// bytes.
//
// ONE ENCODER, NO FENCES. The whole branch, scan included, is a run of
// dispatches on a single ComputeEncoder: Metal's serial dispatch type
// orders them, so the frame recurrence needs no commit between steps.
// The frame LOOP is on the host because a head's [128, 128] fp32 state
// is 64 KB against a 32 KB threadgroup budget -- no threadgroup can own
// a head across frames, so the ordering has to be the queue's.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-h3/vdn-config.h"
#include "generative-models/minimax-h3/vdn-geometry.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/weight-set.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace minimax_h3 {

class MetalVdnBranch {
public:
  // How many frames of per-token work are held live at once.
  //
  // THE PER-TOKEN BUFFERS ARE THE WHOLE MEMORY STORY. At production
  // geometry (100 latent frames x 1008 tokens, 7168 channels) one of
  // them is 2.89 GB in fp32, and the untiled shape holds five: 14.45 GB
  // against 3.67 GB for every per-frame bank put together, or 18.1 GB a
  // block on a box that is also holding a 33B DiT.
  //
  // They do not need to be live. Features feed the PER-FRAME statistics
  // and the readout reads a PER-FRAME state, so both halves can walk the
  // clip a few frames at a time; only the banks are genuinely global,
  // because the scan and the gather are. Tiling takes the per-token half
  // to ~0.14 GB and the total to ~3.8 GB.
  //
  // The tile carries a two-frame HALO on each side, because the short
  // conv's temporal half is a 5-tap stencil: a tile edge that is
  // interior to the clip must read its neighbour, not a zero.
  static constexpr int kFrameTile = 4;
  static constexpr int kConvHalo  = 2;
  // The channel block the matrix-core spatial conv runs dense over. 16
  // is measured, not chosen: see the note on _mma_conv.
  static constexpr int kDwBlock   = 16;
  // The dest tile's width, which covers a 30-column patch grid whole.
  static constexpr int kDwTileW   = 32;
  // The HEIGHTS the loader resolves, and the rule that picks between
  // them: minimise the PADDED row count, ceil(gh / h) * h, and break a
  // tie toward the larger tile.
  //
  // Measured, not derived. A height-8 tile is the fastest of the three
  // where 8 divides the patch grid (1.62x at gh 16) and the slowest
  // where it does not (1.24x at gh 17, where it computes 24 rows for
  // 17); height 6 is the reverse. The rule follows that and nothing
  // else -- in particular NOT "less waste is better", which height 9
  // disproves: same padded count as 6 at gh 17, and 60% slower.
  static constexpr int kDwHeights[3] = {4, 6, 8};

  // Alignment of one carve out of a lent arena (see set_arena), and the
  // slack arena_bytes() adds for it -- a page each, over more carves
  // than reserve() makes.
  //
  // A PAGE AND NOT A CACHE LINE, deliberately. A private allocation is
  // page-rounded, so every buffer below used to have most of a page of
  // unused bytes behind it, and a kernel that writes a padded tile past
  // its destination's exact size landed in them. Packed tight, that
  // same write lands in the next buffer instead -- silently, and only
  // for the geometries whose tail is ragged. The carve keeps the slack
  // the allocator was providing rather than betting that no kernel
  // needs it.
  static constexpr std::size_t kCarveAlign = 16384;
  static constexpr std::size_t kCarveSlack = 48 * kCarveAlign;

  struct Dims {
    // Frames of per-token work held live at once. A TUNING knob and
    // nothing else: the answer must not depend on it, which is what
    // `tile_size_does_not_change_the_answer` pins. Also the only way to
    // exercise the conv's halo, since a tile as large as the clip never
    // crosses a boundary.
    int frame_tile = kFrameTile;
    // The per-token tensors in bf16: the features, the conv's output,
    // beta, the output gate, and the state the readout reads.
    //
    // THE REFERENCE'S OWN DTYPE MAP. scan.py: "Only the small [F,H,d,d]
    // results are promoted to fp32" -- A, alpha's frame mean and the
    // Cholesky island stay wide here too, and nothing else was ever
    // meant to be. So this is not a precision compromise, it is the
    // precision the released checkpoint was trained at, and vpipe was
    // the one being generous.
    //
    // It is a SWITCH and not a constant because the goldens were taken
    // against an fp32 reference and still check against it exactly at
    // 1e-5; the narrow path is checked against the wide one instead.
    // Mirrors the reference's own `a_fp32`, which exists for the same
    // reason.
    //
    // On a matrix-core GPU this is worth more than the bandwidth: bf16
    // is what the tensor cores take, and an fp32 contraction misses
    // them entirely.
    bool bf16_features = true;
    int heads    = 56;
    int head_dim = 128;      // the BACKBONE's; the branch's is in Config
    int hidden   = 5376;
    int n_layers = 50;
    // Whether the box HAS matrix cores, which the estimate needs and
    // cannot ask: `load()` takes the matrix-core route only when this
    // and `bf16_features` are both true and the libraries validate, and
    // three of its landing buffers exist only on that route. Booking
    // them on a GPU that will not take it over-reserves the arena --
    // 67 MB at 37 frames, silently, and in the one number a graph is
    // refused by. Defaulted TRUE so the figure stays the worst case for
    // a caller that does not know.
    bool matrix_cores = true;
  };

  // The geometry of one forward. `grid_h * grid_w` must equal the
  // tokens per latent frame; the short conv needs the actual grid, and
  // S alone cannot be factored back into one.
  struct Geometry {
    int frames   = 0;
    int grid_h   = 0;
    int grid_w   = 0;
    int text_len = 0;        // 0 disables the text state for this beat
    int tokens_per_frame() const { return grid_h * grid_w; }
    int video_rows() const { return frames * tokens_per_frame(); }
  };

  // The tensors one forward reads and writes. Shaped as the CPU
  // reference documents: x is [video_rows, hidden], q/k/v are
  // [video_rows, heads, head_dim], the text tensors are the prompt's
  // rows, and `out` is [video_rows, heads * head_dim].
  //
  // NOT NECESSARILY TIGHT, AND NOT NECESSARILY FP32. Defaulted, this is
  // exactly what the reference and the goldens hand over. The
  // TRANSFORMER has none of those tensors: q, k and v are three
  // interleaved fields of one fused bf16 projection, grouped per head on
  // the released checkpoint, whose video rows are a run inside a packed
  // sequence that also carries text and audio. Copying them out would
  // cost 8.67 GB a block -- more than the tiling and the bank aliasing
  // together took off the branch -- so the strides below let the kernels
  // read that buffer where it lies instead.
  //
  // `*_off` are BYTE offsets of row 0, which is how a run inside a
  // packed sequence is addressed. The tile offsets are added to them.
  struct Inputs {
    const metal_compute::SharedBuffer* x        = nullptr;
    const metal_compute::SharedBuffer* q_raw    = nullptr;
    const metal_compute::SharedBuffer* k_raw    = nullptr;
    const metal_compute::SharedBuffer* v_raw    = nullptr;
    const metal_compute::SharedBuffer* text_x   = nullptr;
    const metal_compute::SharedBuffer* text_k   = nullptr;
    const metal_compute::SharedBuffer* text_v   = nullptr;

    std::size_t x_off = 0, q_off = 0, k_off = 0, v_off = 0;
    std::size_t text_x_off = 0, text_k_off = 0, text_v_off = 0;

    // Elements between rows / between heads in the q/k/v source. Zero
    // means the tight default (heads * head_dim, head_dim).
    int qkv_row_stride  = 0;
    int qkv_head_stride = 0;
    // bf16 rather than fp32, for q/k/v and for the hidden state. The two
    // are separate flags because the transformer's projection and its
    // normed hidden are separate buffers and need not agree.
    bool qkv_bf16 = false;
    bool x_bf16   = false;
    // `out` narrowed to bf16. The transformer wants it: the readout is
    // about to be projected by a bf16 GEMM into a bf16 residual stream,
    // and fp32 would be 2.89 GB a block for one dispatch. The goldens
    // read fp32, which is the default.
    bool out_bf16 = false;
  };

  static std::unique_ptr<MetalVdnBranch>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const vdn::Config& cfg, const Dims& dims, std::string* err);

  // Bring one block's weights in. Separate from load() because the
  // branch is 4.28 GB over 50 blocks on top of a 33B DiT, so a caller
  // that streams the DiT's blocks wants to stream these the same way.
  bool ensure_block(int layer, std::string* err);
  bool block_ready(int layer) const;

  // Re-read every block instead of keeping it.
  //
  // THIS CHANGES WHICH WEIGHTSET ACCESSOR THE READS TAKE, and that is
  // the whole of it: cached `derived()` entries are owned by the SET, so
  // dropping this object's handles would free nothing and
  // release_block() would be a no-op that reads as a fix. Streaming
  // reads go through stream_derived()/stream_tensor(), which cache
  // nothing and are counted as streaming throughput -- a one-time load
  // reported as streaming traffic misreads as a bounded model thrashing,
  // and the reverse hides a model that really is re-reading.
  //
  // Set it BEFORE the first ensure_block(): blocks already loaded keep
  // whichever accessor brought them in.
  void set_stream_blocks(bool on) { _stream = on; }
  bool stream_blocks() const { return _stream; }

  // Give a block's weights back.
  //
  // SAFE ONLY AFTER THE WORK THAT READS THEM HAS BEEN COMMITTED AND
  // WAITED FOR. An encoded dispatch holds a buffer by pointer, not by
  // reference, so freeing one that a pending command buffer still reads
  // is a use-after-free -- and the transformer runs a RESIDENT stack as
  // a single deferred stream, where no block's work has run when the
  // next begins. Only the streamed path, which commits and waits per
  // block, may call this.
  void release_block(int layer);

  // What the loaded blocks hold right now. Zero for a streaming branch
  // between forwards, which is the point of asking.
  std::size_t resident_bytes() const;

  // Size the scratch for this geometry. Reused across forwards; a
  // changed geometry reallocates.
  bool reserve(const Geometry& g, std::string* err);

  // WHAT THAT WILL COST, before it is spent.
  //
  // reserve() runs at the first forward, which is long after anything
  // that plans memory has had its say -- so a caller that does not ask
  // this sizes a box against the DiT alone and then meets a second
  // allocation of gigabytes. At production geometry the answer is over
  // a gigabyte and it grows LINEARLY with the clip, because seven of
  // the eight largest buffers are [frames, heads, d, d].
  //
  // Static, so it can be asked with no branch built and no weights
  // loaded: it is a function of the geometry and the config, which is
  // everything reserve() consults.
  // WHAT SHARES AND WHAT DOES NOT, since the answer is not obvious and
  // the buffers that matter are the ones that scale.
  //
  // MEASURED (37 latent frames of 510 tokens, the docs pipeline): 1056
  // MB, of which the [Fi, H, d, d] banks are 735 and the per-token half
  // is 304. At 92 frames it is 2220 / 1916 / 304 -- the banks grow with
  // the clip and the per-token half does not, being pinned by the frame
  // tile. So sharing a bank is worth Fi x 3.5 MiB and sharing a
  // per-token buffer is worth 29 MB flat, whatever the clip.
  //
  // Shared today, all of them banks:
  //   L, Li, inv  -> prefix, suffix, state   (the solve's three)
  //   tr          -> A                       (dead after the Cholesky)
  //
  // SIX BANKS IS THE FLOOR without restructuring the solve: after it,
  // tr, inj, prefix, suffix and state are all live into the scan, and
  // `inj` cannot take B because the GEMM that writes inj reads B a whole
  // row at a time -- in place, that is a read-after-write on a row the
  // kernel has already clobbered.
  //
  // NOT SHARED, deliberately: the per-token buffers. `_qf` could take
  // `_kf` (pass 1 ends before pass 2 begins), `_roraw` and `_mmac` could
  // take the two conv rings for the same reason -- about 87 MB, 8% of
  // the total at 37 frames and 4% at 92, i.e. it does not help the case
  // that runs out of memory. Each would also be a lifetime argument
  // spanning half the encode, where the four above are each a few
  // dispatches apart and can be checked by reading them. If the
  // per-token half ever stops being pinned by the tile, revisit.
  static std::size_t scratch_bytes(const Dims& dims, const vdn::Config& cfg,
                                   const Geometry& g);

  // And what the scratch ACTUALLY holds right now -- zero before the
  // first reserve(). Kept beside the estimate for the reason the
  // transformer keeps its pair: an estimate that is never compared with
  // the allocation it predicts drifts silently the first time a buffer
  // is added, and the buffer that gets added is never the small one.
  // See metal_vdn_branch.the_scratch_estimate_matches_the_allocation.
  std::size_t scratch_resident_bytes() const;

  // LEND THE BRANCH SOMEONE ELSE'S MEMORY, and reserve() stops
  // allocating almost all of the above.
  //
  // Every scratch buffer bar the two host-initialised constants is
  // carved from `arena` when one is set and it has the room, and falls
  // back to its own allocation when it does not -- so a bare branch,
  // which is what the goldens and the tests build, is unchanged.
  //
  // THE LENDER IS THE DiT AND THE BUFFER IS ITS ATTENTION ARENA: the
  // one allocation behind qh|kh|vh|oh|ob with the FF intermediate
  // aliased over the whole of it. That arena is dead for exactly the
  // stretch this branch runs in. A block is
  //
  //     norm -> qkv projection -> THE BRANCH -> qk_norm, rope ->
  //     attention -> out projection -> FF
  //
  // and the arena's first writer is the attention, its last reader the
  // FF two steps later. From the previous block's FF to this block's
  // attention nothing in it is live, and the branch runs entirely
  // inside that gap: it reads the fused qkv projection and the normed
  // hidden, which are other buffers, and writes its readout into a
  // third.
  //
  // So it is the DENSE HALF'S memory in the most literal sense. The
  // hybrid replaces full attention with a window plus this branch, and
  // the window wants the arena at an instant the branch does not.
  //
  // Set it BEFORE the first reserve(). A geometry change reallocates
  // the DiT's scratch, which would leave this holding windows into a
  // buffer nothing else uses any more -- valid memory, silently
  // unshared -- so the arena's identity is compared here and a change
  // forces the next reserve() to redo the carve.
  void set_arena(const metal_compute::SharedBuffer& arena);

  // How big that arena has to be for the whole carve to land in it,
  // and what stays this object's own however big it is.
  //
  // pinned_bytes() is `_ones` and `_tState`: both are written by the
  // HOST at reserve() and read by every one of the 50 encodes after
  // it, so they are the two buffers whose contents have to survive the
  // stretch the arena is someone else's. 3.5 MB at the released
  // config, against a gigabyte and up for the rest.
  static std::size_t arena_bytes(const Dims& dims, const vdn::Config& cfg,
                                 const Geometry& g);
  static std::size_t pinned_bytes(const Dims& dims, const vdn::Config& cfg);

  // What the last reserve() actually took out of the arena. Zero with
  // no arena lent, and the check that arena_bytes() is not an
  // underestimate -- see
  // metal_vdn_branch.the_arena_absorbs_the_whole_scratch.
  std::size_t arena_used() const { return _arena_used; }

  // Encode the whole branch for `layer` into `enc`. `bounds` are the
  // UNREBASED per-frame windows -- the anchor skip is applied here,
  // because whether the two frames leave the input is a property of the
  // configuration and not of the caller.
  bool encode(metal_compute::ComputeEncoder& enc, int layer,
              const Geometry& g, const std::vector<vdn::Bound>& bounds,
              const Inputs& in, metal_compute::SharedBuffer& out,
              std::string* err);

  // The SOFTMAX half's per-head gate, applied in place to a bf16
  // attention output.
  //
  // Apart from encode() because it belongs to the other half of the
  // hybrid and runs at a different point in the block -- after
  // attention, before the out projection -- and over the WHOLE packed
  // sequence rather than the video rows. The weights live here because
  // the checkpoint puts them here: `softmax_gate` is a linear-branch
  // tensor, not one of H3's.
  bool encode_softmax_gate(metal_compute::ComputeEncoder& enc, int layer,
                           int rows,
                           const metal_compute::SharedBuffer& x,
                           std::size_t x_off, bool x_bf16, int hidden_stride,
                           metal_compute::SharedBuffer& attn_out,
                           std::size_t out_off, std::string* err);

  // The branch's output projection [hidden, heads*head_dim], bf16 and
  // UNCONVERTED -- the transformer's own GEMM runs it, because that is
  // where the quantisation, the LoRA stack and the steel routing live.
  // Null until the block is loaded.
  const metal_compute::SharedBuffer* out_linear(int layer) const;

  // Per-STAGE timing, for finding where the branch's time goes.
  //
  // The branch is one long run of dispatches on the caller's encoder,
  // so there is nothing to time inside it without splitting -- and the
  // splits need the caller's STREAM, which encode() does not own. Hand
  // it one and every stage boundary below ends the encoder, commits,
  // waits and charges the slice; hand it nothing (the default) and not
  // a line of this changes.
  //
  // The barriers serialise the GPU, so read the SHARES rather than the
  // sum, exactly as with the DiT's own section profile.
  struct Profile {
    double beta = 0, features = 0, stats = 0, alpha = 0, text = 0;
    double conv = 0;   // the spatial half of `features`
    double solve = 0, scan = 0, gather = 0, readout = 0, gate = 0;
    double total() const
    {
      return beta + features + conv + stats + alpha + text + solve
             + scan + gather + readout + gate;
    }
  };
  void set_profile_stream(metal_compute::CommandStream* s)
  {
    _prof_stream = s;
  }
  const Profile& profile() const { return _prof; }
  void clear_profile() { _prof = Profile{}; }

  // The achieved-FLOP/s tally the branch counts its arithmetic into: the
  // host transformer's, since the branch encodes on that model's encoder
  // and commits nothing of its own, so its work is credited at the
  // commits the transformer flushes. Null: nothing is counted.
  //
  // What counts: the feature and gate projections (2MNK), the
  // depthwise spatial and temporal convs (2 * outputs * taps), the
  // per-frame statistics (k^T k and k^T v, 4 * tokens * d^2 a head), the
  // solve (3 * d^3 a frame and head: Cholesky, triangular inverse and
  // their product at d^3/3 each, then a d^3 GEMM's 2 * d^3), the scan's
  // d x d state steps (2 * d^3 each, both directions) and the readout
  // (2 * tokens * d^2 a head). Not counted: elementwise passes, the
  // gather and bridge's blends, and the per-frame decay MLP.
  void set_flops(GpuFlopsTally* t) noexcept { _flops = t; }

  // Non-zero after a forward if any I + A was not positive definite --
  // which is a statement about the PRECISION of the statistics upstream,
  // not about the solve. Read it; a silent NaN reaches the residual
  // stream and the frame it came from is unrecoverable.
  unsigned solve_failures() const;
  void clear_solve_failures();

  const vdn::Config& config() const { return _cfg; }

  // Which matrix-core routes this instance resolved. For tests and the
  // bench: an A/B that measured two arms taking the SAME path is the
  // failure mode these exist to make visible, and "the number did not
  // move" is not evidence either way on its own.
  struct MmaRoutes {
    bool gemm = false, stats = false, readout = false, conv = false;
  };
  MmaRoutes mma_routes() const
  {
    return MmaRoutes{_mma, _mma_stats, _mma_readout, _mma_conv};
  }

private:
  GpuFlopsTally* _flops = nullptr;   // see set_flops()
  void count_(std::uint64_t f) noexcept
  {
    if (_flops != nullptr) { _flops->add(f); }
  }

  struct Block {
    metal_compute::SharedBuffer k_sp, k_tm, v_sp, v_tm;
    metal_compute::SharedBuffer beta_proj;
    metal_compute::SharedBuffer a_down, a_up, a_dt, a_log;
    metal_compute::SharedBuffer g_down, g_up_w, g_up_b;
    metal_compute::SharedBuffer norm;
    metal_compute::SharedBuffer softmax_gate_w, softmax_gate_b;
    metal_compute::SharedBuffer to_out_linear;
    // The bf16 spelling of the four weights the plain GEMMs contract,
    // loaded INSTEAD of the fp32 one when the matrix-core route is on --
    // not beside it. They are bf16 on disk, so this is the raw tensor and
    // the fp32 copies were the transform; a matmul2d takes 16-bit
    // operands and an fp32 contraction misses the matrix units entirely.
    //
    // The values are the SAME: the fp32 path widens these exact bits in
    // the kernel, and bf16 -> f32 is exact. What differs is where the
    // rounding of the PRODUCT happens, which is why the two routes are
    // A/B-able rather than assumed equal.
    metal_compute::SharedBuffer beta_proj_b, g_down_b, g_up_w_b;
    metal_compute::SharedBuffer softmax_gate_wb;
    bool ready = false;
  };

  // Every buffer reserve() hands out (plus the softmax gate's two,
  // which grow lazily elsewhere), in ONE list -- so that counting them
  // and giving them back cannot disagree about which they are. That
  // disagreement is not hypothetical: a geometry with no prompt used to
  // leave the previous one's text banks allocated because the release
  // was written out by hand and missed them.
  std::vector<metal_compute::SharedBuffer*> scratch_buffers_();

  metal_compute::SharedBuffer f32_(const std::string& name,
                                   std::string* err);
  metal_compute::SharedBuffer bf16_(const std::string& name,
                                    std::string* err);
  void gemm_act_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& a, std::size_t aoff,
                 int a_elt, int lda,
                 const metal_compute::SharedBuffer& bmat,
                 const metal_compute::SharedBuffer& bias, int use_bias,
                 metal_compute::SharedBuffer& dst,
                 metal_compute::SharedBuffer& raw, int M, int N, int K,
                 int act, int c_elt);
  static std::size_t block_bytes_(const Block& b);

  // The set is held for this object's LIFETIME: the buffers above are
  // refcounted aliases of it, and the checkpoint unmaps when the last
  // holder goes, not the first.
  std::shared_ptr<WeightSet>   _ws;
  metal_compute::MetalCompute* _mc = nullptr;
  vdn::Config                  _cfg;
  Dims                         _dims;
  bool                         _stream = false;
  std::vector<Block>           _blocks;

  metal_compute::ComputeLibrary _lib_branch, _lib_solve, _lib_mma;
  // THE MATRIX-CORE ROUTE, for the branch's three plain GEMMs alone
  // (beta, the output gate's two, the softmax half's gate). They are
  // y = x W^T with a bf16 x and a bf16 [N, K] weight, which is exactly
  // what dense_gemm_mma already does for every DiT in the tree -- so
  // this borrows that kernel rather than growing a fourth copy of it,
  // and the bias and sigmoid come back as vdn_bias_act_f32.
  //
  // NOT the frame statistics, the solve or the readout: those are fp32
  // by necessity (A is what the Cholesky inverts) or shaped as a batch
  // of 128-wide tiles, and a batched matmul2d over per-(frame, head)
  // slabs is a different kernel with a different answer to measure.
  //
  // Off unless the GPU has matrix cores AND the features are bf16: an
  // fp32 branch is the goldens' path and must stay bit-for-bit what it
  // was. VPIPE_VDN_NO_MMA forces it off for the A/B.
  metal_compute::ComputeFunction _mma_gemm, _mma_gemm_deep, _bias_act;
  bool _mma = false;
  // And the branch's LARGEST arithmetic: the per-frame statistics, two
  // [d, S] x [S, d] products per (frame, head). A separate flag from
  // _mma because it carries one condition of its own -- the kernel's
  // tile IS head_dim, so a checkpoint whose linear_head_dim is not 128
  // keeps the fp32 kernel while the plain GEMMs still go to the matrix
  // cores. It also changes what the FEATURES hold (sqrt(beta) folded
  // in), which is why nothing may read them without asking this.
  metal_compute::ComputeLibrary  _lib_vdn_mma;
  metal_compute::ComputeFunction _stats_mma;
  bool _mma_stats = false;
  // The readout's contraction, and the row-wise tail it cannot carry.
  // Same tile-is-head_dim condition as the statistics, and the same
  // reason: one [S, d] x [d, d] per (frame, head).
  metal_compute::ComputeFunction _readout_mma, _readout_norm;
  bool _mma_readout = false;
  // THE SHORT CONV'S SPATIAL HALF on the MPP hardware convolution op.
  //
  // A depthwise conv cannot be handed to that op as itself -- it takes
  // groups == 1 -- so this runs it DENSE over a block of kDwBlock
  // channels with a block-diagonal weight, doing kDwBlock times the
  // arithmetic on the matrix units for the same answer. MEASURED 1.25x
  // at the generation shape, and a loss at every other block size; the
  // sweep is conv2d_mma.hw_op_depthwise_perf and the reason it is a
  // sweep is that neither direction is monotone.
  //
  // A THIRD flag, not the readout's: it needs the block size to divide
  // the source's channel GROUP (a block straddling two heads would read
  // the next head's channels under this one's weights) and it is the one
  // piece of this that a checkpoint with a different conv size cannot
  // take. VPIPE_VDN_NO_DW_MMA is its own A/B.
  metal_compute::ComputeLibrary  _lib_conv_mma;
  metal_compute::ComputeFunction _conv_dw[3], _dw_weight;
  bool _mma_conv = false;
  // Which of kDwHeights this geometry chose, resolved in reserve().
  int _dw_pick = -1;
  metal_compute::SharedBuffer _dw_k, _dw_v;
  // NO _gate. vdn_gate_f32 still exists and is still checked against
  // the CPU oracle in tests/unit-tests/vdn-linear-branch.cc, but nothing
  // here runs it any more: beta, the output gate and the softmax gate
  // all go through vdn_gemm_act_f32, because all three are matrix
  // multiplies and the per-token spelling ran at DRAM rate.
  metal_compute::ComputeFunction _sp, _ta, _stats, _sym, _mean;
  metal_compute::ComputeFunction _alpha, _bridge, _gather, _readout;
  metal_compute::ComputeFunction _step, _scale, _sgate, _gemm_act;
  metal_compute::ComputeFunction _chol, _trinv, _invtr, _gemm;

  Geometry _geom;
  int      _reserved_tile = 0;
  // The lent arena and how much of it the last carve took. See
  // set_arena(); empty is the ordinary standalone-allocation case.
  metal_compute::SharedBuffer _arena;
  std::size_t                 _arena_used = 0;
  metal_compute::SharedBuffer _qf, _kf, _vf, _beta, _A, _B, _mean_b;
  // The spatial conv's output, as a RING over frames -- one per
  // convolved tensor, because both must survive the other's pass.
  //
  // The tile loop's input window is haloed by two frames on each side
  // (the temporal conv is a 5-tap stencil) while its OUTPUT is the body
  // alone, so consecutive tiles overlap by 2 * kConvHalo frames. Held
  // flat, every one of those frames was convolved TWICE -- 72 frame
  // convolutions for 37 frames of output at the default tile, which is
  // the single largest piece of redundant work the branch had.
  //
  // A ring of kFrameTile + 2 * kConvHalo slots is exactly enough and no
  // more: the frames a tile overwrites are precisely the ones the next
  // tile no longer wants. It costs one extra buffer against the shared
  // scratch it replaces (58 MB at generation geometry) and no copies at
  // all -- the alternative, sliding a flat buffer, would move those same
  // frames every tile.
  metal_compute::SharedBuffer _ring_k, _ring_v;
  // No _L / _Li / _inv: the solve's three intermediates borrow the
  // scan's banks, which are dead until it finishes. See reserve().
  metal_compute::SharedBuffer _alpha_b, _tr, _inj;
  metal_compute::SharedBuffer _prefix, _suffix, _fb, _fa, _state, _gate_b;
  metal_compute::SharedBuffer _tkf, _tvf, _tbeta, _tA, _tB, _tL, _tLi;
  metal_compute::SharedBuffer _tInv, _tTr, _tState, _ones, _fail, _dummy;
  metal_compute::SharedBuffer _bidx, _aidx, _hasb, _hasa, _brb, _bra;
  metal_compute::SharedBuffer _sgate_b;
  // Where the matrix-core route lands a bare product before the bias
  // and the sigmoid. Two of them because the two callers have different
  // widest shapes -- a frame TILE into 7168 columns for the branch's own
  // gate, the whole packed sequence into 56 for the softmax half's.
  metal_compute::SharedBuffer _mmac, _sgate_raw, _roraw;
  // The low-rank chain's intermediate, and a REAL zero bias -- see the
  // note at the allocation for what the one-float `_dummy` was doing.
  metal_compute::SharedBuffer _lo, _zeros;
  std::vector<vdn::Bound>     _cached_bounds;
  metal_compute::CommandStream* _prof_stream = nullptr;
  Profile                       _prof;
};

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
