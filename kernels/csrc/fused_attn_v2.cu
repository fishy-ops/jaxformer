// Fused causal multi-head attention, forward pass, v2 (fp16 WMMA tensor cores).
//
// v1's Nsight profile was unambiguous: 12% occupancy, register-limited at 164
// registers/thread because each thread carried its query vector and output
// accumulator as 64-float register arrays; DRAM sat at 1.5%, so it was
// latency/occupancy-bound, not bandwidth-bound. v2 attacks exactly that:
//
//   * The two matmuls (QK^T and P@V) run on Turing's tensor cores via nvcuda::wmma
//     m16n16k16 half fragments (supported on sm_75; FA2's mma.sync path is not). The
//     accumulators live in warp-distributed fragments, so the per-thread register
//     arrays that capped occupancy are gone.
//   * Q/K/V fragments are loaded straight from global memory rather than staged in
//     shared, which removes v1's shared-store bank conflicts entirely and shrinks the
//     shared footprint to a few KB (only the scores, probabilities, and the running
//     output accumulator live in shared).
//
// One warp owns a 16-row query tile and streams over 16-wide key tiles, keeping the
// same online-softmax recurrence as v1 (running max m, denominator l, rescaled output
// accumulator O). Inputs are fp16; all accumulation is fp32.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <math.h>

using namespace nvcuda;

namespace {
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int HEAD_DIM = 64;         // 4 k-steps of 16
constexpr int TILE = 16;             // query rows and key cols per warp step
// Row stride of the shared output accumulator, padded by one. The scalar softmax
// touches Os[row*STRIDE + d]; with an unpadded stride of 64 (== 0 mod 32) all 16
// active rows collide on the same bank -- a 16-way conflict Nsight measured at
// >550M events. An odd padding makes consecutive rows land in different banks, which
// is the single change that unblocks the tensor cores. Os is not read by WMMA, so
// the padding does not violate any fragment leading-dimension constraint.
constexpr int OS_STRIDE = HEAD_DIM + 1;
}  // namespace

__global__ void fused_attn_v2_kernel(
    const half* __restrict__ Q,  // (B*H, Tp, HEAD_DIM), Tp padded to a multiple of 16
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    int Tp, int T, float scale, bool causal) {

    const int bh = blockIdx.y;
    const int q0 = blockIdx.x * TILE;   // first query row of this warp's tile
    const int lane = threadIdx.x;       // 0..31 (one warp)

    const long base = (long)bh * Tp * HEAD_DIM;
    const half* Qb = Q + base;
    const half* Kb = K + base;
    const half* Vb = V + base;
    float* Ob = O + base;

    __shared__ float Ss[TILE * TILE];   // scores, then scaled/masked scores
    __shared__ half  Ps[TILE * TILE];   // softmax probabilities (fp16 for P@V)
    __shared__ float Os[TILE * OS_STRIDE];  // output accumulator (padded), rescaled across tiles
    __shared__ float Ot[TILE * WMMA_N];    // one P@V n-tile before adding into Os
    __shared__ float Ms[TILE], Ls[TILE];   // running max / denominator per query row

    for (int i = lane; i < TILE * OS_STRIDE; i += 32) Os[i] = 0.0f;
    if (lane < TILE) { Ms[lane] = -INFINITY; Ls[lane] = 0.0f; }
    __syncwarp();

    const int q_last = q0 + TILE - 1;
    const int num_key_tiles = (Tp + TILE - 1) / TILE;

    for (int kt = 0; kt < num_key_tiles; kt++) {
        const int k0 = kt * TILE;
        if (causal && k0 > q_last) break;  // no visible keys in this or later tiles

        // ---- S = Q @ K^T, accumulated over the 4 head-dim steps ----------------
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
        #pragma unroll
        for (int dk = 0; dk < HEAD_DIM; dk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            // col_major load of K yields K^T: element(dim, key) = K[key, dim].
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(a_frag, Qb + (long)q0 * HEAD_DIM + dk, HEAD_DIM);
            wmma::load_matrix_sync(b_frag, Kb + (long)k0 * HEAD_DIM + dk, HEAD_DIM);
            wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
        }
        wmma::store_matrix_sync(Ss, s_frag, TILE, wmma::mem_row_major);
        __syncwarp();

        // ---- scale, causal mask, and the online-softmax update, one row per lane -
        if (lane < TILE) {
            const int r = lane;
            const int qpos = q0 + r;
            float rowmax = -INFINITY;
            #pragma unroll
            for (int c = 0; c < TILE; c++) {
                const int kpos = k0 + c;
                const bool valid = (kpos < T) && (qpos < T) && (!causal || kpos <= qpos);
                const float s = valid ? Ss[r * TILE + c] * scale : -INFINITY;
                Ss[r * TILE + c] = s;
                rowmax = fmaxf(rowmax, s);
            }
            const float m_old = Ms[r];
            const float m_new = fmaxf(m_old, rowmax);
            // exp(-inf) == 0, so a fully-masked tile (rowmax == -inf) leaves the row
            // untouched: corr == 1 when m_new == m_old, and every p below is 0.
            const float corr = __expf(m_old - m_new);
            float rowsum = 0.0f;
            #pragma unroll
            for (int c = 0; c < TILE; c++) {
                const float p = __expf(Ss[r * TILE + c] - m_new);
                Ps[r * TILE + c] = __float2half(p);
                rowsum += p;
            }
            Ls[r] = Ls[r] * corr + rowsum;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) Os[r * OS_STRIDE + d] *= corr;
            Ms[r] = m_new;
        }
        __syncwarp();

        // ---- delta_O = P @ V, added into the rescaled accumulator ---------------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, Ps, TILE);
        #pragma unroll
        for (int nt = 0; nt < HEAD_DIM; nt += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, Vb + (long)k0 * HEAD_DIM + nt, HEAD_DIM);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
            wmma::store_matrix_sync(Ot, o_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            if (lane < TILE) {
                const int r = lane;
                #pragma unroll
                for (int n = 0; n < WMMA_N; n++) Os[r * OS_STRIDE + nt + n] += Ot[r * WMMA_N + n];
            }
            __syncwarp();
        }
    }

    // ---- normalize and write out --------------------------------------------
    if (lane < TILE) {
        const int r = lane;
        const int qpos = q0 + r;
        if (qpos < T) {
            const float inv = (Ls[r] > 0.0f) ? (1.0f / Ls[r]) : 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                Ob[(long)qpos * HEAD_DIM + d] = Os[r * OS_STRIDE + d] * inv;
        }
    }
}

// (B, H, T, 64) fp16 -> (B, H, T, 64) fp32. `causal` toggles the triangular mask.
torch::Tensor fused_attn_v2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "v2 expects fp16 inputs");
    TORCH_CHECK(Q.dim() == 4, "expected (B, H, T, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q/K/V shapes must match");
    const auto B = Q.size(0), H = Q.size(1), T = Q.size(2), Dh = Q.size(3);
    TORCH_CHECK(Dh == HEAD_DIM, "v2 supports head_dim == 64");

    // WMMA reads full 16x16 tiles; pad T up so no fragment load runs off the end.
    const int64_t Tp = ((T + TILE - 1) / TILE) * TILE;
    auto pad = [&](torch::Tensor x) {
        x = x.contiguous();
        if (Tp != T) x = torch::constant_pad_nd(x, {0, 0, 0, Tp - T}, 0);
        return x.contiguous();
    };
    auto Qp = pad(Q).view({B * H, Tp, Dh});
    auto Kp = pad(K).view({B * H, Tp, Dh});
    auto Vp = pad(V).view({B * H, Tp, Dh});

    auto Op = torch::empty({B * H, Tp, Dh}, Q.options().dtype(torch::kFloat32));

    const dim3 grid((int)(Tp / TILE), (int)(B * H));
    const dim3 block(32);
    const float scale = 1.0f / sqrtf((float)Dh);

    fused_attn_v2_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(Qp.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Kp.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Vp.data_ptr<at::Half>()),
        Op.data_ptr<float>(), (int)Tp, (int)T, scale, causal);
    C10_CUDA_CHECK(cudaGetLastError());

    return Op.view({B, H, Tp, Dh}).narrow(2, 0, T).contiguous();
}
