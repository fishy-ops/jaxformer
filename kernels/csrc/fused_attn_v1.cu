// Fused causal multi-head attention, forward pass, v1 (fp32, SIMT).
//
// This is the FlashAttention idea at its simplest: never materialize the T x T score
// matrix. Each block streams over key/value tiles keeping a running softmax
// (max `m`, denominator `l`) and an output accumulator that is rescaled as the max
// moves. Memory is therefore O(T) per query row, not O(T^2), which is the entire
// reason to write this instead of matmul -> softmax -> matmul.
//
// Design, shaped by the target (RTX 2070 Super, Turing sm_75, 64 KB smem/block):
//   * One block per (batch*head, query-tile). blockDim = BR threads, one query row
//     each. The row's query vector and its output accumulator live in registers.
//   * K and V tiles are staged in shared memory (BC x HEAD_DIM each = 32 KB at
//     fp32/64/64, comfortably under the 64 KB ceiling).
//   * Causal skip is structural: key tiles are visited in order, so once a tile
//     starts past this query-tile's last row we break out of the loop entirely,
//     which removes ~half the work at long sequence lengths. Within the diagonal
//     tile the per-key `kpos > qi` test masks the rest of that row.
//
// v1 favors being obviously correct over being fast; tile sizes, vectorized loads,
// and tensor cores come later (v2) and are driven by Nsight Compute, not guessed.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK
#include <math.h>

template <int HEAD_DIM, int BR, int BC>
__global__ void fused_attn_fwd_kernel(
    const float* __restrict__ Q,  // (B*H, T, HEAD_DIM), row-contiguous
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int T, float scale, bool causal) {

    const int bh = blockIdx.y;
    const int tid = threadIdx.x;              // 0 .. BR-1
    const int qi = blockIdx.x * BR + tid;     // global query row this thread owns

    const long base = (long)bh * T * HEAD_DIM;
    const float* Qb = Q + base;
    const float* Kb = K + base;
    const float* Vb = V + base;
    float* Ob = O + base;

    __shared__ float Ks[BC * HEAD_DIM];
    __shared__ float Vs[BC * HEAD_DIM];

    // Per-query-row state kept in registers. The unrolled loops below index qreg/acc
    // with compile-time constants so they stay in registers rather than spilling.
    float qreg[HEAD_DIM];
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;
    float m = -INFINITY;  // running row max
    float l = 0.0f;       // running softmax denominator

    const bool active = qi < T;
    if (active) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) qreg[d] = Qb[(long)qi * HEAD_DIM + d];
    }

    // Last query position in this block's tile; every thread agrees on it, so the
    // causal tile-skip below is uniform and never diverges the loop bound.
    const int q_tile_last = min(blockIdx.x * BR + BR - 1, T - 1);
    const int num_key_tiles = (T + BC - 1) / BC;

    for (int j = 0; j < num_key_tiles; j++) {
        const int kj0 = j * BC;
        if (causal && kj0 > q_tile_last) break;  // no visible keys in this or later tiles

        // Cooperatively stage K and V tiles. Threads stride over BC*HEAD_DIM elements;
        // out-of-range rows (ragged tail) are zeroed and masked out by the score loop.
        for (int idx = tid; idx < BC * HEAD_DIM; idx += BR) {
            const int c = idx / HEAD_DIM;
            const int d = idx % HEAD_DIM;
            const int kpos = kj0 + c;
            if (kpos < T) {
                Ks[idx] = Kb[(long)kpos * HEAD_DIM + d];
                Vs[idx] = Vb[(long)kpos * HEAD_DIM + d];
            } else {
                Ks[idx] = 0.0f;
                Vs[idx] = 0.0f;
            }
        }
        __syncthreads();

        if (active) {
            for (int c = 0; c < BC; c++) {
                const int kpos = kj0 + c;
                if (kpos >= T) break;               // ragged tail
                if (causal && kpos > qi) break;     // keys ordered: rest are masked

                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) s += qreg[d] * Ks[c * HEAD_DIM + d];
                s *= scale;

                // Online softmax: shift by the new running max, rescale the old mass.
                const float m_new = fmaxf(m, s);
                const float corr = expf(m - m_new);
                const float p = expf(s - m_new);
                l = l * corr + p;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    acc[d] = acc[d] * corr + p * Vs[c * HEAD_DIM + d];
                m = m_new;
            }
        }
        __syncthreads();  // all reads of Ks/Vs done before the next tile overwrites them
    }

    if (active) {
        const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) Ob[(long)qi * HEAD_DIM + d] = acc[d] * inv_l;
    }
}

// (B, H, T, HEAD_DIM) fp32 -> same shape. `causal` toggles the triangular mask.
torch::Tensor fused_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "v1 is fp32 only");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4, "expected (B, H, T, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q/K/V shapes must match");

    const auto B = Q.size(0), H = Q.size(1), T = Q.size(2), Dh = Q.size(3);
    TORCH_CHECK(Dh == 64, "v1 supports head_dim == 64 (the model's configuration)");

    auto Qc = Q.contiguous(), Kc = K.contiguous(), Vc = V.contiguous();
    auto O = torch::empty_like(Qc);

    constexpr int HEAD_DIM = 64, BR = 64, BC = 64;
    const dim3 grid((T + BR - 1) / BR, B * H);
    const dim3 block(BR);
    const float scale = 1.0f / sqrtf((float)Dh);

    fused_attn_fwd_kernel<HEAD_DIM, BR, BC><<<grid, block>>>(
        Qc.data_ptr<float>(), Kc.data_ptr<float>(), Vc.data_ptr<float>(),
        O.data_ptr<float>(), (int)T, scale, causal);

    // Surface launch/exec errors at the Python boundary instead of much later.
    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}
