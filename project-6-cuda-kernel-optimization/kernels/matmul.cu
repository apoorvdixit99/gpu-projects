#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// Naive: each output element C[row][col] is computed by one thread, which
// reads an entire row of A and an entire column of B from global memory.
// For an N×N matrix: 2N² threads, each issuing N global reads — O(N³) traffic.
// ---------------------------------------------------------------------------
__global__ void _matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Tiled: shared-memory blocking with TILE×TILE sub-matrices.
// Each block loads a tile of A and a tile of B into shared memory, computes
// the partial dot products for that tile, then advances to the next tile.
// Global memory traffic drops by a factor of TILE vs the naive kernel.
//
// TILE=16 → 2 × 16×16 × 4 bytes = 2 KB shared memory per block (well within
// the 100 KB available on Ada Lovelace).
// ---------------------------------------------------------------------------
__global__ void _matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load tile from A (row-major) and B (row-major) into shared memory.
        // Out-of-bounds threads load 0 to handle non-multiple-of-TILE sizes.
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t * TILE + threadIdx.x < K)
            ? A[row * K + t * TILE + threadIdx.x] : 0.f;
        sB[threadIdx.y][threadIdx.x] =
            (t * TILE + threadIdx.y < K && col < N)
            ? B[(t * TILE + threadIdx.y) * N + col] : 0.f;
        __syncthreads();

        // Compute partial dot product for this tile.
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "float32 required");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
                "Shape mismatch: A is [M,K], B is [K,N]");

    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    _matmul_naive_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "float32 required");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0),
                "Shape mismatch: A is [M,K], B is [K,N]");

    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    _matmul_tiled_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}
