#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Naive: one thread per element, single 32-bit global load per thread.
// Already has coalesced access but leaves GPU bandwidth under-utilized
// because each thread issues only one 32-bit (4-byte) load/store.
// ---------------------------------------------------------------------------
__global__ void _vec_add_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ---------------------------------------------------------------------------
// Optimized: grid-stride loop + float4 vectorized loads.
// Each thread issues one 128-bit (16-byte) load per iteration, loading 4
// floats at once.  The grid-stride loop lets a fixed, SM-saturating grid
// cover arbitrarily large arrays without launching excess blocks.
//
// Requires: N % 4 == 0  (enforced in the host launcher).
// Pointer alignment: PyTorch CUDA tensors are at least 64-byte aligned, so
// reinterpreting float* as float4* is always safe here.
// ---------------------------------------------------------------------------
__global__ void _vec_add_opt_kernel(
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    float4*       __restrict__ C,
    int N4)   // N4 = N / 4
{
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < N4; i += stride) {
        float4 a = A[i];
        float4 b = B[i];
        C[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
torch::Tensor vec_add_naive(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "float32 required");
    auto C = torch::empty_like(A);
    const int N = A.numel();
    const int T = 256;
    _vec_add_naive_kernel<<<(N + T - 1) / T, T>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

torch::Tensor vec_add_opt(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "float32 required");
    TORCH_CHECK(A.numel() % 4 == 0, "N must be divisible by 4 for float4 kernel");
    auto C = torch::empty_like(A);
    const int N4 = A.numel() / 4;
    const int T  = 256;
    // Cap blocks at 1024 — the grid-stride loop handles remaining elements.
    const int B_ = std::min((N4 + T - 1) / T, 1024);
    _vec_add_opt_kernel<<<B_, T>>>(
        reinterpret_cast<const float4*>(A.data_ptr<float>()),
        reinterpret_cast<const float4*>(B.data_ptr<float>()),
        reinterpret_cast<float4*>(C.data_ptr<float>()),
        N4);
    return C;
}
