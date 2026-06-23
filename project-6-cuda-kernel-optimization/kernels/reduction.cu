#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr int THREADS = 256;

// ---------------------------------------------------------------------------
// Naive: interleaved addressing.
// At each step s, threads where tid % (2*s) == 0 are active.
// With s=1: warps contain alternating active/inactive threads → warp divergence.
// With s=32: only thread 0 in a warp is active, max divergence.
// Also causes strided shared memory access that conflicts with bank layout.
// ---------------------------------------------------------------------------
__global__ void _reduce_naive_kernel(
    const float* __restrict__ data,
    float*       __restrict__ out,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? data[i] : 0.f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0)         // divergent: half the threads idle
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Sequential: sequential addressing.
// The active set is always the first s threads: no thread in a warp mixes
// active/inactive → zero warp divergence.
// Access pattern: sdata[tid] += sdata[tid + s] is sequential → no bank conflicts.
// ---------------------------------------------------------------------------
__global__ void _reduce_sequential_kernel(
    const float* __restrict__ data,
    float*       __restrict__ out,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? data[i] : 0.f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Warp shuffle: two elements per thread ("first add during load"), shared
// memory reduction to 64 active values, then warp-level shuffle for the
// final 5 steps.  __shfl_down_sync communicates register values within a
// warp without touching shared memory at all.
//
// Requires blockDim.x >= 64.
// Each block processes 2 * blockDim.x elements → half as many blocks needed.
// ---------------------------------------------------------------------------
__global__ void _reduce_shuffle_kernel(
    const float* __restrict__ data,
    float*       __restrict__ out,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread sums two elements during the load (halves block count).
    float val = 0.f;
    if (i < N)               val += data[i];
    if (i + blockDim.x < N)  val += data[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Shared memory reduction until 64 active elements remain.
    // Loop exits when s == 32 (32 is not > 32), so sdata[0..63] are valid.
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // First warp: fold 64 → 32 from shared memory, then use shuffle.
    // The last __syncthreads() in the loop guarantees sdata[0..63] are visible.
    if (tid < 32) {
        float wval = sdata[tid] + sdata[tid + 32];
        wval += __shfl_down_sync(0xffffffff, wval, 16);
        wval += __shfl_down_sync(0xffffffff, wval, 8);
        wval += __shfl_down_sync(0xffffffff, wval, 4);
        wval += __shfl_down_sync(0xffffffff, wval, 2);
        wval += __shfl_down_sync(0xffffffff, wval, 1);
        if (tid == 0) out[blockIdx.x] = wval;
    }
}

// ---------------------------------------------------------------------------
// Host-side launchers — return partial sums; Python sums to get final value.
// Timing the kernel alone (not the partial-sum collection) is deliberate:
// it isolates the reduction algorithm itself.
// ---------------------------------------------------------------------------
torch::Tensor reduce_naive(torch::Tensor data) {
    TORCH_CHECK(data.is_cuda() && data.dtype() == torch::kFloat32,
                "Requires float32 CUDA tensor");
    int N      = data.numel();
    int blocks = (N + THREADS - 1) / THREADS;
    auto out   = torch::zeros({blocks}, data.options());
    _reduce_naive_kernel<<<blocks, THREADS, THREADS * sizeof(float)>>>(
        data.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

torch::Tensor reduce_sequential(torch::Tensor data) {
    TORCH_CHECK(data.is_cuda() && data.dtype() == torch::kFloat32,
                "Requires float32 CUDA tensor");
    int N      = data.numel();
    int blocks = (N + THREADS - 1) / THREADS;
    auto out   = torch::zeros({blocks}, data.options());
    _reduce_sequential_kernel<<<blocks, THREADS, THREADS * sizeof(float)>>>(
        data.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

torch::Tensor reduce_shuffle(torch::Tensor data) {
    TORCH_CHECK(data.is_cuda() && data.dtype() == torch::kFloat32,
                "Requires float32 CUDA tensor");
    int N      = data.numel();
    int blocks = (N + THREADS * 2 - 1) / (THREADS * 2);
    auto out   = torch::zeros({blocks}, data.options());
    _reduce_shuffle_kernel<<<blocks, THREADS, THREADS * sizeof(float)>>>(
        data.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}
