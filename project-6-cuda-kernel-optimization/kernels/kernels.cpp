#include <torch/extension.h>

// Declarations — implementations live in the companion .cu files.
torch::Tensor vec_add_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor vec_add_opt(torch::Tensor A, torch::Tensor B);

torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B);
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);

torch::Tensor reduce_naive(torch::Tensor data);
torch::Tensor reduce_sequential(torch::Tensor data);
torch::Tensor reduce_shuffle(torch::Tensor data);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Vector addition
    m.def("vec_add_naive", &vec_add_naive,
          "Naive vector addition (one 32-bit load per thread)");
    m.def("vec_add_opt",   &vec_add_opt,
          "Optimized vector addition (float4 grid-stride, 128-bit loads)");

    // Matrix multiplication
    m.def("matmul_naive", &matmul_naive,
          "Naive matmul (global memory, no tiling)");
    m.def("matmul_tiled", &matmul_tiled,
          "Tiled matmul (16x16 shared-memory tiles)");

    // Parallel reduction
    m.def("reduce_naive",      &reduce_naive,
          "Naive reduction (interleaved addressing, warp divergence)");
    m.def("reduce_sequential", &reduce_sequential,
          "Sequential reduction (no divergence, no bank conflicts)");
    m.def("reduce_shuffle",    &reduce_shuffle,
          "Warp-shuffle reduction (float4 load + __shfl_down_sync)");
}
