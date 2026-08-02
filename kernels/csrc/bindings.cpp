// pybind entry point, compiled by cl (not nvcc). Keeping the module here rather than
// in the .cu is the split that builds cleanly with nvcc + MSVC (see
// docs/windows_gpu_setup.md): the .cu stays free of pybind's heaviest templates.
#include <torch/extension.h>

torch::Tensor fused_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_attn_forward,
          "Fused causal multi-head attention forward (v1, fp32)",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("causal") = true);
}
