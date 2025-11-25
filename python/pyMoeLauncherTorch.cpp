#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

#include "iMoeKernelLauncher.h"

namespace py = pybind11;

#define HIP_ERROR_CHECK(condition)                                        \
  {                                                                       \
    const hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                            \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;  \
      TORCH_CHECK(error == hipSuccess);                                   \
    }                                                                     \
  }

#define CHECK_TENSOR(tensor)                                                \
  {                                                                         \
    TORCH_CHECK(tensor.defined(), #tensor " tensor is undefined");          \
    TORCH_CHECK(tensor.is_cuda(), #tensor " must be a CUDA tensor (ROCm)"); \
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,                         \
                #tensor " must be float32");                                \
    TORCH_CHECK(tensor.is_contiguous(), #tensor " must be contiguous");     \
  }

class MoeKernelLauncherWrapper {
 public:
  void launch(const at::Tensor& tokens, const at::Tensor& gate_weights,
              const at::Tensor& ffn1_expert_weights,
              const at::Tensor& ffn2_expert_weights, const at::Tensor& output,
              int tokens_num) {
    if (!ptr_) throw std::runtime_error("Launcher was destroyed");

    CHECK_TENSOR(tokens);
    CHECK_TENSOR(gate_weights);
    CHECK_TENSOR(ffn1_expert_weights);
    CHECK_TENSOR(ffn2_expert_weights);
    CHECK_TENSOR(output);

    const void* t_ptr = tokens.data_ptr();
    const void* gw_ptr = gate_weights.data_ptr();
    const void* f1_ptr = ffn1_expert_weights.data_ptr();
    const void* f2_ptr = ffn2_expert_weights.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    HIP_ERROR_CHECK(ptr_->Launch(t_ptr, gw_ptr, f1_ptr, f2_ptr, out_ptr,
                                 tokens_num, stream));
  }

  void destroy() {
    if (ptr_) {
      hipStream_t stream = at::cuda::getCurrentCUDAStream();
      HIP_ERROR_CHECK(DestroyLauncher(ptr_, stream));
      ptr_ = nullptr;
    }
  }

  void create(int max_tokens) {
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    HIP_ERROR_CHECK(CreateLauncher(&ptr_, stream, max_tokens));
  }

  bool valid() const { return ptr_ != nullptr; }

 private:
  moe::IMoeKernelLauncher* ptr_{nullptr};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyTorch bindings for IMoeKernelLauncher (ROCm/HIP)";

  py::class_<MoeKernelLauncherWrapper>(m, "MoeKernelLauncher")
      .def(py::init())
      .def("launch", &MoeKernelLauncherWrapper::launch, py::arg("tokens"),
           py::arg("gate_weights"), py::arg("ffn1_expert_weights"),
           py::arg("ffn2_expert_weights"), py::arg("output"),
           py::arg("tokens_num"))
      .def("destroy", &MoeKernelLauncherWrapper::destroy)
      .def("create", &MoeKernelLauncherWrapper::create, py::arg("max_tokens"))
      .def_property_readonly("valid", &MoeKernelLauncherWrapper::valid);
}