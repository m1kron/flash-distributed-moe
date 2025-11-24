#include <ATen/cuda/CUDAContext.h>  // maps to HIP on ROCm
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

#include "iMoeKernelLauncher.h"

namespace py = pybind11;

static void check_tensor(const at::Tensor& t, const char* name,
                         bool must_be_cuda = true, bool must_be_f32 = true,
                         bool must_be_contig = true) {
  if (!t.defined()) {
    throw std::invalid_argument(std::string(name) + " tensor is undefined");
  }
  if (must_be_cuda && !t.is_cuda()) {
    throw std::invalid_argument(std::string(name) + " must be a CUDA tensor (ROCm)");
  }
  if (must_be_f32 && t.scalar_type() != at::kFloat) {
    throw std::invalid_argument(std::string(name) + " must be float32");
  }
  if (must_be_contig && !t.is_contiguous()) {
    throw std::invalid_argument(std::string(name) + " must be contiguous");
  }
}

class MoeKernelLauncherWrapper {
 public:
  explicit MoeKernelLauncherWrapper(int max_tokens) {
    hipError_t err = CreateLauncher(&ptr_, /*stream*/ nullptr, max_tokens);
    if (err != hipSuccess || ptr_ == nullptr) {
      throw std::runtime_error("CreateLauncher failed, hipError=" +
                               std::to_string(static_cast<int>(err)));
    }
  }

  ~MoeKernelLauncherWrapper() {
    if (ptr_) {
      DestroyLauncher(ptr_, /*stream*/ nullptr);
      ptr_ = nullptr;
    }
  }

  int launch(const at::Tensor& tokens, const at::Tensor& gate_weights,
             const at::Tensor& ffn1_expert_weights,
             const at::Tensor& ffn2_expert_weights, const at::Tensor& output,
             int tokens_num) {
    if (!ptr_) throw std::runtime_error("Launcher was destroyed");

    check_tensor(tokens, "tokens");
    check_tensor(gate_weights, "gate_weights");
    check_tensor(ffn1_expert_weights, "ffn1_expert_weights");
    check_tensor(ffn2_expert_weights, "ffn2_expert_weights");
    check_tensor(output, "output");

    const int dev_idx = tokens.get_device();
    if (gate_weights.get_device() != dev_idx ||
        ffn1_expert_weights.get_device() != dev_idx ||
        ffn2_expert_weights.get_device() != dev_idx ||
        output.get_device() != dev_idx) {
      throw std::invalid_argument("All tensors must be on the same CUDA device");
    }

    hipStream_t stream = at::cuda::getCurrentCUDAStream(dev_idx).stream();

    const void* t_ptr  = tokens.data_ptr();
    const void* gw_ptr = gate_weights.data_ptr();
    const void* f1_ptr = ffn1_expert_weights.data_ptr();
    const void* f2_ptr = ffn2_expert_weights.data_ptr();
    void* out_ptr      = const_cast<void*>(output.data_ptr());

    hipError_t err = ptr_->Launch(t_ptr, gw_ptr, f1_ptr, f2_ptr, out_ptr, tokens_num, stream);
    return static_cast<int>(err);
  }

  void destroy() {
    if (ptr_) {
      DestroyLauncher(ptr_, /*stream*/ nullptr);
      ptr_ = nullptr;
    }
  }

  bool valid() const { return ptr_ != nullptr; }

 private:
  moe::IMoeKernelLauncher* ptr_{nullptr};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyTorch bindings for IMoeKernelLauncher (ROCm/HIP)";

  py::class_<MoeKernelLauncherWrapper>(m, "MoeKernelLauncher")
      .def(py::init<int>(), py::arg("max_tokens"))
      .def("launch", &MoeKernelLauncherWrapper::launch,
           py::arg("tokens"),
           py::arg("gate_weights"),
           py::arg("ffn1_expert_weights"),
           py::arg("ffn2_expert_weights"),
           py::arg("output"),
           py::arg("tokens_num"))
      .def("destroy", &MoeKernelLauncherWrapper::destroy)
      .def_property_readonly("valid", &MoeKernelLauncherWrapper::valid);

  m.def("create_launcher",
        [](int max_tokens) { return MoeKernelLauncherWrapper(max_tokens); },
        py::arg("max_tokens"));
}