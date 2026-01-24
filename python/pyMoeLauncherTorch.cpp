#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <cstring>

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
  // Returns a 128-byte distributed unique id as Python bytes.
  // If empty is true, returns zero-initialized id without querying ROC-SHMEM.
  static py::bytes getDistributedUniqueId(bool empty) {
    moe::DistributedUniqueId id;
    if (empty) {
      std::memset(id.data, 0, sizeof(id.data));
    } else {
      id = ::getDistributedUniqueId();
    }
    return py::bytes(reinterpret_cast<const char*>(id.data),
                     static_cast<size_t>(sizeof(id.data)));
  }

  // Initializes distributed rocSHMEM using provided unique id, rank and world size
  static void initializeDistributed(const py::bytes& uid, int rank,
                                    int worldSize) {
    std::string s = uid;  // copy bytes into std::string
    TORCH_CHECK(s.size() == sizeof(moe::DistributedUniqueId::data),
                "uid must be 128 bytes");
    moe::DistributedUniqueId id;
    std::memcpy(id.data, s.data(), s.size());
    ::InitializeDistributed(id, rank, worldSize);
  }

  void launch(const at::Tensor& tokens, const at::Tensor& output) {
    if (!ptr_) throw std::runtime_error("Launcher was destroyed");

    CHECK_TENSOR(tokens);
    CHECK_TENSOR(output);

    auto tokensSize = tokens.sizes()[0];
    auto hiddenSize = tokens.sizes()[1];

    TORCH_CHECK(output.sizes()[0] == tokensSize);
    TORCH_CHECK(output.sizes()[1] == hiddenSize);

    const void* t_ptr = tokens.data_ptr();
    void* out_ptr = const_cast<void*>(output.data_ptr());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    HIP_ERROR_CHECK(ptr_->Launch(t_ptr, out_ptr, tokensSize, stream));
  }

  void destroy() {
    if (ptr_) {
      hipStream_t stream = at::cuda::getCurrentCUDAStream();
      HIP_ERROR_CHECK(DestroyLauncher(ptr_, stream));
      ptr_ = nullptr;
    }
  }

  void create(const at::Tensor& gate_weights,
              const at::Tensor& ffn1_expert_weights,
              const at::Tensor& ffn2_expert_weights, int max_tokens) {
    auto expertsSize = gate_weights.sizes()[0];
    auto hiddenSize = gate_weights.sizes()[1];
    auto interSize = ffn2_expert_weights.sizes()[2];

    TORCH_CHECK(ffn1_expert_weights.sizes()[0] == expertsSize);
    TORCH_CHECK(ffn1_expert_weights.sizes()[1] == 2 * interSize);
    TORCH_CHECK(ffn1_expert_weights.sizes()[2] == hiddenSize);

    TORCH_CHECK(ffn2_expert_weights.sizes()[0] == expertsSize);
    TORCH_CHECK(ffn2_expert_weights.sizes()[1] == hiddenSize);
    TORCH_CHECK(ffn2_expert_weights.sizes()[2] == interSize);

    CHECK_TENSOR(gate_weights);
    CHECK_TENSOR(ffn1_expert_weights);
    CHECK_TENSOR(ffn2_expert_weights);

    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    const void* gw_ptr = gate_weights.data_ptr();
    const void* f1_ptr = ffn1_expert_weights.data_ptr();
    const void* f2_ptr = ffn2_expert_weights.data_ptr();

    HIP_ERROR_CHECK(
        CreateLauncher(&ptr_, gw_ptr, f1_ptr, f2_ptr, max_tokens, stream));
  }

  bool valid() const { return ptr_ != nullptr; }

 private:
  moe::IMoeKernelLauncher* ptr_{nullptr};
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyTorch bindings for IMoeKernelLauncher (ROCm/HIP)";

  py::class_<MoeKernelLauncherWrapper>(m, "MoeKernelLauncher")
      .def(py::init())
     .def_static("getDistributedUniqueId",
        &MoeKernelLauncherWrapper::getDistributedUniqueId,
        py::arg("empty") = false,
      "Returns a 128-byte rocSHMEM unique id as bytes. If empty=True, returns zeros.")
      .def_static("initializeDistributed",
                  &MoeKernelLauncherWrapper::initializeDistributed,
                  py::arg("uid"), py::arg("rank"), py::arg("world_size"),
                  "Initialize rocSHMEM distributed context using the provided unique id")
      .def("launch", &MoeKernelLauncherWrapper::launch, py::arg("tokens"),
           py::arg("output"))
      .def("destroy", &MoeKernelLauncherWrapper::destroy)
      .def("create", &MoeKernelLauncherWrapper::create, py::arg("gate_weights"),
           py::arg("ffn1_expert_weights"), py::arg("ffn2_expert_weights"),
           py::arg("max_tokens"))
      .def_property_readonly("valid", &MoeKernelLauncherWrapper::valid);
}