# flash-moe

Project status at last day of active development(28.01.2026).

Flash-moe project aims to speedup Mixture of Experts operator for case where experts weights are distributed to different GPUs(each GPU gets different set of experts, assinged linearly) and each GPU has distict set of tokens. The initial focus was on decode phase where expected number of tokens is small. The main idea is to have a single kernel, which performs gate + experts calculations and also initalizes and transfers tokens to other GPUs at the same time.
THe expected speedup comes from the fact that in this sceario kernel can overlap communication with other GPUS with computations and no all2all barriers are needed. 

The idea is based on https://arxiv.org/abs/2506.04667 paper, but implementation is for AMD GPUs and ecosystem.

The goal of current implementation was to provide POC implementation with focus of being correct and bug-free. Becasue of that all subsystems are not optimized.

Goal of POC:
- proof of the usefullness of the concept
- only MI300x supported
- only Qwen3 MoE implemented
- kernel is integrated with vllm to proove speedup for real world scenarion(Qwen3 model)
- identify potential limitations of this approach

Implemented:
- kernel works for single GPU.
- light-weight GPU task system. Task system is based on global work-queue with tickets mechanism to minimize contention. Each SM is a single worker. Task system allows to dynamically add new tasks to the queue. Implementation is based on linear buffers and allocators for simplicity. Task dependencies are handled externally(no general mechanism was implemented yet).
- qwen3 MoE is split into tasks. Each local token generates around 11 tasks at this point. Support for other MoE would be done by implementing new tasks.
- implementation is fully async: gate calculation can be overlapped with expert calculation and GPU-GPU comunication, dynamic nature of tokens routing is supported
- rocshmem is used for device-to-device communication - it is custom compiled, because by default it relay on MPI lib heavly to do initial handshake. Currently it is configured in a way that MPI is only needed for compilation, but it is not needed for runtime. Initial handshake is done via tcp socket - this is done similarly like in nvshmem.
- integration of the kernel is done with vllm engine - but it is more a hack then a proper solution, which would require to modify vllm source code to do it properly. Kernel is properly injected into data-parallel, expert-parallel vllm execution on simple Qwen3 like model with single MoE operator.
- multiGPU part is not fully implemented. Currenrly kernel properly distributes weights and works if each gpu processes only local kernels. Case when token has to be sent to different GPU to be processed was being implemented at the time where this project was suspended/stopped. Current PR(not passing CI) was meant to add it.
- implemented basic unit, functional and integration tests, all works with CI
- implemented basic benchmarks
