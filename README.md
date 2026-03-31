# flash-distributed-moe
Implementation of flashDmoe paper: https://arxiv.org/abs/2506.04667 paper for AMD's GPU ecoshystem. 

Uses rocshmem for inter-kernel communication.
Implementes Qwen3 MoE, the goal is to integrate it with vllm.

WORK IN PROGRESS.

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
