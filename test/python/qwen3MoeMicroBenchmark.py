import torch

import vllmMinimalEnv.vllmMinimalEnv
from benchmark.benchmark import Benchmark

from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

if __name__ == "__main__":
    vllm_env = vllmMinimalEnv.vllmMinimalEnv.VllmMinimalEnv()
    vllm_env.setup_vllm_env_with_default_configs()
    vllm_env.moe_dtype

    quant_config = None
    prefix = "test"
    enable_eplb = False
    moe = vllm_env.createMoeInstance(
        Qwen3MoeSparseMoeBlock.__name__, quant_config, prefix, enable_eplb
    )

    hiddenStateTensor = torch.rand(
        (16, 2048), dtype=vllm_env.moe_dtype, device=torch.device(vllm_env.device)
    )
    output = torch.empty_like(hiddenStateTensor)

    def RunModule(input, moe=moe):
        return moe(input)

    Benchmark.DoGpuBenchmark(RunModule, hiddenStateTensor, 10, 100)

    vllm_env.shutdown()
