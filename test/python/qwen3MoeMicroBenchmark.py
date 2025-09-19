import torch

import vllmMinimalEnv.vllmMinimalEnv
from benchmark.benchmark import Benchmark

from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

def DoBenchmarkForMoe(moe, vllmEnv, batchSize, hiddenSize):
    
    hiddenStateTensor = torch.rand(
        (batchSize, hiddenSize), dtype=vllmEnv.moe_dtype, device=torch.device(vllmEnv.device)
    )
    
    moe.prepareCudaGraphAndReplaceForward(hiddenStateTensor)

    def RunModule(input, moe=moe):
        return moe(input)

    avgMs = Benchmark.DoGpuBenchmark(RunModule, hiddenStateTensor, 10, 100)
    
    print(f"Benchmark of {moe.module.__class__.__name__}, [{batchSize},{hiddenSize}]: {avgMs} ms!")

if __name__ == "__main__":
    vllmEnv = vllmMinimalEnv.vllmMinimalEnv.VllmMinimalEnv()
    vllmEnv.setup_vllm_env_with_default_configs()
    vllmEnv.moe_dtype

    quant_config = None
    prefix = "test"
    enable_eplb = False
    moe = vllmEnv.createMoeInstance(
        Qwen3MoeSparseMoeBlock.__name__, quant_config, prefix, enable_eplb
    )
    
    DoBenchmarkForMoe(moe, vllmEnv, 16, 2048)
    DoBenchmarkForMoe(moe, vllmEnv, 2048, 2048)
    DoBenchmarkForMoe(moe, vllmEnv, 16000, 2048)

    vllmEnv.shutdown()
