import torch

import vllmMinimalEnv.vllmMinimalEnv
from benchmark.benchmark import Benchmark
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from flashMoeOp import FlashMoeBlockWrapper

HIDDEN_SIZE = 2048
NUM_EXPERTS = 128
INTERMEDIATE_SIZE = 768


def PrepareRandomWeights(wanted_dtype=torch.float32):
    hiddenSize = HIDDEN_SIZE
    experts = NUM_EXPERTS
    inter = INTERMEDIATE_SIZE

    gateWeights = torch.empty(
        (experts, hiddenSize),
        dtype=wanted_dtype,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(gateWeights, a=-0.1, b=0.1)

    ffn1Weights = torch.empty(
        (experts, 2 * inter, hiddenSize),
        dtype=wanted_dtype,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(ffn1Weights, a=-0.1, b=0.1)

    ffn2Weights = torch.empty(
        (experts, hiddenSize, inter),
        dtype=wanted_dtype,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(ffn2Weights, a=-0.1, b=0.1)

    return gateWeights, ffn1Weights, ffn2Weights


def DoBenchmarkForMoe(moe, vllmEnv, batchSize, hiddenSize, enableCudaGraph):
    hiddenStateTensor = torch.rand(
        (batchSize, hiddenSize),
        dtype=vllmEnv.moe_dtype,
        device=torch.device(vllmEnv.device),
    )

    if enableCudaGraph:
        moe.prepareCudaGraphAndReplaceForward(hiddenStateTensor)

    def RunModule(input, moe=moe):
        return moe(input)

    avgMs = Benchmark.DoGpuBenchmark(RunModule, hiddenStateTensor, 10, 100)

    print(
        f"Benchmark of {moe.module.__class__.__name__}, [{batchSize},{hiddenSize}]: {avgMs} ms!"
    )
    
    moe.resetForward()


if __name__ == "__main__":
    vllmEnv = vllmMinimalEnv.vllmMinimalEnv.VllmMinimalEnv()
    wanted_dtype=torch.float32
    vllmEnv.setup_vllm_env_with_default_configs(wanted_dtype=wanted_dtype)

    gateW, ffn1W, ffn2W = PrepareRandomWeights(wanted_dtype=wanted_dtype)

    moe = vllmEnv.createMoeInstance(Qwen3MoeSparseMoeBlock, None, "test", False)

    # Set weights:
    moe.gate.weight.data = gateW
    moe.experts.w13_weight.data = ffn1W
    moe.experts.w2_weight.data = ffn2W

    DoBenchmarkForMoe(moe, vllmEnv, 1, HIDDEN_SIZE, True)
    DoBenchmarkForMoe(moe, vllmEnv, 16, HIDDEN_SIZE, True)

    moe.module = FlashMoeBlockWrapper(moe.module, 16)
    
    flashMoe = moe
    enableCudaGraph = False  # Currrently cuda graph fails on flash-moe - probably async mem cpy problem(?)

    DoBenchmarkForMoe(flashMoe, vllmEnv, 1, 2048, enableCudaGraph)
    DoBenchmarkForMoe(flashMoe, vllmEnv, 16, 2048, enableCudaGraph)

    vllmEnv.shutdown()
