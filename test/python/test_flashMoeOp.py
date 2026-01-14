import torch
import vllmMinimalEnv.vllmMinimalEnv
import pytest
from flashMoeOp import FlashMoeBlockWrapper
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

torch.manual_seed(95447)

batchSizes = [1, 16, 48]


@pytest.mark.parametrize("batchSize", batchSizes)
def test_FlashMoeWrapper(batchSize):

    # Prepare data:
    batchSize = batchSize
    hiddenSize = 2048
    experts = 128
    inter = 768

    gateWeights = torch.empty(
        (experts, hiddenSize),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(gateWeights, a=-0.1, b=0.1)

    ffn1Weights = torch.empty(
        (experts, 2 * inter, hiddenSize),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(ffn1Weights, a=-0.1, b=0.1)

    ffn2Weights = torch.empty(
        (experts, hiddenSize, inter),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(ffn2Weights, a=-0.1, b=0.1)

    tokens = torch.empty(
        (batchSize, hiddenSize),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(tokens, a=-1, b=1)

    tokens_copy = tokens.detach().clone()

    #######################################################################
    # Run inference:

    vllmEnv = vllmMinimalEnv.vllmMinimalEnv.VllmMinimalEnv()
    vllmEnv.setup_vllm_env_with_default_configs(wanted_dtype=torch.float32)

    quant_config = None
    prefix = "test"
    enable_eplb = False

    moe = vllmEnv.createMoeInstance(
        Qwen3MoeSparseMoeBlock, quant_config, prefix, enable_eplb
    )

    moe.gate.weight.data = gateWeights
    moe.experts.w13_weight.data = ffn1Weights
    moe.experts.w2_weight.data = ffn2Weights

    moeWrappered = FlashMoeBlockWrapper(moe, batchSize)

    outRef = moe(tokens)
    out = moeWrappered(tokens_copy)

    vllmEnv.shutdown()

    assert torch.allclose(out, outRef, rtol=1e-03, atol=1e-02)


if __name__ == "__main__":
    for batchSize in batchSizes:
        test_FlashMoeWrapper(batchSize)
