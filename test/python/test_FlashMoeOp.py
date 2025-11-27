from flashMoeOp import FlashMoeBlockWrapper
import torch
import vllmMinimalEnv.vllmMinimalEnv

from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

torch.manual_seed(95447)


def PrintFirst(tensor, str):
    print(f"{str}:", tensor[0, :5].tolist())


def test_Moeop():

    batchSize = 1
    hiddenSize = 2048
    experts = 128
    inter = 768

    gateWeights = torch.empty(
        (experts, hiddenSize),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(gateWeights, a=1, b=1)  # a=-0.1, b=0.1)

    ffn1Weights = torch.empty(
        (experts, 2 * inter, hiddenSize),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(ffn1Weights, a=1, b=1)  # a=-0.1, b=0.1)

    ffn2Weights = torch.empty(
        (experts, hiddenSize, inter),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(ffn2Weights, a=1, b=1)  # a=-0.1, b=0.1)

    print(f"ffn2Weights:", ffn2Weights[0, 0, :5].tolist())

    tokens = torch.empty(
        (batchSize, hiddenSize),
        dtype=torch.float32,
        device=torch.device("cuda"),
    )
    torch.nn.init.uniform_(tokens, a=1, b=1)  # a=-0.1, b=0.1)

    tokens_copy = tokens.detach().clone()

    #######################################################################

    vllmEnv = vllmMinimalEnv.vllmMinimalEnv.VllmMinimalEnv()
    vllmEnv.setup_vllm_env_with_default_configs(wanted_dtype=torch.float32)

    quant_config = None
    prefix = "test"
    enable_eplb = False

    moe = vllmEnv.createMoeInstance(
        Qwen3MoeSparseMoeBlock, quant_config, prefix, enable_eplb
    )

    moe.gate.weight.data = gateWeights
    moe.experts.w13_weight = torch.nn.Parameter(ffn1Weights)
    moe.experts.w2_weight = torch.nn.Parameter(ffn2Weights)

    for item, tensor in moe.state_dict().items():
        print(f"{item}:", tensor.shape)

    moeWrappered = FlashMoeBlockWrapper(moe)

    PrintFirst(tokens, "tokens")
    outRef = moe(tokens)

    PrintFirst(tokens_copy, "tokens_copy")

    out = moeWrappered(tokens_copy)

    vllmEnv.shutdown()

    close = torch.allclose(out, outRef)
    print(f"torch.allclose: {close}")

    PrintFirst(out, "out")
    PrintFirst(outRef, "outRef")


if __name__ == "__main__":
    test_Moeop()
