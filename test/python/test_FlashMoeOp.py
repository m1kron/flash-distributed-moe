import torch
import flashMoeLauncher

from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
import vllmMinimalEnv.vllmMinimalEnv


class FlashMoeBlock(Qwen3MoeSparseMoeBlock):

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__(config, quant_config, prefix, enable_eplb)
        self.launcher = flashMoeLauncher.MoeKernelLauncher()
        self.launcher.create(48)

    def __del__(self):
        self.launcher.destroy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.

        # out = super().forward(hidden_states)
        
        tokens = hidden_states
        gateWeights = self.gate.weight
        ffn1ExpertWeights = self.experts.w13_weight
        ffn2ExpertWeights = self.experts.w2_weight
        
        output = torch.empty_like(tokens)
        tokensNum = tokens.shape[0]
        
        print(f"tokens: {tokens.shape}")
        print(f"gateWeights: {gateWeights.shape}")
        print(f"ffn1ExpertWeights: {ffn1ExpertWeights.shape}")
        print(f"ffn2ExpertWeights: {ffn2ExpertWeights.shape}")

        self.launcher.launch(tokens, gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, output)

        return output

def test_Moeop():
    vllmEnv = vllmMinimalEnv.vllmMinimalEnv.VllmMinimalEnv()
    vllmEnv.setup_vllm_env_with_default_configs(wanted_dtype=torch.float32)
    vllmEnv.moe_dtype

    quant_config = None
    prefix = "test"
    enable_eplb = False
    moe = vllmEnv.createMoeInstance(FlashMoeBlock, quant_config, prefix, enable_eplb)
    
    batchSize = 16
    hiddenSize = 2048
    
    hiddenStateTensor = torch.rand(
        (batchSize, hiddenSize), dtype=vllmEnv.moe_dtype, device=torch.device(vllmEnv.device)
    )
    
    out = moe(hiddenStateTensor)
    
    print(out.shape)

    vllmEnv.shutdown()


if __name__ == "__main__":
    test_Moeop()
