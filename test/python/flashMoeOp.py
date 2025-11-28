import flashMoeLauncher
import torch
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock


# Initial Flash moe wrapper for vllm.
class FlashMoeBlockWrapper(torch.nn.Module):

    def __init__(
        self,
        qwen3MoeSparseMoeBlockModule: Qwen3MoeSparseMoeBlock,
    ):
        super().__init__()
        self.qwen3MoeSparseMoeBlockModule = qwen3MoeSparseMoeBlockModule
        self.launcher = flashMoeLauncher.MoeKernelLauncher()
        self.launcher.create(1)

    def __del__(self):
        self.launcher.destroy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        tokens = hidden_states
        gateWeights = self.qwen3MoeSparseMoeBlockModule.gate.weight
        ffn1ExpertWeights = self.qwen3MoeSparseMoeBlockModule.experts.w13_weight
        ffn2ExpertWeights = self.qwen3MoeSparseMoeBlockModule.experts.w2_weight

        self.launcher.launch(
            tokens, gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, tokens
        )

        return tokens
