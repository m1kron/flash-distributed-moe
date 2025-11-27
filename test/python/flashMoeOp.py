import flashMoeLauncher
import torch
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock


class FlashMoeBlockWrapper(torch.nn.Module):

    def __init__(
        self,
        qwen3MoeSparseMoeBlockModule,
    ):
        super().__init__()
        self.qwen3MoeSparseMoeBlockModule = qwen3MoeSparseMoeBlockModule
        self.launcher = flashMoeLauncher.MoeKernelLauncher()
        self.launcher.create(1)

    def __del__(self):
        self.launcher.destroy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.

        # out = super().forward(hidden_states)

        tokens = hidden_states
        gateWeights = self.qwen3MoeSparseMoeBlockModule.gate.weight
        ffn1ExpertWeights = self.qwen3MoeSparseMoeBlockModule.experts.w13_weight
        ffn2ExpertWeights = self.qwen3MoeSparseMoeBlockModule.experts.w2_weight

        print(f"ffn2ExpertWeights:", ffn2ExpertWeights[0, 0, :5].tolist())

        output = torch.empty_like(tokens)

        # print(f"tokens: {tokens.shape}")
        # print(f"gateWeights: {gateWeights.shape}")
        # print(f"ffn1ExpertWeights: {ffn1ExpertWeights.shape}")
        # print(f"ffn2ExpertWeights: {ffn2ExpertWeights.shape}")

        self.launcher.launch(
            tokens, gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, output
        )

        return output
