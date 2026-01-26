import flashMoeLauncher
import torch
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock


# Initial Flash moe wrapper for vllm.
class FlashMoeBlockWrapper(torch.nn.Module):

    def __init__(self, qwen3MoeSparseMoeBlockModule: Qwen3MoeSparseMoeBlock, maxTokens):
        super().__init__()
        self.qwen3MoeSparseMoeBlockModule = qwen3MoeSparseMoeBlockModule
        self.launcher = flashMoeLauncher.MoeKernelLauncher()

        gateWeights = self.qwen3MoeSparseMoeBlockModule.gate.weight
        ffn1ExpertWeights = self.qwen3MoeSparseMoeBlockModule.experts.w13_weight
        ffn2ExpertWeights = self.qwen3MoeSparseMoeBlockModule.experts.w2_weight

        uniqueid = self.launcher.getDistributedUniqueId(empty=True)
        self.launcher.create(
            gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, maxTokens, uniqueid, 0, 1
        )
        self.maxTokens = maxTokens

    def __del__(self):
        self.launcher.destroy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        
        if num_tokens <= self.maxTokens:
            tokens = hidden_states

            # Inplace calc -> output_mem = input_mem
            self.launcher.launch(tokens, tokens)

            return tokens
        else:
            return super().forward(hidden_states)
