import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ROCM_USE_AITER"] = "1"

from vllm import EngineArgs
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
import torch
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context

# Wrapper for moe module, propvides correct vllm context
# and option to execute via cuda graph.
class VllmMinimalEnvMoeExecutionWrapper(torch.nn.Module):
    def __init__(self, module, vllm_config):
        super().__init__()
        self.module = module
        self.vllm_config = vllm_config
        self.staticInput = torch.empty(1,1)
        self.staticOutput = torch.empty(1,1)
        self.graph = None
        self.forwardCallBack = self._forwardStandard

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
    def prepareCudaGraphAndReplaceForward(self, input):
        self.staticInput = torch.empty_like(input)
        self.staticInput.copy_(input)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                _ = self._forwardStandard(self.staticInput)
        torch.cuda.current_stream().wait_stream(s)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
             self.staticOutput = self._forwardStandard(self.staticInput)
             
        self.forwardCallBack = self._forwardCudaGraph
             
    def forward(self, input):
        return self.forwardCallBack(input)

    def _forwardStandard(self, input):
        with set_forward_context(None, self.vllm_config):
            return self.module(input)
        
    def _forwardCudaGraph(self, input):
        self.graph.replay()
        return self.staticOutput

# Setups minimal vllm enviroment to run vllm ops, in particular MoE layer instance.
class VllmMinimalEnv:
    engine_args: EngineArgs
    vllm_config: VllmConfig
    moe_dtype: torch.dtype
    moe: Qwen3MoeSparseMoeBlock
    config: Qwen3MoeConfig

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ops = []

    def setup_vllm_env_with_default_configs(self, wanted_dtype=None):
        self.engine_args = EngineArgs(model="Qwen/Qwen3-30B-A3B", dtype="float16")
        self.vllm_config = self.engine_args.create_engine_config()

        if wanted_dtype == None:
            wanted_dtype = self.vllm_config.model_config.dtype
            
        self.moe_dtype = wanted_dtype

        self.config = Qwen3MoeConfig()
        self.config.architectures = ["Qwen3MoeForCausalLM"]
        self.config.torch_dtype = self.moe_dtype
        self.config.bos_token_id = 151643
        self.config.eos_token_id = 151645
        self.config.has_no_defaults_at_init = False
        torch.set_default_dtype(self.moe_dtype)

        with set_current_vllm_config(self.vllm_config):
            self._setup_model_parallel()

    def createMoeInstance(self, klass, quant_config, prefix, enable_eplb):
        with set_current_vllm_config(self.vllm_config):
            moe = klass(
                config=self.config,
                quant_config=quant_config,
                prefix=prefix,
                enable_eplb=enable_eplb,
            )
        moe.to(self.device)
        return VllmMinimalEnvMoeExecutionWrapper(moe, self.vllm_config)

    def generate(self, input):
        with set_forward_context(None, self.vllm_config):
            output = self.moe(input)
            torch.cuda.synchronize()
        return output

    def _setup_model_parallel(self):
        world_size = 1
        rank = 0
        distributed_init_method = "env://"
        local_rank = -1
        backend = "nccl"
        tensor_parallel_size = 1
        pipeline_parallel_size = 1
        init_distributed_environment(
            world_size, rank, distributed_init_method, local_rank, backend
        )

        ensure_model_parallel_initialized(tensor_parallel_size, pipeline_parallel_size)

    def shutdown(self):
        destroy_model_parallel()
        destroy_distributed_environment()
