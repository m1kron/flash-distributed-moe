# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for Data Parallel Inference with MoE models.

Usage:
rocm 7.0, vllm 0.11.2:

NCCL_DEBUG=WARN HIP_VISIBLE_DEVICES=4,5,6,7 python3 benchmark_dp.py -dp=4 --all2all-backend=allgather_reducescatter --disable-nccl-for-dp-synchronization

Optional benchmark parameters:
    --num-warmup-iters: Number of warmup iterations (default: 5)
    --num-benchmark-iters: Number of benchmark iterations (default: 50)
    --total-prompts: Total number of prompts per iteration (default: 4)
    --input-length: Input sequence length (default: 16)
    --output-length: Maximum output tokens (default: 10)
    
The idea of the benchmark is to setup data parallel, expert parallel vllm offline inference,
where the each rank processes a subset of the total prompts, and measure the throughput and latency.
"""

# import torch
# import rocprofsys

# @rocprofsys.profile()
# def warmup_omnitrace():
#     """Dummy method getting omnitrace initialized before hip runtime"""
#     torch.manual_seed(42)
    
# warmup_omnitrace()

import os
import tempfile
from time import sleep, time
import json
import numpy as np
import torch
from multiprocessing import Queue
import statistics

from vllm import LLM, EngineArgs, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port

SEED = 95447
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
np.random.seed(SEED)
import vllm.model_executor.models.qwen3_moe

import torch.distributed as dist

import flashMoeLauncher

class FlashMoeWrapper(vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock):

    def __init__(
        self,
        vllm_config,
        prefix: str = "",
    ):
        super().__init__(vllm_config, prefix)

        ep_group = self.ep_group
        ep_rank =self.ep_rank
        ep_size =self.ep_size
        
        maxTokens = 8
        
        self.launcher = flashMoeLauncher.MoeKernelLauncher()
        self.maxTokens = maxTokens
        
        if ep_size == 1:
            print("FlashMoeWrapper: Initializing single process")
            gateWeights = self.gate.weight
            ffn1ExpertWeights = self.experts.w13_weight
            ffn2ExpertWeights = self.experts.w2_weight

            self.launcher.create(
                gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, maxTokens
            )
        else:
            print(f"FlashMoeWrapper: Initializing distributed ep_size={ep_size}, ep_rank={ep_rank}")
            # Get distributed unique id as bytes and broadcast using broadcast_object_list        
            uniqueid = self.launcher.getDistributedUniqueId(empty=True)
            if ep_rank == 0:
                uniqueid = self.launcher.getDistributedUniqueId()
                broadcast_objects = [uniqueid]
            else:
                broadcast_objects = [None]

            dist.broadcast_object_list(broadcast_objects, 
                                    src=dist.get_process_group_ranks(ep_group)[0], 
                                    group=ep_group)
            dist.barrier(ep_group)
            
            uniqueid = broadcast_objects[0]

            self.launcher.initializeDistributed(
                uniqueid,
                ep_rank,
                ep_size
            )
            
    def __del__(self):
        self.launcher.destroy()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        if self.ep_size == 1 and num_tokens < self.maxTokens:
            tokens = hidden_states

            #print("Using Flash Moe Kernel Launcher for MoE forward")
            # Inplace calc -> output_mem = input_mem
            self.launcher.launch(tokens, tokens)

            return tokens
        else:
           return super().forward(hidden_states)

orginal_process_weights_after_loading = vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod.process_weights_after_loading

def custom_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    # Flash moe works now only with orginal weights layout, so no need to process weights
    print("Processing weights after loading - no-op for FlashMoeWrapper")

# Monkey patch:
# vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock = FlashMoeWrapper
# vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod.process_weights_after_loading = custom_process_weights_after_loading
# -------------------

WANTED_MOE_DTYPE = "float32"
os.environ["VLLM_USE_V1"] = "1"

if WANTED_MOE_DTYPE == "float16":
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
else:
    os.environ["VLLM_ROCM_USE_AITER"] = "0"
    
def create_qwen3_moe_config(
    output_dir: str,
    hidden_size: int = 2048,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 4,
    num_experts: int = 128,
    moe_intermediate_size: int = 768,
    num_experts_per_tok: int = 8,
    vocab_size: int = 32000,
    max_position_embeddings: int = 4096,
    dtype: str = WANTED_MOE_DTYPE,
) -> str:
    """Create a minimal Qwen3MoE config.json with dummy weights."""
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 4,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": hidden_size // num_attention_heads,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "mlp_only_layers": [],
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "use_sliding_window": False,
        "sliding_window": None,
        "torch_dtype": dtype,
        "dtype": dtype,
        "transformers_version": "4.40.0",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return output_dir


def create_parser():
    parser = FlexibleArgumentParser(description="Data Parallel Inference Benchmark")

    # Add all engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model=None,
        enable_expert_parallel=True,
        skip_tokenizer_init=True,
        load_format="dummy",
        gpu_memory_utilization=0.1,
        disable_log_stats=True,
        # Needed for data parallel on amd ----- 
        all2all_backend="allgather_reducescatter",
        disable_nccl_for_dp_synchronization=True,
        # ----------------------
        enforce_eager=True,
        dtype=WANTED_MOE_DTYPE,
    )

    return parser

def benchmark_worker(
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    engine_args,
    benchmark_config,
    result_queue
):
    """Worker function for each DP rank that runs the benchmark."""
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    num_warmup_iters = benchmark_config["num_warmup_iters"]
    num_benchmark_iters = benchmark_config["num_benchmark_iters"]
    total_prompts = benchmark_config["total_prompts"]
    input_length = benchmark_config["input_length"]
    output_length = benchmark_config["output_length"]

    # Generate prompts
    all_prompts = [
        {"prompt_token_ids": np.random.randint(1, 30000, size=input_length).tolist()}
        for _ in range(total_prompts)
    ]

    # Distribute prompts across DP ranks
    floor = len(all_prompts) // dp_size
    remainder = len(all_prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    prompts = all_prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(prompts) == 0:
        prompts = [{"prompt_token_ids": [1] * input_length}]

    print(f"[Rank {global_dp_rank}] Processing {len(prompts)} prompts per iteration")

    # Create sampling params
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_length,
        detokenize=False,
        seed=SEED,
    )

    # Create LLM instance
    print(f"[Rank {global_dp_rank}] Initializing LLM...")
    llm_init_start = time()
    llm = LLM(**engine_args)
    llm_init_time = time() - llm_init_start
    print(f"[Rank {global_dp_rank}] LLM initialized in {llm_init_time:.2f}s")

    # Warmup iterations
    print(f"[Rank {global_dp_rank}] Running {num_warmup_iters} warmup iterations...")
    for i in range(num_warmup_iters):
        warmup_start = time()
        _ = llm.generate(prompts, sampling_params, use_tqdm=False)
        warmup_time = time() - warmup_start
        
    # Synchronize before benchmark (barrier via sleep)
    torch.cuda.synchronize()
    sleep(0.5)

    # Benchmark iterations
    print(f"[Rank {global_dp_rank}] Running {num_benchmark_iters} benchmark iterations...")
    iter_times = []
    total_tokens_generated = 0

    for i in range(num_benchmark_iters):
        torch.cuda.synchronize()
        iter_start = time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        iter_time = time() - iter_start
        iter_times.append(iter_time)

        # Count tokens generated
        for output in outputs:
            total_tokens_generated += len(output.outputs[0].token_ids)

    # Calculate statistics for this rank
    avg_time = statistics.mean(iter_times)
    std_time = statistics.stdev(iter_times) if len(iter_times) > 1 else 0.0
    min_time = min(iter_times)
    max_time = max(iter_times)
    tokens_per_second = total_tokens_generated / sum(iter_times)
    prompts_per_second = (len(prompts) * num_benchmark_iters) / sum(iter_times)

    result = {
        "rank": global_dp_rank,
        "llm_init_time": llm_init_time,
        "iter_times": iter_times,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "prompts_per_second": prompts_per_second,
        "num_prompts": len(prompts),
        "outputs": outputs,
    }

    result_queue.put(result)

def print_aggregate_results(results, benchmark_config, name):
    """Print aggregated benchmark results from all ranks."""
    print("\n" + "=" * 70)
    print(f"AGGREGATE BENCHMARK RESULTS for {name}:")
    print("=" * 70)

    print(f"\nBenchmark Configuration:")
    print(f"  Data Parallel Size: {len(results)}")
    print(f"  Warmup Iterations: {benchmark_config['num_warmup_iters']}")
    print(f"  Benchmark Iterations: {benchmark_config['num_benchmark_iters']}")
    print(f"  Total Prompts (per iter): {benchmark_config['total_prompts']}")
    print(f"  Input Length: {benchmark_config['input_length']}")
    print(f"  Output Length: {benchmark_config['output_length']}")

    # Aggregate metrics
    all_avg_times = [r["avg_time"] for r in results]
    all_init_times = [r["llm_init_time"] for r in results]
    total_tokens = sum(r["total_tokens_generated"] for r in results)
    total_prompts_processed = sum(r["num_prompts"] for r in results)
    total_time = sum(r["avg_time"] for r in results) / len(results)  # avg across ranks

    # Total throughput (sum across all ranks)
    total_tokens_per_sec = sum(r["tokens_per_second"] for r in results)
    total_prompts_per_sec = sum(r["prompts_per_second"] for r in results)

    print(f"\nPer-Rank Statistics:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  Rank {r['rank']}: Avg={r['avg_time']:.4f}s, "
              f"Tokens/s={r['tokens_per_second']:.2f}, "
              f"Init={r['llm_init_time']:.2f}s")
    
    print(f"\nPer-Rank outputs:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  Rank {r['rank']} outputs:")
        for i, out in enumerate(r["outputs"]):
            print(f"   Prompt {i}:   {out.outputs[0].token_ids}")

    print(f"\nAggregate Throughput:")
    print(f"  Total Tokens Generated (per iter): {total_tokens // benchmark_config['num_benchmark_iters']}")
    print(f"  Total Prompts Processed (per iter): {total_prompts_processed}")
    print(f"  Combined Tokens/Second: {total_tokens_per_sec:.2f}")
    print(f"  Combined Prompts/Second: {total_prompts_per_sec:.2f}")

    print(f"\nLatency Statistics:")
    print(f"  Avg Iteration Time: {statistics.mean(all_avg_times):.4f}s")
    print(f"  Std Iteration Time: {statistics.stdev(all_avg_times) if len(all_avg_times) > 1 else 0:.4f}s")
    print(f"  Max Init Time: {max(all_init_times):.2f}s")

    print("=" * 70)

def RunBenchmark(model_dir, benchmark_config, name, enable_flash_moe):
    
    if(enable_flash_moe):
        # Monkey patch:
        vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock = FlashMoeWrapper
        vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod.process_weights_after_loading = custom_process_weights_after_loading
    
    
    parser = create_parser()
    args = vars(parser.parse_args())

    # Extract DP-specific args
    dp_size = args.pop("data_parallel_size")
    dp_num_nodes = 1 #< Only one node.
    dp_node_rank = 0 #< Only one node.
    timeout = 300
    dp_master_ip = "127.0.0.1"
    dp_master_port_val = get_open_port()

    # Remaining args are engine args
    engine_args = args
    engine_args["model"] = model_dir

    assert dp_size % dp_num_nodes == 0, "dp_size should be divisible by dp_num_nodes"
    dp_per_node = dp_size // dp_num_nodes

    from multiprocessing import Process

    if current_platform.is_rocm():
        from multiprocessing import set_start_method
        set_start_method("spawn", force=True)

    # Create result queue for collecting benchmark results
    result_queue = Queue()

    print("\n" + "=" * 70)
    print(f"STARTING DATA PARALLEL BENCHMARK: {name}")
    print("=" * 70)
    print(f"DP Size: {dp_size}, TP Size: {engine_args.get('tensor_parallel_size', 1)}")
    print(f"Model Dir: {model_dir}")
    print("=" * 70 + "\n")

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(dp_node_rank * dp_per_node, (dp_node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=benchmark_worker,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port_val,
                engine_args,
                benchmark_config,
                result_queue,
            ),
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within {timeout}s.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    # Collect results from all ranks
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return exit_code, results, name


if __name__ == "__main__":
    # Create model directory with config
    model_dir = tempfile.mkdtemp(prefix="qwen3_moe_benchmark_")
    create_qwen3_moe_config(
        output_dir=model_dir,
        hidden_size=2048,
        num_hidden_layers=1,
        num_experts=128,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
    )
    
    # Extract benchmark-specific args
    benchmark_config = {
        "num_warmup_iters": 1,
        "num_benchmark_iters": 1,
        "total_prompts": 4,
        "input_length": 1,
        "output_length": 16,
    }

    exit_code1, results1, name1 = RunBenchmark(model_dir, benchmark_config, "VLLM REF", True)
    
    exit_code2, results2, name2 = RunBenchmark(model_dir, benchmark_config, "VLLM REF2", True)
    
    if exit_code1 == 0:
        print_aggregate_results(results1, benchmark_config, name1)

    if exit_code2 == 0:
        print_aggregate_results(results2, benchmark_config, name2)
        
    # Compare results:
    if exit_code1 == 0 and exit_code2 == 0:
        print("\n" + "=" * 70)
        print("COMPARING RESULTS BETWEEN RUNS")
        print("=" * 70)

        for r1, r2 in zip(sorted(results1, key=lambda x: x["rank"]), sorted(results2, key=lambda x: x["rank"])):
            outputs1 = [out.outputs[0].token_ids for out in r1["outputs"]]
            outputs2 = [out.outputs[0].token_ids for out in r2["outputs"]]
            assert outputs1 == outputs2, f"Outputs differ for rank {r1['rank']}"
        
        print("All outputs match between runs.")

    exit(exit_code1)    