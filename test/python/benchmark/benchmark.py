import time
import numpy as np
import torch


class Benchmark:

    @staticmethod
    def DoGpuBenchmark(func, input, warmup, iters):

        print("Warming up...")
        for _ in range(warmup):
            func(input)

        torch.cuda.synchronize()
        print("Profiling..")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(iters):
            func(input)
        end.record()
        torch.cuda.synchronize()

        avg = start.elapsed_time(end) / iters

        print(f"Done! Avg latency for '{func.__name__}': {avg} [ms]")

    @staticmethod
    def _SingleIter(func, input):
        start_time = time.perf_counter_ns()
        func(input)
        end_time = time.perf_counter_ns()
        latency = end_time - start_time
        return latency / 1000000  # [ms]
