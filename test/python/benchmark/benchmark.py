import time
import numpy as np
import torch


class Benchmark:
    @staticmethod
    def DoGpuBenchmark(func, input, warmup, iters):
        for _ in range(warmup):
            func(input)

        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(iters):
            func(input)
        end.record()
        torch.cuda.synchronize()

        avgMs = start.elapsed_time(end) / iters

        return avgMs
