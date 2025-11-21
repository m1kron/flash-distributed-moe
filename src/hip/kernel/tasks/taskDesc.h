#pragma once
#include "src/hip/taskSystem/taskManager.h"

namespace moe {

enum TaskType : int { FFN1, FFN2, REDUCE };

// Moe taask descriptor.
struct __align__(16) MoeTaskDesc {
  const void* __restrict__ ffn1ExpertWeights;
  const void* __restrict__ ffn2ExpertWeights;
  const void* __restrict__ tokens;
  void* __restrict__ ffn1Output;
  void* __restrict__ ffn2Output;
  void* __restrict__ finalOutput;
  float expertWeight;
  int tokenIdx;
  int expertIdx;
  int blockTileColStartIdx;
  int topkSlotIdx;
  TaskType taskType;
};

// TODO: Buffer size as param.
// TODO: do sth with __constant__ scattered over multiple files..
using TaskManagerType = TaskManager<MoeTaskDesc, 8192>;
__constant__ TaskManagerType globalTaskManager;
}  // namespace moe