# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FlashMoE patch module.

This module applies monkey patches to vLLM when FLASH_MOE_ENABLED=1 environment
variable is set. It's designed to work with multiprocessing 'spawn' mode where
child processes don't inherit patches from the parent.

The key mechanism is using sitecustomize.py which Python imports automatically
at startup in EVERY process. We create a temp directory with sitecustomize.py
and prepend it to PYTHONPATH - this ensures all child/grandchild processes
get patched.

Usage:
    from flash_moe_patch import enable_flash_moe_patching
    
    # Call this BEFORE creating any subprocesses
    enable_flash_moe_patching()
    
    # Now any subprocess (including grandchildren) will be patched
"""

import os
import sys
import tempfile
import atexit
import shutil

_PATCHED = False
_SITECUSTOMIZE_DIR = None


def _get_flash_moe_wrapper_class():
    """Lazily import and return the FlashMoeWrapper class."""
    import torch
    import torch.distributed as dist
    import vllm.model_executor.models.qwen3_moe
    import flashMoeLauncher

    class FlashMoeWrapper(vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock):

        def __init__(
            self,
            vllm_config,
            prefix: str = "",
        ):
            super().__init__(vllm_config, prefix)

            ep_group = self.ep_group
            ep_rank = self.ep_rank
            ep_size = self.ep_size

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

                dist.broadcast_object_list(
                    broadcast_objects,
                    src=dist.get_process_group_ranks(ep_group)[0],
                    group=ep_group
                )
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

                # Inplace calc -> output_mem = input_mem
                self.launcher.launch(tokens, tokens)

                return tokens
            else:
                return super().forward(hidden_states)

    return FlashMoeWrapper


def _custom_process_weights_after_loading(self, layer) -> None:
    """Flash moe works now only with original weights layout, so no need to process weights."""
    print("Processing weights after loading - no-op for FlashMoeWrapper")


def apply_flash_moe_patch():
    """
    Apply the FlashMoE monkey patches to vLLM.
    
    This function is idempotent - calling it multiple times has no effect
    after the first successful patch.
    """
    global _PATCHED
    
    if _PATCHED:
        return
    
    try:
        import vllm.model_executor.models.qwen3_moe
        import vllm.model_executor.layers.fused_moe
        
        FlashMoeWrapper = _get_flash_moe_wrapper_class()
        
        vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock = FlashMoeWrapper
        vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod.process_weights_after_loading = _custom_process_weights_after_loading
        
        _PATCHED = True
        print(f"[PID {os.getpid()}] FlashMoE patch applied successfully")
    except ImportError as e:
        print(f"[PID {os.getpid()}] Warning: Could not apply FlashMoE patch: {e}")


def is_patched():
    """Check if the FlashMoE patch has been applied."""
    return _PATCHED


# The sitecustomize.py content that will be created in a temp directory.
# This gets executed at Python startup in EVERY process.
_SITECUSTOMIZE_CODE = '''
import os
import sys

def _apply_flash_moe_patch():
    """Apply FlashMoE patches at Python startup."""
    if os.environ.get("FLASH_MOE_ENABLED", "0") != "1":
        return
    
    # Check if already patched
    if getattr(sys, "_flash_moe_patched", False):
        return
    
    try:
        import torch
        import torch.distributed as dist
        import vllm.model_executor.models.qwen3_moe
        import vllm.model_executor.layers.fused_moe
        import flashMoeLauncher

        _OriginalQwen3MoeSparseMoeBlock = vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock

        class FlashMoeWrapper(_OriginalQwen3MoeSparseMoeBlock):

            def __init__(self, vllm_config, prefix: str = ""):
                super().__init__(vllm_config, prefix)

                ep_group = self.ep_group
                ep_rank = self.ep_rank
                ep_size = self.ep_size

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
                    uniqueid = self.launcher.getDistributedUniqueId(empty=True)
                    if ep_rank == 0:
                        uniqueid = self.launcher.getDistributedUniqueId()
                        broadcast_objects = [uniqueid]
                    else:
                        broadcast_objects = [None]

                    dist.broadcast_object_list(
                        broadcast_objects,
                        src=dist.get_process_group_ranks(ep_group)[0],
                        group=ep_group
                    )
                    dist.barrier(ep_group)

                    uniqueid = broadcast_objects[0]

                    self.launcher.initializeDistributed(uniqueid, ep_rank, ep_size)

            def __del__(self):
                self.launcher.destroy()

            def forward(self, hidden_states):
                num_tokens, _ = hidden_states.shape
                if self.ep_size == 1 and num_tokens < self.maxTokens:
                    tokens = hidden_states
                    self.launcher.launch(tokens, tokens)
                    return tokens
                else:
                    return super().forward(hidden_states)

        def _custom_process_weights_after_loading(self, layer) -> None:
            print("Processing weights after loading - no-op for FlashMoeWrapper")

        vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock = FlashMoeWrapper
        vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod.process_weights_after_loading = _custom_process_weights_after_loading

        sys._flash_moe_patched = True
        print(f"[PID {os.getpid()}] FlashMoE patch applied via sitecustomize")
    except ImportError as e:
        # vLLM not yet imported or not available - that's OK, we'll use import hook
        pass

# Try to apply immediately (in case vLLM is already imported)
_apply_flash_moe_patch()

# Also install an import hook to patch when vllm.model_executor.models.qwen3_moe is imported
class _FlashMoeImportHook:
    """Import hook that applies patch when qwen3_moe module is imported."""
    
    def find_module(self, fullname, path=None):
        if fullname == "vllm.model_executor.models.qwen3_moe":
            return self
        return None
    
    def load_module(self, fullname):
        # Remove ourselves temporarily to avoid recursion
        sys.meta_path = [h for h in sys.meta_path if not isinstance(h, _FlashMoeImportHook)]
        
        # Import the real module
        import importlib
        module = importlib.import_module(fullname)
        
        # Re-add ourselves
        sys.meta_path.insert(0, _FlashMoeImportHook())
        
        # Apply the patch
        _apply_flash_moe_patch()
        
        return module

if os.environ.get("FLASH_MOE_ENABLED", "0") == "1":
    if not any(isinstance(h, _FlashMoeImportHook) for h in sys.meta_path):
        sys.meta_path.insert(0, _FlashMoeImportHook())
'''


def _cleanup_sitecustomize_dir():
    """Remove the temporary sitecustomize directory."""
    global _SITECUSTOMIZE_DIR
    if _SITECUSTOMIZE_DIR and os.path.exists(_SITECUSTOMIZE_DIR):
        try:
            shutil.rmtree(_SITECUSTOMIZE_DIR)
        except Exception:
            pass
        _SITECUSTOMIZE_DIR = None


def enable_flash_moe_patching():
    """
    Enable FlashMoE patching for ALL subprocesses (including grandchildren).
    
    This works by:
    1. Setting FLASH_MOE_ENABLED=1 environment variable
    2. Creating a temp directory with sitecustomize.py 
    3. Prepending that directory to PYTHONPATH
    
    Python automatically imports sitecustomize.py at startup, so all 
    subprocesses will have the patch applied.
    
    Call this ONCE in your main process before spawning any subprocesses.
    """
    global _SITECUSTOMIZE_DIR
    
    # Set the env var
    os.environ["FLASH_MOE_ENABLED"] = "1"
    
    # Create temp directory for sitecustomize.py if not already created
    if _SITECUSTOMIZE_DIR is None:
        _SITECUSTOMIZE_DIR = tempfile.mkdtemp(prefix="flash_moe_patch_")
        
        # Write sitecustomize.py
        sitecustomize_path = os.path.join(_SITECUSTOMIZE_DIR, "sitecustomize.py")
        with open(sitecustomize_path, "w") as f:
            f.write(_SITECUSTOMIZE_CODE)
        
        # Register cleanup
        atexit.register(_cleanup_sitecustomize_dir)
    
    # Prepend to PYTHONPATH (for child processes)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if _SITECUSTOMIZE_DIR not in current_pythonpath:
        if current_pythonpath:
            os.environ["PYTHONPATH"] = f"{_SITECUSTOMIZE_DIR}:{current_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = _SITECUSTOMIZE_DIR
    
    # Also add to sys.path for current process
    if _SITECUSTOMIZE_DIR not in sys.path:
        sys.path.insert(0, _SITECUSTOMIZE_DIR)
    
    # Apply patch in current process too
    apply_flash_moe_patch()
    
    print(f"[PID {os.getpid()}] FlashMoE patching enabled (sitecustomize at {_SITECUSTOMIZE_DIR})")


def disable_flash_moe_patching():
    """Disable FlashMoE patching for future subprocesses."""
    os.environ["FLASH_MOE_ENABLED"] = "0"
    _cleanup_sitecustomize_dir()
    
    # Remove from PYTHONPATH
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if _SITECUSTOMIZE_DIR and _SITECUSTOMIZE_DIR in current_pythonpath:
        parts = [p for p in current_pythonpath.split(":") if p != _SITECUSTOMIZE_DIR]
        os.environ["PYTHONPATH"] = ":".join(parts)


# Auto-apply patch if environment variable is set (for sitecustomize bootstrap)
if os.environ.get("FLASH_MOE_ENABLED", "0") == "1":
    apply_flash_moe_patch()
