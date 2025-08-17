import math
import os
import re
import sys
from sys import stderr
from typing import Dict, List, Tuple


import torch
import torch.nn as nn
from safetensors.torch import load_file
from tqdm import tqdm
from transformers.models.auto.configuration_auto import AutoConfig
from huggingface_hub import snapshot_download

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead


# Add local path for now
df11_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "DF11")
if df11_path not in sys.path:
    sys.path.insert(0, df11_path)

import dfloat11_decode_v2

logger = init_logger(__name__)


class TensorManager:
    """
    Static utility class that manages tensor allocation and reuse
    to minimize memory allocation overhead during tensor reconstruction.
    """
    # Static class variables to store tensors
    _tensors = {}  # Maps device to tensor

    @staticmethod
    def allocate_bfloat16(device, n_elements):
        """
        Get a bfloat16 tensor with at least n_elements on the specified device.

        If a tensor already exists on the device and is larger than n_elements,
        a slice of the tensor with exactly n_elements is returned. If n_elements 
        exceeds the size of the existing tensor, the existing tensor is deallocated 
        and a larger one is allocated.

        Args:
            device: The device to allocate the tensor on (e.g., 'cuda:0')
            n_elements: The exact number of elements required

        Returns:
            A bfloat16 tensor with exactly n_elements on the specified device
        """
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check if we already have a tensor for this device
        if device in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[device]
            
            # If existing tensor is large enough, return a slice of it
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]
            
            # Otherwise, delete the existing tensor to free up memory
            del TensorManager._tensors[device]
            torch.cuda.empty_cache()  # Ensure memory is freed
        
        # Allocate a new tensor
        new_tensor = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        print(f'Allocated {n_elements} bf16 on device {device}', file=stderr)
        
        # Store the tensor
        TensorManager._tensors[device] = new_tensor
        
        return new_tensor

    @staticmethod
    def clear_device(device=None):
        """
        Clear tensors for a specific device or all devices if none specified.
        
        Args:
            device: The device to clear tensors from, or None to clear all devices
        """
        if device is None:
            # Clear all devices
            TensorManager._tensors.clear()
        else:
            # Convert device to torch.device if it's a string
            if isinstance(device, str):
                device = torch.device(device)
                
            # Remove specific device
            if device in TensorManager._tensors:
                del TensorManager._tensors[device]
                
        torch.cuda.empty_cache()  # Ensure memory is freed

def get_hook(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly.
    
    This hook reconstructs full-precision weights from compressed representations
    using a custom CUDA kernel during the forward pass.
    
    Args:
        threads_per_block: CUDA thread configuration 
        bytes_per_thread: Number of bytes processed per CUDA thread
        
    Returns:
        A forward pre-hook function for PyTorch modules
    """
    threads_per_block = tuple(threads_per_block)

    def decode_hook(module: nn.Module, _):
        device = module.luts.device
        
        # Load offloaded tensors to GPU if not already there
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name, tensor in module.offloaded_tensors.items():
                if not (
                    hasattr(module, tensor_name) and (getattr(module, tensor_name).device == device)
                ):
                    module.register_buffer(tensor_name, tensor.to(device, non_blocking=True))

        # Get dimensions for tensor reconstruction
        n_elements = module.sign_mantissa.numel()
        n_bytes = module.encoded_exponent.numel()
        n_luts = module.luts.shape[0]
        
        # Get output tensor for reconstructed weights
        reconstructed = TensorManager.allocate_bfloat16(device, n_elements)

        # Configure CUDA grid dimensions for the kernel launch
        blocks_per_grid = (
            int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))),
        )

        dfloat11_decode_v2.decode(
            module.luts.data_ptr(),
            module.encoded_exponent.data_ptr(),
            module.sign_mantissa.data_ptr(),
            module.output_positions.data_ptr(),
            module.gaps.data_ptr(),
            reconstructed.data_ptr(),
            n_luts,
            n_bytes,
            n_elements,
            blocks_per_grid[0],
            threads_per_block[0],
            module.shared_mem_size,
        )
        
        # Inject reconstructed weights into the appropriate module
        if isinstance(module, nn.Linear):
            logger.debug(f"DF11: Linear module - setting weight with shape {module.out_features}x{module.in_features}")
            module.weight = reconstructed.view(
                module.out_features, module.in_features
            )
        elif isinstance(module, ParallelLMHead):
            logger.debug(f"DF11: ParallelLMHead - setting _df11_weight with shape {module.num_embeddings}x{module.embedding_dim}")
            module._df11_weight = reconstructed.view(
                module.num_embeddings, module.embedding_dim
            )
        elif isinstance(module, (nn.Embedding, VocabParallelEmbedding)):
            logger.debug(f"DF11: Embedding-like module - setting weight with shape {module.num_embeddings}x{module.embedding_dim}")
            module.weight = reconstructed.view(
                module.num_embeddings, module.embedding_dim
            )
        elif hasattr(module, 'weight_injection_modules'):
            # Handle special case where weights need to be split across multiple submodules
            weights = torch.tensor_split(reconstructed, module.split_positions)
            for sub_module, weight in zip(module.weight_injection_modules, weights):
                sub_module.weight = weight.view(sub_module.out_features, sub_module.in_features)
        else:
            # Fallback: try to find out_features and in_features attributes
            if hasattr(module, 'out_features') and hasattr(module, 'in_features'):
                module.weight = reconstructed.view(module.out_features, module.in_features)
            elif hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
                module.weight = reconstructed.view(module.num_embeddings, module.embedding_dim)
            else:
                logger.error(f"DF11: Unknown module type {type(module).__name__} for weight injection")

        # Delete tensors from GPU if offloading is enabled
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name in module.offloaded_tensors.keys():
                if hasattr(module, tensor_name):
                    tmp = getattr(module, tensor_name)
                    delattr(module, tensor_name)
                    del tmp

    return decode_hook


class DF11ModelLoader(BaseModelLoader):

    COMPRESSED_SUFFIXES = {
        "encoded_exponent",
        "sign_mantissa",
        "luts",
        "output_positions",
        "gaps",
        "split_positions",
    }


    def download_model(self, model_config: ModelConfig) -> None:
        # will find a better way to handle it, prob chaing the df11 config file
        model_path = model_config.model
        if os.path.exists(model_path):
            return
        logger.info("Downloading DF11 model %s from HF Hub", model_path)
        local_dir = model_path.replace("/", "__")
        snapshot_download(model_path, local_dir=local_dir, repo_type="model")
        model_config.model = local_dir  # mutate so load_weights sees the correct path


    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_path = model_config.model
        cfg = AutoConfig.from_pretrained(model_path)
        if not hasattr(cfg, "dfloat11_config"):
            raise ValueError("Checkpoint lacks dfloat11_config; not a DF11 model")
        df_cfg: Dict = cfg.dfloat11_config
        threads_per_block: Tuple[int, ...] = tuple(df_cfg["threads_per_block"])
        bytes_per_thread: int = int(df_cfg["bytes_per_thread"])
        pattern_dict: Dict[str, List[str]] = df_cfg["pattern_dict"]

  
        compressed: Dict[str, Dict[str, torch.Tensor]] = {}
        safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        for fname in tqdm(safetensor_files, desc="Reading DF11 safetensors"):
            loaded = load_file(os.path.join(model_path, fname))
            for tname, tvalue in loaded.items():
                layer_path, component = self.parse_compressed_tensor_name(tname)
                if layer_path is None:
                    if tname in dict(model.named_parameters()):
                        param = dict(model.named_parameters())[tname]
                        if param.shape == tvalue.shape:
                            param.data.copy_(tvalue)
                    elif tname in dict(model.named_buffers()):
                        buf = dict(model.named_buffers())[tname]
                        if buf.shape == tvalue.shape:
                            buf.copy_(tvalue)
                    continue
                compressed.setdefault(layer_path, {})[component] = tvalue

   
        decode_hook = get_hook(threads_per_block, bytes_per_thread)
        for layer_path, comp in compressed.items():
            module = self.resolve_module(model, layer_path)
            # Buffers
            for name, tensor in comp.items():
                if name == "split_positions":
                    setattr(module, name, tensor.tolist())
                else:
                    module.register_buffer(name, tensor)
            # shared memory value
            if "output_positions" in comp:
                op = comp["output_positions"]
                op_u32 = op.view(torch.uint32)

                # casting
                diff_max = (op_u32[1:].to(torch.int64) - op_u32[:-1].to(torch.int64)).max().item()
                shared_mem = threads_per_block[0] * 4 + 4 + diff_max * 2
                module.shared_mem_size = shared_mem 
            
            # Set up decompression for encoded weights (v2)
            if "encoded_exponent" in comp:
                
                # Register the decode hook to decompress weights during forward pass
                module.register_forward_pre_hook(decode_hook)

                # Configure weight injection based on module type
                pattern_matched = False
                for pattern, attr_names in pattern_dict.items():
                    if re.fullmatch(pattern, layer_path):
                        pattern_matched = True                        
                        
                        # Check ParallelLMHead first since it inherits from VocabParallelEmbedding
                        if isinstance(module, ParallelLMHead):
                            if hasattr(module, 'weight'):
                                tmp = module.weight
                                delattr(module, 'weight')
                                del tmp
                            
                            # Create a property that triggers reconstruction when accessed
                            def weight_property(self):
                                # Trigger the decode hook manually
                                decode_hook(self, None)
                                # Return the reconstructed weight
                                return self._df11_weight
                            
                            # Add the property to the instance
                            module.__class__.weight = property(weight_property)
                        elif isinstance(module, (nn.Embedding, VocabParallelEmbedding)):
                            # Remove weight attribute from embedding layer
                            if hasattr(module, 'weight'):
                                tmp = module.weight
                                delattr(module, 'weight')
                                del tmp
                        elif isinstance(module, nn.Linear):
                            # Remove weight attribute from linear layer  
                            if hasattr(module, 'weight'):
                                tmp = module.weight
                                delattr(module, 'weight')
                                del tmp
                        else:
                            # Handle special case for multi-module weight injection
                            setattr(module, 'weight_injection_modules', [])
                            for attr_path in attr_names:
                                parts = attr_path.split('.')
                                target = module
                                for p in parts:
                                    target = getattr(target, p)

                                if hasattr(target, 'weight'):
                                    tmp = target.weight
                                    delattr(target, 'weight')
                                    del tmp
                                module.weight_injection_modules.append(target)
                        break
                
                if not pattern_matched:
                    print(f"DF11: WARNING - No pattern matched for {layer_path}, module type: {type(module).__name__}", flush=True)

        # move everything to the desired single GPU
        target = self.load_config.device or "cuda"
        model.to(target)
        logger.info("DF11 model loaded onto %s", target)


    @staticmethod
    def parse_compressed_tensor_name(name):
        """Returns layer_path, component """
        for suffix in DF11ModelLoader.COMPRESSED_SUFFIXES:
            if name.endswith("." + suffix):
                return name[: -len(suffix) - 1], suffix
        return None, None

    @staticmethod
    def resolve_module(root: nn.Module, dotted: str) -> nn.Module:
        mod = root
        for part in dotted.split("."):
            if not hasattr(mod, part):
                raise RuntimeError(f"Failed to resolve module path {dotted} (missing {part})")
            mod = getattr(mod, part)
        return mod

    @staticmethod
    def prepare_weight_injection(module: nn.Module, layer_path: str, pattern_dict: Dict[str, List[str]]):
        logger.debug(f"DF11: Preparing weight injection for {layer_path}, module type: {type(module).__name__}")
        
        # Find pattern match
        matched_attrs: List[str] | None = None
        for pattern, attrs in pattern_dict.items():
            if re.fullmatch(pattern, layer_path):
                matched_attrs = attrs
                logger.debug(f"DF11: Pattern '{pattern}' matched for {layer_path} with attrs: {attrs}")
                break
        if isinstance(module, (nn.Linear, nn.Embedding, VocabParallelEmbedding, ParallelLMHead)):
            logger.debug(f"DF11: Standard module ({type(module).__name__}) - removing weight attribute")
            if hasattr(module, "weight"):
                delattr(module, "weight")
            return
        # merged module: holds multiple Linear sub-modules
        if matched_attrs is None:
            logger.warning("Compressed module %s not matched in pattern_dict", layer_path)
            logger.debug(f"DF11: Available patterns: {list(pattern_dict.keys())}")
            return
        # Check if pattern has empty attributes list (treat as standard module)
        if not matched_attrs:  # Empty list means it's a standard module like embed_tokens or lm_head
            logger.debug(f"DF11: Pattern matched with empty attrs - treating as standard module")
            if hasattr(module, "weight"):
                delattr(module, "weight")
            return
        module.weight_injection_modules = [] 
        for attr_path in matched_attrs:
            sub = module
            ok = True
            for part in attr_path.split("."):
                if hasattr(sub, part):
                    sub = getattr(sub, part)
                else:
                    logger.debug("DF11: attribute %s not found under %s while preparing injection for %s", part, sub.__class__.__name__, layer_path)
                    ok = False
                    break
            if not ok:
                continue
            if hasattr(sub, "weight"):
                delattr(sub, "weight")
            module.weight_injection_modules.append(sub)
        if not module.weight_injection_modules:
            logger.warning("DF11: no sub-modules found for weight injection on %s", layer_path)

