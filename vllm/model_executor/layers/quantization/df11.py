import os
import sys
import math
from typing import Optional

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.utils import direct_register_custom_op

from torch._dynamo import disable as dynamo_disable, allow_in_graph as dynamo_allow_in_graph

logger = init_logger(__name__)


# will fix it by df11 plugin
try:
    # __file__ = .../vllm/model_executor/layers/quantization/df11.py
    repo_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )
    )
    df11_path = os.path.join(repo_root, "DF11")
    if df11_path not in sys.path:
        sys.path.insert(0, df11_path)
except Exception:
    pass

# df11_extension should be installed seperately, but will fix later -> vLLM plugin
def import_df11_extension():
    try:
        import importlib
        # first, try regular import
        return importlib.import_module("dfloat11_decode_v2")
    except Exception:
        # try to load from DF11 directory explicitly if present
        # wil move to extension
        candidates = [
            os.path.join(df11_path, "dfloat11_decode_v2.cpython-312-x86_64-linux-gnu.so"),
            os.path.join(df11_path, "build", "lib.linux-x86_64-cpython-312",
                         "dfloat11_decode_v2.cpython-312-x86_64-linux-gnu.so"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                sys.path.insert(0, os.path.dirname(cand))
                try:
                    import importlib
                    return importlib.import_module("dfloat11_decode_v2")
                except Exception:
                    continue
        return None

dfloat11_decode_v2 = import_df11_extension()
if dfloat11_decode_v2 is None:
    logger.warning(
        "DF11: dfloat11_decode_v2 extension not found.")


def apply_df11_embedding(
    indices: torch.Tensor,
    luts: torch.Tensor,
    encoded_exponent: torch.Tensor,
    sign_mantissa: torch.Tensor,
    output_positions: torch.Tensor,
    gaps: torch.Tensor,
    threads_per_block: int,
    bytes_per_thread: int,
    shared_mem_size: int,
    num_embeddings: int,
    embedding_dim: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Decode to a scratch vector then gather embeddings
    assert dfloat11_decode_v2 is not None, "DF11 extension missing at runtime"
    device = encoded_exponent.device
    n_elements = sign_mantissa.numel()
    n_bytes = encoded_exponent.numel()
    n_luts = int(luts.shape[0])
    out = TensorManager.get(device, n_elements)
    threads = int(threads_per_block)
    blocks = int(math.ceil(n_bytes / (threads * int(bytes_per_thread))))
    dfloat11_decode_v2.decode(
        luts.data_ptr(),
        encoded_exponent.data_ptr(),
        sign_mantissa.data_ptr(),
        output_positions.data_ptr(),
        gaps.data_ptr(),
        out.data_ptr(),
        n_luts,
        n_bytes,
        n_elements,
        blocks,
        threads,
        shared_mem_size,
    )
    weight_2d = out.view(num_embeddings, embedding_dim)
    if dtype is None:
        dtype = weight_2d.dtype
    return torch.embedding(weight_2d.to(dtype), indices)

try:
    def fake_df11_embedding(
        indices: torch.Tensor,
        luts: torch.Tensor,
        encoded_exponent: torch.Tensor,
        sign_mantissa: torch.Tensor,
        output_positions: torch.Tensor,
        gaps: torch.Tensor,
        threads_per_block: int,
        bytes_per_thread: int,
        shared_mem_size: int,
        num_embeddings: int,
        embedding_dim: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        # Create a fake output tensor with the correct shape and dtype for Dynamo
        # This follows the same way as vLLM's models
        out_dtype = dtype or torch.bfloat16
        # Support arbitrary-shaped indices; append embedding_dim as last dim
        out_shape = tuple(list(indices.shape) + [int(embedding_dim)])
        # Allocate on the same device as indices
        return torch.empty(out_shape, dtype=out_dtype, device=indices.device)

    direct_register_custom_op(
        op_name="apply_df11_embedding",
        op_func=apply_df11_embedding,
        mutates_args=[],
        fake_impl=fake_df11_embedding,
    )
    apply_df11_embedding_op = torch.ops.vllm.apply_df11_embedding
except Exception as e:  # pragma: no cover
    logger.warning("DF11: failed to register custom embedding op: %s", e)
    apply_df11_embedding_op = None


# Linear decode custom op (decodes to a 1D bf16 tensor)
def df11_decode(
    luts: torch.Tensor,
    encoded_exponent: torch.Tensor,
    sign_mantissa: torch.Tensor,
    output_positions: torch.Tensor,
    gaps: torch.Tensor,
    threads_per_block: int,
    bytes_per_thread: int,
    shared_mem_size: int,
    n_elements: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert dfloat11_decode_v2 is not None, "DF11 extension missing at runtime"
    device = encoded_exponent.device
    n_bytes = encoded_exponent.numel()
    n_luts = int(luts.shape[0])
    out = TensorManager.get(device, int(n_elements))
    threads = int(threads_per_block)
    blocks = int(math.ceil(n_bytes / (threads * int(bytes_per_thread))))
    dfloat11_decode_v2.decode(
        luts.data_ptr(),
        encoded_exponent.data_ptr(),
        sign_mantissa.data_ptr(),
        output_positions.data_ptr(),
        gaps.data_ptr(),
        out.data_ptr(),
        n_luts,
        n_bytes,
        int(n_elements),
        blocks,
        threads,
        int(shared_mem_size),
    )
    if dtype is not None and out.dtype != dtype:
        return out.to(dtype)
    return out


def fake_df11_decode(
    luts: torch.Tensor,
    encoded_exponent: torch.Tensor,
    sign_mantissa: torch.Tensor,
    output_positions: torch.Tensor,
    gaps: torch.Tensor,
    threads_per_block: int,
    bytes_per_thread: int,
    shared_mem_size: int,
    n_elements: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    fake_dtype = dtype or torch.bfloat16
    return torch.empty((int(n_elements), ), dtype=fake_dtype, device=encoded_exponent.device)

try:
    direct_register_custom_op(
        op_name="df11_decode",
        op_func=df11_decode,
        mutates_args=[],
        fake_impl=fake_df11_decode,
    )
    df11_decode_op = torch.ops.vllm.df11_decode
except Exception as e:  # pragma: no cover
    logger.warning("DF11: failed to register decode op: %s", e)
    df11_decode_op = None


class TensorManager:

    buffers = {}

    @staticmethod
    def get(device: torch.device, n_elements: int) -> torch.Tensor:
        if isinstance(device, str):
            device = torch.device(device)
        buf = TensorManager.buffers.get(device)
        if buf is None or buf.numel() < n_elements:
            buf = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
            TensorManager.buffers[device] = buf
        return buf[:n_elements]

class DF11Config(QuantizationConfig):
    """Placeholder config
    Here, we will add df11 config -> load_format="df11"
    """

    def get_name(self) -> str:  # type: ignore[override]
        return "df11"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:  # type: ignore[override]
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:  # type: ignore[override]
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:  # type: ignore[override]
        return []

    @classmethod
    def from_config(cls, config: dict) -> "DF11Config":  # type: ignore[override]
        return cls()

    # We do not use global get_quant_method for DF11; loader assigns methods.


class DF11LinearMethod(LinearMethodBase):
    """Linear method that decodes DF11-compressed weights on-the-fly.

    The compressed tensors must be registered as buffers on the layer:
    - luts (int16/uint16), encoded_exponent (uint8), sign_mantissa (uint16),
      output_positions (uint32), gaps (uint16), and attribute shared_mem_size.
    """

    def __init__(self, *, threads_per_block: tuple[int, ...],
                 bytes_per_thread: int):
        super().__init__()
        self.threads_per_block = tuple(threads_per_block)
        self.bytes_per_thread = int(bytes_per_thread)

    def create_weights(self, layer):
        """Do not materialize dense weights for DF11 layers.

        Keeping a dummy attribute avoids downstream assumptions. We register a
        tiny non-trainable Parameter to satisfy potential hasattr checks,
        but DF11 apply() ignores it.
        """
        if not hasattr(layer, "weight"):
            layer.register_parameter(
                "weight",
                Parameter(torch.empty(0, dtype=torch.bfloat16),
                          requires_grad=False),
            )

    def decode_into(self, layer) :
        # Use the custom op so Dynamo can trace with a fake implementation
        n_elements = int(layer.sign_mantissa.numel())
        if df11_decode_op is not None:
            return df11_decode_op(
                layer.luts,
                layer.encoded_exponent,
                layer.sign_mantissa,
                layer.output_positions,
                layer.gaps,
                int(self.threads_per_block[0]),
                int(self.bytes_per_thread),
                int(layer.shared_mem_size),
                int(n_elements),
                torch.bfloat16,
            )
        # Fallback to direct extension
        assert dfloat11_decode_v2 is not None, "DF11 CUDA extension is not available"
        device = layer.encoded_exponent.device
        out = TensorManager.get(device, n_elements)
        n_bytes = layer.encoded_exponent.numel()
        n_luts = layer.luts.shape[0]
        blocks_per_grid = int(math.ceil(n_bytes / (self.threads_per_block[0] * self.bytes_per_thread)))
        dfloat11_decode_v2.decode(
            layer.luts.data_ptr(),
            layer.encoded_exponent.data_ptr(),
            layer.sign_mantissa.data_ptr(),
            layer.output_positions.data_ptr(),
            layer.gaps.data_ptr(),
            out.data_ptr(),
            n_luts,
            n_bytes,
            n_elements,
            blocks_per_grid,
            self.threads_per_block[0],
            layer.shared_mem_size,
        )
        return out

    @dynamo_disable
    @dynamo_allow_in_graph
    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Decode compressed weight to scratch and GEMM.
        decoded = self.decode_into(layer)

        # Determine expected 2D shape from layer attributes.
        if hasattr(layer, "output_size") and hasattr(layer, "input_size"):
            out_features = int(layer.output_size)
            in_features = int(layer.input_size)
        elif hasattr(layer, "out_features") and hasattr(layer, "in_features"):
            out_features = int(layer.out_features)
            in_features = int(layer.in_features)
        else:
            # Fallback to square root-based guess; should not happen for vLLM
            in_features = x.shape[-1]
            out_features = decoded.numel() // in_features

        weight_2d = decoded.view(out_features, in_features)
        return dispatch_unquantized_gemm()(layer, x, weight_2d, bias)

    # Embedding layers use a separate method implementation below


class DF11EmbeddingMethod(DF11LinearMethod):
    """Embedding method for DF11 that supports both embedding() and apply()."""

    def embedding(self, layer: VocabParallelEmbedding, indices: torch.Tensor):
        # Use a lightweight custom op wrapper to avoid Dynamo stepping into
        # Python decoding code paths. The op itself calls DF11 decode and then
        # performs embedding gather.
        if apply_df11_embedding_op is not None:
            return apply_df11_embedding_op(
                indices,
                layer.luts,
                layer.encoded_exponent,
                layer.sign_mantissa,
                layer.output_positions,
                layer.gaps,
                int(self.threads_per_block[0]),
                int(self.bytes_per_thread),
                int(layer.shared_mem_size),
                int(layer.num_embeddings),
                int(layer.embedding_dim),
                torch.bfloat16,
            )
        # Fallback to Python path (may be slower) WIP
        decoded = self.decode_into(layer)
        weight_2d = decoded.view(layer.num_embeddings, layer.embedding_dim)
        return torch.embedding(weight_2d, indices)


class DF11LinearSplitMethod(DF11LinearMethod):
    """DF11 method that decodes from a parent holder and slices for submodule.

    Some checkpoints store compressed weight at a parent module and split the
    dense weights across child linear modules. For each child, we assign this
    method with a reference to the parent module and per-child slice offsets.
    """

    def __init__(self, *, threads_per_block: tuple[int, ...],
                 bytes_per_thread: int, parent: torch.nn.Module,
                 start_index: int, end_index: int):
        super().__init__(threads_per_block=threads_per_block,
                         bytes_per_thread=bytes_per_thread)
        self._parent = parent
        self._start = int(start_index)
        self._end = int(end_index)

    @dynamo_disable
    def decode_into(self, layer: torch.nn.Module) -> torch.Tensor:  # type: ignore[override]
        # Decode using parent buffers, then slice the flat vector
        full = super().decode_into(self._parent)
        return full[self._start:self._end]



# Helper used by V1 layers to avoid inlining DF11LinearMethod.apply
def df11_apply_linear(layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
    # Decode using the df11 custom op path (also works with fake tensors)
    # Avoid calling any decode_into bound methods to keep Dynamo happy.
    method = getattr(layer, "quant_method", None)
    # Determine source module for buffers (parent for split method)
    src = getattr(method, "_parent", layer)
    # Total elements to decode from source
    n_elements = int(src.sign_mantissa.numel())
    if df11_decode_op is not None:
        full = df11_decode_op(
            src.luts,
            src.encoded_exponent,
            src.sign_mantissa,
            src.output_positions,
            src.gaps,
            int(method.threads_per_block[0] if hasattr(method, "threads_per_block") else 256),
            int(method.bytes_per_thread if hasattr(method, "bytes_per_thread") else 16),
            int(src.shared_mem_size),
            int(n_elements),
            torch.bfloat16,
        )
    else:
        # Fallback to extension decode
        assert dfloat11_decode_v2 is not None, "DF11 CUDA extension is not available"
        device = src.encoded_exponent.device
        full = TensorManager.get(device, n_elements)
        n_bytes = src.encoded_exponent.numel()
        n_luts = src.luts.shape[0]
        threads = int(method.threads_per_block[0])
        blocks = int(math.ceil(n_bytes / (threads * int(method.bytes_per_thread))))
        dfloat11_decode_v2.decode(
            src.luts.data_ptr(),
            src.encoded_exponent.data_ptr(),
            src.sign_mantissa.data_ptr(),
            src.output_positions.data_ptr(),
            src.gaps.data_ptr(),
            full.data_ptr(),
            n_luts,
            n_bytes,
            n_elements,
            blocks,
            threads,
            int(src.shared_mem_size),
        )
    # Slice if split method
    if hasattr(method, "_start") and hasattr(method, "_end"):
        decoded = full[int(method._start):int(method._end)]
    else:
        decoded = full
    # Infer 2D weight shape
    if hasattr(layer, "output_size") and hasattr(layer, "input_size"):
        out_features = int(layer.output_size)
        in_features = int(layer.input_size)
    elif hasattr(layer, "out_features") and hasattr(layer, "in_features"):
        out_features = int(layer.out_features)
        in_features = int(layer.in_features)
    else:
        in_features = x.shape[-1]
        out_features = decoded.numel() // in_features
    weight_2d = decoded.view(out_features, in_features)
    return dispatch_unquantized_gemm()(layer, x, weight_2d, bias)
