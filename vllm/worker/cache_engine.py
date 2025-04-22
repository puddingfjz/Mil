"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm._C import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]



# <> 
import os



class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]


        # <>
        self.continuous_gpu_cache = None

        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"self.continuous_gpu_cache 1: {self.continuous_gpu_cache}")


        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

        # <> added parameters
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"KV cache layout (when initialing it): {len(self.gpu_cache), len(self.gpu_cache[0]), len(self.gpu_cache[0][0])}")


    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache_vllm(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache




    # <> we allocate a continuous memory for the KV cache. This will be ok for the 80G GPU memory (as the tensor index is in Long type)
    # this function already allocates continuous memory for KV cache, we need to change the data layout of the KV cache again
    def allocate_gpu_cache_continuous(self) -> List[KVCache]:
        key_blk_size_per_layer = self.num_gpu_blocks
        for i in self.get_key_block_shape():
            key_blk_size_per_layer = key_blk_size_per_layer * i

        value_blk_size_per_layer = self.num_gpu_blocks
        for i in self.get_value_block_shape():
            value_blk_size_per_layer = value_blk_size_per_layer * i

        # allocate a whole KV cache memory
        whole_cache = torch.empty(
                size=(self.num_layers*(key_blk_size_per_layer+value_blk_size_per_layer), ),
                dtype=self.dtype,
                device="cuda",
            )

        self.continuous_gpu_cache = whole_cache
        
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"self.continuous_gpu_cache 2: {self.continuous_gpu_cache.shape}")

        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for layer_i in range(self.num_layers):
            key_blocks = torch.narrow(whole_cache, 
                    0, 
                    layer_i*(key_blk_size_per_layer+value_blk_size_per_layer),
                    key_blk_size_per_layer).view(self.num_gpu_blocks, *key_block_shape)
            
            value_blocks = torch.narrow(whole_cache, 
                    0, 
                    layer_i*(key_blk_size_per_layer+value_blk_size_per_layer)+key_blk_size_per_layer,
                    value_blk_size_per_layer).view(self.num_gpu_blocks, *value_block_shape)

            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache



    # <> support all kinds of gpu cache allocation function
    def allocate_gpu_cache(self) -> List[KVCache]:
        # <>
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS']:{os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS']}")

        if (os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] == 'True'):
            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                print(f"self.cache_engine.continuous_gpu_cache:{self.continuous_gpu_cache}")
            return self.allocate_gpu_cache_continuous()
        else:
            return self.allocate_gpu_cache_vllm()




    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
                device="cpu",
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
                device="cpu",
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)





    # <> change the KV cache organization to make space for model weights
    def update_KV_cache_organization(self, block_num_reduced: int) -> None:

        # new_block_num = self.num_gpu_blocks - block_num_reduced
        # Fix bug: the current block number is not self.num_gpu_blocks but len(self.gpu_cache[0][0])
        # as there may already be some blocks reduced
        new_block_num  = len(self.gpu_cache[0][0]) - block_num_reduced
        whole_cache = self.continuous_gpu_cache.view(-1)

        print(f"block_num_reduced: {block_num_reduced}, new_block_num: {new_block_num}")
        
        key_blk_size_per_layer = new_block_num
        for i in self.get_key_block_shape():
            key_blk_size_per_layer = key_blk_size_per_layer * i

        value_blk_size_per_layer = new_block_num
        for i in self.get_value_block_shape():
            value_blk_size_per_layer = value_blk_size_per_layer * i

        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()

        for layer_i in range(self.num_layers):
            key_blocks = torch.narrow(whole_cache, 
                    0, 
                    layer_i*(key_blk_size_per_layer+value_blk_size_per_layer),
                    key_blk_size_per_layer).view(new_block_num, *key_block_shape)
            
            value_blocks = torch.narrow(whole_cache, 
                    0, 
                    layer_i*(key_blk_size_per_layer+value_blk_size_per_layer)+key_blk_size_per_layer,
                    value_blk_size_per_layer).view(new_block_num, *value_block_shape)

            gpu_cache.append((key_blocks, value_blocks))

            
            # if int(os.getenv("LOCAL_RANK", "0")) == 0:
            #     print(f"layer_i {layer_i} key & value cache address: {key_blocks.data_ptr(), value_blocks.data_ptr()}")

        return gpu_cache


    
    
    # <> directly reorganize the KV cache blocks to make space for model weights
    def reorganize_blocks_deprecated(self, src_to_dsts: Dict[int, List[int]], block_num_reduced: int) -> None:      

        # first update the KV cache organization to get gpu_cache_reorganized
        gpu_cache_reorganized = self.update_KV_cache_organization(block_num_reduced)

        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        new_key_caches = [key_cache for key_cache, _ in gpu_cache_reorganized]
        new_value_caches = [value_cache for _, value_cache in gpu_cache_reorganized]

        cache_ops.reorganize_blocks(key_caches, value_caches, src_to_dsts, new_key_caches, new_value_caches)
        
        # udpate gpu cache
        self.gpu_cache = gpu_cache_reorganized





    
    # <> directly reorganize the KV cache blocks to make space for model weights
    def reorganize_blocks(self, src_to_dsts: Dict[int, List[int]], block_num_reduced: int) -> None:      

        # first update the KV cache organization to get gpu_cache_reorganized
        gpu_cache_reorganized = self.update_KV_cache_organization(block_num_reduced)

        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.reorganize_blocks(self.continuous_gpu_cache, src_to_dsts[0], src_to_dsts[1], self.gpu_cache[0][0][0].numel())
        
        # udpate gpu cache
        self.gpu_cache = gpu_cache_reorganized










    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
