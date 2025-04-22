"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest


# <>
from vllm.core.block_manager import KVBlkPerLayerWeight



class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = ModelRunner(model_config,
                                        parallel_config,
                                        scheduler_config,
                                        device_config,
                                        lora_config=self.lora_config,
                                        kv_cache_dtype=kv_cache_dtype,
                                        is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

        # <> 
        # TODO (): seems the worker process can inherit the os environment variables from the main process, so we do not need to set "os.environ['CHANGE_KV_LAYOUT']" here
        # self.tot_gpu_num = tot_gpu_num
        os.environ['LOCAL_RANK'] = str(local_rank)
        print(f"os.environ['CUDA_VISIBLE_DEVICES']:{os.environ['CUDA_VISIBLE_DEVICES']}, LOCAL_RANK: {os.environ['LOCAL_RANK']}  self.model: {self.model_config.model}")
        # self.tot_ordered_gpus = 'None' # os.environ['TOT_ORDERED_GPUS']
        self.tot_ordered_gpus = os.getenv("TOT_ORDERED_GPUS", 'None')
        print(f"os.environ['TOT_ORDERED_GPUS']:{self.tot_ordered_gpus}  self.model: {self.model_config.model}")
        # print(f"JUST FOR DEBUG: device_config.device in init worker: {device_config.device}")

    def init_model(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)



            # <> as the order in os.environ["CUDA_VISIBLE_DEVICES"] may be wrong due to Ray, we need to get the correct gpus
            # TODO (): check the correctness of this part.
            if self.tot_ordered_gpus != 'None':
                ori_gpu_orders = self.tot_ordered_gpus.split(',')
                curr_gpu_orders = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
                print(f"ori_gpu_orders: {ori_gpu_orders}, curr_gpu_orders: {curr_gpu_orders}")
                for i, gpu_i in enumerate(curr_gpu_orders):
                    if gpu_i == ori_gpu_orders[self.local_rank]:
                        self.device = torch.device(f"cuda:{i}")
                        print(f"After cuda remapping, torch.current_device set to {i}")
                        break
            else:
                self.device = torch.device(f"cuda:{self.local_rank}")



            # self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            # <> For DEBUG
            print(f'rank: {self.rank}, local_rank: {self.local_rank}, visible gpus: {os.environ["CUDA_VISIBLE_DEVICES"]}')


            _check_if_gpu_supports_dtype(self.model_config.dtype)
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.

        # <> For Profiling
        import time
        start = time.perf_counter()

        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method)
        
        end = time.perf_counter()
        print(f"init_distributed_environment time: {end - start}s")

        if not self.parallel_config.disable_custom_all_reduce:


            # <> For DEBUG
            print(f"init_custom_ar -------------", flush=True)
            start = time.perf_counter()

            init_custom_ar()
            
            end = time.perf_counter()
            print(f"init_custom_ar time: {end - start}s")
        # Initialize the model.

        # <> For DEBUG
        print(f"set_random_seed -------------", flush=True)

        set_random_seed(self.model_config.seed)



    # <>
    def disable_p2ps(self):
        from vllm._C import cache_ops
        for cache_device_i in self.model_runner.model.model.cache_device_ids:        
            cache_ops.disable_P2P_access(cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())





    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)


        # <> store the information of cache_block_size in class 
        KVBlkPerLayerWeight.block_size = cache_block_size
        if os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] == 'True':
            assert KVBlkPerLayerWeight.layer_weight_size>0, KVBlkPerLayerWeight.layer_weight_size
        KVBlkPerLayerWeight.blk_num_per_layer = (KVBlkPerLayerWeight.layer_weight_size + KVBlkPerLayerWeight.block_size - 1) // KVBlkPerLayerWeight.block_size
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"\n\nblk_num_per_layer: {KVBlkPerLayerWeight.blk_num_per_layer}\n\n")
            print(f"total_gpu_memory: {total_gpu_memory}, gpu_memory_utilization:{gpu_memory_utilization}, peak_memory:{peak_memory}, cache_block_size:{cache_block_size}")



        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        # return num_gpu_blocks, num_cpu_blocks

        # <> also return the information of KVBlkPerLayerWeight
        return num_gpu_blocks, num_cpu_blocks, (KVBlkPerLayerWeight.blk_num_per_layer, KVBlkPerLayerWeight.cached_layer_num)
    


    # <> fake profiling 
    def fake_profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> None:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)

        # store total gpu memory and KV cache dtype
        _, total_gpu_memory = torch.cuda.mem_get_info()
        KVBlkPerLayerWeight.tot_gpu_mem = total_gpu_memory

        # <> store the information of cache_block_size in class 
        KVBlkPerLayerWeight.block_size = cache_block_size
        if os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] == 'True':
            assert KVBlkPerLayerWeight.layer_weight_size>0, KVBlkPerLayerWeight.layer_weight_size
        KVBlkPerLayerWeight.blk_num_per_layer = (KVBlkPerLayerWeight.layer_weight_size + KVBlkPerLayerWeight.block_size - 1) // KVBlkPerLayerWeight.block_size
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"\n\nblk_num_per_layer: {KVBlkPerLayerWeight.blk_num_per_layer}\n\n")
            print(f"gpu_memory_utilization:{gpu_memory_utilization}, cache_block_size:{cache_block_size}")

        # <> also return the information of KVBlkPerLayerWeight ==> do not need to return KVBlkPerLayerWeight
        # return num_gpu_blocks, num_cpu_blocks, (KVBlkPerLayerWeight.blk_num_per_layer, KVBlkPerLayerWeight.cached_layer_num)





    # <>
    @torch.inference_mode()
    def profile_per_iter_latency(
        self,
        is_prompt: bool,
        sampling_params_dict: Dict[str, float], 
        set_max_num_batched_tokens: float = float('inf'),
        set_max_num_seqs: float = float('inf'),
        set_seqlens: List[int] = list(),
    ) -> Tuple[int, int]:
        """Profile the per iteration latency of the model.
        Args:
            is_prompt: controls which stage (prefill or decoding stage) to profile
            sampling_params_dict: the sampling param dictionary to be used.

            set_max_num_batched_tokens: the max_num_batched_tokens (< the limit by KV cache)
            set_max_num_seqs: the max_num_seqs (< the limit by the scheduler config)
        """
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_per_iter_latency(
            is_prompt, sampling_params_dict, self.gpu_cache, 
            set_max_num_batched_tokens,set_max_num_seqs,
            set_seqlens)
    







    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

        # <> support dynamically increasing on-card layer weights
        if os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] == 'True':
            print(f"self.cache_engine.continuous_gpu_cache:{self.cache_engine.continuous_gpu_cache.shape}")
            self.model_runner.init_extra_weight_cache_from_KV_cache([self.cache_engine.continuous_gpu_cache])


    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        # <> added parameters
        blocks_to_reorganize: Dict[int, List[int]],
        blk_num_reduced: int,
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True


        # <>
        if blocks_to_reorganize:
            self.cache_engine.reorganize_blocks(blocks_to_reorganize, blk_num_reduced)
            # update gpu cache
            self.gpu_cache = self.cache_engine.gpu_cache
            issued_cache_op = True


        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,

        # <> added parameter: support dynamically increasing on-card layer weights
        load_more_layer_on_card_num: int = 0,
        blocks_to_reorganize: Optional[Dict[int, List[int]]] = None,

    ) -> Optional[SamplerOutput]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            assert blocks_to_reorganize is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
                # <> added dict content
                "blocks_to_reorganize": blocks_to_reorganize,
                "load_more_layer_on_card_num": load_more_layer_on_card_num,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

            # <>
            blocks_to_reorganize = data["blocks_to_reorganize"]
            load_more_layer_on_card_num = data["load_more_layer_on_card_num"]


        # TODO () check the correctness when there is driver worker
        # <> update KVBlkPerLayerWeight when there are multiple workers
        KVBlkPerLayerWeight.load_more_layer_on_card_num = load_more_layer_on_card_num
        blk_num_reduced = KVBlkPerLayerWeight.blk_num_per_layer * load_more_layer_on_card_num


        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy, blocks_to_reorganize, blk_num_reduced)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()


def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():

        # <> For DEBUG
        print(f"init_distributed_environment 1-------------")

        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:

        # <> For DEBUG
        print(f"init_distributed_environment 3-------------")


        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)
    

    # <> For DEBUG
    print(f"finish init model-------------", flush=True)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
