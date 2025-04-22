# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.config import LoRAConfig


# <>
from vllm._C import cache_ops
import time
import os


KVCache = Tuple[torch.Tensor, torch.Tensor]


class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()


        # # <> For DEBUG
        # self.infer_count = -1


    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)


        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'gate_up_proj: {x.shape} -> {gate_up.shape}')


        # <> For DEBUG
        # self.infer_count += 1
        # if (int(os.getenv("LOCAL_RANK", "0")) == 1):
        #     print(f"self.infer_count: {self.infer_count}")


        # <> For DEBUG
        # seq_to_check = 136 # 136 vllm 156 ours
        # step_to_print = 155 # 155 vllm  128 ours
        # if os.environ['USE_VLLM']=='False':
        #     seq_to_check = 156 # 136 vllm 156 ours
        #     step_to_print = 128 # 155 vllm  128 ours            
        # if (int(os.getenv("LOCAL_RANK", "0")) == 1)\
        #     and (self.infer_count >= step_to_print):
        #     print(f"x shape 1: {x.shape}")
        #     print(f"x 1: {x[seq_to_check][-1].tolist()}")
        #     print(f"gate_up shape: {gate_up.shape}")
        #     print(f"gate_up: {gate_up[seq_to_check][-1].tolist()}")
        #     print(f"gate_up_proj linear_weights: {self.gate_up_proj.linear_weights['weight'][0].tolist()}")
        #     print(f"gate_up_proj linear_weights address: {self.gate_up_proj.linear_weights['weight'].data_ptr()}")


        x = self.act_fn(gate_up)

        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'act_fn: {gate_up.shape} -> {x.shape}')

        # if (int(os.getenv("LOCAL_RANK", "0")) == 1)\
        #     and (self.infer_count >= step_to_print):
        #     print(f"x shape 2: {x.shape}")
        #     print(f"x 2: {x[seq_to_check][-1].tolist()}")
        #     with open('tmp_ours.log', 'a') as f:
        #         f.write(f"x 2: {x[seq_to_check][-1].tolist()}\n")




        x, _ = self.down_proj(x)


        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'down_proj: -> {x.shape}')

        # if (int(os.getenv("LOCAL_RANK", "0")) == 1)\
        #     and (self.infer_count >= step_to_print):
        #     print(f"x shape 3: {x.shape}")
        #     print(f"x 3: {x[seq_to_check][-1].tolist()}")
        #     print(f"down_proj linear_weights: {self.down_proj.linear_weights['weight'][0].tolist()}")
        #     print(f"down_proj linear_weights address: {self.down_proj.linear_weights['weight'].data_ptr()}")
        #     with open('tmp_ours.log', 'a') as f:
        #         f.write(f"x 3: {x[seq_to_check][-1].tolist()}\n")
        #         f.write(f"down_proj linear_weights: {self.down_proj.linear_weights['weight'][0].tolist()}\n")


        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   self.scaling,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        # <> For DEBUG
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        #     print(f"hidden_states 3: {hidden_states[-1].tolist()}")
        #     print(f"qkv_proj.linear_weights: {self.qkv_proj.linear_weights['weight'][0].tolist()}")
            # print(f"qkv_proj.linear_weights address: {self.qkv_proj.linear_weights['weight'].data_ptr()}")


        qkv, _ = self.qkv_proj(hidden_states)

        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'qkv_proj: {hidden_states.shape} -> {qkv.shape}')


        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'rotary_emb: -> {q.shape}, {k.shape}')

        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)

        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'attn: -> {attn_output.shape}')

        output, _ = self.o_proj(attn_output)

        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'o_proj: -> {output.shape}')

        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)



        # # <> For DEBUG
        # self.infer_count = -1

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention


        # <> For DEBUG
        # self.infer_count += 1
        # if (int(os.getenv("LOCAL_RANK", "0")) == 1) and (type(kv_cache[0])!=type(None)):
        #     print(f"self.infer_count: {self.infer_count}")


        # <> For DEBUG
        # seq_to_check = 136 # 136 vllm 156 ours
        # step_to_print = 155 # 155 vllm  128 ours
        # if os.environ['USE_VLLM']=='False':
        #     seq_to_check = 156 # 136 vllm 156 ours
        #     step_to_print = 128 # 155 vllm  128 ours            
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_cache[0])!=type(None))\
        #     and (self.infer_count >= step_to_print):
        #     print(f"hidden_states 1 shape: {hidden_states.shape}")
        #     print(f"hidden_states 1: {hidden_states[seq_to_check][-1].tolist()}")


        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'input_layernorm: -> {hidden_states.shape}')

        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_cache[0])!=type(None))\
        #     and (self.infer_count >= step_to_print):
        #     print(f"hidden_states 1.5: {hidden_states[seq_to_check][-1].tolist()}")
        #     # print(f"input_layernorm.weight: {self.input_layernorm.weight.tolist()}")


        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )


        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'self_attn: -> {hidden_states.shape}')


        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_cache[0])!=type(None))\
        #     and (self.infer_count >= step_to_print):
        #     print(f"hidden_states 2: {hidden_states[seq_to_check][-1].tolist()}")


        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)


        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'post_attention_layernorm: -> {hidden_states.shape}')


        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_cache[0])!=type(None))\
        #     and (self.infer_count >= step_to_print):
        #     print(f"hidden_states 3: {hidden_states[seq_to_check][-1].tolist()}")


        hidden_states = self.mlp(hidden_states)


        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'mlp: -> {hidden_states.shape}')

        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 1) and (type(kv_cache[0])!=type(None))\
        #     and (self.infer_count >= step_to_print):
        #     print(f"hidden_states 4: {hidden_states[seq_to_check][-1].tolist()}")


        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)





        # ======================================================================

        # below are only used when we consider weights offloading


        # <> to support loading parameters layer by layer
        self.cache_device_ids = [1] # example: [1] means we use GPU 1 as our cache
        # consider when there are multiple workers due to parallelism

        self.set_cache_device_ids()

        # print(f'In model, os.environ["LOCAL_RANK"]:{os.getenv("LOCAL_RANK", "0")}, all card num: {torch.cuda.device_count()}, worker_num:{torch.distributed.get_world_size()}, os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}, os.environ["TOT_ORDERED_GPUS"]:{os.environ["TOT_ORDERED_GPUS"]}  current_device: {torch.cuda.current_device()}, self.cache_device_ids: {self.cache_device_ids}')
        print(f'In model, os.environ["WEIGHT_LOAD_DEGREE"]: {os.environ["WEIGHT_LOAD_DEGREE"]}, os.environ["LOCAL_RANK"]:{os.getenv("LOCAL_RANK", "0")}, all card num: {torch.cuda.device_count()}, worker_num:{torch.distributed.get_world_size()}, os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}, current_device: {torch.cuda.current_device()}, self.cache_device_ids: {self.cache_device_ids}')

        # Initialize the stream for loading parameters.
        # self.param_stream = torch.cuda.Stream()
        # assert self.param_stream != torch.cuda.current_stream()
        # prepare a stream for each cache GPU
        self.param_streams = [torch.cuda.Stream() for _ in self.cache_device_ids]
        for param_stream in self.param_streams:
            assert param_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.load_param_events = [[torch.cuda.Event() for _ in self.layers] for _ in self.cache_device_ids]
        self.layer_waiting_for_params = [False for _ in self.layers]
        self.layer_num = len(self.layers)


        # <> For offloading weights layer by layer
        # compute the size of parameters for each layer
        self.weight_cache: torch.Tensor = torch.Tensor([])
        # self.weight_cache_cpu: torch.Tensor = torch.Tensor([])
        self.weight_cache_cpu: List[List[torch.Tensor]] = []
        self.weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list() # (int, int, size_tuple): (start position, offset, tensor shape)
        self.weight_num_per_layer: int = -1
        
        # because of the high memory transfer cost: almost 1/2 decoding time per iter, only allows 1 layer to be transferred
        # self.weight_cache_block_num: int = self.layer_num - 1 # (self.layer_num + 1)//2
        # we set a parameter: pipeline degree to control weight_cache_block_num
        # self.pipeline_degree: int = 20 # 1 iter decoding time = pipeline_degree * 1 layer weight loading time
        self.pipeline_degree: int = int(os.environ['WEIGHT_LOAD_DEGREE'])
        self.pipeline_interval = (self.layer_num + self.pipeline_degree - 1) // self.pipeline_degree # get ceiling value to ensure enough pipeline interval
        self.pipeline_degree = (self.layer_num + self.pipeline_interval - 1) // self.pipeline_interval # get the interval number
        self.weight_cache_block_num = self.layer_num - (self.pipeline_degree - 1) # assuming all layers involved in weight offloading will take the same weight cache block
        self.weight_cache_block_num += 1 # use another weight cache block so that we do not need to wait the current layer to finish


        # support dynamically increasing the layer weights kept on the comp card
        # these first two variables are only used when we change the number of on-card layers in the current forward round
        self.new_pipeline_degree = None
        self.new_pipeline_interval = None
        self.extra_weight_cache: Dict[int, torch.Tensor] = dict()
        self.extra_weight_cache_blk_i = self.weight_cache_block_num



        # example: if totally 5 layers, the cache blk num is 3, then l0 -> blk0, l1 -> blk1, l2 -> blk2, l3->blk0, l4->blk1
        # self.weight_cache_block_idx: List[int] = list(range(self.weight_cache_block_num)) \
        #     + list(range(self.layer_num - self.weight_cache_block_num))
        # example: 5 layers, 4 param cache blk, l0->blk0, l1->blk1, l2-blk2, l3->blk0, l4->blk3
        # self.weight_cache_block_idx: List[int] = list(range((self.layer_num + 1)//2)) + [0] + list(range((self.layer_num + 1)//2, self.layer_num-1))
        # no weight loading during inference
        # self.weight_cache_block_idx: List[int] = list(range(self.layer_num))
        # compute the weight_cache_block_idx based on weight_cache_block_num automatically
        self.weight_cache_block_idx: List[int] = [0] * self.layer_num
        # example: layer_num=5 pipeline_degree=2 pipeline_interval=3 then weight_cache_block_idx=[0,1,2]+[0,3]
        for interval_i in range(self.pipeline_degree):
            self.weight_cache_block_idx[ interval_i*self.pipeline_interval:(interval_i+1)*self.pipeline_interval ] =\
                range(interval_i*self.pipeline_interval - interval_i, (interval_i+1)*self.pipeline_interval - interval_i)
            # self.weight_cache_block_idx[ interval_i*self.pipeline_inteval] = 0
            self.weight_cache_block_idx[ interval_i*self.pipeline_interval] = (interval_i % 2) * (self.weight_cache_block_num - 1)

        print(f"self.weight_cache_block_idx: {self.weight_cache_block_idx}")


        self.buffer_params = dict() # stores the params in the weight cache blocks which we use to load weights during inference


        # stores the parameters we will store in self.weight_cache in order.
        self.layer_params = [list() for _ in self.layers]

        self.use_vllm = (os.environ['USE_VLLM'] == 'True')


        # when there is tensor parallelism, to support using cache gpus, 
        # we need ensure the weight loading happen after certain computation finishes
        self.synch_comp_events = [[torch.cuda.Event() for _ in self.layers] for _ in self.cache_device_ids]
        for device_events in self.synch_comp_events:
            device_events[((-1)%self.pipeline_degree)*self.pipeline_interval].record(stream=torch.cuda.current_stream())

        # ======================================================================



    # <>
    def set_cache_device_ids(self) -> None:
        '''
        Set the cache_device_ids automatically.
        Policy: given all the visible GPUs, the first worker_num GPUs is for computation (each for a worker), 
                while the remaining GPUs used by all the workers as cache.
        '''
        worker_num = torch.distributed.get_world_size()
        cache_gpu_num = torch.cuda.device_count() - worker_num
        # we need do cuda order remapping because ray would mess it up
        # self.cache_device_ids = list(range(worker_num+torch.cuda.current_device()*cache_gpu_num, worker_num+(torch.cuda.current_device()+1)*cache_gpu_num))

        if os.getenv("TOT_ORDERED_GPUS", 'None') != 'None':
            # to support reschedule, we need TOT_ORDERED_GPUS
            cand_cache_device_names = os.environ['TOT_ORDERED_GPUS'].split(',')[worker_num:]
            current_device_names_ordered = {gpu_i:i for i, gpu_i in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(','))}
            cand_cache_device_ids = [current_device_names_ordered[cuda_name] for cuda_name in cand_cache_device_names]
            self.cache_device_ids = cand_cache_device_ids
            
            print(f"self.cache_device_ids: {self.cache_device_ids} layer_num: {len(self.layers)}")
            
            return

        if worker_num == 1:
            # although the logic is the same as when worker_num > 1, but as in this branch, we do not have os.environ['TOT_ORDERED_GPUS'], 
            # we deal with it seperately
            self.cache_device_ids = list(range(worker_num+torch.cuda.current_device()*cache_gpu_num, worker_num+(torch.cuda.current_device()+1)*cache_gpu_num))
        else:
            # in the latest vllm code, we already make worker's visible gpus in the same order as in the main process
            self.cache_device_ids = list(range(worker_num, worker_num+cache_gpu_num))
            return
            # 
            cand_cache_device_names = os.environ['TOT_ORDERED_GPUS'].split(',')[worker_num:]
            cand_cache_device_ids = list()
            current_device_names_ordered = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            for cuda_name in cand_cache_device_names:
                for i, to_check in enumerate(current_device_names_ordered):
                    if cuda_name == to_check:
                        cand_cache_device_ids.append(i)
                        break
            self.cache_device_ids = cand_cache_device_ids



    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        
        # # <> For DEBUG
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        #     prefix = ''
        #     # print(f"In _paged_attention: use_v1: {use_v1}, key_cache.size: {key_cache.size()}, layer_i: {layer_i}")
        #     with open(f'blk_table_info_{prefix}2_model.log', 'a') as f:
        #         if type(input_metadata.block_tables) != type(None):
        #             f.write(f"{input_metadata.block_tables.tolist()}\n")


        if self.use_vllm:
            return self.forward_ori(input_ids, positions, kv_caches, input_metadata)
        else:
            return self.forward_ours(input_ids, positions, kv_caches, input_metadata)



    # the original code from vllm with all parameters on GPU
    def forward_ori(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 1) and (type(kv_caches[0][0])!=type(None)):
        #     print(f"input_ids: {input_ids.tolist()}")
        #     print(f"hidden_states 0 shape: {hidden_states.shape}")
        #     print(f"hidden_states 0: {hidden_states[0][-1].tolist()}")

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_caches[0][0])!=type(None)):
        #     print(f"hidden_states 5: {hidden_states[0][-1].tolist()}")
        #     print(f"last norm.weight: {self.norm.weight.tolist()}")
        #     print(f"last norm.weight address: {self.norm.weight.data_ptr()}")


        return hidden_states




    # <> support dynamically increase the on-card layer weights when there are too few requests and enough space
    def pre_increase_oncard_layers(self) -> None:
        '''
        Change the related parameters when we want to increase on-card layers.
        Changed:
            self.new_pipeline_degree, self.new_pipeline_interval, self.weight_cache_block_idx
        '''
        from vllm.core.block_manager import KVBlkPerLayerWeight
        # KVBlkPerLayerWeight.cached_layer_num will remain 0 after we load all layers on card, so the code is wrong
        # more_layer_num = self.layer_num - self.weight_cache_block_num - KVBlkPerLayerWeight.cached_layer_num
        more_layer_num = KVBlkPerLayerWeight.load_more_layer_on_card_num
        # the driver worker should not update cached_layer_num again
        # KVBlkPerLayerWeight.cached_layer_num = KVBlkPerLayerWeight.cached_layer_num - more_layer_num

        if more_layer_num == 0:
            # we do not need to keep more layer weights on the comp card
            return
        
        # we first assume all the layers will be kept on card
        
        self.new_pipeline_degree = self.pipeline_degree - more_layer_num
        self.new_pipeline_interval = (self.layer_num + self.new_pipeline_degree - 1) // self.new_pipeline_degree # get ceiling value to ensure enough pipeline interval
        # self.pipeline_degree = (self.layer_num + self.pipeline_inteval - 1) // self.pipeline_inteval # get the interval number

        # <> For DEBUG
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(f"In pre_increase_oncard_layers: self.layer_num:{self.layer_num}, self.weight_cache_block_num:{self.weight_cache_block_num}, KVBlkPerLayerWeight.cached_layer_num Invalid:{KVBlkPerLayerWeight.cached_layer_num}, more_layer_num:{more_layer_num}, self.pipeline_degree:{self.pipeline_degree}, self.new_pipeline_degree:{self.new_pipeline_degree}, self.new_pipeline_interval:{self.new_pipeline_interval}", flush=True)

        # TODO (): we currently assume layer_num % pipeline_degree == 0
        assert self.new_pipeline_degree == (self.layer_num + self.new_pipeline_interval - 1) // self.new_pipeline_interval # get the interval number
        

        # which layers will be on spare KV cache memory
        # TODO (): we assume new pipeline_interval % ori_pipeline_interval == 0
        # extra_i = self.weight_cache_block_num
        if self.new_pipeline_degree == 2:
            # in this case, there will be no layer cached on other gpus, so we directly keep previous cached layer 0,1 in the same place
            # and move other cached layers in KV cache memory.
            for interval_i in range(self.pipeline_degree):
                layer_i = interval_i * self.pipeline_interval
                if interval_i >= 2:
                    # this layer will be stored on spare KV cache memory
                    self.weight_cache_block_idx[layer_i] = self.extra_weight_cache_blk_i
                    self.extra_weight_cache_blk_i += 1
                # else:
                    # we do not want to change the weight cache block idx of layer 0
                    # we also do not need to change the weight cache block idx of layer 1*pipeline_interval
                    # self.weight_cache_block_idx[layer_i] = ((self.weight_cache_block_idx[0]//(self.weight_cache_block_num - 1)+interval_i)%2)\
                    #      * (self.weight_cache_block_num - 1)
        else:
            for interval_i in range(self.pipeline_degree):
                layer_i = interval_i * self.pipeline_interval
                if layer_i % self.new_pipeline_interval != 0:
                    # this layer will be stored on spare KV cache memory
                    self.weight_cache_block_idx[layer_i] = self.extra_weight_cache_blk_i
                    self.extra_weight_cache_blk_i += 1
                else:
                    # we do not want to change the weight cache block idx of layer 0
                    # we want to make the second newly cached layer in the same weight cache block as layer 0
                    if layer_i // self.new_pipeline_interval < 2:
                        self.weight_cache_block_idx[layer_i] = self.weight_cache_block_idx[0]
                    else:  
                        self.weight_cache_block_idx[layer_i] = \
                            ((self.weight_cache_block_idx[0]//(self.weight_cache_block_num - 1) + \
                            layer_i//self.new_pipeline_interval-1)%2) * (self.weight_cache_block_num - 1)

        # <> For DEBUG
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(f"self.weight_cache_block_num: {self.weight_cache_block_num}, self.extra_weight_cache_blk_i:{self.extra_weight_cache_blk_i}, In pre_increase_oncard_layers: {self.weight_cache_block_idx}")
        print(max(self.weight_cache_block_idx), max(self.extra_weight_cache))
        # if max(self.weight_cache_block_idx) > max(self.extra_weight_cache):
        #     assert False
        # update KVBlkPerLayerWeight.load_more_layer_on_card_num --> we will reset KVBlkPerLayerWeight.load_more_layer_on_card_num in llm_engine.step()
        # KVBlkPerLayerWeight.load_more_layer_on_card_num = 0

        

    # <> support dynamically increase the on-card layer weights when there are too few requests and enough space
    def post_increase_oncard_layers(self) -> None:
        '''
        Change the related parameters when we want to increase on-card layers.
        Changed:
            self.pipeline_degree, self.pipeline_interval, self.new_pipeline_degree, self.new_pipeline_interval
        '''
        if self.new_pipeline_degree == None:
            # this round does not increase the on-card layer weights, so nothing to update
            return
        self.pipeline_degree = self.new_pipeline_degree
        self.pipeline_interval = self.new_pipeline_interval
        self.new_pipeline_degree = None
        self.new_pipeline_interval = None



    # <> In this function, we will try offloading 
    def forward_ours(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        
        # <> For profiling
        # torch.cuda.synchronize()
        # time1 = time.perf_counter()

        hidden_states = self.embed_tokens(input_ids)
        residual = None


        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'embed_tokens: {input_ids.shape} -> {hidden_states.shape}')

        # <> For profiling
        # torch.cuda.synchronize()
        # time2 = time.perf_counter()
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        #     print(f"embed_tokens latency: {time2-time1}s abs: {time2}s")         


        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 1) and (type(kv_caches[0][0])!=type(None)):
        #     print(f"input_ids: {input_ids.tolist()}")
        #     print(f"hidden_states 0 shape: {hidden_states.shape}")
        #     print(f"hidden_states 0: {hidden_states[0][-1].tolist()}")


        # support dynamically increase on-card layer weight amount
        self.pre_increase_oncard_layers()

        for i in range(len(self.layers)):

            # loading weight -----------------------------------------------------------------------------------------
            # param_event = self.load_param_events[i] if self.layer_waiting_for_params[i] else None
            # if param_event is not None:
            #     param_event.wait()
            #     # update the status of this layer params
            #     self.layer_waiting_for_params[i] = False
            # 

            # # we can directly load the next layer to be loaded directly without waiting for this layer to finish its computation
            # if (i%self.pipeline_inteval) == 0:
            #     # self.load_layer_params(((self.layer_num + 1)//2)-i, self.weight_cache_block_idx[i], i)
            #     to_load_layer_i = ((i//self.pipeline_inteval+1)%self.pipeline_degree)*self.pipeline_inteval
            #     self.load_layer_params(to_load_layer_i, i)
            #     self.layer_waiting_for_params[ to_load_layer_i ] = True


            # <> For DEBUG
            # if int(os.getenv("LOCAL_RANK", "0")) == 0:
            #     print(f"layer {i}, self.layer_waiting_for_params[i]: {self.layer_waiting_for_params[i]}")


            # waiting for the param loading if there is
            if self.layer_waiting_for_params[i]:
                for device_events in self.load_param_events:
                    device_events[i].wait()
                # update the status of this layer params
                self.layer_waiting_for_params[i] = False                

            # if int(os.getenv("LOCAL_RANK", "0")) == 0:
            #     print(f"layer_i: {i}, {self.weight_cache_block_idx[i]}, {self.weight_cache[self.weight_cache_block_idx[i]]}")
            #     print(f"- layer_i: - {i}, {self.weight_cache_block_idx[i]}, {self.weight_cache_cpu[i]}")
            # loading weight END -----------------------------------------------------------------------------------------



            # print(f'in computation for layer {i}:')
            # for param in self.layer_params[i]:
            #     print(param)


            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )


            # loading weight -----------------------------------------------------------------------------------------
            # # <> after finishing this layer, start loading the params of a waiting layer
            # self.load_layer_params((i + self.weight_cache_block_num)%self.layer_num, self.weight_cache_block_idx[i], i)
            # # currently just a fixed mode, may consider dynamic weight_cache space later
            # self.layer_waiting_for_params[ (i + self.weight_cache_block_num)%self.layer_num ] = True

            # currently only allow one layer on CPU
            # if i in [0, ((self.layer_num + 1)//2)]:

            # if (i%self.pipeline_inteval) == 0:
            #     # self.load_layer_params(((self.layer_num + 1)//2)-i, self.weight_cache_block_idx[i], i)
            #     to_load_layer_i = ((i//self.pipeline_inteval+1)%self.pipeline_degree)*self.pipeline_inteval
            #     self.load_layer_params(to_load_layer_i, self.weight_cache_block_idx[i], i)
            #     self.layer_waiting_for_params[ to_load_layer_i ] = True
            # loading weight END -----------------------------------------------------------------------------------------


            # record finish computation----------------------
            if (i%self.pipeline_interval) == 0:
                for device_events in self.synch_comp_events:
                    device_events[i].record(stream=torch.cuda.current_stream())
            # -----------------------------------------------



            # we can directly load the next layer to be loaded directly without waiting after this layer finish its computation
            if (i%self.pipeline_interval) == 0:
                # self.load_layer_params(((self.layer_num + 1)//2)-i, self.weight_cache_block_idx[i], i)
                # supporse layer 0,2,4,6 particite the weight caching, after finish layer 0, we can load layer 4 (as layer 2 is already in memory)
                to_load_layer_i = ((i//self.pipeline_interval+2)%self.pipeline_degree)*self.pipeline_interval

                if not((self.new_pipeline_degree == 2) and (to_load_layer_i//self.pipeline_interval < 2)):
                    # when trying load all cached layers on card and the layer to load is the first two cached layers, do not need to do the load
                    if to_load_layer_i != i:
                        # the weights of a new layer need to be loaded
                        self.load_layer_params(to_load_layer_i, i)
                        # self.layer_waiting_for_params[ to_load_layer_i ] = True



        hidden_states, _ = self.norm(hidden_states, residual)

        # if (int(os.getenv("LOCAL_RANK", "0")) == 0):
        #     print(f'norm:  -> {hidden_states.shape}')

        # support dynamically increase on-card layer weight amount
        self.post_increase_oncard_layers()

        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_caches[0][0])!=type(None)):
        #     print(f"hidden_states 5: {hidden_states[0][-1].tolist()}")
        #     print(f"last norm.weight: {self.norm.weight.tolist()}")
        #     print(f"last norm.weight address: {self.norm.weight.data_ptr()}")


        return hidden_states



    def load_layer_params_no_dynamicIncreaseOnCardWeight(self, layer_i:int, last_last_layer_i:int) -> None:

        # consider we have multiple GPUs as cache============================================================
        
        # print(f"loading layer weights: {self.weight_cache_cpu[layer_i][0].element_size(), self.weight_cache_cpu[layer_i][0].numel()}, {len(self.weight_cache_cpu), len(self.weight_cache_cpu[layer_i]), self.weight_cache_cpu[layer_i][0].shape, self.weight_cache_cpu[layer_i][0].dtype}")

        dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]
        last_last_layer_weight_blk_idx = self.weight_cache_block_idx[last_last_layer_i]
        # NOTE we have changed to: last layer and layer i should point to the same weight cache blk id, i.e., layer_i will replace last_layer_i, 
        # in fact, last_layer_i is last_last_layer_i
        if dst_weight_blk_idx != last_last_layer_weight_blk_idx:
            # we need to load layer weight in another buffer
            self.weight_cache_block_idx[layer_i] = self.weight_cache_block_num - 1 - dst_weight_blk_idx
            dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]

            for param, to_replace in zip(self.layer_params[layer_i], self.buffer_params[dst_weight_blk_idx]):
                param.data = to_replace.data


        # loading
        for part_i, cache_device_i in enumerate(self.cache_device_ids):
            param_event = self.load_param_events[part_i][layer_i]
            to_synch_comp_event = self.synch_comp_events[part_i][last_last_layer_i]

            # use nvlink to copy from another device
            with torch.cuda.stream(self.param_streams[part_i]):

                # the loading stream should wait the comp to finish
                to_synch_comp_event.wait()

                # in distributed inference, the current device may not be cuda:0
                cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.weight_cache[dst_weight_blk_idx][part_i],
                    layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())
                param_event.record(stream=self.param_streams[part_i])


        self.layer_waiting_for_params[ layer_i ] = True

        # <> For DEBUG
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        #     print(f"load layer {layer_i}, weight_blk_i {dst_weight_blk_idx}, waiting layer {last_last_layer_i}")


        return






    def load_layer_params_when_dynamicIncreaseOnCardWeight(self, layer_i:int, last_last_layer_i:int) -> None:
        '''
            We need to support the dynamic increase of on-card layer weights: we will change their para data pointer here.
        '''

        # consider we have multiple GPUs as cache============================================================
        
        # print(f"loading layer weights: {self.weight_cache_cpu[layer_i][0].element_size(), self.weight_cache_cpu[layer_i][0].numel()}, {len(self.weight_cache_cpu), len(self.weight_cache_cpu[layer_i]), self.weight_cache_cpu[layer_i][0].shape, self.weight_cache_cpu[layer_i][0].dtype}")

        dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]

        # we are trying to increase the on-card layer weights in this forward round
        # we need to first update para.data address
        for param, to_replace in zip(self.layer_params[layer_i], self.buffer_params[dst_weight_blk_idx]):
            param.data = to_replace.data
        
        # compute the real last_last_layer_i
        if layer_i//self.new_pipeline_interval == 1:
            # last_last_layer_i = self.pipeline_interval
            # layer 0 must have been computed, but layer self.pipeline_interval may not
            last_last_layer_i = 0
        elif layer_i//self.new_pipeline_interval == 2:
            # in this case, layer self.pipeline_interval must have been computed
            last_last_layer_i = self.pipeline_interval
        else:
            last_last_layer_i = ((layer_i//self.new_pipeline_interval-2)%self.new_pipeline_degree)*self.new_pipeline_interval

        need_to_wait = False
        if (layer_i%self.new_pipeline_interval == 0) and (self.new_pipeline_degree!=2): # when new pipe degree == 2, do not need to wait for computation to finish
            need_to_wait = True

        # loading
        for part_i, cache_device_i in enumerate(self.cache_device_ids):
            param_event = self.load_param_events[part_i][layer_i]
            to_synch_comp_event = self.synch_comp_events[part_i][last_last_layer_i]

            # use nvlink to copy from another device
            with torch.cuda.stream(self.param_streams[part_i]):

                # the loading stream should wait the comp to finish
                if need_to_wait:
                    to_synch_comp_event.wait()

                # in distributed inference, the current device may not be cuda:0
                cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.extra_weight_cache[dst_weight_blk_idx][part_i],
                    layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())
                param_event.record(stream=self.param_streams[part_i])


        self.layer_waiting_for_params[ layer_i ] = True


        # <> For DEBUG
        # if int(os.getenv("LOCAL_RANK", "0")) == 0:
        #     print(f"load layer {layer_i}, weight_blk_i {dst_weight_blk_idx}, waiting layer {last_last_layer_i}, need_to_wait: {need_to_wait}")


        return




    def load_layer_params(self, layer_i:int, last_last_layer_i:int) -> None:
        '''
            We need to support the dynamic increase of on-card layer weights: we will change their para data pointer here.
        '''
        if self.new_pipeline_degree == None:
            self.load_layer_params_no_dynamicIncreaseOnCardWeight(layer_i, last_last_layer_i)
        else:
            if layer_i == 0:
                last_last_layer_i = ((-2)%self.new_pipeline_degree)*self.new_pipeline_interval
                # we may need to change the parameter pointer for layer 0
                self.load_layer_params_no_dynamicIncreaseOnCardWeight(layer_i, last_last_layer_i)
            else:
                self.load_layer_params_when_dynamicIncreaseOnCardWeight(layer_i, last_last_layer_i)
                if layer_i//self.pipeline_interval == 1:
                    # we need to load the second newly cached layer as well
                    layer_i = self.new_pipeline_interval
                    last_last_layer_i = ((-1)%self.new_pipeline_degree)*self.new_pipeline_interval
                    self.load_layer_params_no_dynamicIncreaseOnCardWeight(layer_i, last_last_layer_i)




class LlamaForCausalLM(nn.Module):
    supports_lora = True

    def __init__(
        self,
        config: LlamaConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        
        print(f"LlamaForCausalLM LlamaConfig: {config}")

        self.config = config
        self.linear_method = linear_method
        self.model = LlamaModel(config, linear_method, lora_config=lora_config)
        unpadded_vocab_size = config.vocab_size
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.sampler = Sampler(unpadded_vocab_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:


        # <> For profiling
        torch.cuda.synchronize()
        time1 = time.perf_counter()

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        

        # <> For profiling
        torch.cuda.synchronize()
        time2 = time.perf_counter()
        # in distributed inference, only the first worker will print information
        # if torch.cuda.current_device() == 0:
        if int(os.getenv("LOCAL_RANK", "0")) == 0:
            print(f"iter latency: {time2-time1}s abs: {time1, time2}s")



        # <> For DEBUG
        # if (int(os.getenv("LOCAL_RANK", "0")) == 0) and (type(kv_caches[0][0])!=type(None)):
        #     print(f"hidden_states end: {hidden_states[-4:][-1].tolist()}")



        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens




    # <> this function prints the parameter shape of the model
    def print_param_shapes(self):
        params_dict = dict(self.named_parameters())
        for name, param in params_dict.items():
            print(f"name: {name}, param: {param.shape}")









    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        if os.environ['USE_VLLM'] == 'True':
            self.load_weights_ori(model_name_or_path, cache_dir, load_format, revision)
        else:
            # self.load_weights_ours(model_name_or_path, cache_dir, load_format, revision)
            self.preprocess_for_offloading_weights_fast(model_name_or_path, cache_dir, load_format, revision)



        # <> print the parameter shapes of the model
        self.print_param_shapes()




    def load_weights_ori(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)




    def load_weights_ours(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):

        # <> For offloading weights layer by layer
        # first do some preprocess
        self.preprocess_for_offloading_weights(model_name_or_path, cache_dir, load_format, revision)


        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            # <> For Profiling
            # print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # <> For Profiling
                # print(f"specific weight loader: {name.replace(weight_name, param_name), loaded_weight.shape, loaded_weight.device}")
                
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) < self.model.weight_cache_block_num):
                # currently only allow one layer on CPU
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) != ((self.model.layer_num + 1)//2)):
                # determine the condition automatically based on pipeline_interval
                if ('model.layers.' not in name) or \
                    (self.get_layer_id_from_name(name) in [0, self.model.pipeline_interval]) or \
                    ((self.get_layer_id_from_name(name)%self.model.pipeline_interval)!=0):
                    # if True:
                    # print('loading-----')
                    # param = params_dict[name.replace(weight_name, param_name)]

                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)

                    # print(f"layerBylayer_param_tensor of {name}: {param.data}")
                    # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                    # <> For DEBUG
                    if int(os.getenv("LOCAL_RANK", "0")) == 0:
                        print(f"param.data.data_ptr() of {name}: {param.data.data_ptr()}, shape: {param.data.shape} device: {param.data.device} is contiguous: {param.is_contiguous()}")



                break
            else:

                # <> For Profiling
                # print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) < self.model.weight_cache_block_num):
                # if ('model.layers.' not in name) or (self.get_layer_id_from_name(name) != ((self.model.layer_num + 1)//2)):
                if ('model.layers.' not in name) or \
                    (self.get_layer_id_from_name(name) in [0, self.model.pipeline_interval]) or \
                    ((self.get_layer_id_from_name(name)%self.model.pipeline_interval)!=0):
                    # if True:
                    # print('loading-----')
                    # param = params_dict[name]
                    
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

                    # print(f"layerBylayer_param_tensor of {name}: {param.data}")
                    # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                    # <> For DEBUG
                    if int(os.getenv("LOCAL_RANK", "0")) == 0:
                        print(f"param.data.data_ptr() of {name}: {param.data.data_ptr()}, shape: {param.data.shape} device: {param.data.device} is contiguous: {param.is_contiguous()} visible gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")



        # print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")






    # <> For offloading weights layer by layer
    # Set up: (1) weight_num_per_layer, (2) weight_range_in_blk, (3) weight cache blocks
    # (4) making the param tensors pointing to the new GPU physical addresses

    # NOTE: we cannot assume the parameter tensors for the same layer will appear together;
    #       we cannot assume the parameter tensors for different layers appear in the same order

    # NOTE: we need to ensure each parameter tensor is aligned in the our weigh cache, to avoid computation efficiency degradation.
                        
    # NOTE: support the case where there is no cache devices or no weight to be cache on other gpus

    def preprocess_for_offloading_weights(self,
             model_name_or_path: str,
             cache_dir: Optional[str] = None,
             load_format: str = "auto",
             revision: Optional[str] = None):


        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())


        visited_param_names = [list() for _ in self.model.layers]

        print(f"init model size: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        # we only need to consider layer params.
        # we do weight loading.
        # we collect layer parameters in order (no repetition).
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            # <> For Profiling
            # print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

            if 'model.layers.' not in name:
                # we only need to consider layer params
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # <> For Profiling
                # print(f"specific weight loader: {name.replace(weight_name, param_name), loaded_weight.shape, loaded_weight.device}")

                full_para_name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if full_para_name.endswith(".bias") and full_para_name not in params_dict:
                    continue                
                param = params_dict[full_para_name]

                # to avoid OOM
                # param.data = param.data.cpu()

                # need load weight so that we can get packed weight_cache_cpu
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                # print(f"ori_param_tensor of {name}: {param.data}")

                # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                # store this parameter
                layer_i = self.get_layer_id_from_name(name)
                if full_para_name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(full_para_name)
                break
            else:

                # <> For Profiling
                # print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                if name.endswith(".bias") and name not in params_dict:
                    continue  
                param = params_dict[name]

                # to avoid OOM
                # param.data = param.data.cpu()

                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

                # print(f"ori_param_tensor of {name}: {param.data}")

                # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                # store this parameter
                layer_i = self.get_layer_id_from_name(name)
                if name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(name)
                


        # we need to ensure the params and param_names for different layers are in the same order
        # name: 'model.layers.30.input_layernorm.weight'
        for layer_i in range(self.model.layer_num):
            param_names = [name.split('.')[3:] for name in visited_param_names[layer_i]]
            order = sorted(range(len(param_names)), key=lambda i: param_names[i])
            params = self.model.layer_params[layer_i]
            self.model.layer_params[layer_i] = [params[i] for i in order]
            visited_param_names[layer_i] = [visited_param_names[layer_i][i] for i in order]



        weight_num_per_layer = 0
        weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list()
        params_dtype = None

        alignment_unit = 128 // self.model.layer_params[0][0].element_size()

        for param in self.model.layer_params[0]:
            # for these information we only need to consider layer 0, as other layers are the same as it
            # we also avoid repeat param as some weights may be packed into one param
            weight_range_in_blk.append((weight_num_per_layer, param.nelement(), param.shape))
            # weight_num_per_layer += param.nelement()

            # NOTE that we need to make the parameter tensor aligned
            print(f"tensor size: {param.nelement()} -> {(((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)} alignment_unit: {alignment_unit}")
            weight_num_per_layer = weight_num_per_layer + (((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)

            assert (params_dtype == None) or (params_dtype == param.dtype)
            params_dtype = param.dtype


        print(f"visited_param_names[30]: {visited_param_names[30]}")
        print(f"weight_range_in_blk: {weight_range_in_blk}")

        # weight_num_per_layer should be the multiple of 1, 2, 3, i.e., multiple of 6
        # NOTE that weight_num_per_layer should also be multiple of alignment_unit
        need_be_multiple_of = len(self.model.cache_device_ids) * alignment_unit # 3 is from the number of cache gpus
        # TODO <> we need to consider the case where there is no cache device [in this case, there will also be no weights on cached gpus]
        if need_be_multiple_of == 0:
            need_be_multiple_of = 1

        print(f"weight_num_per_layer: {weight_num_per_layer} -> multiple of {need_be_multiple_of} ==> {((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of}")

        # to_pad_each_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of - weight_num_per_layer
        weight_num_per_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of

        print(f"weight_num_per_layer: {weight_num_per_layer}")
        print(f"weight_cache_cpu size: {weight_num_per_layer*self.model.layer_num}")

        # obtain weights_cache_cpu and release the param tensors on GPU
        # make sure weight_cache_cpu is a pinned memory
        weight_cache_cpu = torch.empty(weight_num_per_layer*self.model.layer_num, dtype=self.model.layer_params[0][0].data.cpu().dtype, pin_memory=True)
        print(f"weight_cache_cpu size: {weight_cache_cpu.size()}")
        

        # torch.cat(\
        #     [param.data.cpu().view(-1) for params in self.model.layer_params for param in params] \
        #     + [torch.tensor(range(to_pad_each_layer*self.model.layer_num), dtype=weight_cache_cpu.dtype)], \
        #     out=weight_cache_cpu)
        # 
        # As we make each param tensor mem aligned, they may be empty places between tensors.
        weight_cache_cpu = weight_cache_cpu.view(self.model.layer_num, -1)
        for layer_i, params in enumerate(self.model.layer_params):
            for param, (offset, size, tensor_shape) in zip(params, weight_range_in_blk):
                print(f"offset: {offset}, size: {size}, tensor_shape: {tensor_shape}, param shape: {param.data.shape}, param size: {param.data.size()}")
                weight_cache_cpu[layer_i][offset:offset+size] = param.data.cpu().view(-1)

        print(f"weight_cache_cpu size: {weight_cache_cpu.size()}")

        # TODO () deal with the case where cache_device_ids == 0
        if len(self.model.cache_device_ids) > 0:
            weight_cache_cpu = weight_cache_cpu.view(self.model.layer_num, len(self.model.cache_device_ids), -1) #weight_num_per_layer)

        # we need to use gpus as a weight cache
        for layer_i, layer_weights in enumerate(weight_cache_cpu):
            self.model.weight_cache_cpu.append(list())
            
            # if int(os.getenv("LOCAL_RANK", "0")) == 0:
            #     print(f"--layer_i-- {layer_i}, -- {layer_weights}")
            
            # NOTE: we do not need to store the parameter of every layer on the cache GPUs, because some layers are kept only on the compute GPUs.
            # NOTE: () if there is no weight to be cached on other gpus, we also do not need to store them
            if ((layer_i % self.model.pipeline_interval) != 0) or (self.model.layer_num == self.model.weight_cache_block_num):
                # this layer will not be cached on other GPUs.
                print(f"layer_i: {layer_i} is not cached, pipeline_interval: {self.model.pipeline_interval}")
                continue

            print(f"\n\n\n\n\n\n store weights on cache gpus\n\n\n\n\n\n\n {self.model.layer_num, self.model.weight_cache_block_num}")
            for part_weights, cache_device_i in zip(layer_weights, self.model.cache_device_ids):
                self.model.weight_cache_cpu[-1].append( part_weights.to(cache_device_i) )

        # self.model.weight_cache_cpu = self.model.weight_cache_cpu.to(1)
        # TODO () deal with the case where cache_device_ids == 0
        if len(self.model.weight_cache_cpu[0]) > 0:
            print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")
        else:
            print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0])}")

        for params in self.model.layer_params:
            for param in params:
                param.data = torch.Tensor([])

        torch.cuda.empty_cache()
        print(f"release param tensors: {torch.cuda.memory_allocated()/1024/1024/1024} GB")

        self.model.weight_num_per_layer = weight_num_per_layer
        self.model.weight_range_in_blk = weight_range_in_blk
        self.model.weight_cache = torch.empty(self.model.weight_cache_block_num * self.model.weight_num_per_layer,
                                       device=torch.cuda.current_device(),
                                       dtype=params_dtype)
        
        # store the weight_num_per_layer infor in the class KVBlkPerLayerWeight
        from vllm.core.block_manager import KVBlkPerLayerWeight
        KVBlkPerLayerWeight.layer_weight_size = self.model.weight_num_per_layer * self.model.weight_cache.element_size()
        KVBlkPerLayerWeight.cached_layer_num = self.model.layer_num - self.model.weight_cache_block_num
        KVBlkPerLayerWeight.layer_num = self.model.layer_num

        
        print(f"after allocate param cache: {torch.cuda.memory_allocated()/1024/1024/1024} GB")

        # make the layer param tensors point to the new address in self.weight_cache
        for layer_i in range(len(self.model.layer_params)):
            params = self.model.layer_params[layer_i]
            for param, rng_info in zip(params, self.model.weight_range_in_blk):
                param.data = torch.narrow(self.model.weight_cache, 
                    0, 
                    self.model.weight_cache_block_idx[layer_i]*self.model.weight_num_per_layer + rng_info[0], 
                    rng_info[1]).view(rng_info[2])


        # prepare buffer_params
        # TODO (): if there is no weight to be cached on other gpus, we do not need to prepare buffer_params
        for weight_cache_block_idx in [0, self.model.weight_cache_block_num - 1]:
            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.weight_cache, 
                    0, 
                    weight_cache_block_idx*self.model.weight_num_per_layer + rng_info[0], 
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)



        # self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, self.model.weight_num_per_layer)
        # TODO () deal with the case where cache_device_ids == 0
        if len(self.model.cache_device_ids) > 0:
            self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, len(self.model.cache_device_ids), -1)

        torch.cuda.empty_cache()


        for cache_device_i in self.model.cache_device_ids:
            # cache_ops.init_P2P_access(cache_device_i, 0, 0)
            # in distributed inference, the current device may not be cuda:0
            cache_ops.init_P2P_access(cache_device_i, torch.cuda.current_device(), torch.cuda.current_device()) 

        # print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")












    # <> For offloading weights layer by layer
    # Set up: (1) weight_num_per_layer, (2) weight_range_in_blk, (3) weight cache blocks
    # (4) making the param tensors pointing to the new GPU physical addresses

    # NOTE: we cannot assume the parameter tensors for the same layer will appear together;
    #       we cannot assume the parameter tensors for different layers appear in the same order

    # NOTE: we need to ensure each parameter tensor is aligned in the our weigh cache, to avoid computation efficiency degradation.
            
    # NOTE: support the case where there is no cache devices or no weight to be cache on other gpus

    # NOTE: this is the fast version: we first get the parameter information, then do data transfer for only one time
            


    def preprocess_for_offloading_weights_fast(self,
             model_name_or_path: str,
             cache_dir: Optional[str] = None,
             load_format: str = "auto",
             revision: Optional[str] = None):


        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())


        visited_param_names = [list() for _ in self.model.layers]

        print(f"init model size: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        # we only need to consider layer params.
        # we do weight loading.
        # we collect layer parameters in order (no repetition).
        
        # store the weights in a list of dicts for later use
        loaded_weights = [dict() for _ in range(self.model.layer_num)]

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):

            # <> For Profiling
            # print(f"layer info: {name, loaded_weight.shape, loaded_weight.device}")

            # if 'model.layers.' not in name:
            #     # we only need to consider layer params
            #     continue

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # <> For Profiling
                # print(f"specific weight loader: {name.replace(weight_name, param_name), loaded_weight.shape, loaded_weight.device}")

                full_para_name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if full_para_name.endswith(".bias") and full_para_name not in params_dict:
                    continue                
                param = params_dict[full_para_name]

                # to avoid OOM
                # param.data = param.data.cpu()

                # need load weight so that we can get packed weight_cache_cpu
                # weight_loader = param.weight_loader
                # weight_loader(param, loaded_weight, shard_id)
                # if this weight is not layer weight, directly load it
                if 'model.layers.' not in name:
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break

                # print(f"ori_param_tensor of {name}: {param.data}")

                # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                # store this parameter
                layer_i = self.get_layer_id_from_name(name)
                if full_para_name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(full_para_name)

                # store the weight in the dictionary
                if full_para_name not in loaded_weights[layer_i]:
                    loaded_weights[layer_i][full_para_name] = list()
                loaded_weights[layer_i][full_para_name].append((shard_id, loaded_weight))
                # print(f"{full_para_name} is is_pinned: {loaded_weight.is_pinned()}")


                break
            else:

                # <> For Profiling
                # print(f"other weight loader: {name, loaded_weight.shape, loaded_weight.device}")
                if name.endswith(".bias") and name not in params_dict:
                    continue  
                param = params_dict[name]

                # to avoid OOM
                # param.data = param.data.cpu()

                # weight_loader = getattr(param, "weight_loader",
                #                         default_weight_loader)
                # weight_loader(param, loaded_weight)
                # if this weight is not layer weight, directly load it
                if 'model.layers.' not in name:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    # print(f"ori_param_tensor of {name}: {param.data}")

                    # print(f"{name}: {param.data.size(), param.data_ptr(), param.is_contiguous()}")

                    # store this parameter
                    layer_i = self.get_layer_id_from_name(name)
                    if name not in visited_param_names[layer_i]:
                        self.model.layer_params[layer_i].append(param)
                        visited_param_names[layer_i].append(name)

                    # store the weight in the dictionary
                    if name not in loaded_weights[layer_i]:
                        loaded_weights[layer_i][name] = list()
                    loaded_weights[layer_i][name].append((None, loaded_weight))
                    # print(f"{name} is is_pinned: {loaded_weight.is_pinned()}")


        # we need to ensure the params and param_names for different layers are in the same order
        # name: 'model.layers.30.input_layernorm.weight'
        for layer_i in range(self.model.layer_num):
            # param_names = [name.split('.')[3:] for name in visited_param_names[layer_i]]
            # order = sorted(range(len(param_names)), key=lambda i: param_names[i])
            params = self.model.layer_params[layer_i]
            # order by the param data size
            order = sorted(range(len(params)), key=lambda i: params[i].nelement())
            self.model.layer_params[layer_i] = [params[i] for i in order]
            visited_param_names[layer_i] = [visited_param_names[layer_i][i] for i in order]



        weight_num_per_layer = 0
        weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list()
        params_dtype = None

        alignment_unit = 128 // self.model.layer_params[0][0].element_size()

        for param in self.model.layer_params[0]:
            # for these information we only need to consider layer 0, as other layers are the same as it
            # we also avoid repeat param as some weights may be packed into one param
            weight_range_in_blk.append((weight_num_per_layer, param.nelement(), param.shape))
            # weight_num_per_layer += param.nelement()

            # NOTE that we need to make the parameter tensor aligned
            print(f"tensor size: {param.nelement()} -> {(((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)} alignment_unit: {alignment_unit}")
            weight_num_per_layer = weight_num_per_layer + (((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)

            assert (params_dtype == None) or (params_dtype == param.dtype)
            params_dtype = param.dtype


        # print(f"visited_param_names[30]: {visited_param_names[30]}")
        # print(f"weight_range_in_blk: {weight_range_in_blk}")

        # weight_num_per_layer should be the multiple of 1, 2, 3, i.e., multiple of 6
        # NOTE that weight_num_per_layer should also be multiple of alignment_unit
        need_be_multiple_of = len(self.model.cache_device_ids) * alignment_unit # 3 is from the number of cache gpus
        # TODO <> we need to consider the case where there is no cache device [in this case, there will also be no weights on cached gpus]
        if need_be_multiple_of == 0:
            need_be_multiple_of = 1

        print(f"weight_num_per_layer: {weight_num_per_layer} -> multiple of {need_be_multiple_of} ==> {((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of}")

        # to_pad_each_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of - weight_num_per_layer
        weight_num_per_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of

        print(f"weight_num_per_layer: {weight_num_per_layer}")
        print(f"weight_cache_cpu size: {weight_num_per_layer*self.model.layer_num}")


        # ==========================================


        for params in self.model.layer_params:
            for param in params:
                param.data = torch.Tensor([])

        torch.cuda.empty_cache()
        print(f"release param tensors: {torch.cuda.memory_allocated()/1024/1024/1024} GB")



        self.model.weight_num_per_layer = weight_num_per_layer
        self.model.weight_range_in_blk = weight_range_in_blk
        self.model.weight_cache = torch.empty(self.model.weight_cache_block_num * self.model.weight_num_per_layer,
                                       device=torch.cuda.current_device(),
                                       dtype=params_dtype)
        
        # store the weight_num_per_layer infor in the class KVBlkPerLayerWeight
        from vllm.core.block_manager import KVBlkPerLayerWeight
        KVBlkPerLayerWeight.layer_weight_size = self.model.weight_num_per_layer * self.model.weight_cache.element_size()
        KVBlkPerLayerWeight.cached_layer_num = self.model.layer_num - self.model.weight_cache_block_num
        KVBlkPerLayerWeight.layer_num = self.model.layer_num

        
        print(f"after allocate param cache: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


    
        layer_num_on_other_gpus = self.model.pipeline_degree if self.model.layer_num > self.model.weight_cache_block_num else 0
        layer_size_on_other_gpus = (self.model.weight_num_per_layer // len(self.model.cache_device_ids)) if len(self.model.cache_device_ids) > 0 else 0

        # first prepare self.model.weight_cache_cpu
        self.model.weight_cache_cpu = [list() for _ in range(self.model.layer_num)]        
        for cache_device_i in self.model.cache_device_ids:
            partial_cache = torch.empty((layer_num_on_other_gpus, layer_size_on_other_gpus),
                                device=torch.device(f'cuda:{cache_device_i}'),
                                dtype=params_dtype)
            # cache_on_other_gpus.append(partial_cache)
            for cached_layer_i in range(layer_num_on_other_gpus):
                self.model.weight_cache_cpu[cached_layer_i*self.model.pipeline_interval].append(partial_cache[cached_layer_i])

        # the list of parameters which will be split and stored on two cached gpus
        cross_boundary_param_info = list()

        # make the layer param tensors point to the new address in self.weight_cache
        for layer_i in range(len(self.model.layer_params)):
            params = self.model.layer_params[layer_i]
            on_cache_gpus = (layer_i % self.model.pipeline_interval == 0) and (self.model.layer_num > self.model.weight_cache_block_num)
            for param, rng_info in zip(params, self.model.weight_range_in_blk):
                if not on_cache_gpus:
                    param.data = torch.narrow(self.model.weight_cache, 
                        0, 
                        self.model.weight_cache_block_idx[layer_i]*self.model.weight_num_per_layer + rng_info[0], 
                        rng_info[1]).view(rng_info[2])
                    # print(f"param.data.shape 1: {param.data.shape, layer_i, rng_info, param.device}")
                else:
                    start_partial_cache_i = rng_info[0]//layer_size_on_other_gpus
                    end_partial_cache_i = (rng_info[0]+rng_info[1]-1)//layer_size_on_other_gpus
                    if start_partial_cache_i != end_partial_cache_i:
                        # this parameter will be split
                        param.data = torch.empty(rng_info[2], dtype=params_dtype, device=torch.device('cpu'), pin_memory=True)
                        param_data_rng_end = 0
                        for partial_cache_i in range(start_partial_cache_i, end_partial_cache_i + 1):
                            rng = None
                            if partial_cache_i == start_partial_cache_i:
                                rng = [(0, layer_size_on_other_gpus-(rng_info[0]%layer_size_on_other_gpus)), (rng_info[0]%layer_size_on_other_gpus, layer_size_on_other_gpus)]
                                param_data_rng_end += (layer_size_on_other_gpus-(rng_info[0]%layer_size_on_other_gpus))
                            elif partial_cache_i == end_partial_cache_i:
                                rng = [(param_data_rng_end, rng_info[1]), (0, (rng_info[0]+rng_info[1]-1)%layer_size_on_other_gpus+1)]
                                param_data_rng_end += (rng_info[1]-param_data_rng_end)
                            else:
                                rng = [(param_data_rng_end, param_data_rng_end + layer_size_on_other_gpus), (0, layer_size_on_other_gpus)]
                                param_data_rng_end += layer_size_on_other_gpus
                            cross_boundary_param_info.append(\
                                (param.data, layer_i, partial_cache_i, # the cache gpu to use
                                rng))
                        # print(f"param.data.shape 2: {param.data.shape, param.shape, start_partial_cache_i, '-', end_partial_cache_i, layer_i, rng_info, layer_size_on_other_gpus}")
                    else:
                        # this parameter will be on one cache gpu
                        param.data = torch.narrow(self.model.weight_cache_cpu[layer_i][start_partial_cache_i], 
                            0, 
                            rng_info[0]%layer_size_on_other_gpus, 
                            rng_info[1]).view(rng_info[2])
                        # print(f"param.data.shape 3: {param.data.shape, start_partial_cache_i, layer_i, rng_info, layer_size_on_other_gpus}")




        # prepare buffer_params
        # TODO (): if there is no weight to be cached on other gpus, we do not need to prepare buffer_params
        for weight_cache_block_idx in [0, self.model.weight_cache_block_num - 1]:
            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.weight_cache, 
                    0, 
                    weight_cache_block_idx*self.model.weight_num_per_layer + rng_info[0], 
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)



        # self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, self.model.weight_num_per_layer)
        # TODO () deal with the case where cache_device_ids == 0
        if len(self.model.cache_device_ids) > 0:
            self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, len(self.model.cache_device_ids), -1)

        torch.cuda.empty_cache()


        for cache_device_i in self.model.cache_device_ids:
            # cache_ops.init_P2P_access(cache_device_i, 0, 0)
            # in distributed inference, the current device may not be cuda:0
            cache_ops.init_P2P_access(cache_device_i, torch.cuda.current_device(), torch.cuda.current_device()) 

        # print(f"weight_cache_cpu shape: {len(self.model.weight_cache_cpu), len(self.model.weight_cache_cpu[0]), self.model.weight_cache_cpu[0][0].shape}")



        # cache weights on other gpus now--------------------------------        
        # load layer weight to correct place
        for layer_i in range(self.model.layer_num):
            param_names = visited_param_names[layer_i]
            params = self.model.layer_params[layer_i]
            # loaded_weights[layer_i][full_para_name].append((shard_id, loaded_weight))
            weight_loader_info = loaded_weights[layer_i]

            # load weight to proper space
            for name, param in zip(param_names, params):
                for shard_id, loaded_weight in weight_loader_info[name]:
                    # print(f"{name}, {shard_id}, {param.shape}, {loaded_weight.shape}")
                    if shard_id != None:
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)
                    # print(f"layer_i{layer_i}, {param.device}")
            

        # print(f"-3 layer {31}: param: {self.model.layer_params[31][-1].device, self.model.layer_params[31][-1].data_ptr()}")


        # move the cross boundary parameters from cpu to corresponding gpus       
        for param_data, layer_i, partial_cache_i, (param_rng, cache_rng) in cross_boundary_param_info:
            # print(f"cross_boundary_param_info: {layer_i, partial_cache_i, (param_rng, cache_rng)}")
            # deal with the first part
            narrow_param_data = torch.narrow(self.model.weight_cache_cpu[layer_i][partial_cache_i], 
                    0, cache_rng[0], cache_rng[1]-cache_rng[0])
            # print(f"-2 layer {31}: param: {self.model.layer_params[31][-1].device, self.model.layer_params[31][-1].data_ptr(), param_data.device, param_data.data_ptr()}")
            narrow_param_cpu = param_data.view(-1)
            # print(f"-2 layer {31}: param: {self.model.layer_params[31][-1].device, self.model.layer_params[31][-1].data_ptr(), param_data.device, param_data.data_ptr()}")
            narrow_param_cpu = torch.narrow(narrow_param_cpu, 0, param_rng[0], param_rng[1]-param_rng[0])
            # print(f"narrow_param_cpu len: layer: {layer_i} {narrow_param_cpu.shape}, from {narrow_param_cpu.device} to {narrow_param_data.device}")
            # print(f"-2 layer {31}: param: {self.model.layer_params[31][-1].device, self.model.layer_params[31][-1].data_ptr(), param_data.device, param_data.data_ptr()}")
            narrow_param_data.copy_(narrow_param_cpu)
            


        # print(f"-2 layer {31}: param: {self.model.layer_params[31][-1].device, self.model.layer_params[31][-1].data_ptr()}")

        # update the corresponding param data address for layers cached on other gpus
        for i in range(self.model.pipeline_degree):
            layer_i = i * self.model.pipeline_interval
            weight_cache_block_idx = self.model.weight_cache_block_idx[layer_i]
            buffer_param_datas = self.model.buffer_params[weight_cache_block_idx]
            for param, param_data in zip(self.model.layer_params[layer_i], buffer_param_datas):
                param.data = param_data


        # print(f"-1 layer {31}: param: {self.model.layer_params[31][-1].device}")
        
        # load the first two cache layer to the current device
        if layer_num_on_other_gpus > 0:
            for layer_i, weight_cache_block_idx in zip([0, self.model.pipeline_interval], [0, self.model.weight_cache_block_num - 1]):
                for part_i, cache_device_i in enumerate(self.model.cache_device_ids): 
                    cache_ops.load_layer_weights(self.model.weight_cache_cpu[layer_i][part_i], self.model.weight_cache[weight_cache_block_idx][part_i],
                        layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())

        # torch.cuda.empty_cache()

        # for layer_i in range(self.model.layer_num):
        #     for param in self.model.layer_params[layer_i]:
        #         print(f"layer {layer_i}: param: {param.device}")
        #     print(f"layer {layer_i}: cached weight {self.model.weight_cache_cpu[layer_i]}")
        

















    def get_layer_id_from_name(self, name:str) -> int:
        # 'model.layers.24.input_layernorm.weight'
        terms = name.split('.')
        return int(terms[2])





    # <> support dynamically increaing on-card layer weights
    def init_extra_weight_cache_in_KV_cache(self, kv_caches: List[KVCache]) -> None:
        '''
            prepare parameter data whose address is in the KV cache.
            Only called after we change the KV cache layout and want to dynamically increase the on-card layer weights.
            Initialize:  self.model.buffer_params, self.model.extra_weight_cache
        '''
        # prepare buffer parameters in the extra cache from the KV cache
        from vllm.core.block_manager import KVBlkPerLayerWeight
        extra_weight_cache_blk_num = KVBlkPerLayerWeight.cached_layer_num

        cache = kv_caches[0].view(-1)
        assert cache.element_size() == self.model.weight_cache.element_size(), "the KV cache dtype size and the weight cache dtype size are different"

        for weight_cache_block_idx in range(self.model.weight_cache_block_num, self.model.weight_cache_block_num+extra_weight_cache_blk_num):
            # set self.model.extra_weight_cache
            print(f"set extra_weight_cache[{weight_cache_block_idx}]")
            self.model.extra_weight_cache[weight_cache_block_idx] = torch.narrow(cache, 
                    0, 
                    len(cache) - (weight_cache_block_idx+1-self.model.weight_cache_block_num)*self.model.weight_num_per_layer,
                    self.model.weight_num_per_layer)

            # set self.model.buffer_params
            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.extra_weight_cache[weight_cache_block_idx], 
                    0, 
                    rng_info[0], 
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)
            
            # change the view
            self.model.extra_weight_cache[weight_cache_block_idx] = self.model.extra_weight_cache[weight_cache_block_idx].view(len(self.model.cache_device_ids), -1)


            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                print(f"self.model.extra_weight_cache[{weight_cache_block_idx}]: {self.model.extra_weight_cache[weight_cache_block_idx].data_ptr()}")

        # for convenience, we also store the weight cache block 0 and (self.model.weight_cache_block_num - 1) in extra_weight_cache
        # TODO (): if there is no weight to be cached on other gpus, we do not need to prepare extra_weight_cache
        self.model.extra_weight_cache[0] = self.model.weight_cache[0]
        self.model.extra_weight_cache[self.model.weight_cache_block_num - 1] = self.model.weight_cache[self.model.weight_cache_block_num - 1]


