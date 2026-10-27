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


from vllm._C import cache_ops
import time
import os


PRINT_LOGS=True


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


    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)


        x = self.act_fn(gate_up)


        x, _ = self.down_proj(x)


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


            assert self.total_num_kv_heads % tp_size == 0
        else:


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


        qkv, _ = self.qkv_proj(hidden_states)


        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)


        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)


        output, _ = self.o_proj(attn_output)


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


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)


        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )


        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)


        hidden_states = self.mlp(hidden_states)


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


        self.cache_device_ids = [1]


        self.set_cache_device_ids()


        print(f'In model, os.environ["WEIGHT_LOAD_DEGREE"]: {os.environ["WEIGHT_LOAD_DEGREE"]}, os.environ["LOCAL_RANK"]:{os.getenv("LOCAL_RANK", "0")}, all card num: {torch.cuda.device_count()}, worker_num:{torch.distributed.get_world_size()}, os.environ["CUDA_VISIBLE_DEVICES"]: {os.environ["CUDA_VISIBLE_DEVICES"]}, current_device: {torch.cuda.current_device()}, self.cache_device_ids: {self.cache_device_ids}')


        self.param_streams = [torch.cuda.Stream() for _ in self.cache_device_ids]
        for param_stream in self.param_streams:
            assert param_stream != torch.cuda.current_stream()

        self.load_param_events = [[torch.cuda.Event() for _ in self.layers] for _ in self.cache_device_ids]
        self.layer_waiting_for_params = [False for _ in self.layers]
        self.layer_num = len(self.layers)


        self.weight_cache: torch.Tensor = torch.Tensor([])

        self.weight_cache_cpu: List[List[torch.Tensor]] = []
        self.weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list()
        self.weight_num_per_layer: int = -1


        self.pipeline_degree: int = int(os.environ['WEIGHT_LOAD_DEGREE'])
        self.pipeline_interval = (self.layer_num + self.pipeline_degree - 1) // self.pipeline_degree
        self.pipeline_degree = (self.layer_num + self.pipeline_interval - 1) // self.pipeline_interval
        self.weight_cache_block_num = self.layer_num - (self.pipeline_degree - 1)
        self.weight_cache_block_num += 1


        self.new_pipeline_degree = None
        self.new_pipeline_interval = None
        self.extra_weight_cache: Dict[int, torch.Tensor] = dict()
        self.extra_weight_cache_blk_i = self.weight_cache_block_num


        self.weight_cache_block_idx: List[int] = [0] * self.layer_num

        for interval_i in range(self.pipeline_degree):
            self.weight_cache_block_idx[ interval_i*self.pipeline_interval:(interval_i+1)*self.pipeline_interval ] =\
                range(interval_i*self.pipeline_interval - interval_i, (interval_i+1)*self.pipeline_interval - interval_i)

            self.weight_cache_block_idx[ interval_i*self.pipeline_interval] = (interval_i % 2) * (self.weight_cache_block_num - 1)

        print(f"self.weight_cache_block_idx: {self.weight_cache_block_idx}")


        self.buffer_params = dict()


        self.layer_params = [list() for _ in self.layers]

        self.use_vllm = (os.environ['USE_VLLM'] == 'True')


        self.synch_comp_events = [[torch.cuda.Event() for _ in self.layers] for _ in self.cache_device_ids]
        for device_events in self.synch_comp_events:
            device_events[((-1)%self.pipeline_degree)*self.pipeline_interval].record(stream=torch.cuda.current_stream())


    def set_cache_device_ids(self) -> None:
        '''
        Set the cache_device_ids automatically.
        Policy: given all the visible GPUs, the first worker_num GPUs is for computation (each for a worker), 
                while the remaining GPUs used by all the workers as cache.
        '''
        worker_num = torch.distributed.get_world_size()
        cache_gpu_num = torch.cuda.device_count() - worker_num


        if os.getenv("TOT_ORDERED_GPUS", 'None') != 'None':

            cand_cache_device_names = os.environ['TOT_ORDERED_GPUS'].split(',')[worker_num:]
            current_device_names_ordered = {gpu_i:i for i, gpu_i in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(','))}
            cand_cache_device_ids = [current_device_names_ordered[cuda_name] for cuda_name in cand_cache_device_names]
            self.cache_device_ids = cand_cache_device_ids

            print(f"self.cache_device_ids: {self.cache_device_ids} layer_num: {len(self.layers)}")

            return

        if worker_num == 1:


            self.cache_device_ids = list(range(worker_num+torch.cuda.current_device()*cache_gpu_num, worker_num+(torch.cuda.current_device()+1)*cache_gpu_num))
        else:

            self.cache_device_ids = list(range(worker_num, worker_num+cache_gpu_num))
            return

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


        if self.use_vllm:
            return self.forward_ori(input_ids, positions, kv_caches, input_metadata)
        else:
            return self.forward_ours(input_ids, positions, kv_caches, input_metadata)


    def forward_ori(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)


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


        return hidden_states


    def pre_increase_oncard_layers(self) -> None:
        '''
        Change the related parameters when we want to increase on-card layers.
        Changed:
            self.new_pipeline_degree, self.new_pipeline_interval, self.weight_cache_block_idx
        '''
        from vllm.core.block_manager import KVBlkPerLayerWeight


        more_layer_num = KVBlkPerLayerWeight.load_more_layer_on_card_num


        if more_layer_num == 0:

            return


        self.new_pipeline_degree = self.pipeline_degree - more_layer_num
        self.new_pipeline_interval = (self.layer_num + self.new_pipeline_degree - 1) // self.new_pipeline_degree


        print(f"In pre_increase_oncard_layers: self.layer_num:{self.layer_num}, self.weight_cache_block_num:{self.weight_cache_block_num}, KVBlkPerLayerWeight.cached_layer_num Invalid:{KVBlkPerLayerWeight.cached_layer_num}, more_layer_num:{more_layer_num}, self.pipeline_degree:{self.pipeline_degree}, self.new_pipeline_degree:{self.new_pipeline_degree}, self.new_pipeline_interval:{self.new_pipeline_interval}", flush=True)


        assert self.new_pipeline_degree == (self.layer_num + self.new_pipeline_interval - 1) // self.new_pipeline_interval


        if self.new_pipeline_degree == 2:


            for interval_i in range(self.pipeline_degree):
                layer_i = interval_i * self.pipeline_interval
                if interval_i >= 2:

                    self.weight_cache_block_idx[layer_i] = self.extra_weight_cache_blk_i
                    self.extra_weight_cache_blk_i += 1


        else:
            for interval_i in range(self.pipeline_degree):
                layer_i = interval_i * self.pipeline_interval
                if layer_i % self.new_pipeline_interval != 0:

                    self.weight_cache_block_idx[layer_i] = self.extra_weight_cache_blk_i
                    self.extra_weight_cache_blk_i += 1
                else:


                    if layer_i // self.new_pipeline_interval < 2:
                        self.weight_cache_block_idx[layer_i] = self.weight_cache_block_idx[0]
                    else:
                        self.weight_cache_block_idx[layer_i] = \
                            ((self.weight_cache_block_idx[0]//(self.weight_cache_block_num - 1) + \
                            layer_i//self.new_pipeline_interval-1)%2) * (self.weight_cache_block_num - 1)


        print(f"self.weight_cache_block_num: {self.weight_cache_block_num}, self.extra_weight_cache_blk_i:{self.extra_weight_cache_blk_i}, In pre_increase_oncard_layers: {self.weight_cache_block_idx}")
        print(max(self.weight_cache_block_idx), max(self.extra_weight_cache))


    def post_increase_oncard_layers(self) -> None:
        '''
        Change the related parameters when we want to increase on-card layers.
        Changed:
            self.pipeline_degree, self.pipeline_interval, self.new_pipeline_degree, self.new_pipeline_interval
        '''
        if self.new_pipeline_degree == None:

            return
        self.pipeline_degree = self.new_pipeline_degree
        self.pipeline_interval = self.new_pipeline_interval
        self.new_pipeline_degree = None
        self.new_pipeline_interval = None


    def forward_ours(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:


        hidden_states = self.embed_tokens(input_ids)
        residual = None


        self.pre_increase_oncard_layers()

        for i in range(len(self.layers)):


            if self.layer_waiting_for_params[i]:
                for device_events in self.load_param_events:
                    device_events[i].wait()

                self.layer_waiting_for_params[i] = False


            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )


            if (i%self.pipeline_interval) == 0:
                for device_events in self.synch_comp_events:
                    device_events[i].record(stream=torch.cuda.current_stream())


            if (i%self.pipeline_interval) == 0:


                to_load_layer_i = ((i//self.pipeline_interval+2)%self.pipeline_degree)*self.pipeline_interval

                if not((self.new_pipeline_degree == 2) and (to_load_layer_i//self.pipeline_interval < 2)):

                    if to_load_layer_i != i:

                        self.load_layer_params(to_load_layer_i, i)


        hidden_states, _ = self.norm(hidden_states, residual)


        self.post_increase_oncard_layers()


        return hidden_states


    def load_layer_params_no_dynamicIncreaseOnCardWeight(self, layer_i:int, last_last_layer_i:int) -> None:


        dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]
        last_last_layer_weight_blk_idx = self.weight_cache_block_idx[last_last_layer_i]


        if dst_weight_blk_idx != last_last_layer_weight_blk_idx:

            self.weight_cache_block_idx[layer_i] = self.weight_cache_block_num - 1 - dst_weight_blk_idx
            dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]

            for param, to_replace in zip(self.layer_params[layer_i], self.buffer_params[dst_weight_blk_idx]):
                param.data = to_replace.data


        for part_i, cache_device_i in enumerate(self.cache_device_ids):
            param_event = self.load_param_events[part_i][layer_i]
            to_synch_comp_event = self.synch_comp_events[part_i][last_last_layer_i]


            with torch.cuda.stream(self.param_streams[part_i]):


                to_synch_comp_event.wait()


                cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.weight_cache[dst_weight_blk_idx][part_i],
                    layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())
                param_event.record(stream=self.param_streams[part_i])


        self.layer_waiting_for_params[ layer_i ] = True


        return


    def load_layer_params_when_dynamicIncreaseOnCardWeight(self, layer_i:int, last_last_layer_i:int) -> None:
        '''
            We need to support the dynamic increase of on-card layer weights: we will change their para data pointer here.
        '''


        dst_weight_blk_idx = self.weight_cache_block_idx[layer_i]


        for param, to_replace in zip(self.layer_params[layer_i], self.buffer_params[dst_weight_blk_idx]):
            param.data = to_replace.data


        if layer_i//self.new_pipeline_interval == 1:


            last_last_layer_i = 0
        elif layer_i//self.new_pipeline_interval == 2:

            last_last_layer_i = self.pipeline_interval
        else:
            last_last_layer_i = ((layer_i//self.new_pipeline_interval-2)%self.new_pipeline_degree)*self.new_pipeline_interval

        need_to_wait = False
        if (layer_i%self.new_pipeline_interval == 0) and (self.new_pipeline_degree!=2):
            need_to_wait = True


        for part_i, cache_device_i in enumerate(self.cache_device_ids):
            param_event = self.load_param_events[part_i][layer_i]
            to_synch_comp_event = self.synch_comp_events[part_i][last_last_layer_i]


            with torch.cuda.stream(self.param_streams[part_i]):


                if need_to_wait:
                    to_synch_comp_event.wait()


                cache_ops.load_layer_weights( self.weight_cache_cpu[layer_i][part_i], self.extra_weight_cache[dst_weight_blk_idx][part_i],
                    layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())
                param_event.record(stream=self.param_streams[part_i])


        self.layer_waiting_for_params[ layer_i ] = True


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

                self.load_layer_params_no_dynamicIncreaseOnCardWeight(layer_i, last_last_layer_i)
            else:
                self.load_layer_params_when_dynamicIncreaseOnCardWeight(layer_i, last_last_layer_i)
                if layer_i//self.pipeline_interval == 1:

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


        global PRINT_LOGS
        if PRINT_LOGS:
            torch.cuda.synchronize()
            time1 = time.perf_counter()

        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)


        if PRINT_LOGS:
            torch.cuda.synchronize()
            time2 = time.perf_counter()


            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                print(f"iter latency: {time2-time1}s abs: {time1, time2}s")


        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens


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

            self.preprocess_for_offloading_weights_fast(model_name_or_path, cache_dir, load_format, revision)


        self.print_param_shapes()


    def load_weights_ori(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [

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


                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

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


        self.preprocess_for_offloading_weights(model_name_or_path, cache_dir, load_format, revision)


        stacked_params_mapping = [

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


                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue


                if ('model.layers.' not in name) or \
                    (self.get_layer_id_from_name(name) in [0, self.model.pipeline_interval]) or \
                    ((self.get_layer_id_from_name(name)%self.model.pipeline_interval)!=0):


                    name = name.replace(weight_name, param_name)

                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)


                    if int(os.getenv("LOCAL_RANK", "0")) == 0:
                        print(f"param.data.data_ptr() of {name}: {param.data.data_ptr()}, shape: {param.data.shape} device: {param.data.device} is contiguous: {param.is_contiguous()}")


                break
            else:


                if ('model.layers.' not in name) or \
                    (self.get_layer_id_from_name(name) in [0, self.model.pipeline_interval]) or \
                    ((self.get_layer_id_from_name(name)%self.model.pipeline_interval)!=0):


                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


                    if int(os.getenv("LOCAL_RANK", "0")) == 0:
                        print(f"param.data.data_ptr() of {name}: {param.data.data_ptr()}, shape: {param.data.shape} device: {param.data.device} is contiguous: {param.is_contiguous()} visible gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")


    def preprocess_for_offloading_weights(self,
             model_name_or_path: str,
             cache_dir: Optional[str] = None,
             load_format: str = "auto",
             revision: Optional[str] = None):


        stacked_params_mapping = [

            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())


        visited_param_names = [list() for _ in self.model.layers]

        print(f"init model size: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):


            if 'model.layers.' not in name:

                continue

            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):


                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue


                full_para_name = name.replace(weight_name, param_name)

                if full_para_name.endswith(".bias") and full_para_name not in params_dict:
                    continue
                param = params_dict[full_para_name]


                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)


                layer_i = self.get_layer_id_from_name(name)
                if full_para_name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(full_para_name)
                break
            else:


                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]


                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


                layer_i = self.get_layer_id_from_name(name)
                if name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(name)


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


            weight_range_in_blk.append((weight_num_per_layer, param.nelement(), param.shape))


            print(f"tensor size: {param.nelement()} -> {(((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)} alignment_unit: {alignment_unit}")
            weight_num_per_layer = weight_num_per_layer + (((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)

            assert (params_dtype == None) or (params_dtype == param.dtype)
            params_dtype = param.dtype


        print(f"visited_param_names[30]: {visited_param_names[30]}")
        print(f"weight_range_in_blk: {weight_range_in_blk}")


        need_be_multiple_of = len(self.model.cache_device_ids) * alignment_unit

        if need_be_multiple_of == 0:
            need_be_multiple_of = 1

        print(f"weight_num_per_layer: {weight_num_per_layer} -> multiple of {need_be_multiple_of} ==> {((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of}")


        weight_num_per_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of

        print(f"weight_num_per_layer: {weight_num_per_layer}")
        print(f"weight_cache_cpu size: {weight_num_per_layer*self.model.layer_num}")


        weight_cache_cpu = torch.empty(weight_num_per_layer*self.model.layer_num, dtype=self.model.layer_params[0][0].data.cpu().dtype, pin_memory=True)
        print(f"weight_cache_cpu size: {weight_cache_cpu.size()}")


        weight_cache_cpu = weight_cache_cpu.view(self.model.layer_num, -1)
        for layer_i, params in enumerate(self.model.layer_params):
            for param, (offset, size, tensor_shape) in zip(params, weight_range_in_blk):
                print(f"offset: {offset}, size: {size}, tensor_shape: {tensor_shape}, param shape: {param.data.shape}, param size: {param.data.size()}")
                weight_cache_cpu[layer_i][offset:offset+size] = param.data.cpu().view(-1)

        print(f"weight_cache_cpu size: {weight_cache_cpu.size()}")


        if len(self.model.cache_device_ids) > 0:
            weight_cache_cpu = weight_cache_cpu.view(self.model.layer_num, len(self.model.cache_device_ids), -1)


        for layer_i, layer_weights in enumerate(weight_cache_cpu):
            self.model.weight_cache_cpu.append(list())


            if ((layer_i % self.model.pipeline_interval) != 0) or (self.model.layer_num == self.model.weight_cache_block_num):

                print(f"layer_i: {layer_i} is not cached, pipeline_interval: {self.model.pipeline_interval}")
                continue

            print(f"\n\n\n\n\n\n store weights on cache gpus\n\n\n\n\n\n\n {self.model.layer_num, self.model.weight_cache_block_num}")
            for part_weights, cache_device_i in zip(layer_weights, self.model.cache_device_ids):
                self.model.weight_cache_cpu[-1].append( part_weights.to(cache_device_i) )


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


        from vllm.core.block_manager import KVBlkPerLayerWeight
        KVBlkPerLayerWeight.layer_weight_size = self.model.weight_num_per_layer * self.model.weight_cache.element_size()
        KVBlkPerLayerWeight.cached_layer_num = self.model.layer_num - self.model.weight_cache_block_num
        KVBlkPerLayerWeight.layer_num = self.model.layer_num


        print(f"after allocate param cache: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        for layer_i in range(len(self.model.layer_params)):
            params = self.model.layer_params[layer_i]
            for param, rng_info in zip(params, self.model.weight_range_in_blk):
                param.data = torch.narrow(self.model.weight_cache,
                    0,
                    self.model.weight_cache_block_idx[layer_i]*self.model.weight_num_per_layer + rng_info[0],
                    rng_info[1]).view(rng_info[2])


        for weight_cache_block_idx in [0, self.model.weight_cache_block_num - 1]:
            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.weight_cache,
                    0,
                    weight_cache_block_idx*self.model.weight_num_per_layer + rng_info[0],
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)


        if len(self.model.cache_device_ids) > 0:
            self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, len(self.model.cache_device_ids), -1)

        torch.cuda.empty_cache()


        for cache_device_i in self.model.cache_device_ids:


            cache_ops.init_P2P_access(cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())


    def preprocess_for_offloading_weights_fast(self,
             model_name_or_path: str,
             cache_dir: Optional[str] = None,
             load_format: str = "auto",
             revision: Optional[str] = None):


        stacked_params_mapping = [

            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())


        visited_param_names = [list() for _ in self.model.layers]

        print(f"init model size: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        loaded_weights = [dict() for _ in range(self.model.layer_num)]

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):


            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):


                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue


                full_para_name = name.replace(weight_name, param_name)

                if full_para_name.endswith(".bias") and full_para_name not in params_dict:
                    continue
                param = params_dict[full_para_name]


                if 'model.layers.' not in name:
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break


                layer_i = self.get_layer_id_from_name(name)
                if full_para_name not in visited_param_names[layer_i]:
                    self.model.layer_params[layer_i].append(param)
                    visited_param_names[layer_i].append(full_para_name)


                if full_para_name not in loaded_weights[layer_i]:
                    loaded_weights[layer_i][full_para_name] = list()
                loaded_weights[layer_i][full_para_name].append((shard_id, loaded_weight))


                break
            else:


                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]


                if 'model.layers.' not in name:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:


                    layer_i = self.get_layer_id_from_name(name)
                    if name not in visited_param_names[layer_i]:
                        self.model.layer_params[layer_i].append(param)
                        visited_param_names[layer_i].append(name)


                    if name not in loaded_weights[layer_i]:
                        loaded_weights[layer_i][name] = list()
                    loaded_weights[layer_i][name].append((None, loaded_weight))


        for layer_i in range(self.model.layer_num):


            params = self.model.layer_params[layer_i]

            order = sorted(range(len(params)), key=lambda i: params[i].nelement())
            self.model.layer_params[layer_i] = [params[i] for i in order]
            visited_param_names[layer_i] = [visited_param_names[layer_i][i] for i in order]


        weight_num_per_layer = 0
        weight_range_in_blk: List[Tuple[int, int, torch.Size]] = list()
        params_dtype = None

        alignment_unit = 128 // self.model.layer_params[0][0].element_size()

        for param in self.model.layer_params[0]:


            weight_range_in_blk.append((weight_num_per_layer, param.nelement(), param.shape))


            print(f"tensor size: {param.nelement()} -> {(((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)} alignment_unit: {alignment_unit}")
            weight_num_per_layer = weight_num_per_layer + (((param.nelement() + alignment_unit - 1) // alignment_unit)*alignment_unit)

            assert (params_dtype == None) or (params_dtype == param.dtype)
            params_dtype = param.dtype


        need_be_multiple_of = len(self.model.cache_device_ids) * alignment_unit

        if need_be_multiple_of == 0:
            need_be_multiple_of = 1

        print(f"weight_num_per_layer: {weight_num_per_layer} -> multiple of {need_be_multiple_of} ==> {((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of}")


        weight_num_per_layer = ((weight_num_per_layer+need_be_multiple_of-1)//need_be_multiple_of)*need_be_multiple_of

        print(f"weight_num_per_layer: {weight_num_per_layer}")
        print(f"weight_cache_cpu size: {weight_num_per_layer*self.model.layer_num}")


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


        from vllm.core.block_manager import KVBlkPerLayerWeight
        KVBlkPerLayerWeight.layer_weight_size = self.model.weight_num_per_layer * self.model.weight_cache.element_size()
        KVBlkPerLayerWeight.cached_layer_num = self.model.layer_num - self.model.weight_cache_block_num
        KVBlkPerLayerWeight.layer_num = self.model.layer_num


        print(f"after allocate param cache: {torch.cuda.memory_allocated()/1024/1024/1024} GB")


        layer_num_on_other_gpus = self.model.pipeline_degree if self.model.layer_num > self.model.weight_cache_block_num else 0
        layer_size_on_other_gpus = (self.model.weight_num_per_layer // len(self.model.cache_device_ids)) if len(self.model.cache_device_ids) > 0 else 0


        self.model.weight_cache_cpu = [list() for _ in range(self.model.layer_num)]
        for cache_device_i in self.model.cache_device_ids:
            partial_cache = torch.empty((layer_num_on_other_gpus, layer_size_on_other_gpus),
                                device=torch.device(f'cuda:{cache_device_i}'),
                                dtype=params_dtype)

            for cached_layer_i in range(layer_num_on_other_gpus):
                self.model.weight_cache_cpu[cached_layer_i*self.model.pipeline_interval].append(partial_cache[cached_layer_i])


        cross_boundary_param_info = list()


        for layer_i in range(len(self.model.layer_params)):
            params = self.model.layer_params[layer_i]
            on_cache_gpus = (layer_i % self.model.pipeline_interval == 0) and (self.model.layer_num > self.model.weight_cache_block_num)
            for param, rng_info in zip(params, self.model.weight_range_in_blk):
                if not on_cache_gpus:
                    param.data = torch.narrow(self.model.weight_cache,
                        0,
                        self.model.weight_cache_block_idx[layer_i]*self.model.weight_num_per_layer + rng_info[0],
                        rng_info[1]).view(rng_info[2])

                else:
                    start_partial_cache_i = rng_info[0]//layer_size_on_other_gpus
                    end_partial_cache_i = (rng_info[0]+rng_info[1]-1)//layer_size_on_other_gpus
                    if start_partial_cache_i != end_partial_cache_i:

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
                                (param.data, layer_i, partial_cache_i,
                                rng))

                    else:

                        param.data = torch.narrow(self.model.weight_cache_cpu[layer_i][start_partial_cache_i],
                            0,
                            rng_info[0]%layer_size_on_other_gpus,
                            rng_info[1]).view(rng_info[2])


        for weight_cache_block_idx in [0, self.model.weight_cache_block_num - 1]:
            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.weight_cache,
                    0,
                    weight_cache_block_idx*self.model.weight_num_per_layer + rng_info[0],
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)


        if len(self.model.cache_device_ids) > 0:
            self.model.weight_cache = self.model.weight_cache.view(self.model.weight_cache_block_num, len(self.model.cache_device_ids), -1)

        torch.cuda.empty_cache()


        for cache_device_i in self.model.cache_device_ids:


            cache_ops.init_P2P_access(cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())


        for layer_i in range(self.model.layer_num):
            param_names = visited_param_names[layer_i]
            params = self.model.layer_params[layer_i]

            weight_loader_info = loaded_weights[layer_i]


            for name, param in zip(param_names, params):
                for shard_id, loaded_weight in weight_loader_info[name]:

                    if shard_id != None:
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                    else:
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)


        for param_data, layer_i, partial_cache_i, (param_rng, cache_rng) in cross_boundary_param_info:


            narrow_param_data = torch.narrow(self.model.weight_cache_cpu[layer_i][partial_cache_i],
                    0, cache_rng[0], cache_rng[1]-cache_rng[0])

            narrow_param_cpu = param_data.view(-1)

            narrow_param_cpu = torch.narrow(narrow_param_cpu, 0, param_rng[0], param_rng[1]-param_rng[0])


            narrow_param_data.copy_(narrow_param_cpu)


        for i in range(self.model.pipeline_degree):
            layer_i = i * self.model.pipeline_interval
            weight_cache_block_idx = self.model.weight_cache_block_idx[layer_i]
            buffer_param_datas = self.model.buffer_params[weight_cache_block_idx]
            for param, param_data in zip(self.model.layer_params[layer_i], buffer_param_datas):
                param.data = param_data


        if layer_num_on_other_gpus > 0:
            for layer_i, weight_cache_block_idx in zip([0, self.model.pipeline_interval], [0, self.model.weight_cache_block_num - 1]):
                for part_i, cache_device_i in enumerate(self.model.cache_device_ids):
                    cache_ops.load_layer_weights(self.model.weight_cache_cpu[layer_i][part_i], self.model.weight_cache[weight_cache_block_idx][part_i],
                        layer_i, cache_device_i, torch.cuda.current_device(), torch.cuda.current_device())


    def get_layer_id_from_name(self, name:str) -> int:

        terms = name.split('.')
        return int(terms[2])


    def init_extra_weight_cache_in_KV_cache(self, kv_caches: List[KVCache]) -> None:
        '''
            prepare parameter data whose address is in the KV cache.
            Only called after we change the KV cache layout and want to dynamically increase the on-card layer weights.
            Initialize:  self.model.buffer_params, self.model.extra_weight_cache
        '''

        from vllm.core.block_manager import KVBlkPerLayerWeight
        extra_weight_cache_blk_num = KVBlkPerLayerWeight.cached_layer_num

        cache = kv_caches[0].view(-1)
        assert cache.element_size() == self.model.weight_cache.element_size(), "the KV cache dtype size and the weight cache dtype size are different"

        for weight_cache_block_idx in range(self.model.weight_cache_block_num, self.model.weight_cache_block_num+extra_weight_cache_blk_num):

            print(f"set extra_weight_cache[{weight_cache_block_idx}]")
            self.model.extra_weight_cache[weight_cache_block_idx] = torch.narrow(cache,
                    0,
                    len(cache) - (weight_cache_block_idx+1-self.model.weight_cache_block_num)*self.model.weight_num_per_layer,
                    self.model.weight_num_per_layer)


            self.model.buffer_params[weight_cache_block_idx] = list()
            for rng_info in self.model.weight_range_in_blk:
                param_data = torch.narrow(self.model.extra_weight_cache[weight_cache_block_idx],
                    0,
                    rng_info[0],
                    rng_info[1]).view(rng_info[2])
                self.model.buffer_params[weight_cache_block_idx].append(param_data)


            self.model.extra_weight_cache[weight_cache_block_idx] = self.model.extra_weight_cache[weight_cache_block_idx].view(len(self.model.cache_device_ids), -1)


            if int(os.getenv("LOCAL_RANK", "0")) == 0:
                print(f"self.model.extra_weight_cache[{weight_cache_block_idx}]: {self.model.extra_weight_cache[weight_cache_block_idx].data_ptr()}")


        self.model.extra_weight_cache[0] = self.model.weight_cache[0]
        self.model.extra_weight_cache[self.model.weight_cache_block_num - 1] = self.model.weight_cache[self.model.weight_cache_block_num - 1]
