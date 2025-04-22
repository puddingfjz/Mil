"""
This file contains the search method to find the best exec plans
for the given set of models and the given set of requests.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import itertools
import fake_scheduling
from my_per_iter_latency_estimator import CostTable, get_cost_table, get_cost_table_from_serialized_data
import get_my_cost_table_directly, benchmarks.get_my_cost_table_directly2 as get_my_cost_table_directly2
import output_length_sampler

from vllm.transformers_utils.config import get_config
from vllm.worker.cache_engine import CacheEngine
from collections import defaultdict

from model_size_database import model_sizes
import time


import ray
import os

from functools import partial

_ENGINE_ARGS_LIST = dict()
_FAKE_SCHEDULING_RES = dict() 
_MAX_SEQ_NUM = 0
_CHECKED_SEQ_NUM = 0
_MODEL_ID = 0
_COST_MODEL_REF = None
_hf_config_DICT = dict()
_MODEL_CHECK_ORDER = 0

class InferenceArgs:
    """ The args of the inference setting. """
    def __init__(self, 
        scheduler_config,
        cache_config,
    ) -> None:
        self.prompt_limit = min(scheduler_config.max_model_len,
                                scheduler_config.max_num_batched_tokens)
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens

        
        

        self.max_seq_num = scheduler_config.max_num_seqs
        self.block_size=cache_config.block_size



class MyModelInfor:
    """ My model information class. Contains basic model information. """
    def __init__(
        self,
        model_id: int,
        cost_table: CostTable,
        model_path, 
        outlen_generator,
        sample_config, trust_remote_code, revision,
        data_byte, 
        inp_lens: List[int], 
        out_lens: List[int] = list(), 
        inp_seq_ids: List[int] = list(), 
        
        inp_req_from_which_models: Dict[int, List[int]] = None, 
        inp_req_from_which_model_which_out_reqs: Dict[int, Dict[int, int]] = None,
        
        independent_srcs: bool = None,
    ) -> None:
        self.data_byte = data_byte
        self.model_name = None
        self.model_path = model_path
        self.set_model_name_from_path()
        self.trust_remote_code=trust_remote_code
        self.revision=revision
        
        self.hf_config=None
        self.layer_num = None
        self.set_hf_config()

        
        
        
        
        
        
        
        self.sample_config = sample_config

        
        self.inp_lens = tuple(inp_lens)
        self.out_lens = tuple(outlen_generator(
            model_id, self.model_name, inp_lens)) if len(out_lens) == 0 else tuple(out_lens)
        

        print(f"{self.model_name} avg output len: {sum(self.out_lens)/len(self.out_lens)}")
        print(f"{self.model_name} max output len: {max(self.out_lens)}")

        self.remaining_decode_flops = None
        self.set_remaining_decode_flops(cost_table)

        self.model_id = model_id

        self.input_model_ids: List[int] = list()

        
        self.inp_base_model_ids: List[int] = list()


        
        self.ori_tot_inp_num: int = len(inp_lens) 
        
        self.inp_seq_ids = np.asarray(inp_seq_ids) if len(inp_seq_ids)>0 else np.asarray(range(self.ori_tot_inp_num))

        self.ori_tot_remaining_decode_flops = self.remaining_decode_flops

        self.ori_inp_seq_ids = sorted(self.inp_seq_ids)
    
        self.inp_req_from_which_models = inp_req_from_which_models
        self.inp_req_from_which_model_which_out_reqs = inp_req_from_which_model_which_out_reqs
        
        self.independent_srcs = independent_srcs

        self.can_be_vertically_fused_topologically = False 

        
        self.check_order:int = int(1e9)


    
    

    def get_unfinished_srcs(self):
        assert self.independent_srcs
        alive_seq_ids = set(self.inp_seq_ids)
        unfinished_srcs = list()
        print(f"alive_seq_ids: {alive_seq_ids}")
        
        for from_model_id, req_ids in self.inp_req_from_which_model_which_out_reqs.items():
            print(f"ori from_model_id: {from_model_id}, req_ids: {req_ids}")
            if len(alive_seq_ids.intersection(req_ids)) > 0:
                unfinished_srcs.append(from_model_id)
        return unfinished_srcs

        
    def get_base_model_ids(self):
        return [self.model_id]

    def get_base_models(self):
        return [self]
    
    
    

    def not_started(self):
        return self.ori_tot_remaining_decode_flops == self.remaining_decode_flops

    def set_model_name_from_path(self):
        pos = self.model_path.find('/')
        model_name = self.model_path[pos+1:]
        self.model_name = model_name

    def set_hf_config(self):

        global _hf_config_DICT

        key = (self.model_path, self.trust_remote_code, self.revision)
        hf_config = None
        if key in _hf_config_DICT:
            hf_config = _hf_config_DICT[key]
        else:
            hf_config = get_config(*key)
            _hf_config_DICT[key] = hf_config
        
        
        self.hf_config = hf_config
        L: int = hf_config.num_hidden_layers
        self.layer_num = L


    def get_hidden_size(self):
        return self.hf_config.hidden_size

    def get_name(self):
        return self.model_name

    def update_inp_out_seqlens(
            self, inp_lens: List[int], out_lens: List[int], inp_seq_ids: List[int],
            cost_table: CostTable, 
            remaining_decode_flops = None):

        
        valid_inds = np.nonzero(out_lens)[0]
        inp_seq_ids = np.asarray(inp_seq_ids)[valid_inds]
        inp_lens = np.asarray(inp_lens)[valid_inds]
        out_lens = np.asarray(out_lens)[valid_inds]
        

        self.inp_seq_ids = inp_seq_ids
        self.inp_lens = tuple(inp_lens)
        self.out_lens = tuple(out_lens)
        self.set_remaining_decode_flops(cost_table, remaining_decode_flops)

    def get_inp_out_seqlens(self):
        return (self.inp_lens, self.out_lens)
    
    def get_inp_seq_ids(self):
        return self.inp_seq_ids


    def get_remaining_flops(self):
        ''' Return flops in TFLOPs '''
        return self.remaining_decode_flops
    
        return fake_scheduling.comp_flops_from_seqlens(
            self.inp_lens, self.out_lens, only_decode=True, cost_table=cost_table, 
            model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)


    def set_remaining_decode_flops(self, cost_table: CostTable, remaining_decode_flops = None):
        ''' Return flops in TFLOPs '''
        if remaining_decode_flops != None:
            self.remaining_decode_flops = remaining_decode_flops
        else:
            self.remaining_decode_flops = fake_scheduling.comp_flops_from_seqlens(
                self.inp_lens, self.out_lens, only_decode=True, cost_table=cost_table, 
                model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)



    
    
    
    
    
    
    
    
    



    def is_finished(self):
        return len(self.inp_lens) == 0


    def get_state(self):
        '''
            The state of the model, i.e., its inference progress, is determined by its remaining seqlens.
        '''
        
        return (self.model_name, self.model_id, self.inp_lens, self.out_lens)


    def __str__(self) -> str:
        return f'{self.model_name}'
















class MyFusedModelInfor(MyModelInfor):
    """ 
        My model information class. Contains basic model information. 
        Contains the informatio of a fused model.
    """
    def __init__(
        self,
        model_list: List[MyModelInfor],
    ) -> None:
        
        global _MODEL_ID

        self.model_list = model_list
        self.model_id = _MODEL_ID
        _MODEL_ID+=1
        
        model_0 = model_list[0]
        self.data_byte = model_0.data_byte
        self.model_name = model_0.model_name
        self.model_path = model_0.model_path

        self.trust_remote_code=model_0.trust_remote_code
        self.revision=model_0.revision
        
        self.hf_config=model_0.hf_config
        self.layer_num = model_0.layer_num

        
        self.sample_config = model_0.sample_config

        
        self.inp_lens = None 
        self.out_lens = None 
        

        
        

        self.remaining_decode_flops = None 

        

        
        self.input_model_ids: List[int] = list()
        self.init_inp_model_ids()

        self.inp_base_model_ids = None
        self.init_inp_base_model_ids()

        
        self.ori_tot_inp_num: int = [model.ori_tot_inp_num for model in self.model_list] 
        
        self.inp_seq_ids = None 

        self.ori_tot_remaining_decode_flops = [model.ori_tot_remaining_decode_flops for model in self.model_list]

        
        self.inp_req_from_which_models = model_list[0].inp_req_from_which_models
        self.inp_req_from_which_model_which_out_reqs = model_list[0].inp_req_from_which_model_which_out_reqs
        
        self.independent_srcs = model_list[0].independent_srcs

        self.can_be_vertically_fused_topologically = model_list[0].can_be_vertically_fused_topologically


        
        self.check_order:int = int(1e9)


    def get_base_model_ids(self):
        return [model.model_id for model in self.model_list]

    def get_base_models(self):
        return self.model_list
    
    def init_inp_model_ids(self):
        for model in self.model_list:
            self.input_model_ids = self.input_model_ids + model.input_model_ids
        
        self.input_model_ids = sorted(set(self.input_model_ids).difference(self.get_base_model_ids()))


    def init_inp_base_model_ids(self):
        inp_base_model_ids = list()
        for model in self.model_list:
            inp_base_model_ids = inp_base_model_ids + model.inp_base_model_ids
        
        self.inp_base_model_ids = sorted(set(inp_base_model_ids).difference(self.get_base_model_ids()))

    
    
    
        
    def not_started(self):
        return self.ori_tot_remaining_decode_flops == [model.remaining_decode_flops for model in self.model_list]





    
    

    
    


    def update_inp_out_seqlens(
            self, inp_lens_for_models: List[int], out_lens_for_models: List[int], inp_seq_ids_for_models: List[int],
            cost_table: CostTable, 
            remaining_decode_flops_for_models = None):

        if remaining_decode_flops_for_models == None:
            remaining_decode_flops_for_models = [None for _ in range(len(self.model_list))]

        
        

        for model, inp_lens, out_lens, inp_seq_ids, remaining_decode_flops in zip(\
            self.model_list, inp_lens_for_models, out_lens_for_models, inp_seq_ids_for_models, remaining_decode_flops_for_models):
            model.update_inp_out_seqlens(inp_lens, out_lens, inp_seq_ids, cost_table, remaining_decode_flops=remaining_decode_flops)



    def get_inp_out_seqlens(self):
        return (tuple([model.inp_lens for model in self.model_list]), tuple([model.out_lens for model in self.model_list]))
    
    def get_inp_seq_ids(self):
        return [model.inp_seq_ids for model in self.model_list]


    def get_remaining_flops(self):
        ''' Return flops in TFLOPs '''
        return [model.remaining_decode_flops for model in self.model_list]
    
        return fake_scheduling.comp_flops_from_seqlens(
            self.inp_lens, self.out_lens, only_decode=True, cost_table=cost_table, 
            model_path=self.model_path, trust_remote_code=self.trust_remote_code, revision=self.revision)


    def set_remaining_decode_flops(self, cost_table: CostTable, remaining_decode_flops = None):
        ''' 
            Return flops in TFLOPs.
            We do not need this function in a fused model.
        '''
        assert False



    def is_finished(self):
        return False not in [model.is_finished() for model in self.model_list]


    def get_state(self):
        '''
            The state of the model, i.e., its inference progress, is determined by its remaining seqlens.
        '''
        assert False, (self.model_id, self.get_base_model_ids())
        
        return (self.model_name, tuple(self.get_base_model_ids()), \
                tuple([model.inp_lens for model in self.model_list]), \
                    tuple([model.out_lens for model in self.model_list]))


    def __str__(self) -> str:
        return f'{self.model_name}'














@ray.remote
def _entry_to_remote_fake_scheduling_for_a_dp_worker(
    dp_inp_lens, dp_out_lens, 
    dp_arrive_times, check_gap,
    max_seq_num, gpu_cache_block_num, 
    max_num_batched_tokens,
    block_size, 
    sort_input, 
    cost_table, 
    model_path, 
    exec_plan_key, 
    sample_config, 
    trust_remote_code, 
    revision
    ):
    
    time1 = time.perf_counter()
    with open(f'tmp_time_logs.log', 'a') as f:
        f.write(f"TIME TO DP FAKE SCHEDULE 0: {os.getpid()} ---abs: {time1}\n")

    
    
    
    cost_table = get_cost_table_from_serialized_data(*cost_table)
    
    
    
    
    
    
    time1 = time.perf_counter()
    with open(f'tmp_time_logs.log', 'a') as f:
        f.write(f"TIME TO DP FAKE SCHEDULE 1: {os.getpid()} ---abs: {time1}\n")
    

    dp_arrive_times=dp_arrive_times.copy()


    
    ret = \
        fake_scheduling.fake_FCFS_schedule(
            inp_lens=list(dp_inp_lens),out_lens=list(dp_out_lens), 
            arrive_times=dp_arrive_times, check_gap=check_gap,
            max_seq_num=max_seq_num, max_block_num=gpu_cache_block_num, 
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size, 
            sort_input=sort_input, 
            cost_estimate_args={
                "cost_table":cost_table, 
                "model_name":model_path, 
                "exec_plan":exec_plan_key, 
                "sample_config":sample_config, 
                "trust_remote_code":trust_remote_code, 
                "revision":revision})
    
    time2 = time.perf_counter()
    
    with open(f'tmp_time_logs.log', 'a') as f:
        f.write(f"TIME TO DP FAKE SCHEDULE 2: {os.getpid()} {time2 - time1}, ---abs: {time2}\n")
    
    return ret













def _entry_to_remote_fake_scheduling(
    dp_size: int,
    dp_inp_lens_list: List[List[int]],
    dp_out_lens_list: List[List[int]],
    arrive_times: List[float],
    
    check_gap: int,
    sort_input: bool,
    max_seq_num: int,
    gpu_cache_block_num: int,
    max_num_batched_tokens: int,
    block_size: int,
    
    
    model_path: str,
    exec_plan_key,
    sample_config,
    trust_remote_code: bool,
    revision: bool
):
    """
        This function is a remote function used to do fake scheduling. --> change to make only fake scheduling remote
    """
    global _COST_MODEL_REF
    cost_table = _COST_MODEL_REF 
    res_ref_list = [None] * dp_size

    

    for dp_id in range(dp_size):
        
        dp_inp_lens = dp_inp_lens_list[dp_id]
        dp_out_lens = dp_out_lens_list[dp_id]            
        dp_arrive_times = arrive_times[dp_id::dp_size]

        time1 = time.perf_counter()
        with open(f'tmp_time_logs.log', 'a') as f:
            f.write(f"TIME TO DP FAKE SCHEDULE -1: {os.getpid()} ---abs: {time1}\n")

        res_ref_list[dp_id] = \
            _entry_to_remote_fake_scheduling_for_a_dp_worker.remote(
                dp_inp_lens, dp_out_lens, 
                dp_arrive_times, check_gap,
                max_seq_num, gpu_cache_block_num, 
                max_num_batched_tokens,
                block_size, 
                sort_input, 
                cost_table, 
                model_path, 
                exec_plan_key, 
                sample_config, 
                trust_remote_code, 
                revision
            )
        continue


        
        dp_inp_lens = dp_inp_lens_list[dp_id]
        dp_out_lens = dp_out_lens_list[dp_id]            
        dp_arrive_times = arrive_times[dp_id::dp_size]

        
        res_ref_list[dp_id] = \
            _entry_to_remote_fake_scheduling_for_a_dp_worker.remote(
                dp_inp_lens, dp_out_lens, 
                dp_arrive_times, check_gap,
                max_seq_num, gpu_cache_block_num, 
                max_num_batched_tokens,
                block_size, 
                sort_input, 
                cost_table, 
                model_path, 
                exec_plan_key, 
                sample_config, 
                trust_remote_code, 
                revision)

    
    return res_ref_list








@ray.remote
def _entry_to_remote_fake_scheduling_vertical_fusion_for_a_dp_worker(
    dp_inp_lens_for_models, dp_out_lens_for_models, 
    dp_arrive_times_for_models, dp_seq_ids_for_models,
    check_gap,
    max_seq_num, gpu_cache_block_num, 
    max_num_batched_tokens,
    block_size, 
    sort_input, 
    cost_table, 
    model_path, 
    exec_plan_key, 
    sample_config, 
    trust_remote_code, 
    revision
    ):
    
    time1 = time.perf_counter()
    with open(f'tmp_time_logs.log', 'a') as f:
        f.write(f"TIME TO DP FAKE SCHEDULE vertical 0: {os.getpid()} ---abs: {time1}\n")

    
    
    
    cost_table = get_cost_table_from_serialized_data(*cost_table)
    
    
    
    
    
    
    time1 = time.perf_counter()
    with open(f'tmp_time_logs.log', 'a') as f:
        f.write(f"TIME TO DP FAKE SCHEDULE vertical 1: {os.getpid()} ---abs: {time1}\n")
    

    ret = \
        fake_scheduling.fake_FCFS_schedule_vertical_fuse(
            inp_lens=list(dp_inp_lens_for_models[0]),out_lens=list(dp_out_lens_for_models[0]), 
            arrive_times=dp_arrive_times_for_models[0], 
            ref_seq_ids=dp_seq_ids_for_models[0],
            
            ref_seq_ids_list=dp_seq_ids_for_models[1:],
            inp_lens_list=dp_inp_lens_for_models[1:],
            out_lens_list=dp_out_lens_for_models[1:],
            arrive_times_list=dp_arrive_times_for_models[1:],
            
            check_gap=check_gap,
            max_seq_num=max_seq_num, max_block_num=gpu_cache_block_num, 
            max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size, 
            sort_input=sort_input, 
            cost_estimate_args={
                "cost_table":cost_table, 
                "model_name":model_path, 
                "exec_plan":exec_plan_key, 
                "sample_config":sample_config, 
                "trust_remote_code":trust_remote_code, 
                "revision":revision}
            )



    time2 = time.perf_counter()
    
    with open(f'tmp_time_logs.log', 'a') as f:
        f.write(f"TIME TO DP FAKE SCHEDULE vertical 2: {os.getpid()} {time2 - time1}, ---abs: {time2}\n")
    
    return ret










def _entry_to_remote_fake_scheduling_vertical_fusion(
    dp_size: int,
    dp_inp_lens_list_for_models: List[List[List[int]]],
    dp_out_lens_list_for_models: List[List[List[int]]],
    dp_inp_seq_ids_list_for_models: List[List[List[int]]],
    dp_arrive_times_list_for_models: List[List[float]],
    
    check_gap: int,
    sort_input: bool,
    max_seq_num: int,
    gpu_cache_block_num: int,
    max_num_batched_tokens: int,
    block_size: int,
    
    
    model_path: str,
    exec_plan_key,
    sample_config,
    trust_remote_code: bool,
    revision: bool
):
    """
        This function is a remote function used to do fake scheduling. --> change to make only fake scheduling remote
    """
    global _COST_MODEL_REF
    cost_table = _COST_MODEL_REF 
    res_ref_list = [None] * dp_size


    for dp_id in range(dp_size):
        
        dp_inp_lens_for_models = dp_inp_lens_list_for_models[dp_id]
        dp_out_lens_for_models = dp_out_lens_list_for_models[dp_id]            
        dp_seq_ids_for_models = dp_inp_seq_ids_list_for_models[dp_id]
        dp_arrive_times_for_models = dp_arrive_times_list_for_models[dp_id]


        time1 = time.perf_counter()
        with open(f'tmp_time_logs.log', 'a') as f:
            f.write(f"TIME TO DP FAKE SCHEDULE vertical -1: {os.getpid()} ---abs: {time1}\n")

        res_ref_list[dp_id] = \
            _entry_to_remote_fake_scheduling_vertical_fusion_for_a_dp_worker.remote(
                dp_inp_lens_for_models, dp_out_lens_for_models, 
                dp_arrive_times_for_models, dp_seq_ids_for_models,
                check_gap,
                max_seq_num, gpu_cache_block_num, 
                max_num_batched_tokens,
                block_size, 
                sort_input, 
                cost_table, 
                model_path, 
                exec_plan_key, 
                sample_config, 
                trust_remote_code, 
                revision
                )


    return res_ref_list














class MyExecPlan:
    """ My execution plan definition. """
    def __init__(
        self,
        model: MyModelInfor, 
        num_worker: int, 
        wld_degree: int, 
        cache_gpu_num: int, 
        mem_per_comp_gpu: float, 
        dp_size: int, 
        param_byte_per_comp_gpu: int, 
        param_byte_per_cache_gpu: int,
        gpu_cache_byte_per_block: int,
        infer_args: InferenceArgs,
        tot_gpu_mem_byte: int,
    ) -> None:
        self.model: MyModelInfor = model
        self.num_worker = num_worker
        self.wld_degree = wld_degree
        self.cache_gpu_num = cache_gpu_num
        self.mem_per_comp_gpu = mem_per_comp_gpu
        self.dp_size = dp_size
        self.param_byte_per_comp_gpu = param_byte_per_comp_gpu
        self.param_byte_per_cache_gpu = param_byte_per_cache_gpu
        self.gpu_cache_byte_per_block = gpu_cache_byte_per_block

        
        self.infer_args: InferenceArgs = infer_args
        
        self.tot_gpu_mem_byte: int = tot_gpu_mem_byte
        
        self.basic_mem_consumption: int = 0
        self.gpu_cache_block_num = None
        self.set_gpu_cache_block_num()
        
       
        
        self.cumsum_latencys_list: List[List[float]] = [list() for _ in range(self.dp_size)]
        self.cum_rng_nums_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.rng_starts_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.rng_ends_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.is_prefill_steps_list: List[List[bool]] = [list() for _ in range(self.dp_size)]

        self.total_latency_list: List[Optional[float]] = [None for _ in range(self.dp_size)]

        self.cache_stop_time_info_list: List[Dict[int, Tuple[Tuple[List[int], List[int]], float]]] \
            = [dict() for _ in range(self.dp_size)]
        
        self.dp_inp_lens_list: List[List[int]] = [list() for _ in range(self.dp_size)]
        self.dp_out_lens_list: List[List[int]] = [list() for _ in range(self.dp_size)]

        
        
        
        self.finish_times_merged: List[float] = list()
        self.finish_times_list: List[List[float]] = [list() for _ in range(self.dp_size)]
        
        
        
        self.inp_arrive_times: List[Tuple[float, int]] = list() 
        self.extra_cost: float = 0.0
        
        self.dp_inp_seq_ids_list: List[List[int]] = [list() for _ in range(self.dp_size)]

        self._FAKE_SCHEDULING_RES_key = None
        self.fake_scheduling_res_ref = None

        self.throughput_till_each_iter_list: List[List[float]] = [list() for _ in range(self.dp_size)]

        
        
        

        self.load_cost_just_for_refer = None 
    


    def copy_the_plan(self):
        return MyExecPlan(
            self.model, 
            self.num_worker, 
            self.wld_degree, 
            self.cache_gpu_num, 
            self.mem_per_comp_gpu, 
            self.dp_size, 
            self.param_byte_per_comp_gpu, 
            self.param_byte_per_cache_gpu,
            self.gpu_cache_byte_per_block,
            self.infer_args,
            self.tot_gpu_mem_byte,
        )


    def get_base_model_ids(self):
        return self.model.get_base_model_ids()
    
    
    
    
    def get_base_models(self):
        return [self.model]
    
    def models_not_started(self):
        return self.model.not_started()

    def get_dp_inp_lens_list_for_models(self):
        return self.dp_inp_lens_list


    def get_finish_times_merged_not_support_different_ori_seq_ids_in_different_models(self, seq_ids: List[int], model_ind: int):

        """
            NOTE: this version does not support different models have different original seq ids to answer.
        """

        assert model_ind == 0, f"Wrong model ind: {model_ind}"

        
        

        if len(self.finish_times_merged) == 0:
            self.finish_times_merged = np.asarray([-1-self.extra_cost]*self.model.ori_tot_inp_num) 
            for dp_inp_seq_ids, finish_times in zip(self.dp_inp_seq_ids_list, self.finish_times_list):
                self.finish_times_merged[dp_inp_seq_ids] = finish_times
        return self.finish_times_merged[seq_ids]
    


    def get_finish_times_merged(self, seq_ids: List[int], model_ind: int):

        assert model_ind == 0, f"Wrong model ind: {model_ind}"

        
        

        if len(self.finish_times_merged) == 0:
            self.finish_times_merged = np.asarray([-1-self.extra_cost]*self.model.ori_tot_inp_num) 
            for dp_inp_seq_ids, finish_times in zip(self.dp_inp_seq_ids_list, self.finish_times_list):
                inds = np.searchsorted(self.model.ori_inp_seq_ids, dp_inp_seq_ids)
                
                self.finish_times_merged[inds] = finish_times


        

        
        
        

        return get_infor_given_seq_ids(
            values=self.finish_times_merged, 
            seq_ids_we_have=self.model.ori_inp_seq_ids, 
            seq_ids_requested=seq_ids, 
            default_value=-1-self.extra_cost)

        inds = np.searchsorted(self.model.ori_inp_seq_ids, seq_ids)
        
        return self.finish_times_merged[inds]



    def merge_new_inp_out_lens_of_data_parallel_workers(
            self, new_inp_out_lens_list: List[List[List[int]]]
        )->List[List[int]]:
        
        
        
        inp_lens_list = [dp_inp_out_lens[0] for dp_inp_out_lens in new_inp_out_lens_list]
        out_lens_list = [dp_inp_out_lens[1] for dp_inp_out_lens in new_inp_out_lens_list]
        inp_seq_ids_list = [dp_inp_seq_ids[dp_inp_out_lens[2]] \
                              for dp_inp_seq_ids, dp_inp_out_lens \
                                in zip(self.dp_inp_seq_ids_list, new_inp_out_lens_list)]
        inp_lens = np.concatenate(inp_lens_list)
        out_lens = np.concatenate(out_lens_list)
        inp_seq_ids = np.concatenate(inp_seq_ids_list)
        
        
        order = np.argsort(inp_seq_ids)
        inp_lens = inp_lens[order]
        out_lens = out_lens[order]
        inp_seq_ids = inp_seq_ids[order]
        return [inp_lens, out_lens, inp_seq_ids]




    def set_gpu_cache_block_num(self):
        '''
            gpu cache block num = (available gpu mem - parameter mem) // mem per block
        '''
        self.gpu_cache_block_num = \
            ((self.tot_gpu_mem_byte * self.mem_per_comp_gpu) \
             - self.param_byte_per_comp_gpu - self.basic_mem_consumption) \
                // self.gpu_cache_byte_per_block
        self.gpu_cache_block_num = int(self.gpu_cache_block_num)
        
        


    def set_extra_cost(self, extra_cost: float):
        self.extra_cost = extra_cost

    
    def estimate_exec_time_no_data_parallel(
            self, cost_table: CostTable):
        '''
            Estimate the total inference of this model for the given inp_lens.
        '''
        
        
        
        

        

        inp_lens, out_lens = self.model.get_inp_out_seqlens()
        

        
        

        
        key = (self.model.model_name, self.get_key(), self.model.get_inp_out_seqlens())
        if key in _FAKE_SCHEDULING_RES:
            (self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends,
                self.is_prefill_steps) = _FAKE_SCHEDULING_RES[key]
            
            self.total_latency = self.cumsum_latencys[-1]

            
            
            
            
            



            return self.total_latency
        else:
            decode_logs, prefill_logs, is_prefill_steps, infer_progress = fake_scheduling.fake_FCFS_schedule(
                inp_lens=list(inp_lens), out_lens=list(out_lens), 
                max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                block_size=self.infer_args.block_size)
        
            
            tot_latency, prefill_latencys, decode_latencys = \
                fake_scheduling.estimate_prefill_and_decode_cost_from_predicted_logs(
                    prefill_logs=prefill_logs, decode_logs=decode_logs, cost_table=cost_table, 
                    model_name=self.model.model_path, exec_plan=self.get_key(), sample_config=self.model.sample_config, 
                    trust_remote_code=self.model.trust_remote_code, revision=self.model.revision)

            
            self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends = \
                fake_scheduling.get_cumLatency_inferRng_info(
                    decode_latencys, prefill_latencys, 
                    is_prefill_steps, infer_progress)

            
            tot_latency = self.cumsum_latencys[-1]


            
            
            self.is_prefill_steps = is_prefill_steps
            
            
            _FAKE_SCHEDULING_RES[key] = (self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends, 
                                         self.is_prefill_steps)

            
            
            
            
            


            self.total_latency = tot_latency
            return tot_latency
        


    def _sort_and_partition_data_parallel(
            self, 
            arrive_times: List[float],
            ):
        inp_lens, out_lens = self.model.get_inp_out_seqlens()
        
        
        
        inp_seq_ids = self.model.get_inp_seq_ids()
        
        arrive_times = np.asarray(arrive_times) - self.extra_cost
        to_sort = list(zip(arrive_times, inp_seq_ids))
        
        
        
        
        
        
        
        
        order = sorted(range(len(to_sort)), key=lambda i: to_sort[i])
        
        

        inp_lens = np.asarray(inp_lens)[order]
        out_lens = np.asarray(out_lens)[order]
        inp_seq_ids = np.asarray(inp_seq_ids)[order]
        arrive_times = arrive_times[order]
        self.inp_arrive_times = to_sort

        for dp_id in range(self.dp_size):
            
            self.dp_inp_lens_list[dp_id] = inp_lens[dp_id::self.dp_size]
            self.dp_out_lens_list[dp_id] = out_lens[dp_id::self.dp_size]
            self.dp_inp_seq_ids_list[dp_id] = inp_seq_ids[dp_id::self.dp_size]

        return inp_lens, out_lens, inp_seq_ids, arrive_times


    def _get_inp_key(
            self,
            arrive_times: List[float],):
        """
            The key contains the inp_lens, out_lens, and the arrive_times.
        """ 
        def _to_tuple(vs):
            return tuple([tuple([tuple(j) for j in i]) for i in vs])

        
        inp_lens_list = list()
        out_lens_list = list()
        arrive_times_list = list()
        for dp_id in range(self.dp_size):
            inp_lens_list.append(tuple(self.dp_inp_lens_list[dp_id]))
            out_lens_list.append(tuple(self.dp_out_lens_list[dp_id]))
            arrive_times_list.append(tuple(arrive_times[dp_id::self.dp_size]))
        return ((tuple(inp_lens_list), tuple(out_lens_list)), tuple(arrive_times_list))


    def _get_valid_max_latency(self, latency_list):
        
        
        print(f"exec plan: {self.model.get_base_model_ids(), self.get_key()}")
        
        latencys = np.asarray(latency_list)
        latencys = latencys[latencys < 1e8]
        if len(latencys) == 0:
            return 0
        else:
            return max(latencys)

    
    """
        Basic idea: 
            1. when generating outputs, we sort the output of all dp workers by (finish time, seq id)
            2. when querying available inputs, whether to sort all the available inputs is controlled by ``sort_input``
    """
    def estimate_exec_time(
            self, cost_table: CostTable,
            
            check_gap: int,
            sort_input: bool,
            arrive_times: List[float],
            
            ):
        '''
            Estimate the total inference of this model for the given inp_lens.
            Input:
                check_gap: query whether there are newly available requests every ``check_gap`` inference steps.
                sort_input: whether to sort the waiting requests when we query available requests.
                arrive_times: the arrive times of all input requests, extra_cost considered.
                extra_cost: the extra time before running the model, e.g., loading the LLM. [stored as self property]
            NOTE:
                1. to support data parallelism + model-level pipeline parallelism, we need limit each dp worker to 
                    query dp_id-th available request, i.e., we split arrive_times like we do to inp_lens. 
            NOTE: 
                2. the output total exec time considers the extra_cost (e.g., loading the LLM)
                3. the arrive times are in the order of mode.inp_seq_ids. SO we need to SORT them!!!
        '''
        
        
        
        
        

        inp_lens, out_lens, inp_seq_ids, arrive_times = self._sort_and_partition_data_parallel(arrive_times)

        
        

        
        
        
        
        

        
        
        key = (self.model.model_name, self.get_key(), *self._get_inp_key(arrive_times))
        self._FAKE_SCHEDULING_RES_key = key

        
        
        
        

        if key in _FAKE_SCHEDULING_RES:

            print(f"REUSE EXISTING FAKE SCHEDULING RESULTS!\n")

            (self.cumsum_latencys_list, self.cum_rng_nums_list, self.rng_starts_list, self.rng_ends_list,
                self.is_prefill_steps_list, self.finish_times_list, 
                self.throughput_till_each_iter_list) = _FAKE_SCHEDULING_RES[key]
            
            
            

            self.total_latency_list = [self._get_valid_max_latency(cumsum_latencys) if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]


            print(f"total_latency_list: {self.total_latency_list}, self.extra_cost: {self.extra_cost}")

            
            
            
            
            

            
            
            
            
            
            


            return self.total_latency_list
        else:

            print(f"DO FAKE SCHEDULING SEARCH!\n")
            

            time1 = time.perf_counter()
            
            self.fake_scheduling_res_ref = _entry_to_remote_fake_scheduling(
                self.dp_size,
                self.dp_inp_lens_list,
                self.dp_out_lens_list,
                arrive_times,
                
                check_gap,
                sort_input,
                max_seq_num=self.infer_args.max_seq_num,
                gpu_cache_block_num=self.gpu_cache_block_num,
                max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                block_size=self.infer_args.block_size, 
                
                
                model_path=self.model.model_path, 
                exec_plan_key=self.get_key_single_dp_worker(),
                sample_config=self.model.sample_config,
                trust_remote_code=self.model.trust_remote_code,
                revision=self.model.revision,
            )
            time2 = time.perf_counter()
            with open(f'tmp_time_logs.log', 'a') as f:
                f.write(f"TIME TO launch remote fake scheduling: exec_plan_key: {self.get_key()}: {time2 - time1}, ---abs: {time2}\n")
            return



           
            

            for dp_id in range(self.dp_size):
                
                
                
                
                
                dp_inp_lens = self.dp_inp_lens_list[dp_id]
                dp_out_lens = self.dp_out_lens_list[dp_id]            
                
                dp_seq_ids = self.dp_inp_seq_ids_list[dp_id]
                dp_arrive_times = arrive_times[dp_id::self.dp_size]

                
                
                
                
                
            
                
                
                
                
                
                
                

                
                
                
                
                
                


                
                (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list[dp_id], 
                    self.rng_starts_list[dp_id], self.rng_ends_list[dp_id], self.is_prefill_steps_list[dp_id], 
                    finish_times, self.throughput_till_each_iter_list[dp_id]) = \
                        fake_scheduling.fake_FCFS_schedule(
                            inp_lens=list(dp_inp_lens),out_lens=list(dp_out_lens), 
                            arrive_times=dp_arrive_times, check_gap=check_gap,
                            max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                            max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                            block_size=self.infer_args.block_size, 
                            sort_input=sort_input, 
                            cost_estimate_args={
                                "cost_table":cost_table, 
                                "model_name":self.model.model_path, 
                                "exec_plan":self.get_key_single_dp_worker(), 
                                "sample_config":self.model.sample_config, 
                                "trust_remote_code":self.model.trust_remote_code, 
                                "revision":self.model.revision})

                
                if len(self.cumsum_latencys_list[dp_id]) == 0:
                    self.total_latency_list[dp_id] = 0
                else:
                    
                    self.total_latency_list[dp_id] = self._get_valid_max_latency(self.cumsum_latencys_list[dp_id])


                
                
                

                
                
                
                self.finish_times_list[dp_id] = finish_times
            
            
            
            _FAKE_SCHEDULING_RES[key] = (self.cumsum_latencys_list, self.cum_rng_nums_list, 
                                         self.rng_starts_list, self.rng_ends_list, 
                                         self.is_prefill_steps_list, self.finish_times_list,
                                         self.throughput_till_each_iter_list) 
                                        

            
            
            
            
            


            return self.total_latency_list


    def get_total_latency_no_data_parallel(self, cost_table: CostTable):
        if self.total_latency == None:
            self.estimate_exec_time_no_data_parallel(cost_table)

        return self.total_latency


    
    
    def get_max_dp_latency_considering_plan_group(self, cost_table: CostTable,
            check_gap: int, sort_input: bool, arrive_times: List[float]):
        '''
            Input:
                extra_cost: the time to prepare (e.g., load) the model before running.
        '''

        print(f"exec_plan: {self.model.get_base_model_ids(), self.get_key()}")
        

        if self.total_latency_list[0] == None:
            self.estimate_exec_time(cost_table, 
                check_gap=check_gap, sort_input=sort_input, arrive_times=arrive_times)

        
        
        if self.fake_scheduling_res_ref != None:
            print("calling remote fake scheduling function-------")
            return None
        
        
        print(f"exec plan latency list: {(self.model.get_base_model_ids(), self.get_key()), self.total_latency_list}, self.extra_cost: {self.extra_cost}")
        
        

        return max(self.total_latency_list) + self.extra_cost
    


    def _wait_for_remote_fake_scheduling_deprecated(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        if self.fake_scheduling_res_ref != None:
            print(f"exec_plan: {str(self)}, start waiting for remote fake scheduling")
            ((self.cumsum_latencys_list, self.cum_rng_nums_list, 
            self.rng_starts_list, self.rng_ends_list, 
            self.is_prefill_steps_list, self.finish_times_list,), self.total_latency_list) = ray.get(self.fake_scheduling_res_ref)

            
            
            
            
            
            
            
            _FAKE_SCHEDULING_RES[self._FAKE_SCHEDULING_RES_key] = (self.cumsum_latencys_list, self.cum_rng_nums_list, 
                                         self.rng_starts_list, self.rng_ends_list, 
                                         self.is_prefill_steps_list, self.finish_times_list,) 


    def _wait_for_remote_fake_scheduling(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        if self.fake_scheduling_res_ref != None:
            with open(f'tmp_time_logs.log', 'a') as f:
                f.write(f"exec_plan: {str(self)}, start waiting for remote fake scheduling: ---abs: {time.perf_counter()}\n")

            for dp_id, dp_res_ref in enumerate(self.fake_scheduling_res_ref):
                (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list[dp_id], 
                        self.rng_starts_list[dp_id], self.rng_ends_list[dp_id], self.is_prefill_steps_list[dp_id], 
                        self.finish_times_list[dp_id], self.throughput_till_each_iter_list[dp_id]) = ray.get(dp_res_ref)
                
                
                if len(self.cumsum_latencys_list[dp_id]) == 0:
                    self.total_latency_list[dp_id] = 0
                else:
                    
                    self.total_latency_list[dp_id] = self._get_valid_max_latency(self.cumsum_latencys_list[dp_id])


            _FAKE_SCHEDULING_RES[self._FAKE_SCHEDULING_RES_key] = (self.cumsum_latencys_list, self.cum_rng_nums_list, 
                                         self.rng_starts_list, self.rng_ends_list, 
                                         self.is_prefill_steps_list, self.finish_times_list,
                                         self.throughput_till_each_iter_list)
            
            time2 = time.perf_counter()
            with open(f'tmp_time_logs.log', 'a') as f:
                f.write(f"TIME TO ESTIMATE COST: exec_plan_key: {self.get_key()}: ---abs: {time2}\n")

        



    def wait_for_remote_fake_scheduling_and_get_max_dp_latency_considering_plan_group(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        self._wait_for_remote_fake_scheduling()
           
        return max(self.total_latency_list) + self.extra_cost


    
    def get_max_dp_latency(self, cost_table: CostTable, sort_input: bool):
        """
            This function is only used in the baseline where we select the best exec plan for each LLM independently.
            NOTE: 
                1. as we select the best exec plan for each LLM independently, 
                we assume all input requests are available.
                i.e., no model-level pipeline is considered here.
        """
        if self.total_latency_list[0] == None:
            arrive_times = [-1]*len(self.model.get_inp_seq_ids())
            self.estimate_exec_time(cost_table, 
                check_gap=1, sort_input=sort_input, arrive_times=arrive_times)

        
        if self.fake_scheduling_res_ref != None:
            print("calling remote fake scheduling function-------")
            return None


        return max(self.total_latency_list)


    def wait_for_remote_fake_scheduling_and_get_max_dp_latency(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        self._wait_for_remote_fake_scheduling()
        
        return max(self.total_latency_list)



    def update_inp_out_seqlens_and_throughput_after_an_infer_stage_no_data_parallel(
            self, stop_time: float, cost_table: CostTable):
        '''
            1. compute valid throughput.
            2. Update the remaining seqlens after it finishes the current infer stage (until stop_time).
        '''

        
        
        stop_time = min(self.cumsum_latencys[-1], stop_time)

        
        stop_iter_i = np.searchsorted(self.cumsum_latencys, stop_time, side='left')
        if stop_iter_i in self.cache_stop_time_info:

            

            return self.cache_stop_time_info[stop_iter_i], stop_iter_i

        actual_stop_time = self.cumsum_latencys[stop_iter_i]

        
        
        
        


        
        finished_lens = fake_scheduling.get_info_at_stop_time(
            self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends, 
            stop_time, stop_iter_i)
        
        
        
        


        
        
        
        
        
        


        
        inp_lens, out_lens = self.model.get_inp_out_seqlens()

        
        print(self)

        valid_throughput = fake_scheduling.comp_valid_throughput_at_stop_time(
            inp_lens,
            finished_lens, actual_stop_time, cost_table,
            self.model.model_path, self.model.trust_remote_code, self.model.revision)

        
        inp_lens = np.asarray(inp_lens) + np.asarray(finished_lens)
        remaining_lens = np.asarray(out_lens) - finished_lens
        valid_indices = (remaining_lens>0)


        

        self.cache_stop_time_info[stop_iter_i] = \
            [(tuple(inp_lens[valid_indices]), tuple(remaining_lens[valid_indices])), valid_throughput]



        

        return self.cache_stop_time_info[stop_iter_i], stop_iter_i
        return (tuple(inp_lens[valid_indices]), tuple(remaining_lens[valid_indices])), valid_throughput











    def get_throughput_at_stop_time_based_on_cached_throughputs(
        self, stage_stop_time: float) -> float:
        """
            return the throughput of the model at the stop time, considering the extra time.
        """
        
        
        throughput = 0
        if stage_stop_time < 0:
            return throughput

        for dp_id in range(self.dp_size):
            cumsum_latencys = self.cumsum_latencys_list[dp_id]

            if len(cumsum_latencys) == 0:
                throughput += 0
                continue                

            
            
            stop_time = min(cumsum_latencys[-1], stage_stop_time)

            stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')
            
            
            
            
            
            flops = self.throughput_till_each_iter_list[dp_id][stop_iter_i] * cumsum_latencys[stop_iter_i]
            throughput += (flops / (stage_stop_time + self.extra_cost))
        
        return throughput

















    
    
    def update_inp_out_seqlens_and_throughput_after_an_infer_stage(
            self, stop_time: float, cost_table: CostTable):
        '''
            1. compute valid throughput.
            2. Update the remaining seqlens after it finishes the current infer stage (until stop_time).
        '''
        new_inp_out_lens_list, valid_throughput_list, stop_iter_i_list = list(), list(), list()
        stage_stop_time = stop_time

        for dp_id in range(self.dp_size):
            cumsum_latencys = self.cumsum_latencys_list[dp_id]
            cache_stop_time_info = self.cache_stop_time_info_list[dp_id]
            cum_rng_nums = self.cum_rng_nums_list[dp_id]
            rng_starts = self.rng_starts_list[dp_id]
            rng_ends = self.rng_ends_list[dp_id]

            if len(cumsum_latencys) == 0:
                new_inp_out_lens_list.append((tuple(np.asarray([])), tuple(np.asarray([])), np.asarray([], dtype=np.int64)))
                valid_throughput_list.append(0)
                stop_iter_i_list.append(0)
                continue                

            
            
            stop_time = min(cumsum_latencys[-1], stage_stop_time)

            

            
            stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')

            
            
            
            
            

            if stop_iter_i in cache_stop_time_info:

                print(f"reuse stop iter i information\n")

                

                new_inp_out_lens, valid_throughput = cache_stop_time_info[stop_iter_i]
                
                
                

                new_inp_out_lens_list.append(new_inp_out_lens)
                valid_throughput_list.append(valid_throughput)
                stop_iter_i_list.append(stop_iter_i)
                continue

                

            actual_stop_time = cumsum_latencys[stop_iter_i]

            
            
            
            


            
            finished_lens = fake_scheduling.get_info_at_stop_time(
                cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, 
                stop_time, stop_iter_i)

            
            
            
            
            


            
            
            
            
            
            


            
            
            dp_inp_lens = self.dp_inp_lens_list[dp_id]
            dp_out_lens = self.dp_out_lens_list[dp_id]


            
            

            
            print(self)

            valid_throughput = fake_scheduling.comp_valid_throughput_at_stop_time(
                dp_inp_lens,
                finished_lens, actual_stop_time, cost_table,
                self.model.model_path, self.model.trust_remote_code, self.model.revision)

            
            dp_inp_lens = np.asarray(dp_inp_lens) + np.asarray(finished_lens)
            remaining_lens = np.asarray(dp_out_lens) - finished_lens
            
            valid_indices = np.nonzero(remaining_lens>0)[0]

            
            

            

            
            cache_stop_time_info[stop_iter_i] = \
                [(tuple(dp_inp_lens[valid_indices]), tuple(remaining_lens[valid_indices]), valid_indices), \
                 valid_throughput]

            new_inp_out_lens_list.append(cache_stop_time_info[stop_iter_i][0])
            valid_throughput_list.append(valid_throughput)
            stop_iter_i_list.append(stop_iter_i)

            
            
            
            
            
            
            


        

        return (new_inp_out_lens_list, valid_throughput_list), stop_iter_i_list








    def update_fake_schedule_output_after_an_infer_stage_no_data_parallel(
            self, 
            old_inp_lens: List[int], new_inp_lens: List[int], new_out_lens: List[int], 
            stop_iter_i: int, cost_table: CostTable):
        '''
            This function is called when the exec plan is selected to run for an infer stage.
            Update:
                _FAKE_SCHEDULING_RES[model_name, exec_plan, new_inp_lens, new_out_lens]
        '''
        
        cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps = \
            fake_scheduling.update_fake_FCFS_schedule_metadata(
                old_inp_lens, new_inp_lens,
                self.cumsum_latencys, self.cum_rng_nums, self.rng_starts, self.rng_ends, 
                self.is_prefill_steps,
                self.infer_args.max_num_batched_tokens, stop_iter_i,
                cost_table, 
                model_name=self.model.model_path, 
                exec_plan=self.get_key(), sample_config=self.model.sample_config, 
                trust_remote_code=self.model.trust_remote_code, revision=self.model.revision
                )
        
        new_key = (self.model.model_name, self.get_key(), (tuple(new_inp_lens), tuple(new_out_lens)))
        _FAKE_SCHEDULING_RES[new_key] = cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps






    def _sort_scheduling_results_by_seq_ids(
            self,
            alive_seq_ids,
            cum_rng_nums, rng_starts, rng_ends, finish_times_of_alive_seqs):
        order = np.argsort(alive_seq_ids)
        new_rng_starts = np.empty_like(rng_starts)
        new_rng_ends = np.empty_like(rng_ends)
        rng_nums = np.diff(cum_rng_nums)[order]
        new_cum_rng_nums = np.cumsum(np.concatenate(([0], rng_nums)))
        for i, ori_ind in enumerate(order):
            new_rng_starts[new_cum_rng_nums[i]:new_cum_rng_nums[i+1]] = \
                rng_starts[cum_rng_nums[ori_ind]:cum_rng_nums[ori_ind+1]]
            new_rng_ends[new_cum_rng_nums[i]:new_cum_rng_nums[i+1]] = \
                rng_ends[cum_rng_nums[ori_ind]:cum_rng_nums[ori_ind+1]]
        new_finish_times_of_alive_seqs = finish_times_of_alive_seqs[order]
        return new_cum_rng_nums, new_rng_starts, new_rng_ends, new_finish_times_of_alive_seqs



    
    def update_fake_schedule_output_after_an_infer_stage(
            self, 
            old_inp_lens_list: List[List[int]], 
            new_inp_out_lens_list: List[List[List[int]]], 
            new_inp_lens_merged: List[int],
            new_out_lens_merged: List[int],
            new_inp_seq_ids_merged: List[int],
            stop_iter_i_list: List[int], 
            cost_table: CostTable, 
            ):
        '''
            This function is called when the exec plan is selected to run for an infer stage.
            Update:
                _FAKE_SCHEDULING_RES[model_name, exec_plan, new_inp_lens, new_out_lens]
            NOTE:
                we only call this function when after an infer stage, all the requests are available.
            NOTE: 
                1. ``new_inp_lens_merged`` is already sorted by seq ids.
                2. we only do update when there are unfinished requests.
        '''
        
        
        
        
        

        
        if (max([i[0] for i in self.inp_arrive_times]) > \
            max([cumsum_latencys[stop_iter_i] if len(cumsum_latencys) > 0 else 0 \
                 for cumsum_latencys, stop_iter_i in \
                 zip(self.cumsum_latencys_list, stop_iter_i_list)])):
            
            return


        
        
        



        cumsum_latencys_list = [list() for _ in range(self.dp_size)]
        cum_rng_nums_list = [list() for _ in range(self.dp_size)]
        rng_starts_list = [list() for _ in range(self.dp_size)]
        rng_ends_list = [list() for _ in range(self.dp_size)]
        is_prefill_steps_list = [list() for _ in range(self.dp_size)]
        
        finish_times_list = [list() for _ in range(self.dp_size)]
        throughput_till_each_iter_list = [list() for _ in range(self.dp_size)]

        for dp_id in range(self.dp_size):
            
            old_inp_lens = old_inp_lens_list[dp_id]
            
            stop_iter_i = stop_iter_i_list[dp_id]

            
            (cumsum_latencys_list[dp_id], cum_rng_nums_list[dp_id], 
                rng_starts_list[dp_id], rng_ends_list[dp_id], is_prefill_steps_list[dp_id], 
                finish_times_of_alive_seqs, alive_old_indices,
                throughput_till_each_iter_list[dp_id]) = \
                    fake_scheduling.update_fake_FCFS_schedule_metadata(
                        old_inp_lens, 
                        self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list[dp_id], 
                        self.rng_starts_list[dp_id], self.rng_ends_list[dp_id], 
                        self.is_prefill_steps_list[dp_id],
                        self.throughput_till_each_iter_list[dp_id],
                        self.infer_args.max_num_batched_tokens, stop_iter_i,
                        cost_table, 
                        model_name=self.model.model_path, 
                        exec_plan=self.get_key_single_dp_worker(), sample_config=self.model.sample_config, 
                        trust_remote_code=self.model.trust_remote_code, revision=self.model.revision
                        )
            

            
            
            
            
            
            
            
            


            
            
            
            
            

            
            
            finish_times_list[dp_id] = finish_times_of_alive_seqs
            assert (new_inp_out_lens_list[dp_id][2] == alive_old_indices).all(), print(new_inp_out_lens_list[dp_id][2], alive_old_indices)


        
        
        
        
        new_key = (self.model.model_name, self.get_key(), 
                   (tuple([tuple(dp_data[0]) for dp_data in new_inp_out_lens_list]), tuple([tuple(dp_data[1]) for dp_data in new_inp_out_lens_list])), \
                   tuple([tuple([-1 - self.extra_cost]*len(dp_data[0])) for dp_data in new_inp_out_lens_list]))
        
        
        _FAKE_SCHEDULING_RES[new_key] = \
            cumsum_latencys_list, cum_rng_nums_list, rng_starts_list, rng_ends_list, is_prefill_steps_list, \
            finish_times_list, throughput_till_each_iter_list
            





    def get_key(self):
        
        
        
        return (self.num_worker, self.mem_per_comp_gpu, self.wld_degree, self.cache_gpu_num, self.dp_size)
    
    def get_key_single_dp_worker(self):
        
        return (self.num_worker, self.mem_per_comp_gpu, self.wld_degree, self.cache_gpu_num)


    def __str__(self) -> str:
        return f"{str(self.model)}, "\
            f"{self.get_key()}"
            
            
            







class MyVerticalFusedExecPlan(MyExecPlan):
    """ 
        My execution plan definition. There may be multiple models fused vertically in this plan. 
        NOTE: we assume the original complete inp seq ids for each model is range(ori_tot_req_num)!
    """
    def __init__(
        self,
        
        fused_model: MyFusedModelInfor,
        
        shared_exec_plan: MyExecPlan,
        
    ) -> None:
        self.model:MyFusedModelInfor = fused_model
        self.model_list: List[MyModelInfor] = fused_model.model_list
        

        exec_plan_0 = shared_exec_plan
        self.num_worker = exec_plan_0.num_worker
        self.wld_degree = exec_plan_0.wld_degree
        self.cache_gpu_num = exec_plan_0.cache_gpu_num
        self.mem_per_comp_gpu = exec_plan_0.mem_per_comp_gpu
        self.dp_size = exec_plan_0.dp_size
        self.param_byte_per_comp_gpu = exec_plan_0.param_byte_per_comp_gpu
        self.param_byte_per_cache_gpu = exec_plan_0.param_byte_per_cache_gpu
        self.gpu_cache_byte_per_block = exec_plan_0.gpu_cache_byte_per_block

        
        self.infer_args: InferenceArgs = exec_plan_0.infer_args
        
        self.tot_gpu_mem_byte: int = exec_plan_0.tot_gpu_mem_byte
        
        self.basic_mem_consumption: int = exec_plan_0.basic_mem_consumption
        self.gpu_cache_block_num = exec_plan_0.gpu_cache_block_num
        


        
        
        
        self.cumsum_latencys_list: List[List[float]] = [list() for _ in range(self.dp_size)]
        self.cum_rng_nums_list_for_models: List[List[int]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.rng_starts_list_for_models: List[List[int]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.rng_ends_list_for_models: List[List[int]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.is_prefill_steps_list: List[List[bool]] = [list() for _ in range(self.dp_size)]

        self.total_latency_list: List[Optional[float]] = [None for _ in range(self.dp_size)]

        self.cache_stop_time_info_list: List[Dict[int, Tuple[Tuple[List[int], List[int]], float]]] \
            = [dict() for _ in range(self.dp_size)]
        
        
        self.dp_inp_lens_list_for_models: List[List[List[int]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.dp_out_lens_list_for_models: List[List[List[int]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        
        

        
        
        
        self.finish_times_merged_for_models: List[List[float]] = [list() for _ in self.model_list]
        self.finish_times_list_for_models: List[List[List[float]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        
        
        
        self.inp_arrive_times_for_models: List[List[Tuple[float, int]]] = [list() for _ in self.model_list] 
        self.extra_cost: float = 0.0
        
        self.dp_inp_seq_ids_list_for_models: List[List[List[int]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]
        self.dp_arrive_times_list_for_models: List[List[List[float]]] = \
            [[list() for model in self.model_list] for _ in range(self.dp_size)]

        
        self.throughput_till_each_iter_list: List[List[float]] = [list() for _ in range(self.dp_size)]


        self._FAKE_SCHEDULING_RES_key = None
        self.fake_scheduling_res_ref = None


        
        
        

        self.load_cost_just_for_refer = None 


    def copy_the_plan(self):
        shared_exec_plan = MyExecPlan(
            self.model, 
            self.num_worker, 
            self.wld_degree, 
            self.cache_gpu_num, 
            self.mem_per_comp_gpu, 
            self.dp_size, 
            self.param_byte_per_comp_gpu, 
            self.param_byte_per_cache_gpu,
            self.gpu_cache_byte_per_block,
            self.infer_args,
            self.tot_gpu_mem_byte,
        )
        return MyVerticalFusedExecPlan(self.model, shared_exec_plan)




    def get_base_model_ids(self):
        return self.model.get_base_model_ids()

    
    
    
    def get_base_models(self):
        return self.model_list
    
    def models_not_started(self):
        
        return self.model.not_started()
        
    def get_dp_inp_lens_list_for_models(self):
        return self.dp_inp_lens_list_for_models

    def get_finish_times_merged_limited_version(self, seq_ids: List[int]):


        if len(self.finish_times_merged_for_models[-1]) == 0:
            self.finish_times_merged_for_models[-1] = np.asarray([-1-self.extra_cost]*self.model_list[-1].ori_tot_inp_num) 
            
            for dp_inp_seq_ids_for_models, finish_times_for_models in \
                zip(self.dp_inp_seq_ids_list_for_models, self.finish_times_list_for_models):
                inds = np.searchsorted(self.model_list[-1].ori_inp_seq_ids, dp_inp_seq_ids_for_models[-1])
                self.finish_times_merged_for_models[-1][inds] = finish_times_for_models[-1]
        
        inds = np.searchsorted(self.model_list[-1].ori_inp_seq_ids, seq_ids)
        return self.finish_times_merged_for_models[-1][inds]




    def get_finish_times_merged(self, seq_ids: List[int], model_ind: int):
        """
            Input: 
                model_ind: we want to get the seq finish times of the ``model_ind``-th base model of the fused model 
        """

        if len(self.finish_times_merged_for_models[model_ind]) == 0:
            self.finish_times_merged_for_models[model_ind] = np.asarray([-1-self.extra_cost]*self.model_list[model_ind].ori_tot_inp_num) 
            
            for dp_inp_seq_ids_for_models, finish_times_for_models in \
                zip(self.dp_inp_seq_ids_list_for_models, self.finish_times_list_for_models):
                inds = np.searchsorted(self.model_list[model_ind].ori_inp_seq_ids, dp_inp_seq_ids_for_models[model_ind])
                
                
                
                
                
                
                self.finish_times_merged_for_models[model_ind][inds] = finish_times_for_models[model_ind]

        
        
        

        


        return get_infor_given_seq_ids(
            values=self.finish_times_merged_for_models[model_ind], 
            seq_ids_we_have=self.model_list[model_ind].ori_inp_seq_ids, 
            seq_ids_requested=seq_ids, 
            default_value=-1-self.extra_cost)


        inds = np.searchsorted(self.model_list[model_ind].ori_inp_seq_ids, seq_ids)
        return self.finish_times_merged_for_models[model_ind][inds]




    def merge_new_inp_out_lens_of_data_parallel_workers(
            self, new_inp_out_lens_list_for_models: List[List[List[List[int]]]]
        )->List[List[int]]:
        """
            Support vertical fusion of models.
        """
        
        
        

        print(f"{str(self)}")
        
        
        data_num = len(new_inp_out_lens_list_for_models[0])
        new_inp_lens_for_models = list()
        new_out_lens_for_models = list()
        new_inp_seq_ids_for_models = list()
        for i in range(len(self.model_list)):
            new_inp_out_lens_list = [[dp_data[data_i][i] for data_i in range(data_num)] for dp_data in new_inp_out_lens_list_for_models]
            dp_inp_seq_ids_list = [dp_data[i] for dp_data in self.dp_inp_seq_ids_list_for_models]

            
            

            inp_lens_list = [dp_inp_out_lens[0] for dp_inp_out_lens in new_inp_out_lens_list]
            out_lens_list = [dp_inp_out_lens[1] for dp_inp_out_lens in new_inp_out_lens_list]
            inp_seq_ids_list = [dp_inp_seq_ids[dp_inp_out_lens[2]] \
                                for dp_inp_seq_ids, dp_inp_out_lens \
                                    in zip(dp_inp_seq_ids_list, new_inp_out_lens_list)]
            inp_lens = np.concatenate(inp_lens_list)
            out_lens = np.concatenate(out_lens_list)
            inp_seq_ids = np.concatenate(inp_seq_ids_list)
            
            
            
            
            

            order = np.argsort(inp_seq_ids)

            

            inp_lens = inp_lens[order]
            out_lens = out_lens[order]
            inp_seq_ids = inp_seq_ids[order]
            new_inp_lens_for_models.append(inp_lens)
            new_out_lens_for_models.append(out_lens)
            new_inp_seq_ids_for_models.append(inp_seq_ids)            
        return [new_inp_lens_for_models, new_out_lens_for_models, new_inp_seq_ids_for_models]



    def set_gpu_cache_block_num(self):
        '''
            gpu cache block num = (available gpu mem - parameter mem) // mem per block
        '''

        """
            We will not call this method directly on a fused exec plan object.
        """
        assert False




    
    
    
    
    
    

    

    def _sort_and_partition_data_parallel(
            self, 
            arrive_times_list: List[List[float]],
            ):
        """
            Update:
                self.dp_inp_seq_ids_list_for_models, 
                self.dp_inp_lens_list_for_models,
                self.dp_out_lens_list_for_models,
                self.dp_arrive_times_list_for_models
        """

        def _sort_and_assign(seq_infos, model_i: int, dp_size: int):
            """
                sort the seqs and assign them to the dp workers.
            """
            inp_lens = [i[0] for i in seq_infos]
            out_lens = [i[1] for i in seq_infos]
            inp_seq_ids = [i[2] for i in seq_infos]
            arrive_times = [i[3] for i in seq_infos]

            to_sort = list(zip(arrive_times, inp_seq_ids))
            order = sorted(range(len(to_sort)), key=lambda i: to_sort[i])
            inp_lens = np.asarray(inp_lens)[order]
            out_lens = np.asarray(out_lens)[order]
            inp_seq_ids = np.asarray(inp_seq_ids)[order]
            arrive_times = np.asarray(arrive_times)[order]

            

            for dp_id in range(dp_size):
                
                self.dp_inp_lens_list_for_models[dp_id][model_i].extend(inp_lens[dp_id::dp_size])
                self.dp_out_lens_list_for_models[dp_id][model_i].extend(out_lens[dp_id::dp_size])
                self.dp_inp_seq_ids_list_for_models[dp_id][model_i].extend(inp_seq_ids[dp_id::dp_size])
                self.dp_arrive_times_list_for_models[dp_id][model_i].extend(arrive_times[dp_id::dp_size])
        
        def _assign_in_consistent_with_previous_assignment(seq_infos, model_i: int, dp_size: int):
            """
                assign seqs to the dp worker which generates the their previous versions.
            """
            inp_lens = np.asarray([i[0] for i in seq_infos])
            out_lens = np.asarray([i[1] for i in seq_infos])
            inp_seq_ids = np.asarray([i[2] for i in seq_infos])
            arrive_times = np.asarray([i[3] for i in seq_infos])
            order = np.argsort(inp_seq_ids)
            for dp_id in range(dp_size):
                last_model_seq_ids = self.dp_inp_seq_ids_list_for_models[dp_id][model_i-1]
                seq_ids_to_add = sorted(set(inp_seq_ids).intersection(last_model_seq_ids))
                inds = np.searchsorted(inp_seq_ids[order], seq_ids_to_add)
                self.dp_inp_lens_list_for_models[dp_id][model_i].extend(inp_lens[order][inds])
                self.dp_out_lens_list_for_models[dp_id][model_i].extend(out_lens[order][inds])
                self.dp_inp_seq_ids_list_for_models[dp_id][model_i].extend(inp_seq_ids[order][inds])
                self.dp_arrive_times_list_for_models[dp_id][model_i].extend(arrive_times[order][inds])


        def _sort_by_seq_ids(dp_size: int):
            """
                Sort the seq info lists by seq ids.
            """
            for dp_id in range(dp_size):
                for model_i in range(len(self.model_list)):
                    order = np.argsort(self.dp_inp_seq_ids_list_for_models[dp_id][model_i])
                    self.dp_inp_lens_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_inp_lens_list_for_models[dp_id][model_i])[order]
                    self.dp_out_lens_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_out_lens_list_for_models[dp_id][model_i])[order]
                    self.dp_inp_seq_ids_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_inp_seq_ids_list_for_models[dp_id][model_i])[order]
                    self.dp_arrive_times_list_for_models[dp_id][model_i] = \
                        np.asarray(self.dp_arrive_times_list_for_models[dp_id][model_i])[order]




        seq_ids_visited = list()
        

        for i in range(len(self.model_list)):
            
            known = list()
            unknown = list()
            model = self.model_list[i]
            inp_lens, out_lens = model.get_inp_out_seqlens()
            inp_seq_ids = model.get_inp_seq_ids()
            for inp_len, out_len, seq_id, arrive_time in zip(inp_lens, out_lens, inp_seq_ids, arrive_times_list[i]):
                if seq_id not in seq_ids_visited:
                    unknown.append((inp_len, out_len, seq_id, arrive_time - self.extra_cost))
                    seq_ids_visited.append(seq_id)
                else:
                    known.append((inp_len, out_len, seq_id, arrive_time - self.extra_cost))
            
            
            

            
            
            assert sorted(inp_seq_ids) == list(inp_seq_ids)
            
            _sort_and_assign(unknown, i, self.dp_size)
            
            _assign_in_consistent_with_previous_assignment(known, i, self.dp_size)


        
        _sort_by_seq_ids(self.dp_size)


    def _get_inp_key_merged_version(
            self,
            arrive_times_list: List[List[float]],):
        """
            The key contains the inp_lens, out_lens, and the arrive_times.
        """ 
        def _to_tuple(vs):
            return tuple([tuple([tuple(j) for j in i]) for i in vs])

        
        inp_lens = list()
        out_lens = list()
        arrive_times = list()
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            inps, outs = model.get_inp_out_seqlens()
            inp_lens.append(tuple(inps))
            out_lens.append(tuple(outs))
            arrive_times.append(tuple(np.asarray(arrive_times_list[i]) - self.extra_cost))
        return ((tuple(inp_lens), tuple(out_lens)), tuple(arrive_times))

        tuple_inp_lens = _to_tuple(self.dp_inp_lens_list_for_models)
        tuple_out_lens = _to_tuple(self.dp_out_lens_list_for_models)
        tuple_arrive_times = _to_tuple(self.dp_arrive_times_list_for_models)
        return ((tuple_inp_lens, tuple_out_lens), tuple_arrive_times)




    def _get_inp_key(
            self,
            arrive_times_list: List[List[float]],):
        """
            The key contains the inp_lens, out_lens, and the arrive_times.
        """ 
        def _to_tuple(vs):
            return tuple([tuple([tuple(j) for j in i]) for i in vs])

        
        
        
        
        
        
        
        
        
        
        

        tuple_inp_lens = _to_tuple(self.dp_inp_lens_list_for_models)
        tuple_out_lens = _to_tuple(self.dp_out_lens_list_for_models)
        tuple_arrive_times = _to_tuple(self.dp_arrive_times_list_for_models)
        return ((tuple_inp_lens, tuple_out_lens), tuple_arrive_times)




    
    """
        Basic idea: 
            1. when generating outputs, we sort the output of all dp workers by (finish time, seq id)
            2. when querying available inputs, whether to sort all the available inputs is controlled by ``sort_input``
    """
    def estimate_exec_time(
            self, cost_table: CostTable,
            
            check_gap: int,
            sort_input: bool,
            
            arrive_times_list: List[List[float]],
            
            ):
        '''
            Estimate the total inference of this model for the given inp_lens.
            Input:
                check_gap: query whether there are newly available requests every ``check_gap`` inference steps.
                sort_input: whether to sort the waiting requests when we query available requests.
                arrive_times: the arrive times of all input requests, extra_cost considered.
                extra_cost: the extra time before running the model, e.g., loading the LLM. [stored as self property]
            NOTE:
                1. to support data parallelism + model-level pipeline parallelism, we need limit each dp worker to 
                    query dp_id-th available request, i.e., we split arrive_times like we do to inp_lens. 
            NOTE: 
                2. the output total exec time considers the extra_cost (e.g., loading the LLM)
                3. the arrive times are in the order of mode.inp_seq_ids. SO we need to SORT them!!!
        '''
        
        
        
        

        

        
        
        


        time1 = time.perf_counter()

        self._sort_and_partition_data_parallel(arrive_times_list)
       
        time2 = time.perf_counter()
        print(f"TIME--_sort_and_partition_data_parallel: {time2 - time1}")

        
        

        
        
        
        
        
        

        
        
        key = (self.model.model_name, self.get_key(), *(self._get_inp_key(arrive_times_list)))
        self._FAKE_SCHEDULING_RES_key = key

        
        if key in _FAKE_SCHEDULING_RES:

            print(f"Reuse fake scheduling results")

            (self.cumsum_latencys_list, 
             self.cum_rng_nums_list_for_models, self.rng_starts_list_for_models, self.rng_ends_list_for_models,
             self.is_prefill_steps_list, 
             self.finish_times_list_for_models, 
             self.throughput_till_each_iter_list) = _FAKE_SCHEDULING_RES[key]
            
            
            

            self.total_latency_list = [self._get_valid_max_latency(cumsum_latencys) if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]

            
            
            
            
            
            
            
            
            
            

            time3 = time.perf_counter()
            print(f"TIME--reuse fake scheduling: {time3 - time2}")

            


            return self.total_latency_list
        else:

            print(f"DO FAKE SCHEDULING SEARCH!\n")

            time1 = time.perf_counter()

            self.fake_scheduling_res_ref = _entry_to_remote_fake_scheduling_vertical_fusion(
                self.dp_size,
                self.dp_inp_lens_list_for_models,
                self.dp_out_lens_list_for_models,
                self.dp_inp_seq_ids_list_for_models,
                self.dp_arrive_times_list_for_models,
                
                check_gap,
                sort_input,
                max_seq_num=self.infer_args.max_seq_num,
                gpu_cache_block_num=self.gpu_cache_block_num,
                max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                block_size=self.infer_args.block_size,
                
                
                model_path=self.model.model_path, 
                exec_plan_key=self.get_key_single_dp_worker(),
                sample_config=self.model.sample_config,
                trust_remote_code=self.model.trust_remote_code,
                revision=self.model.revision
            )

            time2 = time.perf_counter()
            with open(f'tmp_time_logs.log', 'a') as f:
                f.write(f"TIME TO launch remote fake scheduling vertical: exec_plan_key: {self.get_key()}: {time2 - time1}, ---abs: {time2}\n")
            return


            
            
            
            
            
            
            
            
            


            for dp_id in range(self.dp_size):
                
                dp_inp_lens_for_models = self.dp_inp_lens_list_for_models[dp_id]
                dp_out_lens_for_models = self.dp_out_lens_list_for_models[dp_id]            
                dp_seq_ids_for_models = self.dp_inp_seq_ids_list_for_models[dp_id]
                dp_arrive_times_for_models = self.dp_arrive_times_list_for_models[dp_id]


                time3 = time.perf_counter()


                
                
                
                (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list_for_models[dp_id], 
                    self.rng_starts_list_for_models[dp_id], self.rng_ends_list_for_models[dp_id], 
                    self.is_prefill_steps_list[dp_id], 
                    self.finish_times_list_for_models[dp_id], 
                    self.throughput_till_each_iter_list[dp_id]) = \
                        fake_scheduling.fake_FCFS_schedule_vertical_fuse(
                            inp_lens=list(dp_inp_lens_for_models[0]),out_lens=list(dp_out_lens_for_models[0]), 
                            arrive_times=dp_arrive_times_for_models[0], 
                            ref_seq_ids=dp_seq_ids_for_models[0],
                            
                            ref_seq_ids_list=dp_seq_ids_for_models[1:],
                            inp_lens_list=dp_inp_lens_for_models[1:],
                            out_lens_list=dp_out_lens_for_models[1:],
                            arrive_times_list=dp_arrive_times_for_models[1:],
                            
                            check_gap=check_gap,
                            max_seq_num=self.infer_args.max_seq_num, max_block_num=self.gpu_cache_block_num, 
                            max_num_batched_tokens=self.infer_args.max_num_batched_tokens,
                            block_size=self.infer_args.block_size, 
                            sort_input=sort_input, 
                            cost_estimate_args={
                                "cost_table":cost_table, 
                                "model_name":self.model.model_path, 
                                "exec_plan":self.get_key_single_dp_worker(), 
                                "sample_config":self.model.sample_config, 
                                "trust_remote_code":self.model.trust_remote_code, 
                                "revision":self.model.revision}
                            )

                
                if len(self.cumsum_latencys_list[dp_id]) == 0:
                    self.total_latency_list[dp_id] = 0
                else:
                    
                    self.total_latency_list[dp_id] = self._get_valid_max_latency(self.cumsum_latencys_list[dp_id])

                time4 = time.perf_counter()
                print(f"TIME--fake scheduling vertical: {time4 - time3}")

            
            
            
            _FAKE_SCHEDULING_RES[key] = (self.cumsum_latencys_list, self.cum_rng_nums_list_for_models, 
                                         self.rng_starts_list_for_models, self.rng_ends_list_for_models, 
                                         self.is_prefill_steps_list, self.finish_times_list_for_models,
                                         self.throughput_till_each_iter_list) 
                                        

            
            
            
            
            


            time5 = time.perf_counter()  
            print(f"TIME--estimate time cost: {time5 - time1}")

            


            
            if time5-time1 > 20:
                print(f"self.dp_inp_lens_list_for_models: {self.dp_inp_lens_list_for_models}")
                print(f"self.dp_out_lens_list_for_models: {self.dp_out_lens_list_for_models}")
                print(f"self.dp_arrive_times_list_for_models: {self.dp_arrive_times_list_for_models}")

            return self.total_latency_list



    
    
    def get_max_dp_latency_considering_plan_group(self, cost_table: CostTable,
            check_gap: int, sort_input: bool, arrive_times_list: List[List[float]]):
        '''
            Input:
                extra_cost: the time to prepare (e.g., load) the model before running.
        '''

        print(f"exec_plan: {self.model.get_base_model_ids(), self.get_key()}")
        
        

        if self.total_latency_list[0] == None:
            self.estimate_exec_time(cost_table, 
                check_gap=check_gap, sort_input=sort_input, arrive_times_list=arrive_times_list)


        
        if self.fake_scheduling_res_ref != None:
            print("calling remote fake scheduling function vertical -------")
            return None


        print(f"exec plan latency list: {(self.model.get_base_model_ids(), self.get_key()), self.total_latency_list}, self.extra_cost: {self.extra_cost}")
        
        

        return max(self.total_latency_list) + self.extra_cost




    def _wait_for_remote_fake_scheduling(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        if self.fake_scheduling_res_ref != None:
            with open(f'tmp_time_logs.log', 'a') as f:
                f.write(f"exec_plan: {str(self)}, start waiting for remote fake scheduling vertical: ---abs: {time.perf_counter()}\n")

            for dp_id, dp_res_ref in enumerate(self.fake_scheduling_res_ref):               
                (self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list_for_models[dp_id], 
                    self.rng_starts_list_for_models[dp_id], self.rng_ends_list_for_models[dp_id], 
                    self.is_prefill_steps_list[dp_id], 
                    self.finish_times_list_for_models[dp_id], self.throughput_till_each_iter_list[dp_id]) = ray.get(dp_res_ref)


                
                
                
                
                


                
                if len(self.cumsum_latencys_list[dp_id]) == 0:
                    self.total_latency_list[dp_id] = 0
                else:
                    
                    self.total_latency_list[dp_id] = self._get_valid_max_latency(self.cumsum_latencys_list[dp_id])


            
            
            _FAKE_SCHEDULING_RES[self._FAKE_SCHEDULING_RES_key] = (self.cumsum_latencys_list, self.cum_rng_nums_list_for_models, 
                                         self.rng_starts_list_for_models, self.rng_ends_list_for_models, 
                                         self.is_prefill_steps_list, self.finish_times_list_for_models,
                                         self.throughput_till_each_iter_list) 
            
            time2 = time.perf_counter()
            with open(f'tmp_time_logs.log', 'a') as f:
                f.write(f"TIME TO ESTIMATE COST vertical: exec_plan_key: {self.get_key()}: ---abs: {time2}\n")

        



    
    def get_max_dp_latency(self, cost_table: CostTable, sort_input: bool):
        """
            This function is only used in the baseline where we select the best exec plan for each LLM independently.
            NOTE: 
                1. as we select the best exec plan for each LLM independently, 
                we assume all input requests are available.
                i.e., no model-level pipeline is considered here.
        """
        if self.total_latency_list[0] == None:
            arrive_times_list = [[-1]*len(model.get_inp_seq_ids()) for model in self.model_list]
            self.estimate_exec_time(cost_table, 
                check_gap=1, sort_input=sort_input, arrive_times_list=arrive_times_list)

        
        if self.fake_scheduling_res_ref != None:
            print("calling remote fake scheduling function-------")
            return None


        return max(self.total_latency_list)



    
    
    def update_inp_out_seqlens_and_throughput_after_an_infer_stage(
            self, stop_time: float, cost_table: CostTable):
        '''
            1. compute valid throughput.
            2. Update the remaining seqlens after it finishes the current infer stage (until stop_time).
        '''
        new_inp_out_lens_list_for_models, valid_throughput_list, stop_iter_i_list = list(), list(), list()
        stage_stop_time = stop_time

        for dp_id in range(self.dp_size):
            cumsum_latencys = self.cumsum_latencys_list[dp_id]
            cache_stop_time_info = self.cache_stop_time_info_list[dp_id]
            cum_rng_nums_for_models = self.cum_rng_nums_list_for_models[dp_id]
            rng_starts_for_models = self.rng_starts_list_for_models[dp_id]
            rng_ends_for_models = self.rng_ends_list_for_models[dp_id]

            if len(cumsum_latencys) == 0:
                new_inp_out_lens_list_for_models.append(
                    (tuple(np.asarray([[] for _ in range(len(self.model_list))])), 
                     tuple(np.asarray([[] for _ in range(len(self.model_list))])), 
                     np.asarray([[] for _ in range(len(self.model_list))], dtype=np.int64)))
                valid_throughput_list.append(0)
                stop_iter_i_list.append(0)
                continue

            
            
            stop_time = min(cumsum_latencys[-1], stage_stop_time)

            
            stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')
            if stop_iter_i in cache_stop_time_info:

                

                new_inp_out_lens_for_models, valid_throughput = cache_stop_time_info[stop_iter_i]
                
                
                

                new_inp_out_lens_list_for_models.append(new_inp_out_lens_for_models)
                valid_throughput_list.append(valid_throughput)
                stop_iter_i_list.append(stop_iter_i)
                continue

                

            actual_stop_time = cumsum_latencys[stop_iter_i]

            
            
            
            


            
            
            

            

            
            

            finished_lens = np.concatenate([fake_scheduling.get_info_at_stop_time(
                cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, 
                stop_time, stop_iter_i) for cum_rng_nums, rng_starts, rng_ends \
                    in zip(cum_rng_nums_for_models, rng_starts_for_models, rng_ends_for_models)])

            
            
            
            


            
            
            
            
            
            


            
            
            
            

            dp_inp_lens_flattened = np.concatenate( self.dp_inp_lens_list_for_models[dp_id] )
            dp_out_lens_flattened = np.concatenate( self.dp_out_lens_list_for_models[dp_id] )

            

            
            print(self)

            
            valid_throughput = fake_scheduling.comp_valid_throughput_at_stop_time(
                dp_inp_lens_flattened,
                finished_lens, actual_stop_time, cost_table,
                self.model.model_path, self.model.trust_remote_code, self.model.revision)

            
            dp_inp_lens_flattened = np.asarray(dp_inp_lens_flattened) + np.asarray(finished_lens)
            remaining_lens = np.asarray(dp_out_lens_flattened) - np.asarray(finished_lens)
            valid_indices = (remaining_lens>0)

            indptr = np.cumsum([0]+[len(_) for _ in self.dp_inp_lens_list_for_models[dp_id]])
            
            dp_inp_lens_for_models = [dp_inp_lens_flattened[indptr[i]:indptr[i+1]] for i in range(len(self.model_list))]
            remaining_lens_for_models = [remaining_lens[indptr[i]:indptr[i+1]] for i in range(len(self.model_list))]
            valid_indices_for_models = [valid_indices[indptr[i]:indptr[i+1]] for i in range(len(self.model_list))]

            
            dp_inp_lens_for_models = [tuple(vs[inds]) for vs, inds in zip(dp_inp_lens_for_models, valid_indices_for_models)]
            remaining_lens_for_models = [tuple(vs[inds]) for vs, inds in zip(remaining_lens_for_models, valid_indices_for_models)]

            
            

            
            cache_stop_time_info[stop_iter_i] = \
                [(tuple(dp_inp_lens_for_models), tuple(remaining_lens_for_models), valid_indices_for_models), \
                 valid_throughput]

            new_inp_out_lens_list_for_models.append(cache_stop_time_info[stop_iter_i][0])
            valid_throughput_list.append(valid_throughput)
            stop_iter_i_list.append(stop_iter_i)

            
            
            
            
            
            
            


        return (new_inp_out_lens_list_for_models, valid_throughput_list), stop_iter_i_list






    
    def update_fake_schedule_output_after_an_infer_stage(
            self, 
            old_inp_lens_list: List[List[int]], 
            new_inp_out_lens_list: List[List[List[int]]], 
            new_inp_lens_merged: List[int],
            new_out_lens_merged: List[int],
            new_inp_seq_ids_merged: List[int],
            stop_iter_i_list: List[int], 
            cost_table: CostTable, 
            ):
        '''
            This function is called when the exec plan is selected to run for an infer stage.
            Update:
                _FAKE_SCHEDULING_RES[model_name, exec_plan, new_inp_lens, new_out_lens]
            NOTE:
                we only call this function when after an infer stage, all the requests are available.
            NOTE: 
                1. ``new_inp_lens_merged`` is already sorted by seq ids.
                2. we only do update when there are unfinished requests.
        '''
        def to_tuple(vs):
            return tuple([tuple(_) for _ in vs])

        def _get_max(vs):
            if len(vs) > 0:
                return max(vs)
            else:
                return 0

        
        

        
        
        
        if sum([len(model.get_inp_seq_ids()) for model in self.model_list[:-1]]) == 0:
            
            last_inp_available_time = max(\
                [_get_max(self.dp_arrive_times_list_for_models[dp_id][-1]) \
                 for dp_id in range(self.dp_size)])
            stop_time = max([cumsum_latencys[stop_iter_i] if len(cumsum_latencys) > 0 else 0 \
                             for cumsum_latencys, stop_iter_i in \
                                zip(self.cumsum_latencys_list, stop_iter_i_list)])
            if last_inp_available_time > stop_time:
                
                return
        else:
            
            return


        cumsum_latencys_list = [list() for _ in range(self.dp_size)]
        cum_rng_nums_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        rng_starts_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        rng_ends_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        is_prefill_steps_list = [list() for _ in range(self.dp_size)]
        
        finish_times_list_for_models = [[list() for model in self.model_list] for _ in range(self.dp_size)]
        throughput_till_each_iter_list = [list() for _ in range(self.dp_size)]


        

        for dp_id in range(self.dp_size):
            
            old_inp_lens_for_models = old_inp_lens_list[dp_id]
            old_inp_lens = old_inp_lens_for_models[-1]
            
            stop_iter_i = stop_iter_i_list[dp_id]

            
            (cumsum_latencys_list[dp_id], cum_rng_nums_list_for_models[dp_id][-1], 
                rng_starts_list_for_models[dp_id][-1], rng_ends_list_for_models[dp_id][-1], is_prefill_steps_list[dp_id], 
                finish_times_list_for_models[dp_id][-1], alive_old_indices, 
                throughput_till_each_iter_list[dp_id]) = \
                    fake_scheduling.update_fake_FCFS_schedule_metadata(
                        old_inp_lens, 
                        self.cumsum_latencys_list[dp_id], self.cum_rng_nums_list_for_models[dp_id][-1], 
                        self.rng_starts_list_for_models[dp_id][-1], self.rng_ends_list_for_models[dp_id][-1], 
                        self.is_prefill_steps_list[dp_id],
                        self.throughput_till_each_iter_list[dp_id],
                        self.infer_args.max_num_batched_tokens, stop_iter_i,
                        cost_table, 
                        model_name=self.model.model_path, 
                        exec_plan=self.get_key_single_dp_worker(), sample_config=self.model.sample_config, 
                        trust_remote_code=self.model.trust_remote_code, revision=self.model.revision
                        )
            
            
            
            
            assert (np.nonzero(new_inp_out_lens_list[dp_id][2][-1])[0] == alive_old_indices).all()





        
        

        
        
        new_key = (self.model.model_name, self.get_key(), 
                   (tuple([to_tuple(dp_data[0]) for dp_data in new_inp_out_lens_list]), tuple([to_tuple(dp_data[1]) for dp_data in new_inp_out_lens_list])), \
                   tuple([to_tuple([[-1 - self.extra_cost]*len(_) for _ in dp_data[0]]) for dp_data in new_inp_out_lens_list]))


        
        _FAKE_SCHEDULING_RES[new_key] = \
            cumsum_latencys_list, cum_rng_nums_list_for_models, rng_starts_list_for_models, rng_ends_list_for_models, \
                is_prefill_steps_list, finish_times_list_for_models, \
                throughput_till_each_iter_list
            





    
    
    
    
    
    
    
    
    
    
    


    def __str__(self) -> str:
        return f"{[str(model) for model in self.model_list]}, "\
            f"{self.get_key()}"
        
        
            
            
            













class MyExecPlanGroup:
    """ My execution plan group definition. """
    def __init__(
        self,
        exec_plans: List[MyExecPlan], 
        cost_table: CostTable,
        last_stage_exec_plans: List[MyExecPlan],
        
        check_gap: int,
        sort_input: bool,
        
        base_model_finish_status: Dict[int, bool],
    ) -> None:
        
        print(f"building exec plan group: {[(_.model.get_base_model_ids(), _.get_key()) for _ in exec_plans]}\n", flush=True)

        self.exec_plans = exec_plans
        self.throughput = None
        self.infer_stage_latency = None
        self.comp_throughput = None
        self.base_model_finish_status = base_model_finish_status
        

        
        self.inp_exec_plan_dict: Dict[MyExecPlan, List[MyExecPlan]] = defaultdict(list)
        self._topological_sort()
        

        self.valid_throughputs: List[float] = list()
        
        
        self.tmp_inp_out_lens_list: \
            List[Union[Tuple[List[int], List[int], List[int]],  Tuple[List[int], List[int], List[int]]]] = list()
        self.tmp_remaining_decode_flops_after_infer_stage: List[float] = list()
        self.tmp_stop_iter_i_list: List[int] = list()
        self.compute_infer_stage_data(cost_table=cost_table, 
                                      last_stage_exec_plans=last_stage_exec_plans, 
                                      check_gap=check_gap, sort_input=sort_input)

        
        
        



    def get_involved_base_model_num(self):
        return sum([len(exec_plan.get_base_model_ids()) for exec_plan in self.exec_plans])


    def get_involved_fused_models(self):
        return [exec_plan.model for exec_plan in self.exec_plans if isinstance(exec_plan, MyVerticalFusedExecPlan)]


    
    
    
    def comp_extra_prepare_costs(
            self, cost_table: CostTable, last_stage_exec_plans: List[MyExecPlan]):
        '''
            Compute the extra prepare costs for each exec plan in this group.
            There will be extra prepare cost if:
                (1) the exec plan of the same model changes.
                NOTE: if only the memory changes, will there be extra cost? Yes, but we currently do not consider this.
        '''
        model_exec_plans = { exec_plan.model : exec_plan for exec_plan in last_stage_exec_plans }
        for exec_plan in self.exec_plans:
            model = exec_plan.model
            if model in model_exec_plans:
                last_exec_plan = model_exec_plans[model]
                if exec_plan.get_key() == last_exec_plan.get_key():
                    
                    
                    
                    exec_plan.set_extra_cost(0.0)
                else:
                    
                    
                    
                    
                    
                    
                    exec_plan.set_extra_cost(
                        cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                    )
            
            
            elif _get_vertical_fuse_model_pairs(last_stage_exec_plans, exec_plan)!=None:
                
                exec_plan.set_extra_cost(0.0)
            
            else:
                
                
                
                
                
                
                exec_plan.set_extra_cost(
                    cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                )




    def _topological_sort(self):
        """
            This function returns the list of exec_plans in topological order.
            Update:
                1. self.exec_plans: 
                    sorted list of exec plans;
                2. self.inp_exec_plan_dict: 
                    the dependent input exec plans of each exec plan in this exec plan group.
            NOTE:
                This function is wrong because a model's input models may be finished (so not in the current plan group)
        """
        def get_mapped_model_ids_in_group(node_mapping, model_ids):
            """
                Get the mapped values of the input model_ids accoding to the node mapping.
                I.e., get the corresponding model ids in the new system when accepting the newly fused models.
                NOTE: only get the mapped model id if the id is in the current exec plan group.
            """
            
            return set([node_mapping[i] for i in model_ids if i in node_mapping])

        model_exec_plan_mapping: Dict[int, MyExecPlan] = \
            {exec_plan.model.model_id:exec_plan for exec_plan in self.exec_plans}
        
        
        
        
        node_mapping = {ori:exec_plan.model.model_id \
                             for exec_plan in self.exec_plans for ori in exec_plan.model.get_base_model_ids() }

        
        

        inp_model_ids_dict: Dict[MyExecPlan, List[int]] = dict()

        
        

        
        for exec_plan in self.exec_plans:
            inp_model_ids_this_stage = get_mapped_model_ids_in_group(node_mapping, exec_plan.model.inp_base_model_ids)
            
            inp_exec_plans_this_stage = [model_exec_plan_mapping[model_id] for model_id in inp_model_ids_this_stage]
            self.inp_exec_plan_dict[exec_plan] = inp_exec_plans_this_stage
            inp_model_ids_dict[exec_plan] = inp_model_ids_this_stage
        
        
        

        sorted_plans: List[MyExecPlan] = list()
        sorted_model_ids = set()
        
        while len(sorted_plans) < len(self.exec_plans):
            for model_id, exec_plan in model_exec_plan_mapping.items():
                if model_id in sorted_model_ids:
                    continue
                if inp_model_ids_dict[exec_plan].issubset(sorted_model_ids):
                    sorted_plans.append(exec_plan)
                    sorted_model_ids.add(model_id)

        
        self.exec_plans = sorted_plans








    def _get_arrive_times_base_model(self, inp_seq_ids: List[int], inp_info: List[Tuple[MyExecPlan, int]]):
        """
            Compute the input arrive times for this exec plan.
            NOTE: the finish_times are in the order of model.inp_seq_ids. 
            Input:
                inp_info: list of tuple (inp exec plan, the base model id in the inp exec plan)
        """
               

        if len(inp_info) > 0:
            
            finish_times = np.asarray([inp_plan.get_finish_times_merged(inp_seq_ids, model_ind) \
                                       + inp_plan.extra_cost \
                                    for inp_plan, model_ind in inp_info])

            

            finish_times = np.max(finish_times, axis=0)
        else:
            
            finish_times = np.asarray([-1]*len(inp_seq_ids))
        return finish_times



    
    def _get_arrive_times(self, exec_plan: MyExecPlan):

        

        inp_exec_plans = self.inp_exec_plan_dict[exec_plan]
        in_exec_plans_base_model_ids = [in_exec_plan.get_base_model_ids() for in_exec_plan in inp_exec_plans]
        print(f"in_exec_plans_base_model_ids: {in_exec_plans_base_model_ids}")
        finish_times_list: List[List[float]] = list()

        
        for base_model in exec_plan.get_base_models():
            
            inp_info: List[MyExecPlan, int] = list()

            

            
            
            
            
            for in_model_id in base_model.inp_base_model_ids:
                find:bool = False
                for i, model_ids in enumerate(in_exec_plans_base_model_ids):
                    if in_model_id in model_ids:
                        for j, model_id in enumerate(model_ids):
                            if in_model_id == model_id:
                                inp_info.append((inp_exec_plans[i], j))
                                find = True
                if (not find) and base_model.independent_srcs:
                    
                    inp_info.append((None, None))
            
            
            if base_model.independent_srcs:
                tmp_finish_time_dict: Dict[int, float] = dict()
                for in_model_id, (inp_plan, model_ind) in zip(base_model.inp_base_model_ids, inp_info):
                    
                    inp_seq_ids = base_model.inp_req_from_which_model_which_out_reqs[in_model_id]
                    need_seq_ids = list(set(inp_seq_ids.values()))
                    
                    if model_ind == None:
                        if self.base_model_finish_status[in_model_id]:
                            tmp_finish_time_dict.update({i:-1 for i in inp_seq_ids})
                        else:
                            tmp_finish_time_dict.update({i:1e9 for i in inp_seq_ids})
                    else:
                        
                        
                        
                        tmp_finish_times = self._get_arrive_times_base_model(need_seq_ids, [(inp_plan, model_ind)])
                        _tmp_finish_time_dict = {i:t for i, t in zip(need_seq_ids, tmp_finish_times)}
                        tmp_finish_time_dict.update({i:_tmp_finish_time_dict[inp_seq_ids[i]] for i in inp_seq_ids})

                inp_seq_ids: List[int] = base_model.get_inp_seq_ids()
                finish_times_list.append([tmp_finish_time_dict[i] for i in inp_seq_ids])
                
                
                
                
                continue

            inp_seq_ids: List[int] = base_model.get_inp_seq_ids()
            finish_times_list.append(self._get_arrive_times_base_model(inp_seq_ids, inp_info))

        
        if isinstance(exec_plan, MyVerticalFusedExecPlan):
            

            return finish_times_list
        else:
            

            return finish_times_list[0]









    def compute_infer_stage_data(
            self, cost_table: CostTable, last_stage_exec_plans: List[MyExecPlan], 
            check_gap: int, sort_input: bool):
        '''
            1. Compute the infer stage time.
            2. Update the infer progress of each model involved.
            Modify:
                (1) self.valid_throughputs, (2) self.tmp_inp_out_lens_list
                (3) self.tmp_remaining_decode_flops_after_infer_stage [deleted]
                (4) self.tmp_stop_iter_i_list
                (5) the total latency of the current infer stage.
        '''

        print(f"INIT plan group: {[((exec_plan.model.get_base_model_ids(), exec_plan.get_key()), exec_plan.model.model_id, isinstance(exec_plan, MyVerticalFusedExecPlan)) for exec_plan in self.exec_plans]}")

        
        self.comp_extra_prepare_costs(cost_table, last_stage_exec_plans)
        
        
        
        
        
        
        
        
        
        
        latencys = [exec_plan.get_max_dp_latency_considering_plan_group(
                        cost_table, check_gap, sort_input, self._get_arrive_times(exec_plan),
                        
                    ) for exec_plan in self.exec_plans]
        
        
        return 


        latency = min(latencys)

        print(f"stage latency: {latency}")
        print(f"{[exec_plan.model.get_base_model_ids() for exec_plan in self.exec_plans]}")

        
        
        
        for exec_plan in self.exec_plans:
            extra_cost = exec_plan.extra_cost
            
            
            (new_inp_out_lens, valid_throughput), stop_iter_i = \
                exec_plan.update_inp_out_seqlens_and_throughput_after_an_infer_stage(
                    stop_time=latency-extra_cost, cost_table=cost_table)
            self.tmp_inp_out_lens_list.append(new_inp_out_lens)
            
            
            self.valid_throughputs.append(valid_throughput)
            self.tmp_stop_iter_i_list.append(stop_iter_i)
        
        self.infer_stage_latency = latency

        print(f"throughput: {self.get_throughput()}")




    def wait_remote_fake_scheduling_to_compute_infer_stage_data(
            self, cost_table: CostTable):
        '''
            1. Compute the infer stage time.
            2. Update the infer progress of each model involved.
            Modify:
                (1) self.valid_throughputs, (2) self.tmp_inp_out_lens_list
                (3) self.tmp_remaining_decode_flops_after_infer_stage [deleted]
                (4) self.tmp_stop_iter_i_list
                (5) the total latency of the current infer stage.
            
        '''

        print(f"Continue to INIT plan group: {[(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.model.model_id, isinstance(exec_plan, MyVerticalFusedExecPlan)) for exec_plan in self.exec_plans]}")
       

        time1 = time.perf_counter()

        
        
        latencys = [exec_plan.wait_for_remote_fake_scheduling_and_get_max_dp_latency_considering_plan_group(
                    ) for exec_plan in self.exec_plans]

        
        latencys_to_consider = [v for v, exec_plan in zip(latencys, self.exec_plans) if len(self.inp_exec_plan_dict[exec_plan])==0]
        
        
        latency = min(latencys_to_consider)

        print(f"stage latency: {latency}")
        print(f"{[exec_plan.model.get_base_model_ids() for exec_plan in self.exec_plans]}")

        print(f"comp cost 2/2: {time.perf_counter()-time1}")
        time1 = time.perf_counter()

        
        
        
        for exec_plan in self.exec_plans:
            extra_cost = exec_plan.extra_cost
            
            
            (new_inp_out_lens, valid_throughput), stop_iter_i = \
                exec_plan.update_inp_out_seqlens_and_throughput_after_an_infer_stage(
                    stop_time=latency-extra_cost, cost_table=cost_table)
            self.tmp_inp_out_lens_list.append(new_inp_out_lens)
            
            
            self.valid_throughputs.append(valid_throughput)
            self.tmp_stop_iter_i_list.append(stop_iter_i)
        
        self.infer_stage_latency = latency

        print(f"throughput: {self.get_throughput()}")


        print(f"comp cost 2/2: {time.perf_counter()-time1}")
        time1 = time.perf_counter()


    def get_throughput_no_data_parallel(self):
        '''
        Get the total throughput of the given plan group.
        NOTE: we also consider the extra preparation cost here.
        '''    
        if self.throughput == None:
            assert len(self.valid_throughputs) > 0
            
            throughputs = [v*exec_plan.cumsum_latencys[stop_iter_i]/self.infer_stage_latency \
                for v, exec_plan, stop_iter_i \
                    in zip(self.valid_throughputs, self.exec_plans, self.tmp_stop_iter_i_list)]
            self.throughput = sum(throughputs)

        return self.throughput
    

    
    def get_throughput(self):
        '''
        Get the total throughput of the given plan group.
        NOTE: we also consider the extra preparation cost here.
        '''    
        if self.throughput == None:
            assert len(self.valid_throughputs) > 0
            
            throughputs = [sum([v*cumsum_latencys[stop_iter_i]/self.infer_stage_latency \
                            for v, cumsum_latencys, stop_iter_i in \
                                zip(v_list, exec_plan.cumsum_latencys_list, stop_iter_i_list) \
                                    if len(cumsum_latencys) > 0]) \
                for v_list, exec_plan, stop_iter_i_list \
                    in zip(self.valid_throughputs, self.exec_plans, self.tmp_stop_iter_i_list)]
            self.throughput = sum(throughputs)

        return self.throughput
    

    def get_comp_throughput_only_no_data_parallel(self):
        '''
        Get the total throughput of the given plan group, do not consider the extra preparation cost.
        '''
        if self.comp_throughput == None:
            assert len(self.valid_throughputs) > 0
            self.comp_throughput = sum(self.valid_throughputs)
        return self.comp_throughput

    
    def get_comp_throughput_only(self):
        '''
        Get the total throughput of the given plan group, do not consider the extra preparation cost.
        '''
        if self.comp_throughput == None:
            assert len(self.valid_throughputs) > 0
            self.comp_throughput = sum([sum(v_list) for v_list in self.valid_throughputs])
        return self.comp_throughput

    
    def get_infer_stage_latency(self):
        return self.infer_stage_latency
    

    
    def get_model_states_after_infer_stage(self, cost_table: CostTable):
        
        
        
        

        
        
        
        
        
        

        def get_model_state(exec_plan, inp_out_lens):
            if isinstance(exec_plan, MyVerticalFusedExecPlan):
                print(f"MyVerticalFusedExecPlan")
                return (
                [exec_plan.model.model_name]*len(exec_plan.model.get_base_model_ids()), 
                tuple(exec_plan.model.get_base_model_ids()), 
                tuple([tuple(_) for _ in exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[0]]),
                tuple([tuple(_) for _ in exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[1]]),
                )
                return (
                exec_plan.model.model_name, tuple(exec_plan.model.get_base_model_ids()), 
                tuple(np.concatenate([tuple(np.concatenate(inps)) for inps, outs, indices in inp_out_lens])), 
                tuple(np.concatenate([tuple(np.concatenate(outs)) for inps, outs, indices in inp_out_lens])) 
                )
            elif isinstance(exec_plan, MyExecPlan):
                print(f"MyExecPlan")
                return (
                [exec_plan.model.model_name], 
                tuple(exec_plan.model.get_base_model_ids()), 
                [tuple(exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[0])], 
                [tuple(exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[1])] 
                )
                return (
                exec_plan.model.model_name, tuple(exec_plan.model.get_base_model_ids()), 
                tuple(np.concatenate([inps for inps, outs, indices in inp_out_lens])), 
                tuple(np.concatenate([outs for inps, outs, indices in inp_out_lens])) 
                )

        ret = list()
        for exec_plan, inp_out_lens in zip(self.exec_plans, self.tmp_inp_out_lens_list):
            states = get_model_state(exec_plan, inp_out_lens)
            names, model_ids, inp_lens, out_lens = states
            ret.extend(zip(names, model_ids, inp_lens, out_lens))

        return tuple(sorted(ret))
        
        return tuple(sorted([get_model_state(exec_plan, inp_out_lens) for exec_plan, inp_out_lens \
                in zip(self.exec_plans, self.tmp_inp_out_lens_list)]))


        
        
        
        
        
        
    
    def get_model_states_before_infer_stage(self):
        ret = list()
        for exec_plan in self.exec_plans:
            ret.extend([model.get_state() for model in exec_plan.get_base_models()])
        return tuple(sorted(ret))
        
    

    def update_model_inp_out_lens_no_data_parallel(self, cost_table: CostTable):
        

        
        for exec_plan, inp_out_lens, stop_iter_i in zip(self.exec_plans, self.tmp_inp_out_lens_list, self.tmp_stop_iter_i_list):
            old_inp_lens, _ = exec_plan.model.get_inp_out_seqlens()

            
            exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table)
        
            
            if not exec_plan.model.is_finished():
                
                exec_plan.update_fake_schedule_output_after_an_infer_stage_no_data_parallel(
                    old_inp_lens, *inp_out_lens,
                    stop_iter_i, cost_table)
            
            
            
            
            
            
        
        return
        for i in range(len(self.exec_plans)):
            exec_plan = self.exec_plans[i]
            inp_out_lens = self.tmp_inp_out_lens_list[i]
            remaining_decode_flops = self.tmp_remaining_decode_flops_after_infer_stage[i]
            exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)




    
    def update_model_inp_out_lens(self, cost_table: CostTable):
        

        
        for exec_plan, inp_out_lens, stop_iter_i in \
            zip(self.exec_plans, self.tmp_inp_out_lens_list, self.tmp_stop_iter_i_list):
            
            old_inp_lens = exec_plan.get_dp_inp_lens_list_for_models()

            
            
            merged_inp_out_lens = exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)
            
            print(f"UPDATE MODELS AFTER SELECTING AN EXEC PLAN-------------\n")
            print(f"exec_plan: {str(exec_plan)}, model_ids: {exec_plan.model.get_base_model_ids()} exec_plan.model.independent_srcs: {exec_plan.model.independent_srcs}")
            
            
            
            
            print(f"old inp lens: {old_inp_lens}")
            print(f"merged_inp_out_lens: {merged_inp_out_lens}")

            
            exec_plan.model.update_inp_out_seqlens(*merged_inp_out_lens, cost_table)
        
            
            if not exec_plan.model.is_finished():
                
                exec_plan.update_fake_schedule_output_after_an_infer_stage(
                    old_inp_lens, inp_out_lens, 
                    *merged_inp_out_lens,
                    stop_iter_i, cost_table)
            
            
            
            
            
            
        
        return
        for i in range(len(self.exec_plans)):
            exec_plan = self.exec_plans[i]
            inp_out_lens = self.tmp_inp_out_lens_list[i]
            remaining_decode_flops = self.tmp_remaining_decode_flops_after_infer_stage[i]
            exec_plan.model.update_inp_out_seqlens(*inp_out_lens, cost_table, remaining_decode_flops)



    
    def __str__(self):
        '''
        Get the string to represent a plan group: 
            the exec_plan settings + the remaining_lens after the current infer stage.
        '''
        return str(sorted([f"{str(exec_plan)}: {str(exec_plan.total_latency_list)}s" 
                           for exec_plan in self.exec_plans])) \
                                + ' ' + str(self.infer_stage_latency) \
                                + ' ' + str(self.get_throughput())
    
        return str(sorted([str(exec_plan) 
                           for exec_plan, (inp_lens, out_lens) \
                            in zip(self.exec_plans, self.tmp_inp_out_lens_list)])) \
                                + ' ' + str(self.infer_stage_latency) \
                                + ' ' + str(self.get_throughput())
    
    
    
    def __len__(self):
        return len(self.exec_plans)




class MyExecPlanGroupSeq:
    """ My execution plan group sequence definition. """
    def __init__(
        self,
        tot_flops: float,
        plan_group_seq: List[MyExecPlanGroup], 
        time_seq: List[float],
        last_stage_model_sys_values_seq: List[List],
    ) -> None:
        self.tot_flops = tot_flops
        self.plan_group_seq = plan_group_seq
        self.time_seq = time_seq
        
        self.last_stage_model_sys_values_seq = last_stage_model_sys_values_seq
    
    def get_last_stage_exec_plans(self) -> List[MyExecPlan]:
        if len(self.plan_group_seq) == 0:
            return []
        else:
            return self.plan_group_seq[-1].exec_plans
    
    def get_tot_time(self):
        return sum(self.time_seq)
    

    def get_valid_throughput(self):
        
        if len(self.time_seq) == 0:
            return 0
        else:
            return self.tot_flops / self.get_tot_time()
    
    def get_tmp_throughput_after_adding_a_plan_group(self, plan_group: MyExecPlanGroup):
        flops = sum([group.get_throughput()*group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])
        latency = sum([group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])        
        return flops / latency
    

    def get_tmp_only_comp_throughput_after_adding_a_plan_group(self, plan_group: MyExecPlanGroup):
        flops = sum([group.get_throughput()*group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])
        
        print("DEBUG: ", [group.valid_throughputs for group in self.plan_group_seq+[plan_group]])
        
        latency = sum([group.get_throughput()*group.get_infer_stage_latency()/group.get_comp_throughput_only() \
                       for group in self.plan_group_seq+[plan_group]])        
        return flops / latency
    


    def append_plan_group(self, plan_group):
        self.plan_group_seq.append(plan_group)
    
    def append_exec_time(self, comp_time):
        self.time_seq.append(comp_time)


    def append_last_stage_model_sys_values(self, last_stage_model_sys_values):
        self.last_stage_model_sys_values_seq.append(last_stage_model_sys_values)

    def pop_one_stage(self):
        self.plan_group_seq = self.plan_group_seq[:-1]
        self.time_seq = self.time_seq[:-1]
        self.last_stage_model_sys_values_seq = self.last_stage_model_sys_values_seq[:-1]

    def get_last_stage(self)->Tuple[MyExecPlanGroup, float]:
        return self.plan_group_seq[-1], self.time_seq[-1], self.last_stage_model_sys_values_seq[-1]
    
    def set_plan_group_and_time(self, plan_group_seq, time_seq, last_stage_model_sys_values_seq):
        self.plan_group_seq = plan_group_seq
        self.time_seq = time_seq
        self.last_stage_model_sys_values_seq = last_stage_model_sys_values_seq
    
    def get_stage_throughputs(self):
        return [group.get_throughput() for group in self.plan_group_seq]


    def get_str_using_model_ids(self) -> str:
        if (len(self.plan_group_seq) == 0) or (self.plan_group_seq[0] == None):
            return f"{self.plan_group_seq, self.time_seq}"
        else:
            return f"{[[(exec_plan.model.get_base_model_ids(), exec_plan.get_key()) for exec_plan in group.exec_plans] for group in self.plan_group_seq]} "\
                f"{self.time_seq} "\
                f"{sum(self.time_seq)} "\
                f"{self.get_valid_throughput()}"

    
    def __str__(self) -> str:
        if (len(self.plan_group_seq) == 0) or (self.plan_group_seq[0] == None):
            return f"{self.plan_group_seq, self.time_seq}"
        else:
            return f"{[[str(exec_plan) for exec_plan in group.exec_plans] for group in self.plan_group_seq]} "\
                f"{self.time_seq} "\
                f"{sum(self.time_seq)} "\
                f"{self.get_valid_throughput()}"
        








class MyModelSystem:
    """ My multi-model system class. Contains the computation graph of this system. """
    def __init__(
        self,
        model_list: List[MyModelInfor],
        out_edge_dict: Dict[int, List[int]],
        
        cost_table: CostTable, inp_mergers, outlen_generators, 
        prompt_templates_lens: List[int],
        need_correct_inp_out_lens: bool,
        out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]] = None,
    ) -> None:
        
        self.model_dict: Dict[int, MyModelInfor] = {model.model_id: model for model in model_list}
        
        self.all_level_model_ids: List[List[int]] = list()
        
        self.out_edge_dict = defaultdict(list, out_edge_dict)
        self.cost_table = cost_table
        self.inp_mergers = inp_mergers
        self.outlen_generators = outlen_generators
        
        
        
        
        

        
        
        for model in self.model_dict.values():
            model.input_model_ids = list()
        for inp in out_edge_dict:
            for out in out_edge_dict[inp]:
                self.model_dict[out].input_model_ids.append(inp)

        print(f"out_edge_dict: {out_edge_dict}")
        for model_id, model in self.model_dict.items():
            print(f"inp model ids of model {model_id} = {model.get_base_model_ids()}: {model.input_model_ids}")
        
        
        
        
        

        
        
        if need_correct_inp_out_lens:
            self._get_inp_out_lens_considering_LLM_dependency(
                cost_table=cost_table, inp_mergers=inp_mergers, outlen_generators=outlen_generators, out_req_id_mapping=out_req_id_mapping,
                prompt_templates_lens=prompt_templates_lens)
        
        
        
        
        
        




    def check_finish_states_accuracy(self):
        
        
        for model in self.model_dict.values():
            inp_model_states = [self.model_dict[i].is_finished() for i in model.input_model_ids]
            if model.is_finished() and (False in inp_model_states):
                assert False










    def fuse_similar_models_in_a_chain_fuseAll_by_default(
            self,
            tot_gpu_num, byte_per_gpu, cost_table: CostTable,
            
            check_gap: int, sort_input: bool,
            similar_threshold: float,
            fully_connected_gpu_unit:int):
        """
            This function tries to fuse models in a chain (these models can be fused vertically) which has similar performance given the same GPU resources.
            Output: a new model system with fused models.

            NOTE: call this function before the search starts.
            NOTE: call this function when similar_threshold is infinitely large.
        """

        print(f"\n\nTRYING FUSING SOME MODELS AT THE BEGINNING!\n\n")

        if len(self.all_level_model_ids) == 1:
            return self

        

        visit_model_level = -1
        
        
        comp_throughputs_dict: Dict[int, List[float]] = dict()
        
        fused_models: Dict[int, List[int]] = dict()

        
        base_model_finish_status = {base_model.model_id: True \
                                    for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()}




        
        
        
        
        
        
        
        
        
        




        while True:
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)
            if len(cand_models) == 0:
                break


            
            
            
            
            
            
            
            
            

            for to_fuse_model in cand_models:
                
                
                
                
                
                
                
                
                


                


                is_fused: bool = False
                
                
                to_fuse_inp_model_ids = to_fuse_model.input_model_ids
                for first_model_id, model_ids_fused in fused_models.items():
                    fused_model_inp_base_model_ids = self.model_dict[first_model_id].inp_base_model_ids
                    if (self.model_dict[first_model_id].model_name == to_fuse_model.model_name) and \
                        _meet_vertical_fuse_condition(to_fuse_inp_model_ids, model_ids_fused, fused_model_inp_base_model_ids):
                        
                        
                        to_fuse_model.can_be_vertically_fused_topologically = True

                        
                        
                        
                        if True:
                            
                            fused_models[first_model_id].append(to_fuse_model.model_id)
                            is_fused = True
                            break
            

                
                if not is_fused:
                    
                    fused_models[to_fuse_model.model_id] = [to_fuse_model.model_id]
            
        
        
        print(f"\n\n FINISH FUSING SOME MODELS AT THE BEGINNING!\n\n")
        print(f"comp_throughputs_dict: {comp_throughputs_dict}")
        print(f"fused_models: {fused_models}")
        
        
        
        fused_model_objs = list()
        for model_ids in fused_models.values():
            if len(model_ids) > 1:
                fused_model_objs.append( MyFusedModelInfor([self.model_dict[model_id] for model_id in model_ids]) )
        
        return self.gen_new_model_sys_with_fused_models(fused_model_objs)
        









    def fuse_similar_models_in_a_chain(
            self,
            tot_gpu_num, byte_per_gpu, cost_table: CostTable,
            
            check_gap: int, sort_input: bool,
            similar_threshold: float,
            fully_connected_gpu_unit:int):
        """
            This function tries to fuse models in a chain (these models can be fused vertically) which has similar performance given the same GPU resources.
            Output: a new model system with fused models.

            NOTE: call this function before the search starts.
        """

        print(f"\n\nTRYING FUSING SOME MODELS AT THE BEGINNING!\n\n")

        if len(self.all_level_model_ids) == 1:
            return self

        if similar_threshold == float('inf'):
            return self.fuse_similar_models_in_a_chain_fuseAll_by_default(
                tot_gpu_num, byte_per_gpu, cost_table,
                check_gap, sort_input,
                similar_threshold,
                fully_connected_gpu_unit)

        

        visit_model_level = -1
        
        
        comp_throughputs_dict: Dict[int, List[float]] = dict()
        
        fused_models: Dict[int, List[int]] = dict()

        
        base_model_finish_status = {base_model.model_id: True \
                                    for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()}




        
        plan_groups_dict: Dict[int, List[MyExecPlanGroup]] = dict()
        all_base_models = [base_model for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()]
        for base_model in all_base_models:
            exec_plans = get_possible_exec_plans(
                base_model, tot_gpu_num, byte_per_gpu, cost_table, baseline='ours', sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
            plan_groups = [MyExecPlanGroup([exec_plan], cost_table=cost_table, last_stage_exec_plans=[],
                                            check_gap=check_gap, sort_input=sort_input, base_model_finish_status=base_model_finish_status) for exec_plan in exec_plans]
            plan_groups_dict[base_model.model_id] = plan_groups




        while True:
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)
            if len(cand_models) == 0:
                break


            
            
            
            
            
            
            
            
            

            for to_fuse_model in cand_models:
                
                
                
                
                
                
                plan_groups = plan_groups_dict[to_fuse_model.model_id]
                for plan_group in plan_groups:
                    plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)


                comp_throughput_vecs = np.asarray([plan_group.get_comp_throughput_only() for plan_group in plan_groups])


                is_fused: bool = False
                
                
                to_fuse_inp_model_ids = to_fuse_model.input_model_ids
                for first_model_id, model_ids_fused in fused_models.items():
                    fused_model_inp_base_model_ids = self.model_dict[first_model_id].inp_base_model_ids
                    if (self.model_dict[first_model_id].model_name == to_fuse_model.model_name) and \
                        _meet_vertical_fuse_condition(to_fuse_inp_model_ids, model_ids_fused, fused_model_inp_base_model_ids):
                        
                        
                        to_fuse_model.can_be_vertically_fused_topologically = True

                        
                        diff = np.abs((comp_throughput_vecs-comp_throughputs_dict[first_model_id])/comp_throughputs_dict[first_model_id])
                        if (diff < similar_threshold).all():
                            
                            fused_models[first_model_id].append(to_fuse_model.model_id)
                            is_fused = True
                            break
            

                
                if not is_fused:
                    comp_throughputs_dict[to_fuse_model.model_id] = comp_throughput_vecs
                    fused_models[to_fuse_model.model_id] = [to_fuse_model.model_id]
            
        
        
        print(f"\n\n FINISH FUSING SOME MODELS AT THE BEGINNING!\n\n")
        print(f"comp_throughputs_dict: {comp_throughputs_dict}")
        print(f"fused_models: {fused_models}")
        
        
        
        fused_model_objs = list()
        for model_ids in fused_models.values():
            if len(model_ids) > 1:
                fused_model_objs.append( MyFusedModelInfor([self.model_dict[model_id] for model_id in model_ids]) )
        
        return self.gen_new_model_sys_with_fused_models(fused_model_objs)
        










    def fuse_similar_models_in_a_chain_use_expectation_outlen(
            self,
            tot_gpu_num, byte_per_gpu, 
            cost_table: CostTable, inp_mergers, outlen_generators,
            
            check_gap: int, sort_input: bool,
            similar_threshold: float,
            fully_connected_gpu_unit:int,
            prompt_templates_lens: List[int],
            out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]] = None,):
        """
            This function tries to fuse models in a chain (these models can be fused vertically) which has similar performance given the same GPU resources.
            Output: a new model system with fused models.

            NOTE: call this function before the search starts.
            
        """

        def execpted_outlen_generator(model_id: int, model_name: str, inp_lens: List[int]):
            
            repeat_num = 100
            repeated_inp_lens = np.repeat(inp_lens, repeat_num)
            repeated_out_lens = outlen_generators[model_id](model_id, model_name, repeated_inp_lens)
            out_lens = repeated_out_lens.reshape((-1, repeat_num))
            out_lens = np.mean(out_lens, axis=1).astype(int)
            return out_lens

        print(f"\n\nTRYING FUSING SOME MODELS AT THE BEGINNING!\n\n")

        if len(self.all_level_model_ids) == 1:
            return self


        
        self._get_inp_out_lens_considering_LLM_dependency(cost_table, inp_mergers, [execpted_outlen_generator for model_id in self.model_dict.keys()], out_req_id_mapping, prompt_templates_lens)

        
        

        

        visit_model_level = -1
        
        
        comp_throughputs_dict: Dict[int, List[float]] = dict()
        
        fused_models: Dict[int, List[int]] = dict()

        
        base_model_finish_status = {base_model.model_id: True \
                                    for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()}




        
        plan_groups_dict: Dict[int, List[MyExecPlanGroup]] = dict()
        all_base_models = [base_model for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()]
        for base_model in all_base_models:
            exec_plans = get_possible_exec_plans(
                base_model, tot_gpu_num, byte_per_gpu, cost_table, baseline='ours', sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
            plan_groups = [MyExecPlanGroup([exec_plan], cost_table=cost_table, last_stage_exec_plans=[],
                                            check_gap=check_gap, sort_input=sort_input, base_model_finish_status=base_model_finish_status) for exec_plan in exec_plans]
            plan_groups_dict[base_model.model_id] = plan_groups




        while True:
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)
            if len(cand_models) == 0:
                break


            
            
            
            
            
            
            
            
            

            for to_fuse_model in cand_models:
                
                
                
                
                
                
                plan_groups = plan_groups_dict[to_fuse_model.model_id]
                for plan_group in plan_groups:
                    plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)


                comp_throughput_vecs = np.asarray([plan_group.get_comp_throughput_only() for plan_group in plan_groups])


                is_fused: bool = False
                
                
                to_fuse_inp_model_ids = to_fuse_model.input_model_ids
                for first_model_id, model_ids_fused in fused_models.items():
                    fused_model_inp_base_model_ids = self.model_dict[first_model_id].inp_base_model_ids
                    if (self.model_dict[first_model_id].model_name == to_fuse_model.model_name) and \
                        _meet_vertical_fuse_condition(to_fuse_inp_model_ids, model_ids_fused, fused_model_inp_base_model_ids):
                        
                        
                        to_fuse_model.can_be_vertically_fused_topologically = True

                        
                        diff = np.abs((comp_throughput_vecs-comp_throughputs_dict[first_model_id])/comp_throughputs_dict[first_model_id])
                        if (diff < similar_threshold).all():
                            
                            fused_models[first_model_id].append(to_fuse_model.model_id)
                            is_fused = True
                            break
            

                
                if not is_fused:
                    comp_throughputs_dict[to_fuse_model.model_id] = comp_throughput_vecs
                    fused_models[to_fuse_model.model_id] = [to_fuse_model.model_id]
            
        
        
        print(f"\n\n FINISH FUSING SOME MODELS AT THE BEGINNING!\n\n")
        print(f"comp_throughputs_dict: {comp_throughputs_dict}")
        print(f"fused_models: {fused_models}")
        
        

        
        self._get_inp_out_lens_considering_LLM_dependency(cost_table, inp_mergers, outlen_generators, out_req_id_mapping, prompt_templates_lens)

        
        fused_model_objs = list()
        for model_ids in fused_models.values():
            if len(model_ids) > 1:
                fused_model_objs.append( MyFusedModelInfor([self.model_dict[model_id] for model_id in model_ids]) )
        
        return self.gen_new_model_sys_with_fused_models(fused_model_objs)
        


















    def gen_new_model_sys_with_fused_models(self, fused_model_list: List[MyFusedModelInfor]):
        node_mapping = {ori:fused_model.model_id for fused_model in fused_model_list for ori in fused_model.get_base_model_ids()}
        
        
        for model_id, model in self.model_dict.items():
            base_model_ids = model.get_base_model_ids()
            if base_model_ids[0] in node_mapping:
                assert False not in [node_mapping[_] == node_mapping[base_model_ids[0]] for _ in base_model_ids]
                node_mapping[model_id] = node_mapping[base_model_ids[0]]

        
        new_model_dict = {model_id:model for model_id, model in self.model_dict.items() if model_id not in node_mapping}
        new_model_dict.update({model.model_id:model for model in fused_model_list})
        
        for model_id in self.model_dict:
            if model_id not in node_mapping:
                node_mapping[model_id] = model_id
        new_out_edge_dict = defaultdict(set)
        for src, tgts in self.out_edge_dict.items():
            new_out_edge_dict[node_mapping[src]].update([node_mapping[tgt] for tgt in tgts])
        for k in new_out_edge_dict:
            
            new_out_edge_dict[k] = list(new_out_edge_dict[k].difference({k}))

        
        new_model_sys = MyModelSystem(new_model_dict.values(), new_out_edge_dict, self.cost_table, self.inp_mergers, self.outlen_generators,
                                      prompt_templates_lens=None, 
                                      need_correct_inp_out_lens=False)
        return new_model_sys




    def _get_inp_out_lens_considering_LLM_dependency(
            self, 
            cost_table: CostTable,
            inp_mergers, outlen_generators, 
            out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]],
            prompt_templates_lens: Dict[int, int],
        ):
        """
            Compute the input and output sequence lengths for all LLMs in the system according to the given ``inp_merger``.
            NOTE: the model dependency is considered here.
            NOTE: this function is called when initializing an LLM system.
            INPUT:
                inp_merger: a function whose input is 
                    (1) the output seq lengths from the input models of an LLM (if any),
                    (2) the original inp seq lengths of the LLM (if any)
                    and generates the length of the fused input based on the given inplens.
                outlen_generator: a function which gereates an outlen given an inplen.

            Update:
                update the input and output seq lengths of all LLMs in the system.
        """

        def get_required_outputs_from_inp_model(inp_seq_ids, inp_model_id, out_req_id_mapping):
            outputs = self.model_dict[inp_model_id].get_inp_out_seqlens()[1]
            outputs_inds = self.model_dict[inp_model_id].get_inp_seq_ids()

            if inp_model_id in out_req_id_mapping:
                
                new_output_ids = [out_req_id_mapping[inp_model_id][output_id][0] for output_id in outputs_inds]
                print(f"new_output_ids: {new_output_ids}")
                
                order = np.argsort(new_output_ids)
                new_output_ids, counts = np.unique(new_output_ids, return_counts=True)
                print(f"new_output_ids: {new_output_ids}, counts: {counts}")
                cum_chunk_nums = np.cumsum(np.concatenate(([0], counts)))
                new_outputs = np.asarray(outputs)[order]
                print(f"cum_chunk_nums: {cum_chunk_nums}")
                print(f"new_outputs: {new_outputs}")
                new_outputs = [sum(new_outputs[cum_chunk_nums[i]:cum_chunk_nums[i+1]]) for i in range(len(counts))]
                print(f"new_outputs: {new_outputs}")
                
                outputs = new_outputs
                outputs_inds = new_output_ids

                
                outputs_inds = cum_chunk_nums[1:]-1
                print(f"outputs_inds: {outputs_inds}")

            
            
            

            return get_infor_given_seq_ids(
                values=outputs, 
                seq_ids_we_have=outputs_inds, 
                seq_ids_requested=inp_seq_ids, 
                default_value=0)
            
            inds = np.searchsorted(outputs_inds, inp_seq_ids)
            
            
            
            
            ret = np.zeros(len(inp_seq_ids),dtype=inp_seq_ids.dtype)
            valid_indices1 = inds<len(outputs_inds)
            valid_indices2 = (outputs_inds[inds[valid_indices1]] == inp_seq_ids[valid_indices1])
            inds = inds[valid_indices1][valid_indices2]
            ret_inds = np.arange(len(ret))[valid_indices1][valid_indices2]
            ret[ret_inds] = np.asarray(outputs)[inds]
            
            return ret
            return np.asarray(outputs)[inds]

        
        def consider_prompt_template_len(model_id) -> None:
            
            if model_id in prompt_templates_lens:
                model = self.model_dict[model_id]
                template_len = prompt_templates_lens[model_id]
                ori_inp_lens = model.get_inp_out_seqlens()[0]
                new_inp_lens = [i+template_len for i in ori_inp_lens]
                new_out_lens = outlen_generators[model_id](model_id, model.model_name, new_inp_lens)
                model.update_inp_out_seqlens(new_inp_lens, new_out_lens, model.get_inp_seq_ids(), cost_table)                
        
        
        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=False)
        for model_ids in self.all_level_model_ids:
            for model_id in model_ids:
                model = self.model_dict[model_id]
                if len(model.input_model_ids) == 0:
                    
                    
                    
                    consider_prompt_template_len(model_id)

                    continue
                ori_inp_lens = model.get_inp_out_seqlens()[0]
                
                inp_seq_ids = model.get_inp_seq_ids()

                
                new_inp_lens = None
                if model.independent_srcs:
                    
                    
                    assert model.inp_req_from_which_model_which_out_reqs!=None
                    new_inp_lens = dict()
                    
                    for from_model_id, req_ids in model.inp_req_from_which_model_which_out_reqs.items():
                        assert from_model_id >= 0
                        fetched_seq_lens = get_required_outputs_from_inp_model(list(req_ids.values()), from_model_id, out_req_id_mapping)
                        new_inp_lens.update({i:length for i, length in zip(list(req_ids.keys()), fetched_seq_lens)})
                    new_inp_lens = [new_inp_lens[i] for i in inp_seq_ids]
                else:
                    
                    if model.inp_req_from_which_model_which_out_reqs==None:
                        new_inp_lens = inp_mergers[model_id](
                            [ori_inp_lens] + \
                                [get_required_outputs_from_inp_model(inp_seq_ids, inp_model_id, out_req_id_mapping) for inp_model_id in model.input_model_ids]
                                )
                    else:
                        
                        assert False
                
                
                
                

                
                if len(prompt_templates_lens) > 0:
                    new_inp_lens = [i+prompt_templates_lens[model_id] for i in new_inp_lens]

                new_out_lens = outlen_generators[model_id](model_id, model.model_name, new_inp_lens)
                model.update_inp_out_seqlens(new_inp_lens, new_out_lens, model.get_inp_seq_ids(), cost_table)

                






    
    def get_runnable_models(self, running_model_ids: List[int]):
        """
            A model can be started when all its input models are started.
            Return the models which can be started given the list of running models and the finished models.
            NOTE: the returned models may not depend directly on the running models.
        """
        def is_finished_or_running(model: MyModelInfor, running_model_ids: List[int]):
            return (model.is_finished()) or \
                (model.model_id in running_model_ids)

        to_run: List[MyModelInfor] = list()
        for model in self.model_dict.values():
            if is_finished_or_running(model, running_model_ids):
                continue

            
            inps = model.input_model_ids
            inps_status = [is_finished_or_running(self.model_dict[inp], running_model_ids) for inp in inps]
            if False in inps_status:
                continue

            to_run.append(model)
        
        return to_run
    

    
    
    
    
    
    
    
    
    
    

    
    
    

    
    
    
    

    
        
    
    
    def get_models_at_given_level(self, level_num: int):
        if level_num >= len(self.all_level_model_ids):
            return list()
        return [self.model_dict[model_id] for model_id in self.all_level_model_ids[level_num]]



    def get_runnable_plans_from_cand_plans(
            self, 
            running_plan_group: List[MyExecPlan], 
            cand_models: List[MyModelInfor], cand_exec_plans: List[List[MyExecPlan]]):
        

        print(f"in get_runnable_plans_from_cand_plans: running_plan_group {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in running_plan_group]} cand_models: {[_.get_base_model_ids() for _ in cand_models]}")
        base_model_mapping: Dict[int, int] = {i:j for j, model in self.model_dict.items() for i in model.get_base_model_ids()}


        def is_finished_or_running(model: MyModelInfor, running_model_ids: List[int]):
            return (model.is_finished()) or \
                (set(model.get_base_model_ids()).issubset(running_model_ids))
                

        runnable_exec_plans: List[List[MyExecPlan]] = list()
        running_model_ids: List[int] = [exec_plan.model.model_id for exec_plan in running_plan_group]
        
        for exec_plan in running_plan_group:
            running_model_ids.extend(exec_plan.model.get_base_model_ids())
            print(f"running model ids: {exec_plan.model.model_id} = {exec_plan.get_base_model_ids()}")
        
        for model, exec_plans in zip(cand_models, cand_exec_plans):
            inps = model.input_model_ids

            if model.independent_srcs:
                unfinished_srcs = model.get_unfinished_srcs()
                inps = set([base_model_mapping[src] for src in unfinished_srcs])

            inps_status = [is_finished_or_running(self.model_dict[inp], running_model_ids) for inp in inps]

            print(f"checking model: {model.get_base_model_ids()}  inp model ids: {inps} = {[self.model_dict[_].get_base_model_ids() for _ in inps]}, inps_status: {inps_status}")

            if model.independent_srcs:
                if sum(inps_status) == 0:
                    continue
            elif False in inps_status:
                continue

            

            runnable_exec_plans.append(exec_plans)
        return runnable_exec_plans



    def get_all_level_models(self, aggresive_for_horizontally_fused_model: bool):
        """
            Update the model ids on all levels according to the current model status.
            Update: self.all_level_model_ids
            Input:
                aggresive_for_horizontally_fused_model: if True, we put each horizontally fused model in the earliest level it can be, 
                    i.e., as long as one of its inp model is finished or running, it can run.
                    ==> NOTE: change to 
                    1. for horizontally fused models, we will treat each independent inp src as an individual model instance.
                    E.g., if model A has 2 independent inp srcs, then there can be at most 2 model instances in the sys.
        """
        self.all_level_model_ids = list()
        
        visited: Dict[int, bool] = {model_id: False for model_id in self.model_dict}
        base_model_mapping: Dict[int, int] = {i:j for j, model in self.model_dict.items() for i in model.get_base_model_ids()}


        
        
        finished_model_ids = {model_id: True for model_id, model in self.model_dict.items() if model.is_finished()}
        print(f"finished models: {finished_model_ids}")

        
        while False in visited.values():
            
            
            newly_visited: Dict[int, bool] = {model_id: False for model_id in self.model_dict}
            new_model_ids: List[int] = list()
            for model in self.model_dict.values():
                if visited[model.model_id]:
                    continue
                if model.is_finished():
                    newly_visited[model.model_id] = True
                    continue
                inp_status = [visited[inp] for inp in model.input_model_ids]
                
                
                
                if aggresive_for_horizontally_fused_model and model.independent_srcs:
                    
                    unfinished_srcs = model.get_unfinished_srcs()

                    print(f"unfinished_srcs: {unfinished_srcs}")

                    unfinished_srcs = set([base_model_mapping[src] for src in unfinished_srcs])

                    
                    unfinished_srcs = [inp for inp in unfinished_srcs if not self.model_dict[inp].is_finished()]

                    inp_status = [visited[inp] for inp in unfinished_srcs]

                    print(f"model.model_id: {model.model_id}, unfinished_srcs: {unfinished_srcs}, inp_status: {inp_status}")
                    
                    
                    if (len(inp_status) > 0) and (sum(inp_status) == 0):
                        continue
                    
                elif False in inp_status:
                    
                    continue

                newly_visited[model.model_id] = True
                new_model_ids.append(model.model_id)
            
            visited = {k: visited[k] or newly_visited[k] for k in visited}
            if len(new_model_ids) > 0:
                self.all_level_model_ids.append(new_model_ids)





    def get_model_num(self)->int:
        return len(self.model_dict)




    def get_not_finished_base_model_num(self) -> int:
        not_finished_model_num = sum([not base_model.is_finished() \
                                      for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()])
        return not_finished_model_num


    def get_base_model_finish_status(self) -> Dict[int, bool]:
        """
            Get the dict of the finish status of each base model.
        """
        base_model_finish_status = {base_model.model_id: base_model.is_finished() \
                                      for model in self.model_dict.values() \
                                        for base_model in model.get_base_models()}
        return base_model_finish_status



    def is_finished(self) -> bool:
        """
            Return True if all the models in the system is finished.
        """
        
        return self.get_not_finished_base_model_num() == 0



    
    
    
    
    
    

    def get_base_model_states(self):
        '''
            Get the current inference progress of the given list of models.
            NOTE: the returned progress should be able to be added to a set.
        '''
        return tuple(sorted([base_model.get_state() for model in self.model_dict.values() \
                             for base_model in model.get_base_models()]))
    
    
    def get_model_inp_out_lens(self):
        ori_inp_out_lens_list = [model.get_inp_out_seqlens() for model in self.model_dict.values()]
        return ori_inp_out_lens_list
    
    def get_model_remaining_decode_flops(self):
        ori_remaining_decode_flops_list = [model.get_remaining_flops() for model in self.model_dict.values()]
        return ori_remaining_decode_flops_list

    def get_model_inp_seq_ids(self):
        ori_inp_seq_ids_list = [model.get_inp_seq_ids() for model in self.model_dict.values()]
        return ori_inp_seq_ids_list


    def get_model_inp_model_ids(self):
        ori_inp_model_ids_list = [model.input_model_ids for model in self.model_dict.values()]
        return ori_inp_model_ids_list


    
    
    def recover_model_state(
            self,
            inp_seq_ids_list: List[List[int]],
            inp_out_lens_list: List[Tuple[List[int], List[int]]], 
            cost_table: CostTable, remaining_decode_flops_list: List[float],
            inp_model_ids_list: List[List[int]]):
        
        
        
        
        
        
        for model, inp_seq_ids, inp_out_lens, remaining_decode_flops, inp_model_ids in \
            zip(self.model_dict.values(),inp_seq_ids_list,inp_out_lens_list,remaining_decode_flops_list,inp_model_ids_list):
            model.update_inp_out_seqlens(*inp_out_lens, inp_seq_ids, cost_table, remaining_decode_flops)
            model.input_model_ids = inp_model_ids


    def print_model_list(self):
        print(f"model_list: {[(str(model), model.model_id, model.get_base_model_ids()) for model in self.model_dict.values()]}")




    def _get_good_runnable_exec_plan_keys(
            self,
            cand_plan_group: List[MyExecPlan],
            runnable_exec_plans_list: List[List[MyExecPlan]],
            models_have_in_level_dependency: List[List[int]], 
            
            check_gap: int, sort_input: bool,
            last_stage_exec_plans: List[MyExecPlan],
            cost_table: CostTable,
            tot_gpu_num = 4, 
            ) -> List[List[MyExecPlan]]:
        """
            Get the best exec plan for each runnable models and each assigned comp gpu num.
            NOTE: 
                1. do some exec plan validity checking in advance.
                2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                we do not compute their good exec plans, i.e., we think each exec plan needs to be checked.
        """
        plan_group_objs_list = list()
        available_gpu_num = tot_gpu_num-get_tot_worker_num(cand_plan_group)
        valid_res_list: List[List[Tuple[int, MyExecPlan]]] = list()
        base_model_finish_status = self.get_base_model_finish_status()
        skip_is = list()
        for i, runnable_exec_plans in enumerate(runnable_exec_plans_list):
            if (len(runnable_exec_plans) > 0) and (runnable_exec_plans[0].model.model_id in models_have_in_level_dependency):
                valid_res_list.append(runnable_exec_plans)
                skip_is.append(i)
                continue

            gpu_nums = [get_tot_worker_num([exec_plan]) for exec_plan in runnable_exec_plans]
            plan_group_objs_list.append(
                [MyExecPlanGroup(
                    _get_path_key(cand_plan_group, exec_plan)[1],
                    
                    cost_table, last_stage_exec_plans, check_gap, sort_input,
                    base_model_finish_status=base_model_finish_status
                    ) for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])
            valid_res_list.append([(gpu_num, exec_plan) for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])
        
        
        ret = list()
        for i, valid_res in enumerate(valid_res_list):
            
            if i in skip_is:
                ret.append([exec_plan.get_key() for exec_plan in valid_res])
                continue
            
            good_dict: Dict[int, Tuple[float, MyExecPlan]] = dict()
            for gpu_num, exec_plan in valid_res:
                latency = exec_plan.wait_for_remote_fake_scheduling_and_get_max_dp_latency_considering_plan_group()
                print(f"good plan keys candidate: {(exec_plan.get_key(), latency)}")
                if gpu_num not in good_dict:
                    good_dict[gpu_num] = (latency, exec_plan)
                else:
                    if latency < good_dict[gpu_num][0]:
                        good_dict[gpu_num] = (latency, exec_plan)
            ret.append([exec_plan.get_key() for latency, exec_plan in good_dict.values()])
        

        print(f"good plan keys: {ret}")
        return ret



    def get_candidate_plan_groups_no_horizontal_fused_model(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
        """

        def _directly_discard(
                gen_execplans_baseline: str, not_finished_base_model_num: int, 
                plan_group: List[MyExecPlan], tot_gpu_num: int):
            involved_base_model_num = sum([len(plan.get_base_model_ids()) for plan in plan_group])
            gpu_num_sum = get_tot_worker_num(plan_group)
            return (gen_execplans_baseline=='ours') \
                and (not_finished_base_model_num > involved_base_model_num) \
                    and (gpu_num_sum<tot_gpu_num)

        
        not_finished_base_model_num = self.get_not_finished_base_model_num()

        
        tot_plan_groups: List[List[MyExecPlan]] = [[]]
        new_plan_groups: List[List[MyExecPlan]] = [[]]

        uniq_exec_plan_mapping = dict()
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()

        
        
        

        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=True)

        print(f"all_level_model_ids: {self.all_level_model_ids}")

        base_model_finish_status = self.get_base_model_finish_status()

        visit_model_level = -1
        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)

            

            
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")

                
                
                
                
                


            

            for cand_plan_group in new_plan_groups:
                
                

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)


                
                
                
                good_runnable_exec_plan_keys_list = self._get_good_runnable_exec_plan_keys(
                    cand_plan_group, runnable_exec_plans_list,
                    check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


                
                
                
                
                
                
                
                
                plan_groups = [cand_plan_group]
                old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
                _append_exec_plan(plan_groups, runnable_exec_plans_list, 0, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping, good_runnable_exec_plan_keys_list)
                
                
                
                if len(plan_groups) == 1:
                    
                    if not _directly_discard(
                        gen_execplans_baseline, not_finished_base_model_num, plan_groups[0], tot_gpu_num
                        ):
                        tot_plan_groups.extend(plan_groups)
                else:

                    if visit_model_level+1 >= len(self.all_level_model_ids):
                        
                        plan_groups = [plan_groups[0]]+[plan_group for plan_group in plan_groups[1:] if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_group, tot_gpu_num
                            )]


                    new_uniq_exec_plan_mappinp_keys = [k for k in uniq_exec_plan_mapping if k not in old_uniq_exec_plan_mapping_keys]
                    new_uniq_exec_plans: List[MyExecPlan] = [uniq_exec_plan_mapping[k] for k in new_uniq_exec_plan_mappinp_keys]
                    

                    
                    new_uniq_sub_plan_groups = list()
                    for exec_plan in new_uniq_exec_plans:
                        if isinstance(exec_plan, MyVerticalFusedExecPlan):
                            tmp_cand_plan_group = [plan for plan in cand_plan_group if not set(plan.model.get_base_model_ids()).issubset(exec_plan.model.get_base_model_ids())]
                            new_uniq_sub_plan_groups.append(tmp_cand_plan_group+[exec_plan])
                        else:
                            new_uniq_sub_plan_groups.append(cand_plan_group+[exec_plan])

                    # we first update the good_plan_group_dict
                    print(f"the groups we found a round: # of  groups: {len(plan_groups)}")
                    # for plan_group in plan_groups[1:]:
                    #     print([(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group])
                    print(f"the uniq sub plan groups we found a round: # of  groups: {len(new_uniq_sub_plan_groups)}")

                    # 0. we first init the uniq sub plan groups where no fake scheduling results will be shared
                    uniq_sub_plan_group_objs = [MyExecPlanGroup(
                        plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in new_uniq_sub_plan_groups]
                    for exec_plan in new_uniq_exec_plans:
                        exec_plan._wait_for_remote_fake_scheduling()

                    
                    plan_group_objs = [MyExecPlanGroup(
                        plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups[1:]]
                    good_plan_group_keys = [_update_good_plan_group_dict(
                        group_obj=group_obj, cost_table=cost_table, good_plan_group_dict=good_plan_group_dict
                        ) for group_obj in plan_group_objs]

                    
                    
                    
                    

                    good_plan_group_keys = [_ for _ in good_plan_group_keys if _ != None]
                    to_compare = sorted([_[0] for _ in good_plan_group_dict.values()], reverse=True)[top_k-1] if top_k <= len(good_plan_group_dict) else -1
                    good_plan_groups = [good_plan_group_dict[_][1].exec_plans for _ in good_plan_group_keys if good_plan_group_dict[_][0] > to_compare]
                    
                    
                    
                    
                    
                    tmp_new_plan_groups.extend(good_plan_groups)
                    
                    

                
                
                
                
                
                
                
                
                


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


        
        
        



        
        
        
        

        plan_groups = [_[1] for _ in good_plan_group_dict.values()]

        return plan_groups










    def get_candidate_plan_groups(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
        """

        def _directly_discard(
                gen_execplans_baseline: str, not_finished_base_model_num: int, 
                plan_group: List[MyExecPlan], tot_gpu_num: int):
            involved_base_model_num = sum([len(plan.get_base_model_ids()) for plan in plan_group])
            gpu_num_sum = get_tot_worker_num(plan_group)
            return (gen_execplans_baseline=='ours') \
                and (not_finished_base_model_num > involved_base_model_num) \
                    and (gpu_num_sum<tot_gpu_num)

        
        def _get_model_ids_of_different_levels(models: List[MyModelInfor]) -> Tuple[List[List[int]], Dict[int, List[int]]]:
            sorted_model_ids = list()
            all_model_ids = set([model.model_id for model in models])
            model_ids_of_each_layer = list()
            in_stage_out_edge_dict = defaultdict(list)
            
            while len(sorted_model_ids) < len(models):
                new_layer = list()
                for model in models:
                    if model.model_id in sorted_model_ids:
                        continue
                    in_stage_inp_model_ids = all_model_ids.intersection(model.input_model_ids)
                    if in_stage_inp_model_ids.issubset(sorted_model_ids):
                        new_layer.append(model.model_id)
                        for i in in_stage_inp_model_ids:
                            in_stage_out_edge_dict[i].append(model.model_id)
                sorted_model_ids.extend(new_layer)
                model_ids_of_each_layer.append(new_layer)
            print(f"model_ids_of_each_layer: {model_ids_of_each_layer}, in_stage_out_edge_dict: {in_stage_out_edge_dict}")
            return model_ids_of_each_layer, in_stage_out_edge_dict


        def _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping: Dict[Tuple, Tuple[MyExecPlan, List[MyExecPlan]]], 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
                ):
            
            models_to_estimate: List[int] = list()
            new_uniq_exec_plan_mappinp_keys = [k for k in uniq_exec_plan_mapping if k not in old_uniq_exec_plan_mapping_keys]
            new_uniq_sub_plan_groups = defaultdict(list)
            for k in new_uniq_exec_plan_mappinp_keys:
                plan, path_plans = uniq_exec_plan_mapping[k]
                new_uniq_sub_plan_groups[plan.model.model_id].append(uniq_exec_plan_mapping[k])
                if plan.model not in models_to_estimate:
                    models_to_estimate.append(plan.model)
            
            model_ids_of_each_layer, _ = _get_model_ids_of_different_levels(models_to_estimate)

            print(f"the uniq sub plan groups we found a round: # of  groups: {sum([len(vs) for vs in new_uniq_sub_plan_groups.values()])}") 

            # 0. we first init the uniq sub plan groups where no fake scheduling results will be shared
            for model_ids in model_ids_of_each_layer:
                print(f"uniq_sub_plan_group_objs:")
                for model_id in model_ids: 
                    for plan, plan_group in new_uniq_sub_plan_groups[model_id]:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                plans_to_wait = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    exec_plan._wait_for_remote_fake_scheduling()

        def _has_in_stage_inp_models(plan_group: List[MyExecPlan]):
            horizontally_fused_model_ids = [plan.model.input_model_ids for plan in plan_group if plan.model.independent_srcs]
            all_model_ids = set([plan.model.model_id for plan in plan_group])
            res = [((len(all_model_ids.intersection(inps))>0) or (False not in [self.model_dict[i].is_finished() for i in inps])) \
                   for inps in horizontally_fused_model_ids]
            if False in res:
                return False
            else: 
                return True


        not_finished_base_model_num = self.get_not_finished_base_model_num()

        
        tot_plan_groups: List[List[MyExecPlan]] = [[]]
        new_plan_groups: List[List[MyExecPlan]] = [[]]

        uniq_exec_plan_mapping = dict()
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()

        
        
        

        self.get_all_level_models(aggresive_for_horizontally_fused_model=False)
        model_correct_level_num = dict()
        for level_i, model_ids in enumerate(self.all_level_model_ids):
            model_correct_level_num.update({model_id:level_i for model_id in model_ids})
        
        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=True)       


        print(f"all_level_model_ids: {self.all_level_model_ids}")

        base_model_finish_status = self.get_base_model_finish_status()

        visit_model_level = -1


        time_analysis = [0, 0, 0, 0, 0, 0, 0]

        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)


            time1 = time.perf_counter()

            
            model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(cand_models)
            models_have_in_level_dependency = list()
            for _ in model_ids_of_each_layer[1:]:
                models_have_in_level_dependency.extend(_)

            
            models_have_in_level_dependency.extend([model.model_id for model in cand_models if model_correct_level_num[model.model_id]!=visit_model_level])
            models_have_in_level_dependency = list(set(models_have_in_level_dependency))

            


            
            if (visit_model_level > 0):
                
                
                
                cand_models = [model for model in cand_models if not model.can_be_vertically_fused_topologically]


            
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")

                
                
                
                
                


            
            


            time_analysis[0] += (time.perf_counter() - time1)


            for cand_plan_group in new_plan_groups:
                
                

                time1 = time.perf_counter()

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)


                print(f"time get runnable plans: {time.perf_counter() - time1}")
                time_analysis[1] += (time.perf_counter() - time1)
                time1 = time.perf_counter()

                
                
                
                good_runnable_exec_plan_keys_list = self._get_good_runnable_exec_plan_keys(
                    cand_plan_group, runnable_exec_plans_list,
                    models_have_in_level_dependency,
                    check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


                print(f"time get good plans: {time.perf_counter() - time1}")
                time_analysis[2] += (time.perf_counter() - time1)
                time1 = time.perf_counter()

                
                
                
                
                
                
                
                
                plan_groups = [cand_plan_group]
                old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
                _append_exec_plan(plan_groups, runnable_exec_plans_list, 0, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping, good_runnable_exec_plan_keys_list, 
                                  self.out_edge_dict)
                
                
                

                print(f"time append plans: {time.perf_counter() - time1}")
                time_analysis[3] += (time.perf_counter() - time1)
                time1 = time.perf_counter()


                if len(plan_groups) == 1:
                    

                    
                    if visit_model_level+1 >= len(self.all_level_model_ids):
                        if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_groups[0], tot_gpu_num
                            ):
                            tot_plan_groups.extend(plan_groups)
                else:

                    if visit_model_level+1 >= len(self.all_level_model_ids):
                        
                        plan_groups = [plan_groups[0]]+[plan_group for plan_group in plan_groups[1:] if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_group, tot_gpu_num
                            )]


                    time1 = time.perf_counter()

                    # prune some plan groups where horizontally fused models do not have in-stage input models
                    print(f"the groups we found a round: # of  groups before pruning: {len(plan_groups)}") 
                    plan_groups = plan_groups[:1] + [plan_group for plan_group in plan_groups[1:] if _has_in_stage_inp_models(plan_group)]


                    print(f"the groups we found a round: # of  groups after pruning: {len(plan_groups)}") 
                    for plan_group in plan_groups:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                    

                    time_analysis[4] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                        uniq_exec_plan_mapping, 
                        old_uniq_exec_plan_mapping_keys,
                        cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
                    )

                    print(f"time get costs: {time.perf_counter() - time1}")
                    time_analysis[5] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    
                    
                    
                    

                    
                    
                    
                    
                    
                    
                    
                    

                    
                    
                    
                    
                    

                    
                    
                    
                    
                    
                    

                    
                    plan_group_objs = [MyExecPlanGroup(
                        plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups[1:]]
                    good_plan_group_keys = [_update_good_plan_group_dict(
                        group_obj=group_obj, cost_table=cost_table, good_plan_group_dict=good_plan_group_dict
                        ) for group_obj in plan_group_objs]

                    
                    
                    
                    

                    good_plan_group_keys = [_ for _ in good_plan_group_keys if _ != None]
                    to_compare = sorted([_[0] for _ in good_plan_group_dict.values()], reverse=True)[top_k-1] if top_k <= len(good_plan_group_dict) else -1
                    good_plan_groups = [good_plan_group_dict[_][1].exec_plans for _ in good_plan_group_keys if good_plan_group_dict[_][0] > to_compare]
                    
                    
                    
                    
                    
                    tmp_new_plan_groups.extend(good_plan_groups)
                    
                    


                    print(f"time finish get costs: {time.perf_counter() - time1}")
                    time_analysis[6] += (time.perf_counter() - time1)

                
                
                
                
                
                
                
                
                


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


        
        
        



        
        
        
        

        plan_groups = [_[1] for _ in good_plan_group_dict.values()]

        print(f"time_analysis: {time_analysis}")

        return plan_groups










    def get_candidate_plan_groups_greedy_best_exec_plan_first(
        self, 
        gen_execplans_baseline:str,
        search_method_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
            NOTE: we greedily select the best exec plan that contribute to the overall throughput each time.
            NOTE: we add support for unfinished_model_loading_penalty and non-preemption setting.
        """

        def _directly_discard(
                gen_execplans_baseline: str, not_finished_base_model_num: int, 
                plan_group: List[MyExecPlan], tot_gpu_num: int):
            involved_base_model_num = sum([len(plan.get_base_model_ids()) for plan in plan_group])
            gpu_num_sum = get_tot_worker_num(plan_group)
            return (gen_execplans_baseline=='ours') \
                and (not_finished_base_model_num > involved_base_model_num) \
                    and (gpu_num_sum<tot_gpu_num)

        
        def _get_model_ids_of_different_levels(models: List[MyModelInfor]) -> Tuple[List[List[int]], Dict[int, List[int]]]:
            """
                NOTE: we will sort the models at the same time.
            """
            sorted_model_ids = list()
            sorted_models = list()
            all_model_ids = set([model.model_id for model in models])
            model_ids_of_each_layer = list()
            in_stage_out_edge_dict = defaultdict(list)
            
            while len(sorted_model_ids) < len(models):
                new_layer = list()
                for model in models:
                    if model.model_id in sorted_model_ids:
                        continue
                    in_stage_inp_model_ids = all_model_ids.intersection(model.input_model_ids)
                    if in_stage_inp_model_ids.issubset(sorted_model_ids):
                        new_layer.append(model.model_id)
                        sorted_models.append(model)
                        for i in in_stage_inp_model_ids:
                            in_stage_out_edge_dict[i].append(model.model_id)
                sorted_model_ids.extend(new_layer)
                model_ids_of_each_layer.append(new_layer)
            print(f"model_ids_of_each_layer: {model_ids_of_each_layer}, in_stage_out_edge_dict: {in_stage_out_edge_dict}")
            return sorted_models, model_ids_of_each_layer, in_stage_out_edge_dict


        def _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping: Dict[Tuple, Tuple[MyExecPlan, List[MyExecPlan]]], 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
                ):
            
            models_to_estimate: List[int] = list()
            new_uniq_exec_plan_mappinp_keys = [k for k in uniq_exec_plan_mapping if k not in old_uniq_exec_plan_mapping_keys]
            new_uniq_sub_plan_groups = defaultdict(list)
            for k in new_uniq_exec_plan_mappinp_keys:
                plan, path_plans = uniq_exec_plan_mapping[k]
                new_uniq_sub_plan_groups[plan.model.model_id].append(uniq_exec_plan_mapping[k])
                if plan.model not in models_to_estimate:
                    models_to_estimate.append(plan.model)
            
            _, model_ids_of_each_layer, _ = _get_model_ids_of_different_levels(models_to_estimate)

            print(f"the uniq sub plan groups we found a round: # of  groups: {sum([len(vs) for vs in new_uniq_sub_plan_groups.values()])}") 

            
            for model_ids in model_ids_of_each_layer:
                print(f"uniq_sub_plan_group_objs:")
                for model_id in model_ids: 
                    for plan, plan_group in new_uniq_sub_plan_groups[model_id]:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    print(f"waiting for exec plan: {(exec_plan.model.get_base_model_ids(), exec_plan.get_key())}")
                    exec_plan._wait_for_remote_fake_scheduling()

        def _has_in_stage_inp_models(plan_group: List[MyExecPlan]):
            horizontally_fused_model_ids = [plan.model.input_model_ids for plan in plan_group if plan.model.independent_srcs]
            all_model_ids = set([plan.model.model_id for plan in plan_group])
            res = [((len(all_model_ids.intersection(inps))>0) or (False not in [self.model_dict[i].is_finished() for i in inps])) \
                   for inps in horizontally_fused_model_ids]
            if False in res:
                return False
            else: 
                return True



        def estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(exec_plans: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            
            assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

            
            
            latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]    
            

            all_base_model_ids = np.concatenate([plan.get_base_model_ids() for plan in exec_plans])
            
            
            latencys_to_consider = [v for v, exec_plan in zip(latencys, exec_plans) if len(set(exec_plan.model.inp_base_model_ids).intersection(all_base_model_ids)) == 0]
            infer_stage_latency = min(latencys_to_consider)

            throughput = 0
            for exec_plan in exec_plans:
                
                throughput += exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=infer_stage_latency-exec_plan.extra_cost)

            
            if 'penalty' in search_method_baseline:
                sync_time = max([exec_plan.extra_cost-infer_stage_latency for exec_plan in exec_plans if infer_stage_latency<exec_plan.extra_cost])
                throughput = throughput*infer_stage_latency / (infer_stage_latency + sync_time)

            print(f"In estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, throughput: {throughput}, latencys: {latencys}, extra_times: {[plan.extra_cost for plan in exec_plans]}")


            return throughput


        def _get_correct_tmp_plan_group(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                exec_plan: MyExecPlan,
        ) -> List[MyExecPlan]:
            
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            plan_dict[exec_plan.model.model_id] = exec_plan
            models = [self.model_dict[model_id] for model_id in plan_dict]
            models, model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(models)
            sorted_plan_group = [plan_dict[model.model_id] for model in models]

            
            new_sorted_plan_group: List[MyExecPlan] = list()
            for i, plan in enumerate(sorted_plan_group):
                new_sorted_plan_group.append(
                    _update_uniq_exec_plan_mapping(new_sorted_plan_group, plan, uniq_exec_plan_mapping))
            
            return new_sorted_plan_group



        def _get_best_runnable_exec_plan(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                runnable_exec_plans_list: List[List[MyExecPlan]],
                
                
                check_gap: int, sort_input: bool,
                last_stage_exec_plans: List[MyExecPlan],
                cost_table: CostTable,
                tot_gpu_num = 4, 
                ) -> List[MyExecPlan]:
            """
                Get the best exec plan for each runnable models and each assigned comp gpu num.
                NOTE: 
                    1. do some plan validity check in advance.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    we do not compute their good exec plans, i.e., we think each exec plan needs to be checked.
                NOTE: in this function, we directly return the best exec plan to add
            """
            

            
            base_model_finish_status = self.get_base_model_finish_status()
            old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
            plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]] = list()
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            base_gpu_num = get_tot_worker_num(cand_plan_group)
            for i, runnable_exec_plans in enumerate(runnable_exec_plans_list):
                
                if len(runnable_exec_plans) == 0:
                    continue
                tmp_cand_plan_group = [plan for model_id, plan in plan_dict.items() if model_id != runnable_exec_plans[0].model.model_id]
                available_gpu_num = tot_gpu_num-get_tot_worker_num(tmp_cand_plan_group)

                
                gpu_nums = [get_tot_worker_num([exec_plan]) for exec_plan in runnable_exec_plans]
                
                plan_groups_to_check.extend([
                    (gpu_num + tot_gpu_num - available_gpu_num - base_gpu_num, _get_correct_tmp_plan_group(uniq_exec_plan_mapping, tmp_cand_plan_group, exec_plan))\
                        for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])

            
            
            _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping, 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
            )

            
            if len(plan_groups_to_check) == 0:
                return cand_plan_group
            base_throughput = 0
            if len(cand_plan_group) > 0:
                base_throughput = estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(cand_plan_group)
            plan_group_throughputs = [((estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(plan_group)-base_throughput)/gpu_num, gpu_num) for gpu_num, plan_group in plan_groups_to_check]

            print(f"plan_group_throughputs: {plan_group_throughputs}")
            

            
            best_ind = max(range(len(plan_group_throughputs)), key=lambda i: plan_group_throughputs[i])

            print(f"best_ind: {best_ind}")

            if plan_group_throughputs[best_ind][0] < 0:
                
                return cand_plan_group
            
            best_plan_group = plan_groups_to_check[best_ind][1]
            

            for throughput, (gpu_num, plan_group) in zip(plan_group_throughputs, plan_groups_to_check):
                print(f"candidate to add: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}, throughput: {throughput}")

            print(f"the best_plan_group we found a round: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in best_plan_group]}")

            
            return best_plan_group



        def _prune_different_exec_plans_from_last_stage(exec_plans: List[MyExecPlan]):
            last_stage_plan_dict = {plan.model.model_id for plan in last_stage_exec_plans}
            ret = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]
            return ret


        

        not_finished_base_model_num = self.get_not_finished_base_model_num()

        
        tot_plan_groups: List[List[MyExecPlan]] = [[]]
        new_plan_groups: List[List[MyExecPlan]] = [[]]

        uniq_exec_plan_mapping = dict()
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()

        
        
        

        self.get_all_level_models(aggresive_for_horizontally_fused_model=False)
        model_correct_level_num = dict()
        for level_i, model_ids in enumerate(self.all_level_model_ids):
            model_correct_level_num.update({model_id:level_i for model_id in model_ids})
        
        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=True)       


        print(f"all_level_model_ids: {self.all_level_model_ids}")

        base_model_finish_status = self.get_base_model_finish_status()

        visit_model_level = -1


        time_analysis = [0, 0, 0, 0, 0, 0, 0]

        best_plan_group = []

        first_level_models = self.get_models_at_given_level(0)

        gpu_nums_for_each_model: Dict[int, int] = dict() 

        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            

            cand_models = None
            if visit_model_level == 0:
                cand_models = self.get_models_at_given_level(visit_model_level)
            else:
                running_model_ids = [plan.model.model_id for plan in best_plan_group]
                reachable_model_ids = set(np.concatenate([self.out_edge_dict[model_id] for model_id in running_model_ids]))
                
                
                reachable_model_ids.update(running_model_ids)
                
                reachable_model_ids.update([model.model_id for model in first_level_models])
                print(f"running_model_ids: {running_model_ids}, reachable_model_ids: {reachable_model_ids}")
                cand_models = [self.model_dict[model_id] for model_id in reachable_model_ids]

            time1 = time.perf_counter()


            print(f"visit_model_level: {visit_model_level}, cand_models: {cand_models}")


            

            
            cand_models = [model for model in cand_models \
                if (model in first_level_models) or \
                    ((model not in first_level_models) and (not model.can_be_vertically_fused_topologically))]

            print(f"visit_model_level: {visit_model_level}, cand_models: {cand_models}")

            
            
            
            
            

            
            
            
            

            


            
            
            
            
            
            


            
            if len(cand_models) == 0:
                break


            
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)


                
                if 'no_preemption' in search_method_baseline:
                    exec_plans = _prune_different_exec_plans_from_last_stage(exec_plans)


                
                if model.model_id not in gpu_nums_for_each_model:
                    gpu_nums_for_each_model[model.model_id] = 0
                
                exec_plans = [plan for plan in exec_plans if get_tot_worker_num([plan]) > gpu_nums_for_each_model[model.model_id]]


                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")


                
                
                
                
                

            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)


            
            


            time_analysis[0] += (time.perf_counter() - time1)

            old_best_plan_gpu_num = get_tot_worker_num(best_plan_group)

            for cand_plan_group in [best_plan_group]:
                
                

                time1 = time.perf_counter()

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)


                print(f"time get runnable plans: {time.perf_counter() - time1}")
                time_analysis[1] += (time.perf_counter() - time1)
                time1 = time.perf_counter()

                
                best_plan_group = _get_best_runnable_exec_plan(
                    uniq_exec_plan_mapping,
                    cand_plan_group, runnable_exec_plans_list,
                    
                    check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)

                
                gpu_nums_for_each_model.update({plan.model.model_id:get_tot_worker_num([plan]) for plan in best_plan_group})


                print(f"time get good plans: {time.perf_counter() - time1}")
                time_analysis[2] += (time.perf_counter() - time1)
                time1 = time.perf_counter()

                
                
                
                
                
                
                
                
                


            if get_tot_worker_num(best_plan_group) == old_best_plan_gpu_num:
                break

        
        
        



        
        
        
        

        plan_groups = [best_plan_group]

        time1 = time.perf_counter()
        plan_groups = [MyExecPlanGroup(
            plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
        for plan_group in plan_groups:
            plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)
        print(f"time finish get costs: {time.perf_counter() - time1}")     
        
        

        print(f"time_analysis: {time_analysis}")

        return plan_groups


























    def get_candidate_plan_groups_greedy_best_exec_plan_first_LPT(
        self, 
        gen_execplans_baseline:str,
        search_method_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
            NOTE: we greedily select the best exec plan that contribute to the overall throughput each time.
            NOTE: we add support for unfinished_model_loading_penalty and non-preemption setting.
            LPT stands for ``Longest Processing Time``.
            IDEA: 
                1. sort the nodes by topological levels first and then their longest completion time (using the minimum possible number of GPUs);
                2. adding the nodes into the search space one by one (a node can only be added if its previous nodes are all selected);
                3. every time select the exec plan with the largest node throughput increase / gpu (not the largest stage throughput increase / gpu).
                !  Assume all chain models have been vertically fused.
        """

        def _directly_discard(
                gen_execplans_baseline: str, not_finished_base_model_num: int, 
                plan_group: List[MyExecPlan], tot_gpu_num: int):
            involved_base_model_num = sum([len(plan.get_base_model_ids()) for plan in plan_group])
            gpu_num_sum = get_tot_worker_num(plan_group)
            return (gen_execplans_baseline=='ours') \
                and (not_finished_base_model_num > involved_base_model_num) \
                    and (gpu_num_sum<tot_gpu_num)

        
        def _get_model_ids_of_different_levels(models: List[MyModelInfor]) -> Tuple[List[List[int]], Dict[int, List[int]]]:
            """
                NOTE: we will sort the models at the same time.
            """
            sorted_model_ids = list()
            sorted_models = list()
            all_model_ids = set([model.model_id for model in models])
            model_ids_of_each_layer = list()
            in_stage_out_edge_dict = defaultdict(list)
            
            while len(sorted_model_ids) < len(models):
                new_layer = list()
                for model in models:
                    if model.model_id in sorted_model_ids:
                        continue
                    in_stage_inp_model_ids = all_model_ids.intersection(model.input_model_ids)
                    if in_stage_inp_model_ids.issubset(sorted_model_ids):
                        new_layer.append(model.model_id)
                        sorted_models.append(model)
                        for i in in_stage_inp_model_ids:
                            in_stage_out_edge_dict[i].append(model.model_id)
                sorted_model_ids.extend(new_layer)
                model_ids_of_each_layer.append(new_layer)
            print(f"model_ids_of_each_layer: {model_ids_of_each_layer}, in_stage_out_edge_dict: {in_stage_out_edge_dict}")
            return sorted_models, model_ids_of_each_layer, in_stage_out_edge_dict


        def _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping: Dict[Tuple, Tuple[MyExecPlan, List[MyExecPlan]]], 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
                ):
            
            models_to_estimate: List[int] = list()
            new_uniq_exec_plan_mappinp_keys = [k for k in uniq_exec_plan_mapping if k not in old_uniq_exec_plan_mapping_keys]
            new_uniq_sub_plan_groups = defaultdict(list)
            for k in new_uniq_exec_plan_mappinp_keys:
                plan, path_plans = uniq_exec_plan_mapping[k]
                new_uniq_sub_plan_groups[plan.model.model_id].append(uniq_exec_plan_mapping[k])
                if plan.model not in models_to_estimate:
                    models_to_estimate.append(plan.model)
            
            _, model_ids_of_each_layer, _ = _get_model_ids_of_different_levels(models_to_estimate)

            print(f"the uniq sub plan groups we found a round: # of  groups: {sum([len(vs) for vs in new_uniq_sub_plan_groups.values()])}")

            
            for model_ids in model_ids_of_each_layer:
                print(f"uniq_sub_plan_group_objs:")
                for model_id in model_ids: 
                    for plan, plan_group in new_uniq_sub_plan_groups[model_id]:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    print(f"waiting for exec plan: {(exec_plan.model.get_base_model_ids(), exec_plan.get_key())}")
                    exec_plan._wait_for_remote_fake_scheduling()

        def _has_in_stage_inp_models(plan_group: List[MyExecPlan]):
            horizontally_fused_model_ids = [plan.model.input_model_ids for plan in plan_group if plan.model.independent_srcs]
            all_model_ids = set([plan.model.model_id for plan in plan_group])
            res = [((len(all_model_ids.intersection(inps))>0) or (False not in [self.model_dict[i].is_finished() for i in inps])) \
                   for inps in horizontally_fused_model_ids]
            if False in res:
                return False
            else: 
                return True



        def estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(exec_plans: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            
            assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

            
            
            latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]    
            

            all_base_model_ids = np.concatenate([plan.get_base_model_ids() for plan in exec_plans])
            
            
            latencys_to_consider = [v for v, exec_plan in zip(latencys, exec_plans) if len(set(exec_plan.model.inp_base_model_ids).intersection(all_base_model_ids)) == 0]
            infer_stage_latency = min(latencys_to_consider)

            throughput = 0
            for exec_plan in exec_plans:
                
                throughput += exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=infer_stage_latency-exec_plan.extra_cost)

            
            if 'penalty' in search_method_baseline:
                sync_time = max([exec_plan.extra_cost-infer_stage_latency for exec_plan in exec_plans if infer_stage_latency<exec_plan.extra_cost])
                throughput = throughput*infer_stage_latency / (infer_stage_latency + sync_time)

            print(f"In estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, throughput: {throughput}, latencys: {latencys}, extra_times: {[plan.extra_cost for plan in exec_plans]}")


            return throughput






        def estimate_throughput_sum_if_fake_scheduling_is_done_fast_version_with_penalty(exec_plans: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            
            assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

            throughput = 0
            for exec_plan in exec_plans:
                
                throughput += exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=max(exec_plan.total_latency_list))


            latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]
            print(f"In estimate_throughput_sum_if_fake_scheduling_is_done_fast_version_with_penalty: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, throughput: {throughput}, latencys: {latencys}, extra_times: {[plan.extra_cost for plan in exec_plans]}")


            return throughput






        def _get_delta_score_by_latency(
                plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]], cand_plan_group: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            base_latency_dict = dict()
            if len(cand_plan_group) > 0:
                base_latency_dict = {exec_plan.model.model_id: (max(exec_plan.total_latency_list) + exec_plan.extra_cost, get_tot_worker_num([exec_plan])) for exec_plan in cand_plan_group}

            delta = list()

            for gpu_num, plan_group in plan_groups_to_check:
                
                assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in plan_group], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in plan_group]

                latencys = {exec_plan.model.model_id: (max(exec_plan.total_latency_list) + exec_plan.extra_cost, get_tot_worker_num([exec_plan])) for exec_plan in plan_group}

                print(f"base_latency_dict: {base_latency_dict}")
                print(f"latencys: {latencys}")

                model_id, latency = [(model_id, latency) for model_id, (latency, gpucount) in latencys.items() if model_id not in base_latency_dict or base_latency_dict[model_id][1]!=gpucount][0]

                if model_id not in base_latency_dict:
                    delta.append((latency/gpu_num, gpu_num))
                else:
                    delta.append(((base_latency_dict[model_id][0]-latency)/gpu_num, gpu_num))


            return delta





        def _get_delta_score_by_model_throughput(
                plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]], cand_plan_group: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            def _get_model_throughput_latency_in_a_group(exec_plans:List[MyExecPlan]):
                
                assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

                
                
                latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]    
                

                all_base_model_ids = np.concatenate([plan.get_base_model_ids() for plan in exec_plans])
                
                
                latencys_to_consider = [v for v, exec_plan in zip(latencys, exec_plans) if len(set(exec_plan.model.inp_base_model_ids).intersection(all_base_model_ids)) == 0]
                infer_stage_latency = min(latencys_to_consider)

                infos = {exec_plan.model.model_id: (exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=infer_stage_latency-exec_plan.extra_cost), max(exec_plan.total_latency_list) + exec_plan.extra_cost, get_tot_worker_num([exec_plan])) for exec_plan in exec_plans}

                
                
                
                

                

                return infos

            base_infos = dict()
            if len(cand_plan_group) > 0:
                base_infos = _get_model_throughput_latency_in_a_group(cand_plan_group)

            delta = list()

            for gpu_num, plan_group in plan_groups_to_check:
                infos = _get_model_throughput_latency_in_a_group(plan_group)

                model_id, throughput, latency = [(model_id, throughput, latency) for model_id, (throughput, latency, gpu_count) in infos.items() if model_id not in base_infos or base_infos[model_id][2]!=gpu_count][0]

                if model_id not in base_infos:
                    delta.append((throughput/gpu_num, gpu_num))
                else:
                    delta.append(((throughput - base_infos[model_id][0])/gpu_num, gpu_num))
            return delta









        def _get_delta_score_by_throughput_with_nonpreempt_heuristic(
                plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]], cand_plan_group: List[MyExecPlan]
            ) -> Tuple[List[Tuple[float, int]], List[Tuple[int, List[MyExecPlan]]]]:
            """
                IDEA:
                    the nonpreemption heuristic: 
                        1. for an exec plan, if its 2*extra cost >= computation latency (to complete all its workloads), we call it invalid, i.e., not worth preemption.
                        2. for a model, if all its candidate exec plans except the one in the last stage are ``invalid``, then we must select its last-stage exec plan for it, i.e., preemption is forbiddened in this case.

                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            def _get_model_latency_to_complete(exec_plans:List[MyExecPlan]):
                
                assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

                infos = {exec_plan.model.model_id: (max(exec_plan.total_latency_list), exec_plan.extra_cost, get_tot_worker_num([exec_plan])) for exec_plan in exec_plans}

                print(f"In _get_model_latency_to_complete: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, infos: {infos}")

                return infos


            base_infos = dict()
            if len(cand_plan_group) > 0:
                base_infos = _get_model_latency_to_complete(cand_plan_group)


            plan_groups_dict = dict()
            last_stage_models = set([exec_plan.model.model_id for exec_plan in last_stage_exec_plans])
            pruned_plan_groups_to_check = list()
            no_preempt_latencys = dict()

            for gpu_num, plan_group in plan_groups_to_check:

                
                infos = _get_model_latency_to_complete(plan_group)

                model_id, comp_cost, extra_cost = [(model_id, comp_cost, extra_cost) for model_id, (comp_cost, extra_cost, gpu_count) in infos.items() if model_id not in base_infos or base_infos[model_id][2]!=gpu_count][0]

                if (model_id in last_stage_models) and (extra_cost == 0):
                    
                    no_preempt_latencys[model_id] = comp_cost + extra_cost

                if (model_id in last_stage_models) and (comp_cost <= 2*extra_cost):
                    
                    continue


                
                if 'no_preemption' in search_method_baseline:
                    if (model_id in last_stage_models) and (model_id not in base_infos) and (extra_cost==0):
                        pruned_plan_groups_to_check = [(gpu_num, plan_group)]
                        plan_groups_dict[model_id] = [(gpu_num, plan_group, comp_cost, extra_cost)]
                        break
                    if (model_id in last_stage_models) and (model_id in base_infos):
                        continue
                

                if model_id not in plan_groups_dict:
                    plan_groups_dict[model_id] = list()
                plan_groups_dict[model_id].append((gpu_num, plan_group, comp_cost, extra_cost))
                pruned_plan_groups_to_check.append((gpu_num, plan_group))

            
            pruned_plan_groups_to_check = list()
            pruned_plan_groups_dict = dict()
            for model_id in plan_groups_dict:
                pruned_plan_groups_dict[model_id] = list()
                if model_id not in no_preempt_latencys:
                    pruned_plan_groups_to_check.extend([(gpu_num, plan_group) for gpu_num, plan_group, comp_cost, extra_cost in plan_groups_dict[model_id]])
                else:
                    no_preempt_latency = no_preempt_latencys[model_id]
                    pruned_plan_groups_dict[model_id] = [(gpu_num, plan_group) for gpu_num, plan_group, comp_cost, extra_cost in plan_groups_dict[model_id] if (extra_cost == 0) or (no_preempt_latency/(comp_cost + extra_cost) >= 1.05)]
                    pruned_plan_groups_to_check.extend(pruned_plan_groups_dict[model_id])

            
            for model_id in pruned_plan_groups_dict:
                if (model_id in no_preempt_latencys) and (model_id not in base_infos):
                    if len(pruned_plan_groups_dict[model_id]) == 1:
                        pruned_plan_groups_to_check = pruned_plan_groups_dict[model_id]

            
            plan_group_throughputs = _get_delta_score_by_model_throughput(pruned_plan_groups_to_check, cand_plan_group)
            return plan_group_throughputs, pruned_plan_groups_to_check



            base_throughput = 0
            if len(cand_plan_group) > 0:
                base_throughput = estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(cand_plan_group)
            plan_group_throughputs = [((estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(plan_group)-base_throughput)/gpu_num, gpu_num) for gpu_num, plan_group in pruned_plan_groups_to_check]


            return plan_group_throughputs, pruned_plan_groups_to_check
















        def _get_correct_tmp_plan_group(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                exec_plan: MyExecPlan,
        ) -> List[MyExecPlan]:
            
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            plan_dict[exec_plan.model.model_id] = exec_plan
            models = [self.model_dict[model_id] for model_id in plan_dict]
            models, model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(models)
            sorted_plan_group = [plan_dict[model.model_id] for model in models]

            
            new_sorted_plan_group: List[MyExecPlan] = list()
            for i, plan in enumerate(sorted_plan_group):
                new_sorted_plan_group.append(
                    _update_uniq_exec_plan_mapping(new_sorted_plan_group, plan, uniq_exec_plan_mapping))
            
            return new_sorted_plan_group



        def _get_best_runnable_exec_plan(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                runnable_exec_plans_list: List[List[MyExecPlan]],
                
                
                check_gap: int, sort_input: bool,
                last_stage_exec_plans: List[MyExecPlan],
                cost_table: CostTable,
                tot_gpu_num = 4, 
                ) -> List[MyExecPlan]:
            """
                Get the best exec plan for each runnable models and each assigned comp gpu num.
                NOTE: 
                    1. do some validity check in advance.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    we do not compute their good exec plans, i.e., we think each exec plan needs to be checked.
                NOTE: in this function, we directly return the best exec plan to add
            """
            

            
            base_model_finish_status = self.get_base_model_finish_status()
            old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
            plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]] = list()
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            base_gpu_num = get_tot_worker_num(cand_plan_group)
            for i, runnable_exec_plans in enumerate(runnable_exec_plans_list):
                
                if len(runnable_exec_plans) == 0:
                    continue
                tmp_cand_plan_group = [plan for model_id, plan in plan_dict.items() if model_id != runnable_exec_plans[0].model.model_id]
                available_gpu_num = tot_gpu_num-get_tot_worker_num(tmp_cand_plan_group)

                
                gpu_nums = [get_tot_worker_num([exec_plan]) for exec_plan in runnable_exec_plans]
                
                plan_groups_to_check.extend([
                    (gpu_num + tot_gpu_num - available_gpu_num - base_gpu_num, _get_correct_tmp_plan_group(uniq_exec_plan_mapping, tmp_cand_plan_group, exec_plan))\
                        for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])

            
            
            _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping, 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
            )

            
            if len(plan_groups_to_check) == 0:
                return cand_plan_group
            
            
            
            
            

            
            
            

            
            
            

            
            plan_group_throughputs, plan_groups_to_check = _get_delta_score_by_throughput_with_nonpreempt_heuristic(
                plan_groups_to_check, cand_plan_group)
            if len(plan_groups_to_check) == 0:
                return cand_plan_group

            print(f"plan_group_throughputs: {plan_group_throughputs}")
            

            
            best_ind = max(range(len(plan_group_throughputs)), key=lambda i: plan_group_throughputs[i])

            print(f"best_ind: {best_ind}")

            if plan_group_throughputs[best_ind][0] < 0:
                
                return cand_plan_group
            
            best_plan_group = plan_groups_to_check[best_ind][1]
            

            for throughput, (gpu_num, plan_group) in zip(plan_group_throughputs, plan_groups_to_check):
                print(f"candidate to add: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}, throughput: {throughput}")

            print(f"the best_plan_group we found a round: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in best_plan_group]}")

            
            return best_plan_group







        def _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                runnable_exec_plans_list: List[List[MyExecPlan]],
                
                
                check_gap: int, sort_input: bool,
                last_stage_exec_plans: List[MyExecPlan],
                cost_table: CostTable,
                tot_gpu_num = 4, 
                ) -> Dict[int, float]:
            """
                Compute the latency for each exec plan given a current cand_plan_group.
                Return the latency for each model's min-gpu exec plan in a dict format: {model_id, latency}
            """
            

            
            base_model_finish_status = self.get_base_model_finish_status()
            old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
            plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]] = list()
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            
            for i, runnable_exec_plans in enumerate(runnable_exec_plans_list):
                
                if len(runnable_exec_plans) == 0:
                    continue
                tmp_cand_plan_group = [plan for model_id, plan in plan_dict.items() if model_id != runnable_exec_plans[0].model.model_id]
                available_gpu_num = tot_gpu_num-get_tot_worker_num(tmp_cand_plan_group)

                
                gpu_nums = [get_tot_worker_num([exec_plan]) for exec_plan in runnable_exec_plans]
                
                plan_groups_to_check.extend([
                    (_get_correct_tmp_plan_group(uniq_exec_plan_mapping, tmp_cand_plan_group, exec_plan), exec_plan)\
                        for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])

            
            
            _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping, 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
            )



            min_gpu_latency_dict = {runnable_exec_plans[0].model.model_id: float('inf')  for runnable_exec_plans in runnable_exec_plans_list}
            
            for plan_group, exec_plan in plan_groups_to_check:
                for plan in plan_group:
                    if plan.model.model_id == exec_plan.model.model_id:
                        min_gpu_latency_dict[exec_plan.model.model_id] = max(plan.total_latency_list) + plan.extra_cost
            
            return min_gpu_latency_dict








        
        
        
        





        def _gen_cand_exec_plans_using_more_gpus(cand_models: List[MyModelInfor], gpu_nums_for_each_model: Dict[int, int]):
            exec_plans_list = list()
            last_stage_plan_dict = {plan.model.model_id:plan.get_key() for plan in last_stage_exec_plans}
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)

                
                if 'no_preemption' in search_method_baseline:
                    if model.model_id in last_stage_plan_dict:
                        exec_plans = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]
                    


                
                if model.model_id not in gpu_nums_for_each_model:
                    gpu_nums_for_each_model[model.model_id] = 0
                
                exec_plans = [plan for plan in exec_plans if get_tot_worker_num([plan]) > gpu_nums_for_each_model[model.model_id]]


                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")
            return exec_plans_list







        not_finished_base_model_num = self.get_not_finished_base_model_num()

        
        tot_plan_groups: List[List[MyExecPlan]] = [[]]
        new_plan_groups: List[List[MyExecPlan]] = [[]]

        uniq_exec_plan_mapping = dict()
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()

        
        
        

        self.get_all_level_models(aggresive_for_horizontally_fused_model=False)
        model_correct_level_num = dict()
        for level_i, model_ids in enumerate(self.all_level_model_ids):
            model_correct_level_num.update({model_id:level_i for model_id in model_ids})
        
        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=True)       


        print(f"all_level_model_ids: {self.all_level_model_ids}")

        base_model_finish_status = self.get_base_model_finish_status()

        visit_model_level = -1


        time_analysis = [0, 0, 0, 0, 0, 0, 0]

        best_plan_group = []

        first_level_models = self.get_models_at_given_level(0)

        gpu_nums_for_each_model: Dict[int, int] = dict() 

        tot_cand_models = []


        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)
            is_last_level = (len(cand_models) == 0)

            
            
            
            
            
            
            
            
            
            
            
            
            

            time1 = time.perf_counter()


            print(f"visit_model_level: {visit_model_level}, cand_models: {cand_models}")


            

            
            cand_models = [model for model in cand_models \
                if (model in first_level_models) or \
                    ((model not in first_level_models) and (not model.can_be_vertically_fused_topologically))]

            print(f"visit_model_level: {visit_model_level}, cand_models: {cand_models}")



            
            
            
            
            

            
            
            
            

            


            
            
            
            
            
            


            
            if (len(cand_models) == 0) and (len(tot_cand_models) == 0):
                break


            
            exec_plans_list = _gen_cand_exec_plans_using_more_gpus(cand_models, gpu_nums_for_each_model)
            
            
            
            


            
            
            


            
            
            
            
            


            
            


            
            
            
            
            

            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)



            
            _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                best_plan_group,
                exec_plans_list,
                check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)



            
            min_gpu_plans = [[min(exec_plans, key=lambda plan: get_tot_worker_num([plan]))] for exec_plans in exec_plans_list]
            
            min_gpu_latency_dict: Dict[int, float] = _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                best_plan_group,
                min_gpu_plans,
                check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)
            
            last_stage_plan_dict = {plan.model.model_id for plan in last_stage_exec_plans}
            cand_models = sorted(cand_models, key=lambda model: (model.model_id in last_stage_plan_dict, -model.check_order, min_gpu_latency_dict[model.model_id]), reverse=True)
            


            
            should_stop = False
            while((len(cand_models) > 0) or is_last_level):
                
                if not is_last_level:
                    tot_cand_models.append(cand_models[0])
                    cand_models = cand_models[1:]


                print(f"tot_cand_models: {[model.model_id for model in tot_cand_models]}")


                
                exec_plans_list = _gen_cand_exec_plans_using_more_gpus(tot_cand_models, gpu_nums_for_each_model)

                
                
                


                time_analysis[0] += (time.perf_counter() - time1)

                old_best_plan_gpu_num = get_tot_worker_num(best_plan_group)
                old_best_plan_model_num = len(best_plan_group)

                for cand_plan_group in [best_plan_group]:
                    
                    

                    time1 = time.perf_counter()

                    runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, tot_cand_models, exec_plans_list)


                    print(f"time get runnable plans: {time.perf_counter() - time1}")
                    time_analysis[1] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    
                    best_plan_group = _get_best_runnable_exec_plan(
                        uniq_exec_plan_mapping,
                        cand_plan_group, runnable_exec_plans_list,
                        
                        check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)

                    
                    gpu_nums_for_each_model.update({plan.model.model_id:get_tot_worker_num([plan]) for plan in best_plan_group})


                    print(f"time get good plans: {time.perf_counter() - time1}")
                    time_analysis[2] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    
                    
                    
                    
                    
                    
                    
                    
                    


                
                if not is_last_level:
                    if len(best_plan_group) == old_best_plan_model_num:
                        cand_models = [tot_cand_models[-1]] + cand_models
                        tot_cand_models = tot_cand_models[:-1]           

                
                if get_tot_worker_num(best_plan_group) == old_best_plan_gpu_num:
                    should_stop = True
                    break

            
            
            if should_stop:
                break


        
        global _MODEL_CHECK_ORDER
        for model in tot_cand_models:
            if model.check_order == int(1e9):
                model.check_order = _MODEL_CHECK_ORDER
                _MODEL_CHECK_ORDER += 1
        
        
        
        
        



        
        
        
        

        plan_groups = [best_plan_group]

        time1 = time.perf_counter()
        plan_groups = [MyExecPlanGroup(
            plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
        for plan_group in plan_groups:
            plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)
        print(f"time finish get costs: {time.perf_counter() - time1}")     
        
        

        print(f"time_analysis: {time_analysis}")

        return plan_groups











    def get_candidate_plan_groups_greedy_best_exec_plan_first_LPT_last_stage_models_first(
        self, 
        gen_execplans_baseline:str,
        search_method_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
            NOTE: we greedily select the best exec plan that contribute to the overall throughput each time.
            NOTE: we add support for unfinished_model_loading_penalty and non-preemption setting.
            LPT stands for ``Longest Processing Time``.
            IDEA: 
                1. sort the nodes by topological levels first and then their longest completion time (using the minimum possible number of GPUs);
                2. adding the nodes into the search space one by one (a node can only be added if its previous nodes are all selected);
                3. every time select the exec plan with the largest node throughput increase / gpu (not the largest stage throughput increase / gpu).
                !  Assume all chain models have been vertically fused.
        """

        def _directly_discard(
                gen_execplans_baseline: str, not_finished_base_model_num: int, 
                plan_group: List[MyExecPlan], tot_gpu_num: int):
            involved_base_model_num = sum([len(plan.get_base_model_ids()) for plan in plan_group])
            gpu_num_sum = get_tot_worker_num(plan_group)
            return (gen_execplans_baseline=='ours') \
                and (not_finished_base_model_num > involved_base_model_num) \
                    and (gpu_num_sum<tot_gpu_num)

        
        def _get_model_ids_of_different_levels(models: List[MyModelInfor]) -> Tuple[List[List[int]], Dict[int, List[int]]]:
            """
                NOTE: we will sort the models at the same time.
            """
            sorted_model_ids = list()
            sorted_models = list()
            all_model_ids = set([model.model_id for model in models])
            model_ids_of_each_layer = list()
            in_stage_out_edge_dict = defaultdict(list)
            
            while len(sorted_model_ids) < len(models):
                new_layer = list()
                for model in models:
                    if model.model_id in sorted_model_ids:
                        continue
                    in_stage_inp_model_ids = all_model_ids.intersection(model.input_model_ids)
                    if in_stage_inp_model_ids.issubset(sorted_model_ids):
                        new_layer.append(model.model_id)
                        sorted_models.append(model)
                        for i in in_stage_inp_model_ids:
                            in_stage_out_edge_dict[i].append(model.model_id)
                sorted_model_ids.extend(new_layer)
                model_ids_of_each_layer.append(new_layer)
            print(f"model_ids_of_each_layer: {model_ids_of_each_layer}, in_stage_out_edge_dict: {in_stage_out_edge_dict}")
            return sorted_models, model_ids_of_each_layer, in_stage_out_edge_dict


        def _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping: Dict[Tuple, Tuple[MyExecPlan, List[MyExecPlan]]], 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
                ):
            
            models_to_estimate: List[int] = list()
            new_uniq_exec_plan_mappinp_keys = [k for k in uniq_exec_plan_mapping if k not in old_uniq_exec_plan_mapping_keys]
            new_uniq_sub_plan_groups = defaultdict(list)
            for k in new_uniq_exec_plan_mappinp_keys:
                plan, path_plans = uniq_exec_plan_mapping[k]
                new_uniq_sub_plan_groups[plan.model.model_id].append(uniq_exec_plan_mapping[k])
                if plan.model not in models_to_estimate:
                    models_to_estimate.append(plan.model)
            
            _, model_ids_of_each_layer, _ = _get_model_ids_of_different_levels(models_to_estimate)

            print(f"the uniq sub plan groups we found a round: # of  groups: {sum([len(vs) for vs in new_uniq_sub_plan_groups.values()])}")

            
            for model_ids in model_ids_of_each_layer:
                print(f"uniq_sub_plan_group_objs:")
                for model_id in model_ids: 
                    for plan, plan_group in new_uniq_sub_plan_groups[model_id]:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    print(f"waiting for exec plan: {(exec_plan.model.get_base_model_ids(), exec_plan.get_key())}")
                    exec_plan._wait_for_remote_fake_scheduling()

        def _has_in_stage_inp_models(plan_group: List[MyExecPlan]):
            horizontally_fused_model_ids = [plan.model.input_model_ids for plan in plan_group if plan.model.independent_srcs]
            all_model_ids = set([plan.model.model_id for plan in plan_group])
            res = [((len(all_model_ids.intersection(inps))>0) or (False not in [self.model_dict[i].is_finished() for i in inps])) \
                   for inps in horizontally_fused_model_ids]
            if False in res:
                return False
            else: 
                return True



        def estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(exec_plans: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            
            assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

            
            
            latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]    
            

            all_base_model_ids = np.concatenate([plan.get_base_model_ids() for plan in exec_plans])
            
            
            latencys_to_consider = [v for v, exec_plan in zip(latencys, exec_plans) if len(set(exec_plan.model.inp_base_model_ids).intersection(all_base_model_ids)) == 0]
            infer_stage_latency = min(latencys_to_consider)

            throughput = 0
            for exec_plan in exec_plans:
                
                throughput += exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=infer_stage_latency-exec_plan.extra_cost)

            
            if 'penalty' in search_method_baseline:
                sync_time = max([exec_plan.extra_cost-infer_stage_latency for exec_plan in exec_plans if infer_stage_latency<exec_plan.extra_cost])
                throughput = throughput*infer_stage_latency / (infer_stage_latency + sync_time)

            print(f"In estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, throughput: {throughput}, latencys: {latencys}, extra_times: {[plan.extra_cost for plan in exec_plans]}")


            return throughput





        def estimate_throughput_sum_if_fake_scheduling_is_done_fast_version_with_penalty(exec_plans: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            
            assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

            throughput = 0
            for exec_plan in exec_plans:
                
                throughput += exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=max(exec_plan.total_latency_list))


            latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]
            print(f"In estimate_throughput_sum_if_fake_scheduling_is_done_fast_version_with_penalty: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, throughput: {throughput}, latencys: {latencys}, extra_times: {[plan.extra_cost for plan in exec_plans]}")


            return throughput





        def _get_correct_tmp_plan_group(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                exec_plan: MyExecPlan,
        ) -> List[MyExecPlan]:
            
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            plan_dict[exec_plan.model.model_id] = exec_plan
            models = [self.model_dict[model_id] for model_id in plan_dict]
            models, model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(models)
            sorted_plan_group = [plan_dict[model.model_id] for model in models]

            
            new_sorted_plan_group: List[MyExecPlan] = list()
            for i, plan in enumerate(sorted_plan_group):
                new_sorted_plan_group.append(
                    _update_uniq_exec_plan_mapping(new_sorted_plan_group, plan, uniq_exec_plan_mapping))
            
            return new_sorted_plan_group



        def _get_best_runnable_exec_plan(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                runnable_exec_plans_list: List[List[MyExecPlan]],
                
                
                check_gap: int, sort_input: bool,
                last_stage_exec_plans: List[MyExecPlan],
                cost_table: CostTable,
                tot_gpu_num = 4, 
                ) -> List[MyExecPlan]:
            """
                Get the best exec plan for each runnable models and each assigned comp gpu num.
                NOTE: 
                    1. do some validity check in advance.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    we do not compute their good exec plans, i.e., we think each exec plan needs to be checked.
                NOTE: in this function, we directly return the best exec plan to add
            """
            

            
            base_model_finish_status = self.get_base_model_finish_status()
            old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
            plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]] = list()
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            base_gpu_num = get_tot_worker_num(cand_plan_group)
            for i, runnable_exec_plans in enumerate(runnable_exec_plans_list):
                
                if len(runnable_exec_plans) == 0:
                    continue
                tmp_cand_plan_group = [plan for model_id, plan in plan_dict.items() if model_id != runnable_exec_plans[0].model.model_id]
                available_gpu_num = tot_gpu_num-get_tot_worker_num(tmp_cand_plan_group)

                
                gpu_nums = [get_tot_worker_num([exec_plan]) for exec_plan in runnable_exec_plans]
                
                plan_groups_to_check.extend([
                    (gpu_num + tot_gpu_num - available_gpu_num - base_gpu_num, _get_correct_tmp_plan_group(uniq_exec_plan_mapping, tmp_cand_plan_group, exec_plan))\
                        for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])

            
            
            _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping, 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
            )

            
            if len(plan_groups_to_check) == 0:
                return cand_plan_group
            base_throughput = 0
            if len(cand_plan_group) > 0:
                base_throughput = estimate_throughput_sum_if_fake_scheduling_is_done_fast_version_with_penalty(cand_plan_group)
            plan_group_throughputs = [((estimate_throughput_sum_if_fake_scheduling_is_done_fast_version_with_penalty(plan_group)-base_throughput)/gpu_num, gpu_num) for gpu_num, plan_group in plan_groups_to_check]

            print(f"plan_group_throughputs: {plan_group_throughputs}")
            

            
            best_ind = max(range(len(plan_group_throughputs)), key=lambda i: plan_group_throughputs[i])

            print(f"best_ind: {best_ind}")

            if plan_group_throughputs[best_ind][0] < 0:
                
                return cand_plan_group
            
            best_plan_group = plan_groups_to_check[best_ind][1]
            

            for throughput, (gpu_num, plan_group) in zip(plan_group_throughputs, plan_groups_to_check):
                print(f"candidate to add: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}, throughput: {throughput}")

            print(f"the best_plan_group we found a round: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in best_plan_group]}")

            
            return best_plan_group







        def _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                cand_plan_group: List[MyExecPlan],
                runnable_exec_plans_list: List[List[MyExecPlan]],
                
                
                check_gap: int, sort_input: bool,
                last_stage_exec_plans: List[MyExecPlan],
                cost_table: CostTable,
                tot_gpu_num = 4, 
                ) -> Dict[int, float]:
            """
                Compute the latency for each exec plan given a current cand_plan_group.
                Return the latency for each model's min-gpu exec plan in a dict format: {model_id, latency}
            """
            

            
            base_model_finish_status = self.get_base_model_finish_status()
            old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
            plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]] = list()
            plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in cand_plan_group}
            
            for i, runnable_exec_plans in enumerate(runnable_exec_plans_list):
                
                if len(runnable_exec_plans) == 0:
                    continue
                tmp_cand_plan_group = [plan for model_id, plan in plan_dict.items() if model_id != runnable_exec_plans[0].model.model_id]
                available_gpu_num = tot_gpu_num-get_tot_worker_num(tmp_cand_plan_group)

                
                gpu_nums = [get_tot_worker_num([exec_plan]) for exec_plan in runnable_exec_plans]
                
                plan_groups_to_check.extend([
                    (_get_correct_tmp_plan_group(uniq_exec_plan_mapping, tmp_cand_plan_group, exec_plan), exec_plan)\
                        for gpu_num, exec_plan in zip(gpu_nums, runnable_exec_plans) if gpu_num <=  available_gpu_num])

            
            
            _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping, 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
            )



            min_gpu_latency_dict = {runnable_exec_plans[0].model.model_id: float('inf')  for runnable_exec_plans in runnable_exec_plans_list}
            
            for plan_group, exec_plan in plan_groups_to_check:
                for plan in plan_group:
                    if plan.model.model_id == exec_plan.model.model_id:
                        min_gpu_latency_dict[exec_plan.model.model_id] = max(plan.total_latency_list) + plan.extra_cost
            
            return min_gpu_latency_dict








        def _prune_different_exec_plans_from_last_stage(exec_plans: List[MyExecPlan]):
            last_stage_plan_dict = {plan.model.model_id for plan in last_stage_exec_plans}
            ret = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]
            return ret





        def _gen_cand_exec_plans_using_more_gpus(cand_models: List[MyModelInfor], gpu_nums_for_each_model: Dict[int, int]):
            exec_plans_list = list()
            last_stage_plan_dict = {plan.model.model_id:plan.get_key() for plan in last_stage_exec_plans}
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)

                
                if 'no_preemption' in search_method_baseline:
                    if model.model_id in last_stage_plan_dict:
                        exec_plans = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]
                    


                
                if model.model_id not in gpu_nums_for_each_model:
                    gpu_nums_for_each_model[model.model_id] = 0
                
                exec_plans = [plan for plan in exec_plans if get_tot_worker_num([plan]) > gpu_nums_for_each_model[model.model_id]]


                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")
            return exec_plans_list

        





        not_finished_base_model_num = self.get_not_finished_base_model_num()

        
        tot_plan_groups: List[List[MyExecPlan]] = [[]]
        new_plan_groups: List[List[MyExecPlan]] = [[]]

        uniq_exec_plan_mapping = dict()
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()

        
        
        

        self.get_all_level_models(aggresive_for_horizontally_fused_model=False)
        model_correct_level_num = dict()
        for level_i, model_ids in enumerate(self.all_level_model_ids):
            model_correct_level_num.update({model_id:level_i for model_id in model_ids})
        
        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=True)       


        print(f"all_level_model_ids: {self.all_level_model_ids}")

        base_model_finish_status = self.get_base_model_finish_status()

        visit_model_level = -1


        time_analysis = [0, 0, 0, 0, 0, 0, 0]

        best_plan_group = []

        first_level_models = self.get_models_at_given_level(0)

        gpu_nums_for_each_model: Dict[int, int] = dict() 

        tot_cand_models = []


        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)
            is_last_level = (len(cand_models) == 0)

            
            
            
            
            
            
            
            
            
            
            
            
            

            time1 = time.perf_counter()


            print(f"visit_model_level: {visit_model_level}, cand_models: {cand_models}")


            

            
            cand_models = [model for model in cand_models \
                if (model in first_level_models) or \
                    ((model not in first_level_models) and (not model.can_be_vertically_fused_topologically))]

            print(f"visit_model_level: {visit_model_level}, cand_models: {cand_models}")



            
            
            
            
            

            
            
            
            

            


            
            
            
            
            
            


            
            if (len(cand_models) == 0) and (len(tot_cand_models) == 0):
                break


            
            exec_plans_list = _gen_cand_exec_plans_using_more_gpus(cand_models, gpu_nums_for_each_model)
            
            
            
            


            
            
            


            
            
            
            
            


            
            


            
            
            
            
            

            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)





            
            min_gpu_plans = [[min(exec_plans, key=lambda plan: get_tot_worker_num([plan]))] for exec_plans in exec_plans_list]
            
            min_gpu_latency_dict: Dict[int, float] = _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                best_plan_group,
                min_gpu_plans,
                check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)
            
            
            last_stage_plan_dict = {plan.model.model_id for plan in last_stage_exec_plans}
            cand_models = sorted(cand_models, key=lambda model: (model.model_id in last_stage_plan_dict, min_gpu_latency_dict[model.model_id]), reverse=True)


            
            should_stop = False
            while((len(cand_models) > 0) or is_last_level):
                
                if not is_last_level:
                    tot_cand_models.append(cand_models[0])
                    cand_models = cand_models[1:]


                
                exec_plans_list = _gen_cand_exec_plans_using_more_gpus(tot_cand_models, gpu_nums_for_each_model)

                
                
                


                time_analysis[0] += (time.perf_counter() - time1)

                old_best_plan_gpu_num = get_tot_worker_num(best_plan_group)
                old_best_plan_model_num = len(best_plan_group)

                for cand_plan_group in [best_plan_group]:
                    
                    

                    time1 = time.perf_counter()

                    runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, tot_cand_models, exec_plans_list)


                    print(f"time get runnable plans: {time.perf_counter() - time1}")
                    time_analysis[1] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    
                    best_plan_group = _get_best_runnable_exec_plan(
                        uniq_exec_plan_mapping,
                        cand_plan_group, runnable_exec_plans_list,
                        
                        check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)

                    
                    gpu_nums_for_each_model.update({plan.model.model_id:get_tot_worker_num([plan]) for plan in best_plan_group})


                    print(f"time get good plans: {time.perf_counter() - time1}")
                    time_analysis[2] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    
                    
                    
                    
                    
                    
                    
                    
                    


                
                if not is_last_level:
                    if len(best_plan_group) == old_best_plan_model_num:
                        cand_models = [tot_cand_models[-1]] + cand_models
                        tot_cand_models = tot_cand_models[:-1]           

                
                if get_tot_worker_num(best_plan_group) == old_best_plan_gpu_num:
                    should_stop = True
                    break

            
            
            if should_stop:
                break


        
        
        



        
        
        
        

        plan_groups = [best_plan_group]

        time1 = time.perf_counter()
        plan_groups = [MyExecPlanGroup(
            plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
        for plan_group in plan_groups:
            plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)
        print(f"time finish get costs: {time.perf_counter() - time1}")     
        
        

        print(f"time_analysis: {time_analysis}")

        return plan_groups




















    
    def get_candidate_plan_groups_min_gpu_use_all_gpus(
        self, 
        gen_execplans_baseline:str,
        search_method_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        """
            Get the candidate plan groups following the last_stage_exec_plans.
            NOTE: here we only ensure the validity of the candidate plan groups;
                we do not select good ones from them.
            NOTE: this strategy tries to find the plan group with the most model running concurrently.
                if there are additional gpus, we will try to divede the gpus to the runnable models as evenly as possible.
            NOTE: we add support for unfinished_model_loading_penalty and non-preemption setting.
        """

        def _directly_discard(
                gen_execplans_baseline: str, not_finished_base_model_num: int, 
                plan_group: List[MyExecPlan], tot_gpu_num: int):
            involved_base_model_num = sum([len(plan.get_base_model_ids()) for plan in plan_group])
            gpu_num_sum = get_tot_worker_num(plan_group)
            return (gen_execplans_baseline=='ours') \
                and (not_finished_base_model_num > involved_base_model_num) \
                    and (gpu_num_sum<tot_gpu_num)

        
        def _get_model_ids_of_different_levels(models: List[MyModelInfor]) -> Tuple[List[List[int]], Dict[int, List[int]]]:
            """
                NOTE: we will sort the models at the same time.
            """
            sorted_model_ids = list()
            sorted_models = list()
            all_model_ids = set([model.model_id for model in models])
            model_ids_of_each_layer = list()
            in_stage_out_edge_dict = defaultdict(list)
            
            while len(sorted_model_ids) < len(models):
                new_layer = list()
                for model in models:
                    if model.model_id in sorted_model_ids:
                        continue
                    in_stage_inp_model_ids = all_model_ids.intersection(model.input_model_ids)
                    if in_stage_inp_model_ids.issubset(sorted_model_ids):
                        new_layer.append(model.model_id)
                        sorted_models.append(model)
                        for i in in_stage_inp_model_ids:
                            in_stage_out_edge_dict[i].append(model.model_id)
                sorted_model_ids.extend(new_layer)
                model_ids_of_each_layer.append(new_layer)
            print(f"model_ids_of_each_layer: {model_ids_of_each_layer}, in_stage_out_edge_dict: {in_stage_out_edge_dict}")
            return sorted_models, model_ids_of_each_layer, in_stage_out_edge_dict


        def _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping: Dict[Tuple, Tuple[MyExecPlan, List[MyExecPlan]]], 
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
                ):
            
            models_to_estimate: List[int] = list()
            new_uniq_exec_plan_mappinp_keys = [k for k in uniq_exec_plan_mapping if k not in old_uniq_exec_plan_mapping_keys]
            new_uniq_sub_plan_groups = defaultdict(list)
            for k in new_uniq_exec_plan_mappinp_keys:
                plan, path_plans = uniq_exec_plan_mapping[k]
                new_uniq_sub_plan_groups[plan.model.model_id].append(uniq_exec_plan_mapping[k])
                if plan.model not in models_to_estimate:
                    models_to_estimate.append(plan.model)
            
            _, model_ids_of_each_layer, _ = _get_model_ids_of_different_levels(models_to_estimate)

            print(f"the uniq sub plan groups we found a round: # of  groups: {sum([len(vs) for vs in new_uniq_sub_plan_groups.values()])}")

            
            for model_ids in model_ids_of_each_layer:
                print(f"uniq_sub_plan_group_objs:")
                for model_id in model_ids: 
                    for plan, plan_group in new_uniq_sub_plan_groups[model_id]:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    print(f"waiting for exec plan: {(exec_plan.model.get_base_model_ids(), exec_plan.get_key())}")
                    exec_plan._wait_for_remote_fake_scheduling()
                    

        def _has_in_stage_inp_models(plan_group: List[MyExecPlan]):
            horizontally_fused_model_ids = [plan.model.input_model_ids for plan in plan_group if plan.model.independent_srcs]
            all_model_ids = set([plan.model.model_id for plan in plan_group])
            res = [((len(all_model_ids.intersection(inps))>0) or (False not in [self.model_dict[i].is_finished() for i in inps])) \
                   for inps in horizontally_fused_model_ids]
            if False in res:
                return False
            else: 
                return True



        def estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(exec_plans: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            
            assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

            
            
            latencys = [max(exec_plan.total_latency_list) + exec_plan.extra_cost for exec_plan in exec_plans]    
            

            all_base_model_ids = np.concatenate([plan.get_base_model_ids() for plan in exec_plans])
            
            
            latencys_to_consider = [v for v, exec_plan in zip(latencys, exec_plans) if len(set(exec_plan.model.inp_base_model_ids).intersection(all_base_model_ids)) == 0]
            infer_stage_latency = min(latencys_to_consider)

            throughput = 0
            for exec_plan in exec_plans:
                
                throughput += exec_plan.get_throughput_at_stop_time_based_on_cached_throughputs(stage_stop_time=infer_stage_latency-exec_plan.extra_cost)

            
            if 'penalty' in search_method_baseline:
                sync_time = max([exec_plan.extra_cost-infer_stage_latency for exec_plan in exec_plans if infer_stage_latency<exec_plan.extra_cost])
                throughput = throughput*infer_stage_latency / (infer_stage_latency + sync_time)

            print(f"In estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in exec_plans]}, throughput: {throughput}, latencys: {latencys}")


            return throughput




        def update_real_uniq_exec_plan_mapping(real_uniq_exec_plan_mapping, plan_groups: List[List[MyExecPlan]]):
            """
                Get the uniq_exec_plan_mapping keys for each plan in each plan_group of the input plan_groups.
            """
            new_plan_groups: List[List[MyExecPlan]] = list()
            for plan_group in plan_groups:
                
                plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in plan_group}
                models = [self.model_dict[model_id] for model_id in plan_dict]
                models, model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(models)
                sorted_plan_group = [plan_dict[model.model_id] for model in models]

                
                new_sorted_plan_group: List[MyExecPlan] = list()
                for i, plan in enumerate(sorted_plan_group):
                    new_sorted_plan_group.append(
                        _update_uniq_exec_plan_mapping(new_sorted_plan_group, plan, real_uniq_exec_plan_mapping))
                
                new_plan_groups.append(new_sorted_plan_group)
            return new_plan_groups



        def _prune_different_exec_plans_from_last_stage(exec_plans: List[MyExecPlan], model_id: int):

            last_stage_plan_dict = {plan.model.model_id:plan.get_key() for plan in last_stage_exec_plans}
            
            if model_id in last_stage_plan_dict:
                exec_plans = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]
            
            return exec_plans

            
            
            



        not_finished_base_model_num = self.get_not_finished_base_model_num()

        
        tot_plan_groups: List[List[MyExecPlan]] = [[]]
        new_plan_groups: List[List[MyExecPlan]] = [[]]

        uniq_exec_plan_mapping = dict()
        real_uniq_exec_plan_mapping = dict()
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]] = dict()
        all_plan_group_list: List[List[MyExecPlan]] = list()

        
        
        

        self.get_all_level_models(aggresive_for_horizontally_fused_model=False)
        model_correct_level_num = dict()
        for level_i, model_ids in enumerate(self.all_level_model_ids):
            model_correct_level_num.update({model_id:level_i for model_id in model_ids})
        
        
        self.get_all_level_models(aggresive_for_horizontally_fused_model=True)       


        print(f"all_level_model_ids: {self.all_level_model_ids}")

        base_model_finish_status = self.get_base_model_finish_status()

        visit_model_level = -1


        time_analysis = [0, 0, 0, 0, 0, 0, 0]

        while True:
            tmp_new_plan_groups = []
            visit_model_level += 1
            cand_models = self.get_models_at_given_level(visit_model_level)


            time1 = time.perf_counter()

            
            cand_models, model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(cand_models)
            models_have_in_level_dependency = list()
            for _ in model_ids_of_each_layer[1:]:
                models_have_in_level_dependency.extend(_)

            
            
            models_have_in_level_dependency.extend([model.model_id for model in cand_models if model_correct_level_num[model.model_id]!=visit_model_level])
            models_have_in_level_dependency = list(set(models_have_in_level_dependency))

            


            
            if (visit_model_level > 0):
                
                
                
                cand_models = [model for model in cand_models if not model.can_be_vertically_fused_topologically]


            
            if len(cand_models) == 0:
                break


            
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)

                
                if 'no_preemption' in search_method_baseline:
                    exec_plans = _prune_different_exec_plans_from_last_stage(exec_plans, model.model_id)


                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")

                
                
                
                
                

            
            
            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)


            
            


            time_analysis[0] += (time.perf_counter() - time1)


            for cand_plan_group in new_plan_groups:
                
                

                time1 = time.perf_counter()

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)


                print(f"time get runnable plans: {time.perf_counter() - time1}")
                time_analysis[1] += (time.perf_counter() - time1)
                time1 = time.perf_counter()

                
                
                
                
                
                
                
                


                
                good_runnable_exec_plan_keys_list = [[exec_plan.get_key() for exec_plan in exec_plans] for exec_plans in runnable_exec_plans_list]


                print(f"time get good plans: {time.perf_counter() - time1}")
                time_analysis[2] += (time.perf_counter() - time1)
                time1 = time.perf_counter()

                
                
                
                
                
                
                
                
                plan_groups = [cand_plan_group]
                old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
                _append_exec_plan(plan_groups, runnable_exec_plans_list, 0, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping, good_runnable_exec_plan_keys_list, 
                                  self.out_edge_dict)
                
                


                
                
                
                
                
                
                
                
                

                print(f"time append plans: {time.perf_counter() - time1}")
                time_analysis[3] += (time.perf_counter() - time1)
                time1 = time.perf_counter()


                if len(plan_groups) == 1:
                    

                    
                    if visit_model_level+1 >= len(self.all_level_model_ids):
                        if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_groups[0], tot_gpu_num
                            ):
                            tot_plan_groups.extend(plan_groups)
                else:

                    if visit_model_level+1 >= len(self.all_level_model_ids):
                        
                        plan_groups = [plan_groups[0]]+[plan_group for plan_group in plan_groups[1:] if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_group, tot_gpu_num
                            )]


                    time1 = time.perf_counter()

                    
                    print(f"the groups we found a round: # of  groups before pruning: {len(plan_groups)}")
                    plan_groups = plan_groups[:1] + [plan_group for plan_group in plan_groups[1:] if _has_in_stage_inp_models(plan_group)]


                    print(f"the groups we found a round: # of  groups after pruning: {len(plan_groups)}")
                    for plan_group in plan_groups:
                        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")
                    

                    time_analysis[4] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()


                    
                    

                    
                    
                    
                    
                    

                    print(f"time get costs: {time.perf_counter() - time1}")
                    time_analysis[5] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()

                    
                    
                    
                    

                    
                    
                    
                    
                    
                    
                    
                    

                    
                    
                    
                    
                    

                    
                    
                    
                    
                    
                    

                    
                    
                    

                    

                    
                    
                    

                    
                    
                    
                    

                    
                    
                    
                    


                    
                    


                    
                    
                    
                    
                    

                    good_plan_groups = plan_groups[1:]
                    all_plan_group_list.extend(good_plan_groups)
                    
                    
                    
                    
                    
                    
                    tmp_new_plan_groups.extend(good_plan_groups)
                    
                    


                    print(f"time finish get costs: {time.perf_counter() - time1}")
                    time_analysis[6] += (time.perf_counter() - time1)

                
                
                
                
                
                
                
                
                


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


        
        
        



        
        
        
        

        
        plan_groups = all_plan_group_list

        print(f"\nfinal candidates before pruning:\n")
        for plan_group in plan_groups:
            print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")


        
        
        max_model_num = max([len(plan_group) for plan_group in plan_groups])
        max_gpu_num = tot_gpu_num
        if len(plan_groups) == 1:
            max_gpu_num = get_tot_worker_num(plan_groups[0])
        plan_groups = [plan_group for plan_group in plan_groups if (len(plan_group) == max_model_num) \
                       and (get_tot_worker_num(plan_group)==max_gpu_num)]

        print(f"\nfinal candidates:\n")
        for plan_group in plan_groups:
            print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")


        
        gpu_nums_list = [ [get_tot_worker_num([plan]) for plan in plan_group] for plan_group in plan_groups]
        min_gpu_num_difference = min([max(gpu_nums)-min(gpu_nums) for gpu_nums in gpu_nums_list])
        plan_groups = [plan_group for gpu_nums, plan_group in zip(gpu_nums_list, plan_groups) if max(gpu_nums)-min(gpu_nums) == min_gpu_num_difference]
        print(f"\nfinal candidates: min_gpu_num_difference: {min_gpu_num_difference}\n")
        for plan_group in plan_groups:
            print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}")


        time1 = time.perf_counter()
        
        old_real_uniq_exec_plan_mapping_keys = list(real_uniq_exec_plan_mapping.keys())
        plan_groups = update_real_uniq_exec_plan_mapping(real_uniq_exec_plan_mapping, plan_groups)

        _run_exec_cost_estimation_on_uniq_sub_plan_groups(
            real_uniq_exec_plan_mapping, 
            old_real_uniq_exec_plan_mapping_keys,
            cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
        )


        

        
        max_ind = max(range(len(plan_groups)), key=lambda i: estimate_throughput_if_fake_scheduling_is_done_fast_version_with_penalty(plan_groups[i]))
        plan_groups = [plan_groups[max_ind]]


        plan_groups = [MyExecPlanGroup(
            plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
        for plan_group in plan_groups:
            plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)
        print(f"time finish get costs: {time.perf_counter() - time1}")     
        
        


        print(f"time_analysis: {time_analysis}")

        return plan_groups

















































    def get_candidate_plan_groups_greedy_baseline_adapted_from_MuxServe_best_model_first(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3))->List[MyExecPlanGroup]:
        """
            Greedily select exec plans to run in a exec stage.
            NOTE: sort models by their sizes first, and then select their best exec plans.
            NOTE: when there are multi-level models in the system, we sort the models level by level.
        """
        new_plan_groups = [[]]
        checked_model_ids = list()
        base_model_finish_status = self.get_base_model_finish_status()
        while True:
            cand_plan_group = new_plan_groups[0]
            running_model_ids = [exec_plan.model.model_id for exec_plan in cand_plan_group]
            cand_models = self.get_runnable_models(running_model_ids=running_model_ids)
            cand_models = [model for model in cand_models if model.model_id not in checked_model_ids]

            
            cand_models: List[MyModelInfor] = [get_sorted_models_by_model_size(cand_models)[0]]

            
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input)
                exec_plans_list.append(exec_plans)
            
            
            plan_groups = [cand_plan_group]
            
            _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
                plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu,
                cost_table, last_stage_exec_plans, 
                check_gap, sort_input,)
                           
            new_plan_groups = plan_groups
            if len(cand_models) == 0:
                break

            checked_model_ids.append(cand_models[0].model_id)
        
        
        plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                        check_gap=check_gap, sort_input=sort_input, base_model_finish_status=base_model_finish_status) \
                    for plan_group in new_plan_groups if len(plan_group) > 0]
        return plan_groups




    def get_candidate_plan_groups_greedy_baseline_adapted_from_MuxServe_best_exec_plan_first(
        self, 
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3))->List[MyExecPlanGroup]:
        """
            Greedily select exec plans to run in a exec stage.
            NOTE: sort exec plans from all candidate models by their throughputs.
        """
        new_plan_groups = [[]]
        while True:
            cand_plan_group = new_plan_groups[0]
            ori_group_size = len(cand_plan_group)
            running_model_ids = [exec_plan.model.model_id for exec_plan in cand_plan_group]
            cand_models = self.get_runnable_models(running_model_ids=running_model_ids)

            
            exec_plans_list = list()
            for model in cand_models:
                exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input)
                exec_plans_list.extend(exec_plans)
            
            
            plan_groups = [cand_plan_group]
            
            _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
                plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu,
                cost_table, last_stage_exec_plans, 
                check_gap, sort_input,)
                           
            new_plan_groups = plan_groups
            if len(plan_groups[0]) == ori_group_size:
                break
        
        
        base_model_finish_status = self.get_base_model_finish_status()
        plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                        check_gap=check_gap, sort_input=sort_input, base_model_finish_status=base_model_finish_status) \
                    for plan_group in new_plan_groups if len(plan_group) > 0]
        return plan_groups







    def get_candidate_plan_groups_dispatch(
        self, 
        gen_execplans_baseline:str,
        search_method_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4)->List[MyExecPlanGroup]:
        if search_method_baseline == 'ours':
            return self.get_candidate_plan_groups(
                gen_execplans_baseline, check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline == 'greedy_saturn':
            return self.get_candidate_plan_groups_greedy_best_exec_plan_first(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ['min_gpu_useAll', 'min_gpu_useAll_no_preemption']:
            return self.get_candidate_plan_groups_min_gpu_use_all_gpus(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ['greedy_saturn_LPT', 'greedy_saturn_LPT_no_preemption']:
            return self.get_candidate_plan_groups_greedy_best_exec_plan_first_LPT(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline == 'greedy_saturn_LPT_last_stage_first':
            return self.get_candidate_plan_groups_greedy_best_exec_plan_first_LPT_last_stage_models_first(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        else:
            assert False, f'We current do not support the search_method_baseline: {search_method_baseline}!'






    def remaining_models_are_on_the_last_layer(self)->bool:
        remaining_model_ids = [model_id for model_id, model in self.model_dict.items() if not model.is_finished()]
        output_model_num = sum([len(self.out_edge_dict[model_id]) for model_id in remaining_model_ids])
        fused_model_num = sum([isinstance(self.model_dict[model_id], MyFusedModelInfor) for model_id in remaining_model_ids])
        return (output_model_num == 0) and (fused_model_num == 0)






def get_infor_given_seq_ids(
        values, seq_ids_we_have: List[int], seq_ids_requested: List[int], default_value):
    """
        1. ``values`` containing the values of corresponding to the ``seq_ids_we_have``;
        2. default_value: the value assigned to the requested seq ids which are not in seq_ids_we_have;
    """
    
    
    
    values = np.asarray(values)
    seq_ids_we_have = np.asarray(seq_ids_we_have)
    seq_ids_requested = np.asarray(seq_ids_requested)
    
    ret = np.full(len(seq_ids_requested), default_value, dtype=values.dtype)
    inds = np.searchsorted(seq_ids_we_have, seq_ids_requested)
    valid_indices1 = inds<len(seq_ids_we_have)
    valid_indices2 = (seq_ids_we_have[inds[valid_indices1]] == seq_ids_requested[valid_indices1])
    inds = inds[valid_indices1][valid_indices2]
    ret_inds = np.arange(len(ret))[valid_indices1][valid_indices2]
    ret[ret_inds] = np.asarray(values)[inds]
    
    return ret




def get_factors(v, start_from, smaller_than):
    return [i for i in range(start_from, smaller_than) if v%i==0]



def is_valid_exec_plan(exec_plan: MyExecPlan, cost_table: CostTable):
    '''
    Check whether this exec plan itself is valid, 
    without considering the exec plan combination to be applied together on the GPU cluster.
    Input:
        exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    wld_degree = exec_plan.wld_degree
    mem_per_comp_gpu = exec_plan.mem_per_comp_gpu
    param_byte_per_comp_gpu = exec_plan.param_byte_per_comp_gpu
    param_byte_per_cache_gpu = exec_plan.param_byte_per_cache_gpu
    byte_per_gpu = exec_plan.tot_gpu_mem_byte
    

    

    
    if (wld_degree > 0) and (mem_per_comp_gpu < 0.9):
        
        return False
    
    
    if mem_per_comp_gpu * byte_per_gpu < param_byte_per_comp_gpu:
        
        return False
    
    if byte_per_gpu < param_byte_per_cache_gpu:
        
        return False
    
    
    
    
    
    if not cost_table.can_estimate_cost(exec_plan.model.model_path, exec_plan.get_key_single_dp_worker()):
        
        return False
    
    return True




def _get_possible_exec_plans(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable, fully_connected_gpu_unit:int):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    import math
    exec_plans = list()
    
    
    
    if model.is_finished():
        print(f"model: {model}, model is finished")
        return exec_plans
    
    
    
    for i in range(int(math.log(tot_gpu_num, 2)+1)):
        num_worker = 2**i

        
        if num_worker > fully_connected_gpu_unit:
            continue

        
        if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
            _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
                model_path=model.model_path, tensor_parallel_size=num_worker)
        (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]
        
        infer_args = InferenceArgs(scheduler_config, cache_config)
        
        
        
        gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
        
        param_byte_per_layer, extra_byte = \
            get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)

        
        
        for wld_degree in [2]: 
            if wld_degree < 2:
                
                continue
            
            
            
            param_byte_per_comp_gpu = extra_byte + \
                param_byte_per_layer * (model.layer_num - wld_degree + 2)

            print(f"param_byte_per_comp_gpu: {param_byte_per_comp_gpu/1024/1024/1024}, byte_per_gpu: {byte_per_gpu/1024/1024/1024}")

            
            for cache_gpu_num in range(tot_gpu_num-num_worker+1):
                if (wld_degree > 2) and (cache_gpu_num == 0):
                    
                    continue
                if (wld_degree == 2) and (cache_gpu_num > 0):
                    
                    continue
                
                
                param_byte_per_cache_gpu = 0
                if cache_gpu_num > 0:
                    
                    
                    
                    param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / cache_gpu_num
                
                
                for mem_per_comp_gpu in [j/10 for j in range(1, 10)]:

                    dp_size = 1
                    exec_plan = MyExecPlan(model,
                        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                        param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                        gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                    
                    if isinstance(model, MyFusedModelInfor):
                        exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                    
                    

                    
                    if is_valid_exec_plan(exec_plan, cost_table):
                        exec_plans.append(exec_plan)

                        

                        
                        for dp_size in range(2, tot_gpu_num // num_worker + 1):
                            if dp_size * num_worker + cache_gpu_num > tot_gpu_num:
                                
                                
                                continue

                            
                            exec_plan = MyExecPlan(model,
                                num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                                param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                                gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                            
                            if isinstance(model, MyFusedModelInfor):
                                exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                            
                            
                            exec_plans.append(exec_plan)

    return exec_plans










def _get_possible_exec_plans_naive_baseline_1(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable, sort_input: bool, fully_connected_gpu_unit:int):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    NOTE: generate the exec plan which uses all gpu for computation.
    '''
    exec_plans = list()
    
    
    
    if model.is_finished():
        print(f"This model is finished.")
        return exec_plans
    

    num_worker = tot_gpu_num
    
    num_worker = fully_connected_gpu_unit

    
    if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
        _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
            model_path=model.model_path, tensor_parallel_size=num_worker)
    (model_config, cache_config, parallel_config, scheduler_config,
        device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]
    
    infer_args = InferenceArgs(scheduler_config, cache_config)
    
    
    
    gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
    
    param_byte_per_layer, extra_byte = \
        get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)

    wld_degree = 2

    param_byte_per_comp_gpu = extra_byte + \
        param_byte_per_layer * (model.layer_num - wld_degree + 2)


    cache_gpu_num = 0
    mem_per_comp_gpu = 0.9
    param_byte_per_cache_gpu = 0
    dp_size = 1
    dp_size = tot_gpu_num//num_worker

    exec_plan = MyExecPlan(model,
        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
        param_byte_per_comp_gpu, param_byte_per_cache_gpu,
        gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)

    if isinstance(model, MyFusedModelInfor):
        exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
    
    print(f"gen an exec plan: {str(exec_plan)}")

    
    if is_valid_exec_plan(exec_plan, cost_table):
        exec_plans.append(exec_plan)


    return exec_plans





def _get_possible_exec_plans_naive_baseline(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable, sort_input: bool, fully_connected_gpu_unit:int):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    NOTE: only generate the exec plan which has the highest throughput for the model
    '''
    import math
    exec_plans: List[MyExecPlan] = list()
    
    
    
    if model.is_finished():
        return exec_plans
    
    
    
    for i in range(int(math.log(tot_gpu_num, 2)+1)):
        num_worker = 2**i

        
        if num_worker > fully_connected_gpu_unit:
            continue

        
        if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
            _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
                model_path=model.model_path, tensor_parallel_size=num_worker)
        (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]
        
        infer_args = InferenceArgs(scheduler_config, cache_config)
        
        
        
        gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
        
        param_byte_per_layer, extra_byte = \
            get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)

        
        
        for wld_degree in [2]:
            if wld_degree < 2:
                
                continue
            
            
            
            param_byte_per_comp_gpu = extra_byte + \
                param_byte_per_layer * (model.layer_num - wld_degree + 2)

            
            for cache_gpu_num in range(tot_gpu_num-num_worker+1):
                if (wld_degree > 2) and (cache_gpu_num == 0):
                    
                    continue
                if (wld_degree == 2) and (cache_gpu_num > 0):
                    
                    continue
                
                
                param_byte_per_cache_gpu = 0
                if cache_gpu_num > 0:
                    
                    
                    
                    param_byte_per_cache_gpu = wld_degree * param_byte_per_layer / cache_gpu_num
                
                
                for mem_per_comp_gpu in [j/10 for j in range(1, 10)]:

                    dp_size = 1
                    exec_plan = MyExecPlan(model,
                        num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                        param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                        gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                    
                    if isinstance(model, MyFusedModelInfor):
                        exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                    
                    

                    
                    if is_valid_exec_plan(exec_plan, cost_table):
                        exec_plans.append(exec_plan)

                        

                        
                        for dp_size in range(2, tot_gpu_num // num_worker + 1):
                            if dp_size * num_worker + cache_gpu_num > tot_gpu_num:
                                
                                
                                continue

                            
                            exec_plan = MyExecPlan(model,
                                num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, dp_size,
                                param_byte_per_comp_gpu, param_byte_per_cache_gpu,
                                gpu_cache_byte_per_block, infer_args, tot_gpu_mem_byte=byte_per_gpu)
                            
                            if isinstance(model, MyFusedModelInfor):
                                exec_plan = MyVerticalFusedExecPlan(model, exec_plan)
                            
                            
                            exec_plans.append(exec_plan)
    
    
    
    for exec_plan in exec_plans:
        exec_plan.get_max_dp_latency(cost_table, sort_input)
        
    exec_plans = sorted(exec_plans, \
        key=lambda exec_plan: \
            
            exec_plan.wait_for_remote_fake_scheduling_and_get_max_dp_latency()+\
                cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
    )
    exec_plans = [exec_plans[0]]

    return exec_plans






def get_possible_exec_plans(
        model: MyModelInfor, tot_gpu_num, byte_per_gpu, cost_table: CostTable,
        baseline: str, sort_input: bool, fully_connected_gpu_unit:int):
    '''
    Get the possible execution plan for the model.
    Input:
        can get model_info from model: (layer_num, param_byte_per_layer, extra_param_byte).
    Output:
        each exec_plan: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
    '''
    if baseline == 'ours':
        return _get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit)
    else:
        
        return _get_possible_exec_plans_naive_baseline(model, tot_gpu_num, byte_per_gpu, cost_table, sort_input, fully_connected_gpu_unit)











def select_best_gpus_for_cache_request(request: np.ndarray, resources: np.ndarray, requests: List[List[int]]):
    '''
    Input: 
        request: list of ints: each int is the cache mem required on a cache gpu.
        resources: the available gpu mems on each candidate cache gpu.
        requests: the remaining cache mem requests.
    NOTE: we deal with all cache requirement for a model at a time.
    compute fragment ratio in the way in ATC23.
    '''
    best_choice = None
    best_fail_ratio = float('inf')
    tmp_resources = resources.copy()
    choices = itertools.combinations(range(len(resources)), len(request))
    for choice in choices:
        
        choice_list = list(choice)
        if (resources[choice_list] < request).any():
            continue
        
        tmp_resources[choice_list] = tmp_resources[choice_list] - request
        
        
        fail_ratio = sum([sum((tmp_resources < r[0])*tmp_resources) for r in requests]) / sum(tmp_resources)
        if fail_ratio < best_fail_ratio:
            best_fail_ratio = fail_ratio
            best_choice = choice_list
        tmp_resources[choice_list] = tmp_resources[choice_list] + request
    return best_choice, best_fail_ratio





def get_tot_worker_num(exec_plans: List[MyExecPlan]):
    
    
    return sum([exec_plan.num_worker * exec_plan.dp_size for exec_plan in exec_plans])


def is_valid_exec_plan_combination(exec_plans: List[MyExecPlan], tot_gpu_num, byte_per_gpu):
    '''
    Check whether the exec plan combination can be applied to the GPU cluster.
    Input: 
        exec_plans: a list of exec plans: \
            (num_worker, wld_degree, cache_gpu_num, mem_per_comp_gpu, param_byte_per_comp_gpu, param_byte_per_cache_gpu).
        tot_gpu_num: the total GPU number.
        byte_per_gpu: the total memory of a GPU (in bytes).
    '''
    '''
    refer to ATC23.
    '''

    

    use_cache_gpu_plans = []
    without_cache_gpu_plans = []
    cache_gpu_remaining_bytes = []
    cache_gpu_required_bytes = []
    for exec_plan in exec_plans:
        num_worker = exec_plan.num_worker
        cache_gpu_num = exec_plan.cache_gpu_num
        mem_per_comp_gpu = exec_plan.mem_per_comp_gpu
        param_byte_per_cache_gpu = exec_plan.param_byte_per_cache_gpu
        dp_size = exec_plan.dp_size
        if cache_gpu_num > 0:
            use_cache_gpu_plans.append(exec_plan)
            
            
            
            cache_gpu_required_bytes.extend(
                [np.asarray([param_byte_per_cache_gpu]*cache_gpu_num) for _ in range(num_worker*dp_size)]
                )
        else:
            without_cache_gpu_plans.append(exec_plan)
            
            
            
            cache_gpu_remaining_bytes.extend([byte_per_gpu * (1 - mem_per_comp_gpu)] * num_worker * dp_size)
    
    
    
    tot_worker_num = get_tot_worker_num(exec_plans)
    if tot_worker_num > tot_gpu_num:
        
        return False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    requests = sorted(cache_gpu_required_bytes, key=lambda i: i[0], reverse=True)
    resources = np.asarray(cache_gpu_remaining_bytes)

    

    for i, request in enumerate(requests):

        
        
        
        
        remaining_requests = requests[i+1:]
        
        
        gpus_choice, _ = select_best_gpus_for_cache_request(request, resources, remaining_requests)

        

        if gpus_choice == None:
            
            
            return False
        
        resources[gpus_choice] = resources[gpus_choice] - request


    
    if (resources > 0.1 * byte_per_gpu).any():
        
        return False
    

    
    
    return True



def _can_be_fused_vertically_linear_dependency(plan_group: List[MyExecPlan], to_fuse: MyExecPlan):
    """
        The condition to do vertical fusion:
        1. linear dependency: model 1 -> model 2.
        2. about the same model, has the same exec setting.
        3. the models in both plans have not been started.
    """

    to_fuse_inp_model_ids = to_fuse.model.input_model_ids
    for plan_i, plan in enumerate(plan_group):
        if (len(to_fuse_inp_model_ids) == 1) and (to_fuse_inp_model_ids[0] == plan.get_base_model_ids()[-1]):
            
            if plan.model.get_name() == to_fuse.model.get_name():
                
                if plan.get_key() == to_fuse.get_key():
                    
                    
                    
                    if plan.models_not_started() and to_fuse.models_not_started():
                        

                        print(f"in generate fused model:\n")
                        print([(_.model_id, _) for _ in plan.get_base_models()+[to_fuse.model]])

                        to_fuse_models = sorted(plan.get_base_models()+[to_fuse.model], \
                                                key=lambda model: model.model_id)
                        
                        print([(_.model_id, _) for _ in to_fuse_models])

                        fused_model = MyFusedModelInfor(to_fuse_models)
                        fused = MyVerticalFusedExecPlan(fused_model, to_fuse)
                        return [plan_group[:plan_i]+[fused]+plan_group[plan_i+1:]]
    return []






def _get_path_key(plan_group:List[MyExecPlan], exec_plan):
    """
        Get the information of all models that exec_plan depends directly or indirectly on in the current plan_group.
        Output: path_key and the related plans.
    """
    inp_model_ids = sorted(exec_plan.model.input_model_ids)
    key = list()
    key_plans = list()
    for inp in inp_model_ids:
        for plan in plan_group:
            if (inp == plan.model.model_id) or (inp in plan.model.get_base_model_ids()):
                
                
                key.append(( tuple(plan.model.get_base_model_ids()), plan.get_key() ))
                key_plans.append(plan)
                break
    
    
    key.append(( tuple(exec_plan.model.get_base_model_ids()), exec_plan.get_key() ))
    key_plans.append(exec_plan)
    return tuple(key), key_plans



def _update_good_plan_group_dict(
        group_obj: MyExecPlanGroup,
        cost_table: CostTable,
        
        
        
        good_plan_group_dict: Dict[Tuple[List[int], int], Tuple[float, MyExecPlanGroup]], 
    ) -> Optional[Tuple[List[int], int]]:
    """
        Return None if the given plan group is not good enough, else return the input plan_group key.
    """

    
    
    
    group_obj.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)
    
    
    
    
    
    base_model_ids = sorted(np.concatenate([exec_plan.model.get_base_model_ids() for exec_plan in group_obj.exec_plans]))
    tot_gpu_num = get_tot_worker_num(group_obj.exec_plans)
    key = ( tuple(base_model_ids),  tot_gpu_num)

    
    ret = key
    group_to_print = f"{len(group_obj.exec_plans)}, {[(tuple(plan.model.get_base_model_ids()), plan.get_key()) for plan in group_obj.exec_plans]}"
    if key not in good_plan_group_dict:
        print(f"key not in good_plan_group_dict: {key}-{group_obj.get_throughput()}: {group_to_print}")
        good_plan_group_dict[key] = (group_obj.get_throughput(), group_obj)
    else:
        if (group_obj.get_throughput() > good_plan_group_dict[key][0]) or \
            ( (group_obj.get_throughput() == good_plan_group_dict[key][0]) \
             and (len(group_obj.exec_plans) < len(good_plan_group_dict[key][1].exec_plans)) ):
            
            
            
            good_plan_group_dict[key] = (group_obj.get_throughput(), group_obj)
            print(f"key in good_plan_group_dict, update: {key}-{group_obj.get_throughput()}: {group_to_print}")
        else:
            ret = None
            print(f"key in good_plan_group_dict, discard: ori-{key}-{good_plan_group_dict[key][0]} vs new-{group_obj.get_throughput()}: {group_to_print}")

    return ret





def _meet_vertical_fuse_condition(to_fuse_inp_base_model_ids: List[int], model_ids_fused: List[int], fused_model_inp_base_model_ids: List[int]):
    """
        NOTE: we assume after vertical fusion, 
            the input models of the fused model is the same as the input models of the FIRST BASE model in the fused model.
        INPUT:
            1. to_fuse_inp_base_model_ids: the input base model ids of the model (may already be a fused model) to be fused.
            2. model_ids_fused: the base model ids of the fused model we want to add more models to.
            3. fused_model_inp_base_model_ids: the input base model ids of the fused model we want to add more models to.
    """
    cond1 = (len(to_fuse_inp_base_model_ids) == 1) and \
        (to_fuse_inp_base_model_ids[0] == model_ids_fused[-1])
    
    
    cond2 = sorted(to_fuse_inp_base_model_ids) == sorted(fused_model_inp_base_model_ids+model_ids_fused)
    return (cond1 or cond2)



def _get_vertical_fuse_model_pairs_topology_level(plan_group: List[MyExecPlan], to_fuse: MyExecPlan):
    to_fuse_inp_base_model_ids = to_fuse.get_base_models()[0].inp_base_model_ids
    for plan_i, plan in enumerate(plan_group):
        if _meet_vertical_fuse_condition(to_fuse_inp_base_model_ids, 
                                         model_ids_fused=plan.get_base_model_ids(), 
                                         fused_model_inp_base_model_ids=plan.get_base_models()[0].inp_base_model_ids):
            
            if plan.model.get_name() == to_fuse.model.get_name():
                return plan.model
    return None


def _get_vertical_fuse_model_pairs(plan_group: List[MyExecPlan], to_fuse: MyExecPlan):
    to_fuse_inp_base_model_ids = to_fuse.get_base_models()[0].inp_base_model_ids
    for plan_i, plan in enumerate(plan_group):
        if _meet_vertical_fuse_condition(to_fuse_inp_base_model_ids, 
                                         model_ids_fused=plan.get_base_model_ids(), 
                                         fused_model_inp_base_model_ids=plan.get_base_models()[0].inp_base_model_ids):
            
            if plan.model.get_name() == to_fuse.model.get_name():
                if plan.get_key() == to_fuse.get_key():
                    return plan.model
    return None



def _can_be_fused_vertically(plan_group: List[MyExecPlan], to_fuse: MyExecPlan, uniq_exec_plan_mapping):
    """
        The condition to do vertical fusion:
        1. supported dependency:
            (a) linear dependency: model 1 -> model 2.
            (b) model i depends on all models before it: (model 1, ..., model i-1) -> model i.
        2. about the same model, has the same exec setting.
        3. the models in both plans have not been started.
    """

    
    
    
    
    
    to_fuse_inp_base_model_ids = to_fuse.get_base_models()[0].inp_base_model_ids
    for plan_i, plan in enumerate(plan_group):
        
        
        
        
        
        
        if _meet_vertical_fuse_condition(to_fuse_inp_base_model_ids, 
                                         model_ids_fused=plan.get_base_model_ids(), 
                                         fused_model_inp_base_model_ids=plan.get_base_models()[0].inp_base_model_ids):
            
            if plan.model.get_name() == to_fuse.model.get_name():
                
                if plan.get_key() == to_fuse.get_key():
                    
                    
                    
                    if plan.models_not_started() and to_fuse.models_not_started():
                        

                        
                        

                        
                        
                        
                        
                        
                        to_fuse_models = plan.get_base_models()+to_fuse.get_base_models()
                        
                        

                        
                        
                        path_key = _get_path_key(plan_group, plan)[:-1]
                        path_key = path_key+( (tuple(plan.model.get_base_model_ids()+to_fuse.model.get_base_model_ids()), to_fuse.get_key() ) ,)
                        
                        
                        fused = None
                        if path_key in uniq_exec_plan_mapping:
                            
                            fused = uniq_exec_plan_mapping[path_key]
                            
                        else:
                            
                            fused_model = MyFusedModelInfor(to_fuse_models)
                            fused = MyVerticalFusedExecPlan(fused_model, to_fuse)
                            uniq_exec_plan_mapping[path_key] = fused
                            

                        
                                        
                        
                        
                        return [plan_group[:plan_i]+[fused]+plan_group[plan_i+1:]]
    return []





def _update_uniq_exec_plan_mapping(
        plan_group: List[MyExecPlan], plan: MyExecPlan, uniq_exec_plan_mapping
    ) -> MyExecPlan:
    """
        plan_group is a list of execution plans.
        plan is an execution plan to be appended.
        Check whether the plan will need to be a new uniq plan object after being appended.
        If so, update the uniq_exec_plan_mapping; otherwise, use the plan stored in the uniq_exec_plan_mapping.
    """
    print(f"plan: {plan}")
    print(f"plan_group: {plan_group}")
    path_key, path_plans = _get_path_key(plan_group, plan)
    if path_key in uniq_exec_plan_mapping:
        
        return uniq_exec_plan_mapping[path_key][0]
    else:
        
        exec_plan_to_use = plan.copy_the_plan()
        uniq_exec_plan_mapping[path_key] = exec_plan_to_use, path_plans[:-1]+[exec_plan_to_use]
        return exec_plan_to_use






def _update_plan_group_consider_new_inp_for_horizontally_fused_models(
        plan_group: List[MyExecPlan], new_exec_plan: MyExecPlan, in_stage_out_edge_dict: Dict[int, List[int]],
        uniq_exec_plan_mapping):
    """
        If there are horizontally fused models in plan_group, the new_exec_plan may be a new inp for it, so we may need to instantiate a new plan for it.
    """
    affected_model_ids = list()
    newly_add = [new_exec_plan.model.model_id]
    while True:
        tmp_newly_add = list()
        for model_id in newly_add:
            tmp_newly_add.extend(in_stage_out_edge_dict[model_id])
            affected_model_ids.extend(in_stage_out_edge_dict[model_id])
        newly_add = tmp_newly_add
        if len(newly_add) == 0:
            break

    
    new_plan_group = list()
    for plan in plan_group:
        exec_plan_to_use = plan
        if plan.model.model_id in affected_model_ids:
            
            path_key, path_plans = _get_path_key(plan_group, plan)
            exec_plan_to_use = plan.copy_the_plan()
            uniq_exec_plan_mapping[path_key] = exec_plan_to_use, path_plans
        new_plan_group.append(exec_plan_to_use)
    return new_plan_group





def _append_exec_plan(plan_groups, exec_plans_list, depth_i, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping, good_runnable_exec_plan_keys_list,
                      in_stage_out_edge_dict: Dict[int, List[int]]):
    '''
    Get all the possible exec plans with depth-first search.
    The initial plan_groups is [[]], i.e., containing a group with no exec plan.
    All plan groups are valid if they are put into plan_groups and returned.
    NOTE:
        1. here we use good plen group to add pruning the plan group generation process.
    '''
    
    if depth_i == len(exec_plans_list):
        return
    
    new_plan_groups = list()
    for plan_group in plan_groups:
        

        for exec_plan in exec_plans_list[depth_i]:

            
            
            

            
            
            if _get_vertical_fuse_model_pairs_topology_level(plan_group, exec_plan)!=None:
                continue


            if get_tot_worker_num(plan_group) == tot_gpu_num:
                
                continue

            if exec_plan.get_key() not in good_runnable_exec_plan_keys_list[depth_i]:
                
                continue



            tmp_plan_group = plan_group + [exec_plan]
            
            
            
            if is_valid_exec_plan_combination(tmp_plan_group, tot_gpu_num, byte_per_gpu):

                
                path_key, path_plans = _get_path_key(plan_group, exec_plan)
                
                exec_plan_to_use: MyExecPlan = None
                if path_key in uniq_exec_plan_mapping:
                    exec_plan_to_use, path_plans = uniq_exec_plan_mapping[path_key]
                    
                else:
                    
                    exec_plan_to_use = exec_plan.copy_the_plan()
                    uniq_exec_plan_mapping[path_key] = exec_plan_to_use, path_plans
                    


                plan_group_to_use = _update_plan_group_consider_new_inp_for_horizontally_fused_models(
                    plan_group, exec_plan_to_use, in_stage_out_edge_dict,
                    uniq_exec_plan_mapping)

                tmp_plan_group = plan_group_to_use + [exec_plan_to_use]
                new_plan_groups.append(tmp_plan_group)

                

    plan_groups.extend(new_plan_groups)
    _append_exec_plan(plan_groups, exec_plans_list, depth_i+1, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping, good_runnable_exec_plan_keys_list, 
                      in_stage_out_edge_dict)






def _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
        plan_groups, exec_plans_list, depth_i, tot_gpu_num, byte_per_gpu,
        cost_table: CostTable,
        last_stage_exec_plans: List[MyExecPlan],
        check_gap: int, sort_input: bool,):
    '''
    Get all the possible exec plans with depth-first search.
    The initial plan_groups is [[]], i.e., containing a group with no exec plan.
    All plan groups are valid if they are put into plan_groups and returned.
    '''

    

    
    if depth_i == len(exec_plans_list):
        return
    
    new_plan_groups = list()
    for plan_group in plan_groups:
        
        if get_tot_worker_num(plan_group) == tot_gpu_num:
            
            continue
        for exec_plan in exec_plans_list[depth_i]:
            tmp_plan_group = plan_group + [exec_plan]
            
            
            if is_valid_exec_plan_combination(tmp_plan_group, tot_gpu_num, byte_per_gpu):
                
                new_plan_groups.append(tmp_plan_group)

                

    
    
    

    if len(new_plan_groups) > 0:
        new_plan_groups = [
            MyExecPlanGroup(exec_plans, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans,
                check_gap=check_gap, sort_input=sort_input)\
            for exec_plans in new_plan_groups]
        exec_plans = sorted(new_plan_groups, key=lambda i: i.get_throughput(), reverse=True)[0].exec_plans
        
        plan_groups[0] = exec_plans

    _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
        plan_groups, exec_plans_list, depth_i+1, tot_gpu_num, byte_per_gpu,
        cost_table, last_stage_exec_plans,
        check_gap, sort_input,)

    












def get_one_stage_exec_plans_sorted(
        gen_execplans_baseline:str,
        search_method_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        model_sys: MyModelSystem, 
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4):
    '''
    Get a set of exec plans which can work corrently on the given multi-GPU environment.
    '''

    
    model_sys.print_model_list()

    
    
    
    
    
    
    
    
    
    
    

    time1 = time.perf_counter()

    
    
    
    plan_groups = model_sys.get_candidate_plan_groups_dispatch(
        gen_execplans_baseline, search_method_baseline, 
        check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit
    )
    

    print(f"time-gen_plan_groups 1: {time.perf_counter() -  time1}")
    time1 = time.perf_counter()


    print(f"in get_one_stage_exec_plans_sorted: the plan groups we generated: ")
    for plan_group in plan_groups:
        print(f"{len(plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans], plan_group.get_throughput()}")


    

    
    
    
    
    not_finished_model_num = model_sys.get_not_finished_base_model_num()
    useful_plan_groups = list()
    idle_comp_plan_groups = dict() 
    


    

    
    for plan_group in plan_groups:
        
        

        key = plan_group.get_model_states_before_infer_stage()

        

        if key not in idle_comp_plan_groups:
            
            idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
        else:
            
            if plan_group.get_throughput() > idle_comp_plan_groups[key][0]:
                idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
            elif plan_group.get_throughput() == idle_comp_plan_groups[key][0]:
                
                if get_tot_worker_num(plan_group.exec_plans) < get_tot_worker_num(idle_comp_plan_groups[key][1].exec_plans):
                    idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)


    

    print(f"time-gen_plan_groups 2: {time.perf_counter() -  time1}")
    time1 = time.perf_counter()

    plan_groups: List[MyExecPlanGroup] = [plan_group for _, plan_group in idle_comp_plan_groups.values()]
    idle_comp_plan_groups = dict() 


    
    for plan_group in plan_groups:
        model_ids = tuple(np.concatenate([plan.get_base_model_ids() for plan in plan_group.exec_plans]))
        if get_tot_worker_num(plan_group.exec_plans) == tot_gpu_num:
            idle_comp_plan_groups[model_ids] = plan_group.get_throughput()
    
    plan_groups_to_keep: List[MyExecPlanGroup] = list()
    for plan_group in plan_groups:
        model_ids = set(np.concatenate([plan.get_base_model_ids() for plan in plan_group.exec_plans]))
        if get_tot_worker_num(plan_group.exec_plans) < tot_gpu_num:
            
            discard = False
            for k, v in idle_comp_plan_groups.items():
                if model_ids.issubset(k):
                    if v > plan_group.get_throughput():
                        
                        discard = True
                        break
            if not discard:
                plan_groups_to_keep.append(plan_group)
        else:
            plan_groups_to_keep.append(plan_group)
    
    plan_groups = plan_groups_to_keep
    idle_comp_plan_groups = dict() 


    print(f"time-gen_plan_groups 3: {time.perf_counter() -  time1}")
    time1 = time.perf_counter()


    
    for plan_group in plan_groups:

        


        
        
        
        
        
        
        

        

        
        
        
        
        
        
        
        
        
        key = plan_group.get_model_states_after_infer_stage(cost_table)
        
        
        
        
        
        
        
        




        if key not in idle_comp_plan_groups:
            
            
            

            idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
        else:

            
            

            if plan_group.get_throughput() > idle_comp_plan_groups[key][0]:
                idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)
            elif plan_group.get_throughput() == idle_comp_plan_groups[key][0]:
                
                if get_tot_worker_num(plan_group.exec_plans) < get_tot_worker_num(idle_comp_plan_groups[key][1].exec_plans):
                    idle_comp_plan_groups[key] = (plan_group.get_throughput(), plan_group)                

    for _, plan_group in idle_comp_plan_groups.values():
        useful_plan_groups.append(plan_group)

        
        

    
    

    
    
    
    
    
    
    
    
    
    
    


    print(f"time-gen_plan_groups 4: {time.perf_counter() -  time1}")
    time1 = time.perf_counter()

    uniq_plan_groups = useful_plan_groups

    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    print(f"the uniq plan groups we get:")
    for plan_group in uniq_plan_groups:
        print(f"{[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans], plan_group.infer_stage_latency, plan_group.get_throughput()}")


    
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.get_throughput(), reverse=True)

    return uniq_plan_groups




def get_one_stage_exec_plans_sorted_greedy_baseline_adapted_from_MuxServe(
        gen_execplans_baseline:str,
        check_gap: int, sort_input: bool,
        last_stage_exec_plans: List[MyExecPlan],
        cost_table: CostTable,
        model_sys: MyModelSystem, 
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    '''
    Get a set of exec plans which can work corrently on the given multi-GPU environment.
    NOTE: this function select exec plans in a stage 
        (1) in the order of models from large to small OR 
        (2) in the order of model's remaining flops from large to small [bad idea].
    '''

    uniq_plan_groups = model_sys.get_candidate_plan_groups_greedy_baseline_adapted_from_MuxServe_best_model_first(
        gen_execplans_baseline,
        check_gap, sort_input,
        last_stage_exec_plans,
        cost_table,
        tot_gpu_num, byte_per_gpu)

    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    print(f"the uniq plan groups we get:")
    for plan_group in uniq_plan_groups:
        print(str(plan_group))


    
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.get_throughput(), reverse=True)

    return uniq_plan_groups



    
    
    sorted_models = list()
    for model in model_list:
        
        
        
        param_byte_per_layer, extra_byte = \
                get_per_layer_and_extra_param_and_buffer_byte(model, tp_size=1)
        sorted_models.append((model, param_byte_per_layer*model.layer_num + extra_byte))
    
    model_list = sorted(sorted_models, key=lambda i: i[1], reverse=True)
    model_list = [i[0] for i in model_list]

    print(f"model_list: {[str(model) for model in model_list]}")

    
    exec_plans_list = list()
    for model in model_list:
        exec_plans = get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline)
        exec_plans_list.append(exec_plans)
        
    
    plan_groups = [[]]
    
    _append_exec_plan_baseline_greedy_baseline_adapted_from_MuxServe(
        plan_groups, exec_plans_list, 0, tot_gpu_num, byte_per_gpu,
        cost_table, last_stage_exec_plans)

    


    plan_groups = [MyExecPlanGroup(plan_group, cost_table=cost_table, last_stage_exec_plans=last_stage_exec_plans) \
                   for plan_group in plan_groups if len(plan_group) > 0]

    
    
    uniq_plan_groups = plan_groups

    print(f"len(uniq_plan_groups): {len(uniq_plan_groups)}")
    print(f"the uniq plan groups we get:")
    for plan_group in uniq_plan_groups:
        print(str(plan_group))


    
    uniq_plan_groups = sorted(uniq_plan_groups, key=lambda i: i.get_throughput(), reverse=True)

    return uniq_plan_groups












































def get_total_model_flops(model_list: List[MyModelInfor], cost_table: CostTable):
    flops = 0
    for model in model_list:
        assert not isinstance(model, MyFusedModelInfor)
        i = fake_scheduling.comp_flops_from_seqlens(
            model.inp_lens, model.out_lens, only_decode=False, cost_table=cost_table, 
            model_path=model.model_path, trust_remote_code=model.trust_remote_code, revision=model.revision)
        flops += i
    return flops



















def get_sorted_models_by_model_size(model_list: List[MyModelInfor]):
    sorted_models = list()
    for model in model_list:
        
        
        
        param_byte_per_layer, extra_byte = \
                get_per_layer_and_extra_param_and_buffer_byte(model, tp_size=1)
        sorted_models.append((model, param_byte_per_layer*model.layer_num + extra_byte))
    
    sorted_models = sorted(sorted_models, key=lambda i: i[1], reverse=True)
    sorted_models = [i[0] for i in sorted_models]
    return sorted_models





















def _get_best_model_schedule(
        gen_execplans_baseline: str,
        search_method_baseline: str,
        check_gap: int, sort_input: bool,
        cost_table: CostTable, 
        model_sys: MyModelSystem, 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: dict,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3),
        top_k=float('inf'),
        fully_connected_gpu_unit:int=4):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''
    
    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    if _CHECKED_SEQ_NUM > _MAX_SEQ_NUM:
        return


    print(f"CURRENT PLAN GROUP SEQ: {curr_group_seq.get_str_using_model_ids()}")
    print(f"CURRENT BEST PLAN GROUP SEQ: {best_group_seq.get_str_using_model_ids()}")



    
    
    
    
    
    
    
    if len(curr_group_seq.plan_group_seq) > model_sys.get_model_num():
        
        
        
        print(f"too many stages: {[[(_.get_base_model_ids(), _.get_key()) for _ in group.exec_plans] for group in curr_group_seq.plan_group_seq]}")
        return 



    print(f"finish step 1:      curr depth is within limit")

    
    
    
    if model_sys.is_finished():
        
        print(f"all models finished")
        
        _CHECKED_SEQ_NUM += 1

        
        
        if curr_group_seq.get_tot_time() < best_group_seq.get_tot_time():
            best_group_seq.set_plan_group_and_time(
                curr_group_seq.plan_group_seq, curr_group_seq.time_seq, curr_group_seq.last_stage_model_sys_values_seq)
        return 
    

    if curr_group_seq.get_tot_time() >= best_group_seq.get_tot_time():
        
        return


    print(f"finish step 2")

    
    
    model_states = model_sys.get_base_model_states()


    if model_states in uniq_model_states:
        
        
        

        
        if curr_group_seq.get_tot_time() > uniq_model_states[model_states]:

            print(f"the state has been checked and curr is not the best choice for it")
            print(f"model_states: {model_states}")
            print(f"uniq_model_states: {uniq_model_states}")

            return
        else:
            uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    else:
        
        
        uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    
    
        

    print(f"finish step 3")


    time1 = time.perf_counter()


    
    plan_groups: List[MyExecPlanGroup] = get_one_stage_exec_plans_sorted(
        gen_execplans_baseline, search_method_baseline,
        check_gap, sort_input,
        curr_group_seq.get_last_stage_exec_plans(),
        cost_table, model_sys, gpu_name, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit)
    
    
    
    
    ori_inp_out_lens_list = model_sys.get_model_inp_out_lens()
    ori_remaining_decode_flops_list = model_sys.get_model_remaining_decode_flops()
    ori_inp_seq_ids_list = model_sys.get_model_inp_seq_ids()
    ori_inp_model_ids_list = model_sys.get_model_inp_model_ids()
    ori_model_sys = model_sys


    print(f"time1: {time.perf_counter() - time1}")
    time1 = time.perf_counter()

    
    
    
    
    
    
    
    
    tot_ori_remaining_decode_flops = sum([\
        sum(flops) if isinstance(flops, list) else flops \
         for flops in ori_remaining_decode_flops_list])
    if (model_sys.remaining_models_are_on_the_last_layer()) \
        and ((tot_ori_remaining_decode_flops / plan_groups[0].get_comp_throughput_only() \
              + curr_group_seq.get_tot_time()) >= best_group_seq.get_tot_time()):
        
        print(f"using the highest throughput in plan_groups still cannot beat best_group_seq")
        
        return


    print(f"finish step 4")


    print(f"time2: {time.perf_counter() - time1}")
    time1 = time.perf_counter()


    
    for plan_group in plan_groups:
        print(f"trying adding plan_group: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans]}, models are finished? {[plan.model.is_finished() for plan in plan_group.exec_plans]}")
        if len(plan_group) == 0:
            continue

        
        
        if (len(curr_group_seq.plan_group_seq) > 0):
            print(f"curr_group_seq.plan_group_seq.exec_plans: {[[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plans.exec_plans] for plans in curr_group_seq.plan_group_seq]}")
            model_pairs_to_fuse: List[Tuple[MyModelInfor, MyModelInfor]] = [[_get_vertical_fuse_model_pairs(
                curr_group_seq.plan_group_seq[-1].exec_plans, plan), plan.model] for plan in plan_group.exec_plans]
            model_pairs_to_fuse = [model_pairs for model_pairs in model_pairs_to_fuse if model_pairs[0]!=None]
            
            print(f"model_pairs_to_fuse: {[(i.get_base_model_ids(), j.get_base_model_ids()) for i, j in model_pairs_to_fuse]}")
            
            if len(model_pairs_to_fuse) > 0:

                print(f"\nVERTICAL FUSE AND TAKE 1 STEP BACK!\n")


                time1 = time.perf_counter()

                
                
                last_stage_plan_group, last_stage_time, last_stage_model_sys_values = curr_group_seq.get_last_stage()
                last_ori_inp_out_lens_list, last_ori_remaining_decode_flops_list, \
                    last_ori_inp_seq_ids_list, last_ori_inp_model_ids_list, \
                    last_ori_model_sys = last_stage_model_sys_values
                model_sys = last_ori_model_sys
                model_sys.recover_model_state(
                    last_ori_inp_seq_ids_list,last_ori_inp_out_lens_list, cost_table, last_ori_remaining_decode_flops_list,
                    last_ori_inp_model_ids_list)
                
                
                fused_model_list: List[MyFusedModelInfor] = [MyFusedModelInfor(
                    list(model1.get_base_models())+list(model2.get_base_models())
                    ) for model1, model2 in model_pairs_to_fuse]
                model_sys = model_sys.gen_new_model_sys_with_fused_models(
                    fused_model_list=fused_model_list)

                print(f"fused_model_list: {[_.get_base_model_ids() for _ in fused_model_list]}")
                
                
                print(f"time3: {time.perf_counter() - time1}")
                time1 = time.perf_counter()

                
                curr_group_seq.pop_one_stage()
                _get_best_model_schedule(
                    gen_execplans_baseline, search_method_baseline,
                    check_gap, sort_input,
                    cost_table,
                    model_sys, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit)
                curr_group_seq.append_plan_group(last_stage_plan_group)
                curr_group_seq.append_exec_time(last_stage_time)
                curr_group_seq.append_last_stage_model_sys_values(last_stage_model_sys_values)
                continue

            

        time1 = time.perf_counter()

        
        model_sys = ori_model_sys

        
        
        
        
        if (model_sys.remaining_models_are_on_the_last_layer()) \
            and (curr_group_seq.get_tmp_only_comp_throughput_after_adding_a_plan_group(plan_group) \
                 < best_group_seq.get_valid_throughput()):
            continue


        
        
        
        
        
        model_sys.recover_model_state(
            ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list,
            ori_inp_model_ids_list)
        
        
        
        model_sys = model_sys.gen_new_model_sys_with_fused_models(
            fused_model_list=plan_group.get_involved_fused_models())

        
        

        
        model_sys.check_finish_states_accuracy()

        
        plan_group.update_model_inp_out_lens(cost_table)

        
        model_sys.check_finish_states_accuracy()


        print(f"time4: {time.perf_counter() - time1}")
        time1 = time.perf_counter()


        
        new_iter_model_sys_values = \
                ori_inp_out_lens_list, ori_remaining_decode_flops_list, \
                    ori_inp_seq_ids_list, ori_inp_model_ids_list, \
                    ori_model_sys
        curr_group_seq.append_plan_group(plan_group)
        curr_group_seq.append_exec_time(plan_group.get_infer_stage_latency())
        curr_group_seq.append_last_stage_model_sys_values(new_iter_model_sys_values)
        _get_best_model_schedule(
            gen_execplans_baseline, search_method_baseline,
            check_gap, sort_input,
            cost_table,
            model_sys, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit)
        curr_group_seq.pop_one_stage()
    









def _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe(
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        cost_table: CostTable, 
        model_sys: MyModelSystem, 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: dict,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3)):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''

    
    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    if _CHECKED_SEQ_NUM > _MAX_SEQ_NUM:
        return

    print(f"CURRENT PLAN GROUP SEQ: {str(curr_group_seq)}")
    print(f"CURRENT BEST PLAN GROUP SEQ: {str(best_group_seq)}")



    
    
    
    if len(curr_group_seq.plan_group_seq) > model_sys.get_model_num():
        
        
        assert False, f'{[[str(_) for _ in group.exec_plans] for group in curr_group_seq]}'


    print(f"finish step 1:      curr depth is within limit")

    
    
    
    if model_sys.is_finished():
        
        print(f"all models finished")

        _CHECKED_SEQ_NUM += 1
        
        
        
        if curr_group_seq.get_tot_time() < best_group_seq.get_tot_time():
            best_group_seq.set_plan_group_and_time(
                curr_group_seq.plan_group_seq, curr_group_seq.time_seq, curr_group_seq.last_stage_model_sys_values_seq)
        return 
    

    if curr_group_seq.get_tot_time() >= best_group_seq.get_tot_time():
        
        return


    print(f"finish step 2")

    
    
    
    model_states = model_sys.get_base_model_states()


    if model_states in uniq_model_states:
        
        
        

        
        if curr_group_seq.get_tot_time() > uniq_model_states[model_states]:

            print(f"the state has been checked and curr is not the best choice for it")

            return
        else:
            uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    else:
        
        
        uniq_model_states[model_states] = curr_group_seq.get_tot_time()

    
    
        

    print(f"finish step 3")


    
    plan_groups: List[MyExecPlanGroup] = get_one_stage_exec_plans_sorted_greedy_baseline_adapted_from_MuxServe(
        gen_execplans_baseline,
        check_gap, sort_input,
        curr_group_seq.get_last_stage_exec_plans(),
        cost_table, model_sys, gpu_name, tot_gpu_num, byte_per_gpu)
    
    
    
    
    ori_inp_out_lens_list = model_sys.get_model_inp_out_lens()
    ori_remaining_decode_flops_list = model_sys.get_model_remaining_decode_flops()
    ori_inp_seq_ids_list = model_sys.get_model_inp_seq_ids()
    ori_inp_model_ids_list = model_sys.get_model_inp_model_ids()
    ori_model_sys = model_sys



    
    
    
        
    
        
    


    print(f"finish step 4")


    
    for plan_group in plan_groups:
        
        if len(plan_group) == 0:
            continue

        
        
        
        
        
        

        
        model_sys = ori_model_sys


        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        model_sys.recover_model_state(
            ori_inp_seq_ids_list,ori_inp_out_lens_list, cost_table, ori_remaining_decode_flops_list,
            ori_inp_model_ids_list)
        
        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        
        model_sys = model_sys.gen_new_model_sys_with_fused_models(
            fused_model_list=plan_group.get_involved_fused_models())

        print(f"model_sys model objects: {list(model_sys.model_dict.items())}")
        print(f"plan_group model objects: {[(plan.model.model_id, plan.model) for plan in plan_group.exec_plans]}")

        
        model_sys.check_finish_states_accuracy()

        
        plan_group.update_model_inp_out_lens(cost_table)

        
        model_sys.check_finish_states_accuracy()



        
        
        
        
        
        
        
        
        curr_group_seq.append_plan_group(plan_group)
        curr_group_seq.append_exec_time(plan_group.get_infer_stage_latency())
        _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table,
            model_sys, curr_group_seq, best_group_seq, uniq_model_states, gpu_name, tot_gpu_num, byte_per_gpu)
        curr_group_seq.pop_one_stage()
    







def _get_best_model_schedule_dispatcher(
        search_method_baseline: str,
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        cost_table: CostTable, 
        model_sys: MyModelSystem, 
        curr_group_seq: MyExecPlanGroupSeq, 
        best_group_seq: MyExecPlanGroupSeq, 
        uniq_model_states: dict,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), top_k=float('inf'), fully_connected_gpu_unit: int=4):
    '''
    Input: 
        model_list: (model_name, flops_per_token, (layer_num, param_byte_per_layer, extra_param_byte)).
    Output: the model execution plan for each execution stage and the cost.
    We try enumeration first, backtracking based enumeration (*this one), dynamic programming, ...
    '''
    print(f"search_method_baseline: {search_method_baseline}")
    
    
    if True:
        _get_best_model_schedule(
            gen_execplans_baseline, search_method_baseline,
            check_gap, sort_input,
            cost_table, 
            model_sys, 
            curr_group_seq, 
            best_group_seq, 
            uniq_model_states,
            gpu_name, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit)
    else:
        assert False, "ERROR: we currently do not support greedy search algorithm"
        _get_best_model_schedule_greedy_baseline_adapted_from_MuxServe(
            gen_execplans_baseline,
            check_gap, sort_input,
            cost_table, 
            model_sys, 
            curr_group_seq, 
            best_group_seq, 
            uniq_model_states,
            gpu_name, tot_gpu_num, byte_per_gpu, fully_connected_gpu_unit)        





def get_engin_args(model_path, tensor_parallel_size):
    backend = "ours"
    model = model_path
    tokenizer = None
    quantization = None
    
    n = 1
    use_beam_search = False
    
    seed = 0
    hf_max_batch_size = None
    trust_remote_code = True
    max_model_len = None
    dtype = 'auto'
    enforce_eager = True
    kv_cache_dtype = "auto"
    device = "cuda"
    
    gpu_use_ratio = 0.9
    
   
    
    if tokenizer is None:
        tokenizer = model

    
    from vllm import LLM
    (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config) = LLM.get_engine_configs_only(
        model=model,
        tokenizer=tokenizer,
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        kv_cache_dtype=kv_cache_dtype,
        device=device,
        
        
        
        gpu_memory_utilization=gpu_use_ratio,
        max_num_seqs=512,
        max_paddings=512,
    )
    return (model_config, cache_config, parallel_config, scheduler_config,
            device_config, lora_config)









def get_extra_param_byte_manual_computation(model_info: MyModelInfor, tp_size: int):
    '''
        Input:
            tp_size: the number of tensor parallel workers.
    '''
    assert "Llama" in model_info.model_name, "we only support llama models now"

    hf_config = model_info.hf_config
    V: int = hf_config.vocab_size
    h: int = hf_config.hidden_size

    
    max_position_embeddings = getattr(hf_config, "max_position_embeddings",
                                          8192)

    rotary_dim=hf_config.hidden_size // hf_config.num_attention_heads
    cos_sin_cache = max_position_embeddings*rotary_dim


    extra_param_byte = (2*V*h/tp_size+h+cos_sin_cache) * model_info.data_byte
    return extra_param_byte





def get_param_byte_per_layer_manual_computation(model_info: MyModelInfor, tp_size: int):
    '''
        Input:
            comp_worker_num: the number of tensor parallel workers.
    '''
    assert "Llama" in model_info.model_name, f"we only support llama models now, but the model name is {model_info.model_name}"

    hf_config = model_info.hf_config

    total_num_heads = hf_config.num_attention_heads
    head_dim = hf_config.hidden_size // total_num_heads
    total_num_kv_heads = hf_config.num_key_value_heads

    
    input_size = hf_config.hidden_size
    num_heads = total_num_heads// tp_size
    num_kv_heads = max(total_num_kv_heads//tp_size, 1)
    output_size = (num_heads+2*num_kv_heads)*tp_size*head_dim
    W = output_size/tp_size*input_size

    
    
    W = W + hf_config.hidden_size/tp_size*hf_config.hidden_size

    
    
    
    
    
    

    

    
    W = W + hf_config.hidden_size*hf_config.intermediate_size*2/tp_size


    
    W = W + hf_config.intermediate_size/tp_size * hf_config.hidden_size

    

    
    W = W + hf_config.hidden_size
    
    W = W + hf_config.hidden_size

    

    return W * model_info.data_byte

    

    _, h,I, _ = model_info.model_config
    param_byte_per_layer =  (4*h*h/comp_worker_num+3*I*h/comp_worker_num+2*h) * model_info.data_byte
    return param_byte_per_layer





def get_per_layer_and_extra_param_and_buffer_byte(
        model_info: MyModelInfor, tp_size: int):
    '''
        Input:
            tp_size: the number of tensor parallel workers.

        NOTE: compute according to model.parameters() and model.buffers().
    '''
    if (model_info.model_path, tp_size) not in model_sizes:
        per_layer = 1e12
        extra = 1e12
        return per_layer, extra
        
    per_layer, extra = model_sizes[(model_info.model_path, tp_size)]
    if per_layer == None:
        per_layer = 1e12
        extra = 1e12
    return per_layer, extra





def get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config):
    cache_block_size = CacheEngine.get_cache_block_size(
            cache_config.block_size, cache_config.cache_dtype, model_config, parallel_config)
    return cache_block_size




def get_model_info_objs(
        cost_table: CostTable,
        data_byte: int,
        inp_lens_dict: Dict[str, List[int]],
        model_paths: List[str], 
        inp_seq_ids_dict: Dict[int, int],
        inp_req_from_which_models: Dict[int, Dict[int, List[int]]], 
        
        inp_req_from_which_model_which_out_reqs: Dict[int, Dict[int, Dict[int, int]]],
        
        independent_srcs: Dict[int, bool],
        
        outlen_generators,
        
        sample_config: Tuple[float, float, float, float], 
        trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Get the list of MyModelInfor objects for the given model paths.
        NOTE: 
            inp_seq_ids_dict: stores the ids of the inp seqs each model needs to answer. 
                Support the chain summary case where each LLM stage has different number of inp reqs.
            inp_lens_dict: dict of {model_path: inp_lens}
            1. inp_req_from_which_models: store the inp model of each inp seq for a model if it does not take all out seqs from each inp model
            2. inp_req_from_which_model_which_out_reqs: store the inp model's out req id of each inp seq for a model 
                if it does not take all out seqs from each inp model or 
                if some inp reqs take the same out reqs as input
                ==> we want to use ``inp_req_from_which_model_which_out_reqs`` to replace ``inp_req_from_which_models``.
    '''
    
    
    
    
    out_lens_dict = {i: outlen_generators[i](
            i, model_path[model_path.find('/')+1:], inp_lens_dict[i]) for i, model_path in enumerate(model_paths)}

    
    

    return [MyModelInfor(
                model_id,
                cost_table,
                model_path, 
                outlen_generators[model_id],
                sample_config, trust_remote_code, revision,
                data_byte, 
                inp_lens_dict[model_id], 
                out_lens=out_lens_dict[model_id], 
                inp_seq_ids=inp_seq_ids_dict[model_id],
                
                inp_req_from_which_models=inp_req_from_which_models[model_id] if model_id in inp_req_from_which_models else None, 
                inp_req_from_which_model_which_out_reqs=inp_req_from_which_model_which_out_reqs[model_id] if model_id in inp_req_from_which_model_which_out_reqs else None, 
                
                independent_srcs=independent_srcs[model_id] if model_id in independent_srcs else False,
                
            ) for model_id, model_path in enumerate(model_paths)]





def get_inplens_base_on_log_files(req_num: int):
    import json
    def get_lens(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'output_lens =' in line:
                    pos = len('output_lens =')
                    values = json.loads(line[pos:])
                    return values

    filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_7.log' 
    lens = get_lens(filename)
    inps = [i[0] for i in lens]
    return inps[:req_num]


def get_outlens():
    import json
    def get_lens(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'output_lens =' in line:
                    pos = len('output_lens =')
                    values = json.loads(line[pos:])
                    return values

    filename = './Cost_Model_per_iter/baseline_tp1_llama2_7b_7.log' 
    lens = get_lens(filename)
    outs = [i[2] for i in lens] 
    return outs[:]



def get_best_model_schedule(
        search_method_baseline: str,
        gen_execplans_baseline: str,
        check_gap: int, sort_input: bool,
        model_paths: List[str], 
        
        num_prompts, inp_seq_ids_dict, out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]], 
        inp_req_from_which_models: Dict[int, Dict[int, List[int]]], 
        
        inp_req_from_which_model_which_out_reqs: Dict[int, Dict[int, Dict[int, int]]],
        
        independent_srcs: Dict[int, bool],
        inp_generators, inp_mergers, outlen_generators,
        prompt_templates_lens,
        
        out_edge_dict: Dict[int, List[int]],
        sample_config: Tuple[float, float, float, float],
        trust_remote_code:bool=True, revision:Optional[str] = None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=float('inf'), top_k=float('inf'), similar_threshold: float=0.1, fully_connected_gpu_unit:int=4, 
        machine_name='machine1'):
    """
        NOTE: ``inp_generator``, ``inp_merger``, ``outlen_generator`` are 3 functions about model inp/out lens.
    """

    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM, _MODEL_ID
    _MAX_SEQ_NUM = max_group_seq_num
    _CHECKED_SEQ_NUM = 0

    import time

    
    
    
    
    
    inp_lens_dict = {i:inp_generators[i](num_prompts, i, model_path, inp_seq_ids_dict[i]) for i, model_path in enumerate(model_paths)}
    for k, v in inp_lens_dict.items():
        print(f"len(inp_lens) of model {k}: {v}")


    print(f"finish preparing inp lens: --abs: {time.perf_counter()}")



    time1 = time.perf_counter()
    print(f"begin search: --abs: {time1}")

    
    
    cost_table: CostTable = None
    if machine_name == 'machine1':
        cost_table: CostTable = get_my_cost_table_directly.cost_table
    elif machine_name == 'machine2':
        cost_table: CostTable = get_my_cost_table_directly2.cost_table
    
    global _COST_MODEL_REF
    cost_model_serialized = cost_table.serialize(model_paths=model_paths)
    print(f"np.asarray(cost_model_serialized[0][1]).nbytes: {np.asarray(cost_model_serialized[0][1]).nbytes}")
    _COST_MODEL_REF = ray.put(cost_model_serialized)


    print(f"finish serialize cost model: --abs: {time.perf_counter()}")


    
    model_list: List[MyModelInfor] = get_model_info_objs(
        cost_table,
        data_byte, inp_lens_dict, model_paths, inp_seq_ids_dict, inp_req_from_which_models, inp_req_from_which_model_which_out_reqs,
        independent_srcs, 
        outlen_generators, sample_config, trust_remote_code, revision)
    

    print(f"finish init model objs: --abs: {time.perf_counter()}")

    model_sys = MyModelSystem(model_list=model_list, out_edge_dict=out_edge_dict, 
                              cost_table=cost_table, inp_mergers=inp_mergers, outlen_generators=outlen_generators,
                              prompt_templates_lens=prompt_templates_lens,
                              need_correct_inp_out_lens=True, 
                              out_req_id_mapping=out_req_id_mapping)

    _MODEL_ID = len(model_list)
    
    
    for model in model_sys.model_dict.values():
        model.ori_tot_remaining_decode_flops = model.remaining_decode_flops
        model.inp_base_model_ids = model.input_model_ids

    
    
    
    print(f"begin fusing similar models: --abs: {time.perf_counter()}")
    
    if search_method_baseline == 'naive' or gen_execplans_baseline == 'naive':
        similar_threshold = float('inf')
    
    similar_threshold = float('inf')
    model_sys = model_sys.fuse_similar_models_in_a_chain(
            tot_gpu_num, byte_per_gpu, cost_table,
            check_gap, sort_input,
            similar_threshold,
            fully_connected_gpu_unit)
    
    
    
    
    
    
    
    
    

    total_flops = get_total_model_flops(model_list, cost_table)
    curr_group_seq = MyExecPlanGroupSeq(total_flops, [], [], [])
    best_group_seq = MyExecPlanGroupSeq(total_flops, [None], [float('inf')], [None])    


    time_before_search = time.perf_counter()

    print(f"finish fusing similar models: --abs: {time_before_search}")

    _get_best_model_schedule_dispatcher(
        search_method_baseline,
        gen_execplans_baseline, 
        check_gap, sort_input,
        cost_table, model_sys, curr_group_seq, best_group_seq, dict(), gpu_name, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit)

    
    for plan_group in best_group_seq.plan_group_seq:
        for exec_plan in plan_group.exec_plans:
            exec_plan.load_cost_just_for_refer = cost_table.get_prepare_cost(exec_plan.model.model_name, exec_plan.get_key_single_dp_worker())

    
    time2 = time.perf_counter()
    print(f"Total search time: {time2 - time1}")
    print(f"Total time for preparation before search: {time_before_search - time1}")
    print(f"Best group seq: {best_group_seq.get_str_using_model_ids()}")
    print(f"Best group seq throughputs: {best_group_seq.get_stage_throughputs()}")

    return best_group_seq






def test_data_parallel_improvement(
        search_method_baseline: str,
        gen_execplans_baseline: str,
        model_paths: List[str], 
        sample_config: Tuple[float, float, float, float],
        trust_remote_code:bool=True, revision:Optional[str] = None,
        gpu_name='A100-80G', tot_gpu_num = 4, byte_per_gpu=80*(1024**3), 
        data_byte=2,
        max_group_seq_num=float('inf')):
    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM
    _MAX_SEQ_NUM = max_group_seq_num
    _CHECKED_SEQ_NUM = 0

    import time
    time1 = time.perf_counter()

    
    cost_table = get_cost_table()

    
    inp_lens = get_inplens()

    out_lens_list = []
    inp_lens = sorted(inp_lens, reverse=True)

    for inp_num in [250, 500, 1000]:
        _CHECKED_SEQ_NUM = 0

        
        model_list: List[MyModelInfor] = get_model_info_objs(
            cost_table,
            data_byte, inp_lens, model_paths, sample_config, trust_remote_code, revision)

        
        if out_lens_list == []:
            assert len(model_list[0].out_lens) == 1000
            for model in model_list:
                out_lens_list.append(model.out_lens)
        for model, out_lens in zip(model_list, out_lens_list):
            model.inp_lens = tuple([inp_lens[(1000//inp_num)*i] for i in range(inp_num)])
            model.out_lens = tuple([out_lens[(1000//inp_num)*i] for i in range(inp_num)])

        print(f"len(inp_lens): {len(model.inp_lens)}, len(inp_lens): {len(model.out_lens)}")

        total_flops = get_total_model_flops(model_list, cost_table)
        curr_group_seq = MyExecPlanGroupSeq(total_flops, [], [])
        best_group_seq = MyExecPlanGroupSeq(total_flops, [None], [float('inf')])    

        _get_best_model_schedule_dispatcher(
            search_method_baseline,
            gen_execplans_baseline, 
            cost_table, model_list, curr_group_seq, best_group_seq, dict(), gpu_name, tot_gpu_num, byte_per_gpu)
        
        time2 = time.perf_counter()
        print(f"Total search time: {time2 - time1}")
        print(f"Best group seq: {str(best_group_seq)}")

    return best_group_seq










def get_dependent_exec_plans_for_each_plan(
        plan_group_seq: MyExecPlanGroupSeq
        )->Tuple[List[MyExecPlan], Dict[int, List[int]]]:
    '''
        Return the dependent exec plans for each exec plan in the given plan_group_seq.
        Definition of exec plan dependency:
            (1) exec plan in stage ``i+1'' depend on exec plans in stage ``i'';
            (2) exec plans in the same stage do not depend on each other;
            (3) if a model's exec plans are the same in two consecutive stages, they are regarded as the same one, and it 
            and any other exec plan in its alive stages do not depend on each other.
    '''
    exec_plan_serial_id: Dict[MyExecPlan, int] = dict()
    depend_on: Dict[int, List[int]] = dict()
    uniq_exec_plan_seq: List[MyExecPlan] = list()
    exec_plan_id = 0
    for stage_i in range(len(plan_group_seq.plan_group_seq)):
        plan_group = plan_group_seq.plan_group_seq[stage_i]
        if stage_i == 0:
            
            for exec_plan in plan_group.exec_plans:
                depend_on[exec_plan_id] = list()
                exec_plan_serial_id[exec_plan] = exec_plan_id
                exec_plan_id += 1
                uniq_exec_plan_seq.append(exec_plan)
        else:
            
            last_stage_exec_plans = plan_group_seq.plan_group_seq[stage_i-1].exec_plans
            last_stage_exec_plan_id_dict = {
                (exec_plan.model, exec_plan.get_key()):exec_plan_serial_id[exec_plan]
                for exec_plan in last_stage_exec_plans
                }
            
            
            continue_exec_plan_ids: List[int] = list()
            new_exec_plan_ids: List[int] = list()
            
            for exec_plan in plan_group.exec_plans:
                if (exec_plan.model, exec_plan.get_key()) in last_stage_exec_plan_id_dict:
                    
                    same_exec_plan_id = last_stage_exec_plan_id_dict[(exec_plan.model, exec_plan.get_key())]
                    exec_plan_serial_id[exec_plan] = same_exec_plan_id
                    continue_exec_plan_ids.append(same_exec_plan_id)
                else:
                    
                    exec_plan_serial_id[exec_plan] = exec_plan_id
                    new_exec_plan_ids.append(exec_plan_id)
                    exec_plan_id += 1
                    uniq_exec_plan_seq.append(exec_plan)
            
            
            depend_on_last_stage_exec_plan_ids = \
                list(np.intersect1d(list(last_stage_exec_plan_id_dict.values()), continue_exec_plan_ids))
            for this_plan_id in new_exec_plan_ids:
                depend_on[this_plan_id] = depend_on_last_stage_exec_plan_ids
    

    return uniq_exec_plan_seq, depend_on






def get_arxiv_data_set_chunks(file_name: str, chunk_size: int):
    """
        Just used to check the dataset used in Parrot's experiments.
    """
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from transformers import AutoTokenizer
    
    loader = TextLoader(f"../workloads/arxiv-march-2023/arxiv-sampled/{file_name}.txt")
    docs = loader.load()
    
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=0,
        separator=" ",
    )
    split_docs = text_splitter.split_documents(docs)
    
    return len(split_docs) 








    
    
    
    
    
    
    



