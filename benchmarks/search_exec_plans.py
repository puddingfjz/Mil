"""
This file contains the search method to find the best exec plans
for the given set of models and the given set of requests.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import itertools
import fake_scheduling
from my_per_iter_latency_estimator import CostTable, get_cost_table, get_cost_table_from_serialized_data
import get_my_cost_table_directly, get_my_cost_table_directly_zxcpu
import output_length_sampler

from vllm.transformers_utils.config import get_config
from vllm.worker.cache_engine import CacheEngine
from collections import defaultdict

from model_size_database import model_sizes, peak_mem_dict
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


_Presolved_LP = None


PRINT_LOGS = False
STAGE_SYNCH_TIMES = dict()
STAGE_LATENCY = dict()


USE_PEAK_MEM = False


Cost_Query_Time = 0
Cost_Query_num = 0
Plan_Group_Enu_Time = 0
Plan_Group_Enu_num = 0
Plan_Group_num_After_Pruning = 0
Iter_Search_Time = 0
Select_Best_Time = 0


PRINT_LOGS = False


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


        for from_model_id, req_ids in self.inp_req_from_which_model_which_out_reqs.items():

            if len(alive_seq_ids.intersection(req_ids)) > 0:
                unfinished_srcs.append(from_model_id)
        return unfinished_srcs


    def get_base_model_ids(self):
        return [self.model_id]

    def get_base_models(self):
        return [self]


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
            from artifact.mil_artifact.model_runtime import resolve_model_snapshot
            hf_config = get_config(resolve_model_snapshot(self.model_path), self.trust_remote_code, self.revision)
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
    if PRINT_LOGS:
        with open(f'tmp_time_logs.log', 'a') as f:
            f.write(f"TIME TO DP FAKE SCHEDULE 0: {os.getpid()} ---abs: {time1}\n")


    cost_table = get_cost_table_from_serialized_data(*cost_table)


    time1 = time.perf_counter()
    if PRINT_LOGS:
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

    if PRINT_LOGS:
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
        Submit scheduling simulations to Ray, one task per data-parallel replica.
    """
    global _COST_MODEL_REF
    cost_table = _COST_MODEL_REF
    res_ref_list = [None] * dp_size


    for dp_id in range(dp_size):

        dp_inp_lens = dp_inp_lens_list[dp_id]
        dp_out_lens = dp_out_lens_list[dp_id]
        dp_arrive_times = arrive_times[dp_id::dp_size]

        time1 = time.perf_counter()
        if PRINT_LOGS:
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
    if PRINT_LOGS:
        with open(f'tmp_time_logs.log', 'a') as f:
            f.write(f"TIME TO DP FAKE SCHEDULE vertical 0: {os.getpid()} ---abs: {time1}\n")


    cost_table = get_cost_table_from_serialized_data(*cost_table)


    time1 = time.perf_counter()
    if PRINT_LOGS:
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

    if PRINT_LOGS:
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
        Submit scheduling simulations to Ray, one task per data-parallel replica.
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
        if PRINT_LOGS:
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


        self.additional_extra_time = 0


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

    def get_dp_inp_lens_list_for_models(self):
        return self.dp_inp_lens_list


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


        if PRINT_LOGS:
            print(f"exec plan: {self.model.get_base_model_ids(), self.get_key()}")

        latencys = np.asarray(latency_list)
        latencys = latencys[latencys < 1e8]
        if len(latencys) == 0:
            return 0
        else:
            return max(latencys)


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
                3. Arrival times follow model.inp_seq_ids and must be sorted before simulation.
        '''


        inp_lens, out_lens, inp_seq_ids, arrive_times = self._sort_and_partition_data_parallel(arrive_times)


        key = (self.model.model_name, self.get_key(), *self._get_inp_key(arrive_times))
        self._FAKE_SCHEDULING_RES_key = key


        if key in _FAKE_SCHEDULING_RES:

            if PRINT_LOGS:
                print(f"REUSE EXISTING FAKE SCHEDULING RESULTS!\n")

            (self.cumsum_latencys_list, self.cum_rng_nums_list, self.rng_starts_list, self.rng_ends_list,
                self.is_prefill_steps_list, self.finish_times_list,
                self.throughput_till_each_iter_list) = _FAKE_SCHEDULING_RES[key]


            self.total_latency_list = [self._get_valid_max_latency(cumsum_latencys) if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]


            if PRINT_LOGS:
                print(f"total_latency_list: {self.total_latency_list}, self.extra_cost: {self.extra_cost}")


            return self.total_latency_list
        else:


            global Cost_Query_num
            Cost_Query_num += 1


            if PRINT_LOGS:
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
            if PRINT_LOGS:
                with open(f'tmp_time_logs.log', 'a') as f:
                    f.write(f"TIME TO launch remote fake scheduling: exec_plan_key: {self.get_key()}: {time2 - time1}, ---abs: {time2}\n")
            return




    def get_max_dp_latency_considering_plan_group(self, cost_table: CostTable,
            check_gap: int, sort_input: bool, arrive_times: List[float]):
        '''
            Input:
                extra_cost: the time to prepare (e.g., load) the model before running.
        '''

        if PRINT_LOGS:
            print(f"exec_plan: {self.model.get_base_model_ids(), self.get_key()}")


        if self.total_latency_list[0] == None:
            self.estimate_exec_time(cost_table,
                check_gap=check_gap, sort_input=sort_input, arrive_times=arrive_times)


        if self.fake_scheduling_res_ref != None:
            if PRINT_LOGS:
                print("calling remote fake scheduling function-------")
            return None


        if PRINT_LOGS:
            print(f"exec plan latency list: {(self.model.get_base_model_ids(), self.get_key()), self.total_latency_list}, self.extra_cost: {self.extra_cost}")


        return max(self.total_latency_list) + self.extra_cost


    def _wait_for_remote_fake_scheduling(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        if self.fake_scheduling_res_ref != None:
            if PRINT_LOGS:
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
            if PRINT_LOGS:
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

                if PRINT_LOGS:
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


            if PRINT_LOGS:
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
        Execution plan for vertically fused models.
        Each model's original complete input sequence IDs must be range(ori_tot_req_num).
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


        self.additional_extra_time = 0


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

    def get_dp_inp_lens_list_for_models(self):
        return self.dp_inp_lens_list_for_models

    def get_finish_times_merged(self, seq_ids: List[int], model_ind: int):
        """
            Input: 
                model_ind: index of the base model whose request finish times are returned.
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
                3. Arrival times follow model.inp_seq_ids and must be sorted before simulation.
        '''


        time1 = time.perf_counter()

        self._sort_and_partition_data_parallel(arrive_times_list)

        time2 = time.perf_counter()
        if PRINT_LOGS:
            print(f"TIME--_sort_and_partition_data_parallel: {time2 - time1}")


        key = (self.model.model_name, self.get_key(), *(self._get_inp_key(arrive_times_list)))
        self._FAKE_SCHEDULING_RES_key = key


        if key in _FAKE_SCHEDULING_RES:

            if PRINT_LOGS:
                print(f"Reuse fake scheduling results")

            (self.cumsum_latencys_list,
             self.cum_rng_nums_list_for_models, self.rng_starts_list_for_models, self.rng_ends_list_for_models,
             self.is_prefill_steps_list,
             self.finish_times_list_for_models,
             self.throughput_till_each_iter_list) = _FAKE_SCHEDULING_RES[key]


            self.total_latency_list = [self._get_valid_max_latency(cumsum_latencys) if len(cumsum_latencys)>0 else 0 \
                                       for cumsum_latencys in self.cumsum_latencys_list]


            time3 = time.perf_counter()
            if PRINT_LOGS:
                print(f"TIME--reuse fake scheduling: {time3 - time2}")


            return self.total_latency_list
        else:


            global Cost_Query_num
            Cost_Query_num += 1


            if PRINT_LOGS:
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
            if PRINT_LOGS:
                with open(f'tmp_time_logs.log', 'a') as f:
                    f.write(f"TIME TO launch remote fake scheduling vertical: exec_plan_key: {self.get_key()}: {time2 - time1}, ---abs: {time2}\n")
            return




    def get_max_dp_latency_considering_plan_group(self, cost_table: CostTable,
            check_gap: int, sort_input: bool, arrive_times_list: List[List[float]]):
        '''
            Input:
                extra_cost: the time to prepare (e.g., load) the model before running.
        '''

        if PRINT_LOGS:
            print(f"exec_plan: {self.model.get_base_model_ids(), self.get_key()}")


        if self.total_latency_list[0] == None:
            self.estimate_exec_time(cost_table,
                check_gap=check_gap, sort_input=sort_input, arrive_times_list=arrive_times_list)


        if self.fake_scheduling_res_ref != None:
            if PRINT_LOGS:
                print("calling remote fake scheduling function vertical -------")
            return None


        if PRINT_LOGS:
            print(f"exec plan latency list: {(self.model.get_base_model_ids(), self.get_key()), self.total_latency_list}, self.extra_cost: {self.extra_cost}")


        return max(self.total_latency_list) + self.extra_cost


    def _wait_for_remote_fake_scheduling(self):
        """
            Wait for the latency list if remote fake scheduling is called.
        """
        if self.fake_scheduling_res_ref != None:
            if PRINT_LOGS:
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
            if PRINT_LOGS:
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


            if PRINT_LOGS:
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


class MyDummyExecPlanGroup:
    """ My execution plan group definition. """
    def __init__(
        self,
        exec_plans: List[MyExecPlan],
    ) -> None:
        self.exec_plans = exec_plans


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


        consider_whole_stage_synch_time=False,
    ) -> None:


        self.consider_whole_stage_synch_time = consider_whole_stage_synch_time


        if PRINT_LOGS:
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


    def get_involved_fused_models(self):
        return [exec_plan.model for exec_plan in self.exec_plans if isinstance(exec_plan, MyVerticalFusedExecPlan)]


    def comp_extra_prepare_costs(
            self, cost_table: CostTable, last_stage_exec_plans: List[MyExecPlan]):
        '''
            Compute the extra prepare costs for each exec plan in this group.
            There will be extra prepare cost if:
                (1) the exec plan of the same model changes.
                NOTE: Preparation overhead from memory-only changes is not modeled.
        '''
        model_exec_plans = { exec_plan.model : exec_plan for exec_plan in last_stage_exec_plans }


        if self.consider_whole_stage_synch_time:
            last_stage_latency = 0
            if tuple(last_stage_exec_plans) in STAGE_LATENCY:
                last_stage_latency = STAGE_LATENCY[tuple(last_stage_exec_plans)]
            remaining_extra_costs = { exec_plan.model.model_id : max(exec_plan.extra_cost - last_stage_latency, 0) for exec_plan in last_stage_exec_plans }


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

            # Account for vertical model fusion when assigning preparation cost.
            elif _get_vertical_fuse_model_pairs(last_stage_exec_plans, exec_plan)!=None:

                exec_plan.set_extra_cost(0.0)

            else:


                exec_plan.set_extra_cost(
                    cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                )


        if self.consider_whole_stage_synch_time:

            ori_extra_costs = { exec_plan.model.model_id:exec_plan.extra_cost for exec_plan in self.exec_plans }

            synch_time = max([0] + [remaining_extra_costs[model_id] for model_id in remaining_extra_costs if ((model_id not in ori_extra_costs) or (ori_extra_costs[model_id] != 0))])
            STAGE_SYNCH_TIMES[tuple(self.exec_plans)] = synch_time
            for exec_plan in self.exec_plans:
                if exec_plan.extra_cost == 0:
                    exec_plan.set_extra_cost( remaining_extra_costs[exec_plan.model.model_id] )
                else:
                    exec_plan.set_extra_cost( synch_time + exec_plan.extra_cost )


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
            NOTE: finish_times follow the order of model.inp_seq_ids.
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

        """
            Each base model in a fused plan exposes its own outputs to downstream models.
        """


        inp_exec_plans = self.inp_exec_plan_dict[exec_plan]
        in_exec_plans_base_model_ids = [in_exec_plan.get_base_model_ids() for in_exec_plan in inp_exec_plans]

        finish_times_list: List[List[float]] = list()


        for base_model in exec_plan.get_base_models():

            inp_info: List[MyExecPlan, int] = list()


            # Only some upstream base models may be finished or running.

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


        if PRINT_LOGS:
            print(f"INIT plan group: {[((exec_plan.model.get_base_model_ids(), exec_plan.get_key()), exec_plan.model.model_id, isinstance(exec_plan, MyVerticalFusedExecPlan)) for exec_plan in self.exec_plans]}")


        self.comp_extra_prepare_costs(cost_table, last_stage_exec_plans)


        arrive_times_list = [self._get_arrive_times(exec_plan) for exec_plan in self.exec_plans]
        arrive_times_list = [[vs - exec_plan.additional_extra_time for vs in arrive_times] \
                                if isinstance(arrive_times, list) \
                                else (arrive_times - exec_plan.additional_extra_time)\
                            for arrive_times, exec_plan in zip(arrive_times_list, self.exec_plans)]


        latencys = [exec_plan.get_max_dp_latency_considering_plan_group(
                        cost_table, check_gap, sort_input, arrive_times,

                    ) for arrive_times, exec_plan in zip(arrive_times_list, self.exec_plans)]


        return




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
            # NOTE: here we need to continue the computation after waiting for the remote fake scheduling calls to finish
        '''

        if PRINT_LOGS:
            print(f"Continue to INIT plan group: {[(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.model.model_id, isinstance(exec_plan, MyVerticalFusedExecPlan)) for exec_plan in self.exec_plans]}")


        time1 = time.perf_counter()


        latencys = [exec_plan.wait_for_remote_fake_scheduling_and_get_max_dp_latency_considering_plan_group(
                    ) for exec_plan in self.exec_plans]


        latencys_to_consider = [v for v, exec_plan in zip(latencys, self.exec_plans) if len(self.inp_exec_plan_dict[exec_plan])==0]


        latency = min(latencys_to_consider)


        if self.consider_whole_stage_synch_time:
            latency = max(latency, STAGE_SYNCH_TIMES[tuple(self.exec_plans)])
            STAGE_LATENCY[tuple(self.exec_plans)] = latency


        if PRINT_LOGS:
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

        if PRINT_LOGS:
            print(f"throughput: {self.get_throughput()}")


            print(f"comp cost 2/2: {time.perf_counter()-time1}")
        time1 = time.perf_counter()


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
            elif isinstance(exec_plan, MyExecPlan):
                print(f"MyExecPlan")
                return (
                [exec_plan.model.model_name],
                tuple(exec_plan.model.get_base_model_ids()),
                [tuple(exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[0])],
                [tuple(exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)[1])]
                )

        ret = list()
        for exec_plan, inp_out_lens in zip(self.exec_plans, self.tmp_inp_out_lens_list):
            states = get_model_state(exec_plan, inp_out_lens)
            names, model_ids, inp_lens, out_lens = states
            ret.extend(zip(names, model_ids, inp_lens, out_lens))

        return tuple(sorted(ret))



    def get_model_states_before_infer_stage(self):
        ret = list()
        for exec_plan in self.exec_plans:
            ret.extend([model.get_state() for model in exec_plan.get_base_models()])
        return tuple(sorted(ret))


    def update_model_inp_out_lens(self, cost_table: CostTable):


        for exec_plan, inp_out_lens, stop_iter_i in \
            zip(self.exec_plans, self.tmp_inp_out_lens_list, self.tmp_stop_iter_i_list):

            old_inp_lens = exec_plan.get_dp_inp_lens_list_for_models()


            merged_inp_out_lens = exec_plan.merge_new_inp_out_lens_of_data_parallel_workers(inp_out_lens)


            exec_plan.model.update_inp_out_seqlens(*merged_inp_out_lens, cost_table)


            if not exec_plan.model.is_finished():

                exec_plan.update_fake_schedule_output_after_an_infer_stage(
                    old_inp_lens, inp_out_lens,
                    *merged_inp_out_lens,
                    stop_iter_i, cost_table)


        return


    def __str__(self):
        '''
        Get the string to represent a plan group: 
            the exec_plan settings + the remaining_lens after the current infer stage.
        '''
        return str(sorted([f"{str(exec_plan)}: {str(exec_plan.total_latency_list)}s"
                           for exec_plan in self.exec_plans])) \
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


        self.hybridflow_result = None

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

    def get_tmp_only_comp_throughput_after_adding_a_plan_group(self, plan_group: MyExecPlanGroup):
        flops = sum([group.get_throughput()*group.get_infer_stage_latency() for group in self.plan_group_seq+[plan_group]])


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




    def get_models_at_given_level(self, level_num: int):
        if level_num >= len(self.all_level_model_ids):
            return list()
        return [self.model_dict[model_id] for model_id in self.all_level_model_ids[level_num]]


    def get_runnable_plans_from_cand_plans(
            self,
            running_plan_group: List[MyExecPlan],
            cand_models: List[MyModelInfor], cand_exec_plans: List[List[MyExecPlan]]):

        """
            Select runnable plans using model-level dependencies.
            Chain-summary summarization steps are represented by one fused model.
        """


        def is_finished_or_running(model: MyModelInfor, running_model_ids: List[int]):
            return (model.is_finished()) or (model.model_id in running_model_ids)

        runnable_exec_plans: List[List[MyExecPlan]] = list()
        running_model_ids: List[int] = [exec_plan.model.model_id for exec_plan in running_plan_group]


        for model, exec_plans in zip(cand_models, cand_exec_plans):
            inps = model.input_model_ids


            inps_status = [is_finished_or_running(self.model_dict[inp], running_model_ids) for inp in inps]


            if False in inps_status:
                continue


            runnable_exec_plans.append(exec_plans)
        return runnable_exec_plans


    def get_all_level_models(self, aggresive_for_horizontally_fused_model: bool):
        """
        Populate self.all_level_model_ids with unfinished models in dependency order.

        Normally, every input model must have appeared in an earlier level.
        With aggresive_for_horizontally_fused_model enabled, independent-source
        models may enter a level once any unfinished input source has appeared,
        or when no unfinished input sources remain. Finished sources are ignored.
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
                1. Apply early validity checks to execution plans.
                2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``),
                all candidate execution plans must be checked individually.
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
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
                    if PRINT_LOGS:
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
                    1. Apply early validity checks to execution plans.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    all candidate execution plans must be checked individually.
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
            LPT stands for ``Longest Processing Time``.
            Algorithm:
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


            if PRINT_LOGS:
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


            for model_ids in model_ids_of_each_layer:


                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]


                import time
                begin_query = time.perf_counter()


                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    if PRINT_LOGS:
                        print(f"waiting for exec plan: {(exec_plan.model.get_base_model_ids(), exec_plan.get_key())}")
                    exec_plan._wait_for_remote_fake_scheduling()


                end_query = time.perf_counter()
                global Cost_Query_Time
                Cost_Query_Time += (end_query-begin_query)


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

            if PRINT_LOGS:
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
                Algorithm:
                    the nonpreemption heuristic: 
                        1. for an exec plan, if its 2*extra cost >= computation latency (to complete all its workloads), we call it invalid, i.e., not worth preemption.
                        2. for a model, if all its candidate exec plans except the one in the last stage are ``invalid``, then we must select its last-stage exec plan for it, i.e., preemption is forbiddened in this case.

                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """
            def _get_model_latency_to_complete(exec_plans:List[MyExecPlan]):

                assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in exec_plans], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in exec_plans]

                infos = {exec_plan.model.model_id: (max(exec_plan.total_latency_list), exec_plan.extra_cost, get_tot_worker_num([exec_plan])) for exec_plan in exec_plans}


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
                    1. Apply early validity checks to execution plans.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    all candidate execution plans must be checked individually.
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


            print(f"the best_plan_group we found a round: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in best_plan_group]}")


            return best_plan_group


        def _get_best_runnable_exec_plan_random(
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
                    1. Apply early validity checks to execution plans.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    all candidate execution plans must be checked individually.
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


            import random
            best_plan_group = random.choice(plan_groups_to_check)[1]

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


            model_min_gpu_dict = dict()

            for plan_group, exec_plan in plan_groups_to_check:
                if exec_plan.model.model_id in model_min_gpu_dict:
                    if get_tot_worker_num([exec_plan]) >= model_min_gpu_dict[exec_plan.model.model_id]:
                        continue
                model_min_gpu_dict[exec_plan.model.model_id] = get_tot_worker_num([exec_plan])

                for plan in plan_group:
                    if plan.model.model_id == exec_plan.model.model_id:
                        min_gpu_latency_dict[exec_plan.model.model_id] = max(plan.total_latency_list) + plan.extra_cost

            return min_gpu_latency_dict


        def _gen_cand_exec_plans_using_more_gpus(cand_models: List[MyModelInfor], gpu_nums_for_each_model: Dict[int, int],
            tot_exec_plans_list: Dict[int, List[MyExecPlan]]
        ):
            exec_plans_list = list()
            last_stage_plan_dict = {plan.model.model_id:plan.get_key() for plan in last_stage_exec_plans}
            for model in cand_models:


                exec_plans = None
                if model.model_id in tot_exec_plans_list:
                    exec_plans = tot_exec_plans_list[model.model_id]
                else:
                    exec_plans = get_possible_exec_plans(
                        model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
                    tot_exec_plans_list[model.model_id] = exec_plans


                if 'no_preemption' in search_method_baseline:
                    if model.model_id in last_stage_plan_dict:
                        exec_plans = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]


                if model.model_id not in gpu_nums_for_each_model:
                    gpu_nums_for_each_model[model.model_id] = 0

                exec_plans = [plan for plan in exec_plans if get_tot_worker_num([plan]) > gpu_nums_for_each_model[model.model_id]]


                exec_plans_list.append(exec_plans)

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


        tot_exec_plans_list = dict()


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


            exec_plans_list = _gen_cand_exec_plans_using_more_gpus(cand_models, gpu_nums_for_each_model, tot_exec_plans_list)


            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)


            min_gpu_latency_dict: Dict[int, float] = _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                best_plan_group,
                exec_plans_list,
                check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


            if 'randomCost' not in search_method_baseline:


                last_stage_plan_dict = {plan.model.model_id for plan in last_stage_exec_plans}
                cand_models = sorted(cand_models, key=lambda model: (model.model_id in last_stage_plan_dict, -model.check_order, min_gpu_latency_dict[model.model_id]), reverse=True)

            else:
                import random
                random_costs = list(range(len(cand_models)))
                random.shuffle(random_costs)
                random_costs = {model.model_id: cost for model, cost in zip(cand_models, random_costs)}

                last_stage_plan_dict = {plan.model.model_id for plan in last_stage_exec_plans}
                cand_models = sorted(cand_models, key=lambda model: (model.model_id in last_stage_plan_dict, -model.check_order, random_costs[model.model_id]), reverse=True)


            time_analysis[0] += (time.perf_counter() - time1)


            should_stop = False
            while((len(cand_models) > 0) or is_last_level):

                if not is_last_level:
                    tot_cand_models.append(cand_models[0])
                    cand_models = cand_models[1:]


                print(f"tot_cand_models: {[model.model_id for model in tot_cand_models]}")


                time1 = time.perf_counter()


                exec_plans_list = _gen_cand_exec_plans_using_more_gpus(tot_cand_models, gpu_nums_for_each_model, tot_exec_plans_list)


                time_analysis[1] += (time.perf_counter() - time1)

                old_best_plan_gpu_num = get_tot_worker_num(best_plan_group)
                old_best_plan_model_num = len(best_plan_group)

                for cand_plan_group in [best_plan_group]:


                    time1 = time.perf_counter()

                    runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, tot_cand_models, exec_plans_list)


                    print(f"time get runnable plans: {time.perf_counter() - time1}")
                    time_analysis[2] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()


                    if 'randomCost' not in search_method_baseline:
                        best_plan_group = _get_best_runnable_exec_plan(
                            uniq_exec_plan_mapping,
                            cand_plan_group, runnable_exec_plans_list,

                            check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)
                    else:
                        best_plan_group = _get_best_runnable_exec_plan_random(
                            uniq_exec_plan_mapping,
                            cand_plan_group, runnable_exec_plans_list,

                            check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


                    gpu_nums_for_each_model.update({plan.model.model_id:get_tot_worker_num([plan]) for plan in best_plan_group})


                    print(f"time get good plans: {time.perf_counter() - time1}")
                    time_analysis[3] += (time.perf_counter() - time1)
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

        time_analysis[4] += (time.perf_counter() - time1)


        print(f"time finish get costs: {time.perf_counter() - time1}")


        print(f"time_analysis: {time_analysis}")

        return plan_groups


    def get_candidate_plan_groups_oracle(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
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

            if PRINT_LOGS:
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


            for model_ids in model_ids_of_each_layer:


                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    if PRINT_LOGS:
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

            if PRINT_LOGS:
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


        def _sample_cand_exec_plans_to_speedup(runnable_exec_plans_list, cand_plan_group):
            last_stage_models = [plan.model.model_id for plan in last_stage_exec_plans]
            exec_plans_to_keep = list()
            exec_plans_to_sample = list()
            for exec_plans in runnable_exec_plans_list:
                if exec_plans[0].model.model_id in last_stage_models:
                    exec_plans_to_keep.append(exec_plans)
                else:
                    exec_plans_to_sample.append(exec_plans)

            import random
            sampled = random.sample(exec_plans_to_sample, min(len(exec_plans_to_sample), tot_gpu_num-len(cand_plan_group)-len(exec_plans_to_keep)))
            return exec_plans_to_keep + sampled


        def _keep_no_preemption_plan_groups_only(plan_groups, runnable_exec_plans_list):
            last_stage_models = [plan.model.model_id for plan in last_stage_exec_plans]
            cand_model_ids = [exec_plans[0].model.model_id for exec_plans in runnable_exec_plans_list]
            model_ids_to_keep = set(last_stage_models).intersection(cand_model_ids)
            ret = list()
            for plan_group in plan_groups:
                if len(plan_group) < len(model_ids_to_keep):
                    continue
                model_ids = set([plan.model.model_id for plan in plan_group])
                if model_ids_to_keep.issubset(model_ids):
                    ret.append(plan_group)
            return ret


        def _comp_extra_prepare_costs_to_consider_stage_synch(
                cost_table: CostTable, exec_plans: List[MyExecPlan], last_stage_exec_plans: List[MyExecPlan]):
            '''
                Compute the extra prepare costs for each exec plan in this group.
                There will be extra prepare cost if:
                    (1) the exec plan of the same model changes.
                    NOTE: Preparation overhead from memory-only changes is not modeled.
            '''
            model_exec_plans = { exec_plan.model : exec_plan for exec_plan in last_stage_exec_plans }


            last_stage_latency = 0
            if tuple(last_stage_exec_plans) in STAGE_LATENCY:
                last_stage_latency = STAGE_LATENCY[tuple(last_stage_exec_plans)]
            remaining_extra_costs = { exec_plan.model.model_id : max(exec_plan.extra_cost - last_stage_latency, 0) for exec_plan in last_stage_exec_plans }


            ori_extra_costs = dict()
            for exec_plan in exec_plans:
                model = exec_plan.model
                if model in model_exec_plans:
                    last_exec_plan = model_exec_plans[model]
                    if exec_plan.get_key() == last_exec_plan.get_key():
                        ori_extra_costs[exec_plan.model.model_id] = 0.0
                    else:

                        ori_extra_costs[exec_plan.model.model_id] = cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())
                else:

                    ori_extra_costs[exec_plan.model.model_id] = cost_table.get_prepare_cost(model.model_name, exec_plan.get_key_single_dp_worker())


            synch_time = max([0] + [remaining_extra_costs[model_id] for model_id in remaining_extra_costs if ((model_id not in ori_extra_costs) or (ori_extra_costs[model_id] != 0))])

            for exec_plan in exec_plans:
                if ori_extra_costs[exec_plan.model.model_id] == 0:
                    ori_extra_costs[exec_plan.model.model_id] = ( remaining_extra_costs[exec_plan.model.model_id] )
                else:
                    ori_extra_costs[exec_plan.model.model_id] = ( synch_time + ori_extra_costs[exec_plan.model.model_id] )

            return ori_extra_costs


        def _update_uniq_exec_plan_mapping_considerStageSynch(
                plan_group: List[MyExecPlan], plan: MyExecPlan, uniq_exec_plan_mapping, extra_cost_dict,
            ) -> MyExecPlan:
            """
                plan_group is a list of execution plans.
                plan is an execution plan to be appended.
                Check whether the plan will need to be a new uniq plan object after being appended.
                If so, update the uniq_exec_plan_mapping; otherwise, use the plan stored in the uniq_exec_plan_mapping.
            """


            path_key = list()

            model_id2_plan = {exec_plan.model.model_id: exec_plan for exec_plan in plan_group}
            for model_id in list(plan.model.input_model_ids) + [plan.model.model_id]:
                if model_id in model_id2_plan:
                    exec_plan = model_id2_plan[model_id]
                    path_key.append((exec_plan.model.model_id, exec_plan.get_key(), extra_cost_dict[exec_plan.model.model_id]))


            path_key = tuple(path_key)
            if path_key in uniq_exec_plan_mapping:

                return uniq_exec_plan_mapping[path_key][0]
            else:

                exec_plan_to_use = plan.copy_the_plan()

                path_plans = [exec_plan_to_use]
                if len(path_key) > 1:
                    path_plans = uniq_exec_plan_mapping[path_key[:1]][1] + path_plans

                uniq_exec_plan_mapping[path_key] = exec_plan_to_use, path_plans
                return exec_plan_to_use


        def update_real_uniq_exec_plan_mapping_considerStageSynch(
                real_uniq_exec_plan_mapping, plan_groups: List[List[MyExecPlan]],
                cost_table, last_stage_exec_plans):
            """
                Get the uniq_exec_plan_mapping keys for each plan in each plan_group of the input plan_groups.
            """
            new_plan_groups: List[List[MyExecPlan]] = list()
            for plan_group in plan_groups:

                plan_dict: Dict[int, MyExecPlan] = {plan.model.model_id: plan for plan in plan_group}
                models = [self.model_dict[model_id] for model_id in plan_dict]
                models, model_ids_of_each_layer, in_stage_out_edge_dict = _get_model_ids_of_different_levels(models)
                sorted_plan_group = [plan_dict[model.model_id] for model in models]


                extra_cost_dict = _comp_extra_prepare_costs_to_consider_stage_synch(
                cost_table, sorted_plan_group, last_stage_exec_plans)


                new_sorted_plan_group: List[MyExecPlan] = list()
                for i, plan in enumerate(sorted_plan_group):
                    new_sorted_plan_group.append(
                        _update_uniq_exec_plan_mapping_considerStageSynch(sorted_plan_group, plan, real_uniq_exec_plan_mapping, extra_cost_dict))

                new_plan_groups.append(new_sorted_plan_group)
            return new_plan_groups


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")

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


                if 'sampleModels' in search_method_baseline:
                    if unfinished_model_num >= tot_gpu_num:
                        exec_plans = [sorted(exec_plans, key=lambda _: get_tot_worker_num([_]))[0]]


                exec_plans_list.append(exec_plans)


            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)


            time_analysis[0] += (time.perf_counter() - time1)


            for cand_plan_group in new_plan_groups:


                time1 = time.perf_counter()


                begin_enu = time.perf_counter()

                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)


                if 'sampleModels' in search_method_baseline:
                    runnable_exec_plans_list = _sample_cand_exec_plans_to_speedup(runnable_exec_plans_list, cand_plan_group)
                    print(f"number of cand models after sample: {len(runnable_exec_plans_list)}")


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


                end_enu = time.perf_counter()
                global Plan_Group_Enu_Time
                Plan_Group_Enu_Time += (end_enu - begin_enu)


                if 'no_preemption' in search_method_baseline:
                    plan_groups = plan_groups[:1] + _keep_no_preemption_plan_groups_only(plan_groups[1:], runnable_exec_plans_list)


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


                    time1 = time.perf_counter()


                    print(f"the groups we found a round: # of  groups before pruning: {len(plan_groups)}")
                    plan_groups = plan_groups[:1] + [plan_group for plan_group in plan_groups[1:] if _has_in_stage_inp_models(plan_group)]


                    time_analysis[4] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()


                    print(f"time get costs: {time.perf_counter() - time1}")
                    time_analysis[5] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()


                    good_plan_groups = plan_groups[1:]
                    all_plan_group_list.extend(good_plan_groups)
                    print(f"len(all_plan_group_list): {len(all_plan_group_list)}")


                    tmp_new_plan_groups.extend(good_plan_groups)


                    print(f"time finish get costs: {time.perf_counter() - time1}")
                    time_analysis[6] += (time.perf_counter() - time1)


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


            if 'sampleModels' in search_method_baseline:
                model_num_considered = sum([len(_) for _ in self.all_level_model_ids[:visit_model_level+1]])
                print(f"model_num_considered: {model_num_considered}")
                if model_num_considered >= tot_gpu_num:

                    break


        plan_groups = all_plan_group_list


        global Plan_Group_Enu_num
        Plan_Group_Enu_num += len(plan_groups)

        print(f"\nfinal candidates before pruning:\n")
        print(f"\nnumber of final candidates before pruning: {len(plan_groups)}\n")


        time1 = time.perf_counter()

        if 'randomCost' in search_method_baseline:
            import random
            plan_groups = [random.choice(plan_groups)]


        old_real_uniq_exec_plan_mapping_keys = list(real_uniq_exec_plan_mapping.keys())
        plan_groups = update_real_uniq_exec_plan_mapping_considerStageSynch(
                real_uniq_exec_plan_mapping, plan_groups,
                cost_table, last_stage_exec_plans)


        begin_query = time.perf_counter()


        _run_exec_cost_estimation_on_uniq_sub_plan_groups(
            real_uniq_exec_plan_mapping,
            old_real_uniq_exec_plan_mapping_keys,
            cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
        )

        plan_groups = [MyExecPlanGroup(
            plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status, consider_whole_stage_synch_time=True) for plan_group in plan_groups]
        for plan_group in plan_groups:
            plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)


        end_query = time.perf_counter()
        global Cost_Query_Time
        Cost_Query_Time += (end_query - begin_query)


        print(f"time finish get costs: {time.perf_counter() - time1}")


        print(f"time_analysis: {time_analysis}")

        return plan_groups


    def get_candidate_plan_groups_HybridFlow_old(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
        """


        def _get_placements(collection, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict, tot_gpu_num):
            if len(collection) == 1:
                yield [tuple(collection)]
                return
            first = collection[0]
            for smaller in _get_placements(collection[1:], fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict, tot_gpu_num):

                for n, subset in enumerate(smaller):


                    cand_modelset = tuple([first]) + subset
                    if not _GPU_mem_is_enough(
                            [self.model_dict[model_id] for model_id in cand_modelset], [fully_connected_gpu_unit]*len(cand_modelset),
                            tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict)[0]:
                        continue

                    yield smaller[:n] + [cand_modelset] + smaller[n+1:]


                if len(smaller) >= tot_gpu_num:
                    continue


                yield [tuple([first])] + smaller


        def _GPU_mem_is_enough(
                models: List[MyModelInfor], num_workers: List[int],
                tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict):
            '''
            check whether using tp=num_workers[i] for each model models[i], the gpu memory is large enough to hold the models together (colocation).
            Return: (1) whether the GPU memory is enough, (2) the total param_byte_per_comp_gpu per comp GPU
            '''


            tot_mem_demand = 0
            KV_cache_demands = list()
            for model, num_worker in zip(models, num_workers):

                if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
                    _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
                        model_path=model.model_path, tensor_parallel_size=num_worker)
                (model_config, cache_config, parallel_config, scheduler_config,
                    device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]

                infer_args = InferenceArgs(scheduler_config, cache_config)


                gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
                KV_cache_demands.append(gpu_cache_byte_per_block * ((model_config.max_model_len + cache_config.block_size - 1) // cache_config.block_size))


                if (model.model_id, num_worker) not in sufficient_gpu_mem_ratio_dict:

                    sufficient_gpu_mem_ratio_dict[(model.model_id, num_worker)] = (KV_cache_demands[-1] * (infer_args.max_seq_num+1)) / byte_per_gpu


                param_byte_per_layer, extra_byte = \
                    get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)


                for wld_degree in [2]:


                    param_byte_per_comp_gpu = float('inf')
                    if (model.model_path, num_worker) in peak_mem_dict:
                        param_byte_per_comp_gpu = peak_mem_dict[(model.model_path, num_worker)]


                    tot_mem_demand += param_byte_per_comp_gpu


            mem_per_comp_gpu = 0.9

            if mem_per_comp_gpu * byte_per_gpu < max(KV_cache_demands) + tot_mem_demand:

                return False, tot_mem_demand
            else:
                return True, tot_mem_demand


        def _get_min_alloc(placement, tot_gpu_num, fully_connected_gpu_unit, byte_per_gpu, cost_table, sufficient_gpu_mem_ratio_dict, min_alloc_cache):


            import math
            min_allocs = list()
            if len(placement) > tot_gpu_num:
                return None
            for model_ids in placement:
                min_alloc = None


                if tuple(model_ids) in min_alloc_cache:
                    min_alloc = min_alloc_cache[tuple(model_ids)]
                else:

                    modelset = [self.model_dict[model_id] for model_id in model_ids]

                    for i in range(int(math.log(tot_gpu_num, 2)+1)):
                        num_worker = 2**i


                        if num_worker > fully_connected_gpu_unit:
                            break

                        if _GPU_mem_is_enough(
                            modelset, [num_worker]*len(modelset),
                            tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict)[0]:
                            min_alloc  = num_worker
                            break
                        else:
                            continue

                    min_alloc_cache[tuple(model_ids)] = min_alloc

                if min_alloc == None:

                    return None
                else:
                    min_allocs.append(min_alloc)

            if sum(min_allocs) > tot_gpu_num:
                return None

            return min_allocs


        def enum_allocs(tot_gpu_num, min_allocs):

            def backtrack(current_assignment, remaining_machines, additional_allocs_list, modelset_num, ori_remaining_gpu_num):

                if len(current_assignment) == modelset_num:

                    if sum(current_assignment) <= ori_remaining_gpu_num:
                        additional_allocs_list.append(current_assignment.copy())
                    return


                max_machines_for_current = remaining_machines

                for machines in range(max_machines_for_current + 1):
                    current_assignment.append(machines)
                    backtrack(current_assignment, remaining_machines - machines, additional_allocs_list, modelset_num, ori_remaining_gpu_num)
                    current_assignment.pop()

            additional_allocs_list = []
            remaining_gpu_num = tot_gpu_num - sum(min_allocs)
            modelset_num = len(min_allocs)
            backtrack([], remaining_gpu_num, additional_allocs_list, modelset_num, remaining_gpu_num)

            return additional_allocs_list


        def _get_modelset_exec_plan_groups(
            modelset, min_alloc, additional_alloc, exec_plans_dict,
            tot_gpu_num, byte_per_gpu, cost_table,
            uniq_model_plans_dict, model_plan_meta_data, sufficient_gpu_mem_ratio_dict,
        ) -> List[List[MyExecPlan]]:
            import itertools
            import math


            if modelset[0] in model_plan_meta_data:
                if tuple(modelset) in model_plan_meta_data[modelset[0]]:
                    if (min_alloc + additional_alloc) in model_plan_meta_data[modelset[0]][tuple(modelset)]:
                        return model_plan_meta_data[modelset[0]][tuple(modelset)][min_alloc + additional_alloc]


            for model_id in modelset:
                if model_id not in model_plan_meta_data:
                    model_plan_meta_data[model_id] = dict()
                if tuple(modelset) not in model_plan_meta_data[model_id]:
                    model_plan_meta_data[model_id][tuple(modelset)] = dict()
                if (min_alloc + additional_alloc) not in model_plan_meta_data[model_id][tuple(modelset)]:
                    model_plan_meta_data[model_id][tuple(modelset)][min_alloc + additional_alloc] = list()


            valid_exec_plans_list = list()
            for model_id in modelset:
                valid_exec_plans = list()
                for plan in exec_plans_dict[model_id]:
                    if (get_tot_worker_num([plan]) == (min_alloc + additional_alloc)):

                        valid_exec_plans.append(plan)
                valid_exec_plans_list.append(valid_exec_plans)


            cartesian_product = list(itertools.product(*valid_exec_plans_list))

            models = [self.model_dict[model_id] for model_id in modelset]

            cand_modelset_plans = list()
            for plan_group in cartesian_product:
                num_workers = [plan.num_worker for plan in plan_group]
                is_enough, tot_param_byte_per_comp_gpu = _GPU_mem_is_enough(
                    models, num_workers,
                    tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict)

                if is_enough:

                    new_plan_group = list()
                    for plan in plan_group:

                        new_mem_per_comp_gpu = (plan.mem_per_comp_gpu * byte_per_gpu - tot_param_byte_per_comp_gpu + plan.param_byte_per_comp_gpu) / byte_per_gpu


                        new_mem_per_comp_gpu = math.ceil(new_mem_per_comp_gpu * 100) / 100


                        key = (plan.model.model_id, plan.get_key(), new_mem_per_comp_gpu)
                        if key not in uniq_model_plans_dict:

                            new_plan = MyExecPlan(plan.model,
                                    plan.num_worker, plan.wld_degree, plan.cache_gpu_num, new_mem_per_comp_gpu, plan.dp_size,
                                    plan.param_byte_per_comp_gpu, plan.param_byte_per_cache_gpu,
                                    plan.gpu_cache_byte_per_block, plan.infer_args, tot_gpu_mem_byte=byte_per_gpu)

                            if isinstance(plan.model, MyFusedModelInfor):
                                new_plan = MyVerticalFusedExecPlan(plan.model, new_plan)

                            uniq_model_plans_dict[key] = new_plan
                        else:
                            new_plan = uniq_model_plans_dict[key]

                        new_plan_group.append(new_plan)


                        model_plan_meta_data[plan.model.model_id][tuple(modelset)][min_alloc + additional_alloc].append(new_plan)

                    cand_modelset_plans.append(new_plan_group)

            return cand_modelset_plans


        def _auto_parallel_given_model_ids(model_plan_meta_data, best_model_plans_dict, model_ids) -> None:
            """
                为model_ids中的model_id在uniq_model_plans_dict的所有分类中选择最佳plan，并相应地存到best_moedl_plans_dict中。
            """
            for model_id in model_ids:
                best_model_plans_dict[model_id] = dict()
                for modelset in model_plan_meta_data[model_id]:
                    best_model_plans_dict[model_id][modelset] = dict()
                    for gpu_num, plans in model_plan_meta_data[model_id][modelset].items():
                        if len(plans) > 0:
                            best_plan = min(plans, key=lambda plan: max(plan.total_latency_list) + plan.extra_cost)
                            best_model_plans_dict[model_id][modelset][gpu_num] = (best_plan, max(best_plan.total_latency_list) + best_plan.extra_cost)


        def _get_stage1_start_time(stage_policy, model_order_meta, best_model_plans_dict, placement, min_allocs, additional_allocs) -> float:
            if stage_policy == 'model_parallel':
                return 0
            target_labels = None
            if stage_policy == 'summary_first':
                target_labels = [-1]
            else:

                target_labels = [-1, 0]

            finish_times = list()
            for modelset, min_alloc, additional_alloc in zip(placement, min_allocs, additional_allocs):
                tot_latency = 0
                for model_id in modelset:
                    if model_order_meta[model_id] in target_labels:

                        _, latency = best_model_plans_dict[model_id][modelset][min_alloc + additional_alloc]
                        tot_latency += latency
                finish_times.append(tot_latency)
            return max(finish_times)


        def _get_uniq_level1_models_and_start_times(possible_mapping_info, model_plan_meta_data, best_model_plans_dict, stage_policy, cost_table):
            """
                Collect level-1 model plans and their start times for each device mapping.
            """

            def _get_model_start_time(model_in_set_info, model_id, best_model_plans_dict, placement, min_allocs, additional_allocs):
                modelset_i, pre_model_ids = model_in_set_info[model_id]
                start_time = 0
                for pre_model_id in pre_model_ids:
                    _, latency = best_model_plans_dict[pre_model_id][placement[modelset_i]][min_allocs[modelset_i] + additional_allocs[modelset_i]]
                    start_time += latency
                return start_time


            def _get_stage1_model_start_time(model_in_set_info, model_id, best_model_plans_dict, placement, min_allocs, additional_allocs, stage_policy, stage1_start_time, model_order_meta):

                target_labels = None
                if stage_policy == 'model_parallel':
                    target_labels = []
                elif stage_policy == 'summary_first':
                    target_labels = [-1]
                else:

                    target_labels = [-1, 0]

                modelset_i, pre_model_ids = model_in_set_info[model_id]
                start_time = stage1_start_time

                pre_tot_latency = 0
                reach_first_stage1_model = False
                for pre_model_id in pre_model_ids:
                    plan, latency = best_model_plans_dict[pre_model_id][placement[modelset_i]][min_allocs[modelset_i] + additional_allocs[modelset_i]]
                    if model_order_meta[pre_model_id] in target_labels:
                        pre_tot_latency += latency
                        continue
                    if not reach_first_stage1_model:
                        reach_first_stage1_model = True

                        start_time = max(plan.extra_cost + pre_tot_latency, start_time) - plan.extra_cost

                    start_time += latency

                return start_time, reach_first_stage1_model, pre_tot_latency


            model_ids = self.all_level_model_ids[1]
            model_order_meta = np.asarray([0]*(max(self.model_dict)+1))
            for model_id in model_ids:
                model_order_meta[self.model_dict[model_id].input_model_ids] = -1

            model_order_meta[model_ids] = 1


            uniq_model_chains = dict()
            model_chain_meta_dict = dict()

            for placement_i, (placement, min_allocs, additional_allocs_list) in enumerate(possible_mapping_info):
                model_in_set_info = dict()
                model_chain_meta_dict[placement_i] = dict()
                for modelset_i, modelset in enumerate(placement):


                    if (model_order_meta[np.asarray(modelset)] != 0).any():

                        sorted_modelset = sorted(modelset, key=lambda model_id: (model_order_meta[model_id], model_id))
                        for order, model_id in enumerate(sorted_modelset):
                            model_in_set_info[model_id] = (modelset_i, sorted_modelset[:order])


                for additional_allocs_i, additional_allocs in enumerate(additional_allocs_list):


                    model_chain_meta = dict()
                    for model_id in model_ids:
                        model_chain_meta[model_id] = list()


                        model_chain = list()
                        start_times = list()
                        for inp_model_id in self.model_dict[model_id].input_model_ids:
                            modelset_i, pre_model_ids = model_in_set_info[inp_model_id]
                            model_chain.append(best_model_plans_dict[inp_model_id][placement[modelset_i]][min_allocs[modelset_i] + additional_allocs[modelset_i]][0])

                            start_time = _get_model_start_time(model_in_set_info, inp_model_id, best_model_plans_dict, placement, min_allocs, additional_allocs)
                            start_times.append(start_time)


                        stage1_start_time = _get_stage1_start_time(stage_policy, model_order_meta, best_model_plans_dict, placement, min_allocs, additional_allocs)


                        start_time, reach_first_stage1_model, pre_tot_latency = _get_stage1_model_start_time(model_in_set_info, model_id, best_model_plans_dict, placement, min_allocs, additional_allocs, stage_policy, stage1_start_time, model_order_meta)


                        additional_extra_time = start_time - max(start_times)


                        modelset_i, pre_model_ids = model_in_set_info[model_id]
                        cand_plans = model_plan_meta_data[model_id][placement[modelset_i]][min_allocs[modelset_i] + additional_allocs[modelset_i]]
                        for plan in cand_plans:


                            if not reach_first_stage1_model:

                                extra_cost = cost_table.get_prepare_cost(plan.model.model_name, plan.get_key_single_dp_worker())
                                additional_extra_time = max(extra_cost + pre_tot_latency, start_time) - extra_cost - max(start_times)


                            key = (model_chain[0].get_key(), plan.get_key(), additional_extra_time)
                            if key in uniq_model_chains:
                                model_chain_meta[model_id].append( uniq_model_chains[key][-1] )
                                continue


                            new_plan = plan.copy_the_plan()
                            new_plan.additional_extra_time = additional_extra_time

                            uniq_model_chains[ key ] = model_chain + [new_plan]
                            model_chain_meta[model_id].append( new_plan )

                    model_chain_meta_dict[placement_i][additional_allocs_i] = model_chain_meta

            return uniq_model_chains, model_chain_meta_dict


        def _get_best_device_mapping(possible_mapping_info, best_model_plans_dict, model_chain_meta_dict, stage_policy):
            """
                计算每个device mapping方案对应的最佳cost。
            """

            def _get_modelset_time(
                modelset_i, sorted_modelset,
                best_model_plans_dict, model_chain_meta_dict,
                placement_i, placement, min_allocs, additional_allocs_i, additional_allocs,
                stage_policy, stage1_start_time, model_order_meta, plan_dict):

                target_labels = None
                if stage_policy == 'model_parallel':
                    target_labels = []
                elif stage_policy == 'summary_first':
                    target_labels = [-1]
                else:

                    target_labels = [-1, 0]

                start_time = stage1_start_time

                pre_tot_latency = 0
                reach_first_stage1_model = False
                for model_id in sorted_modelset:
                    best_plan, best_latency = None, None

                    if model_id in best_model_plans_dict:

                        best_plan, best_latency = best_model_plans_dict[model_id][placement[modelset_i]][min_allocs[modelset_i] + additional_allocs[modelset_i]]

                        plan_dict[model_id] = best_plan
                    else:

                        best_plan = min(model_chain_meta_dict[placement_i][additional_allocs_i][model_id], key=lambda plan: max(plan.total_latency_list) + plan.extra_cost)
                        best_latency = max(best_plan.total_latency_list) + best_plan.extra_cost

                        plan_dict[model_id] = best_plan


                    if model_order_meta[model_id] in target_labels:
                        pre_tot_latency += best_latency
                        continue
                    if not reach_first_stage1_model:
                        reach_first_stage1_model = True

                        start_time = max(best_plan.extra_cost + pre_tot_latency, start_time) - best_plan.extra_cost

                    start_time += best_latency

                if not reach_first_stage1_model:
                    start_time = pre_tot_latency

                return start_time


            best_cost = float('inf')
            best_mapping = None


            model_ids = []
            if len(self.all_level_model_ids) > 1:
                model_ids = self.all_level_model_ids[1]
            model_order_meta = np.asarray([0]*(max(self.model_dict)+1))
            for model_id in model_ids:
                model_order_meta[np.asarray(self.model_dict[model_id].input_model_ids)] = -1

            model_order_meta[model_ids] = 1


            for placement_i, (placement, min_allocs, additional_allocs_list) in enumerate(possible_mapping_info):
                for additional_allocs_i, additional_allocs in enumerate(additional_allocs_list):


                    stage1_start_time = _get_stage1_start_time(stage_policy, model_order_meta, best_model_plans_dict, placement, min_allocs, additional_allocs)

                    modelset_latencys = list()
                    plan_dict = dict()
                    for modelset_i, modelset in enumerate(placement):


                        sorted_modelset = sorted(modelset, key=lambda model_id: (model_order_meta[model_id], model_id))

                        modelset_latency = _get_modelset_time(
                            modelset_i, sorted_modelset,
                            best_model_plans_dict, model_chain_meta_dict,
                            placement_i, placement, min_allocs, additional_allocs_i, additional_allocs,
                            stage_policy, stage1_start_time, model_order_meta, plan_dict)


                        modelset_latencys.append(modelset_latency)
                    placement_latency = max(modelset_latencys)
                    if placement_latency < best_cost:
                        best_cost = placement_latency
                        best_mapping = [placement, np.asarray(min_allocs)+np.asarray(additional_allocs), plan_dict]


            sorted_placement = list()
            for modelset in best_mapping[0]:
                sorted_modelset = sorted(modelset, key=lambda model_id: (model_order_meta[model_id], model_id))
                sorted_placement.append(sorted_modelset)


            plan_dict = best_mapping[-1]
            stage_num = max([len(modelset) for modelset in sorted_placement])
            plan_group_seq = list()
            for stage_i in range(stage_num):

                plan_group = [plan_dict[modelset[stage_i]] if (stage_i < len(modelset)) else plan_dict[modelset[-1]] for modelset in sorted_placement]
                plan_group = MyDummyExecPlanGroup(plan_group)
                plan_group_seq.append(plan_group)


            best_mapping[0] = sorted_placement
            best_mapping[-1] = plan_group_seq
            return best_cost, best_mapping


        def get_best_device_mapping(stage_policy:str='model_parallel'):
            """
                Find the best device mapping.
                Input:
                    stage_policy: specify whether split the dependent model chains to different stages.
                                'model_parallel': allow model level pipeline parallelism like our method.
                                'summary_first': for chain summary & mixed application, put the summary model in the first stage alone.
                                'evaluator_last': for chain summary & mixed application, put the evaluator model in the last stage alone.
            """

            model_ids = list(self.model_dict.keys())
            sufficient_gpu_mem_ratio_dict = dict()
            all_placements = _get_placements(model_ids, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict, tot_gpu_num)
            best_cost = float('inf')
            best_mapping = None


            exec_plans_dict = dict()
            for model_id, model in self.model_dict.items():
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
                exec_plans_dict[model_id] = exec_plans


            possible_mapping_info = list()
            uniq_model_plans_dict = dict()
            model_plan_meta_data = dict()

            min_alloc_cache = dict()


            import time
            begin_enu = time.perf_counter()

            for placement in all_placements:
                min_allocs = _get_min_alloc(placement, tot_gpu_num, fully_connected_gpu_unit, byte_per_gpu, cost_table, sufficient_gpu_mem_ratio_dict, min_alloc_cache)
                if min_allocs == None:

                    continue

                additional_allocs_list = enum_allocs(tot_gpu_num, min_allocs)


                valid_additional_allocs_list = list()

                for additional_allocs in additional_allocs_list:

                    success = True
                    for model_set_i, modelset in enumerate(placement):

                        cand_modelset_plans = _get_modelset_exec_plan_groups(
                            modelset, min_allocs[model_set_i], additional_allocs[model_set_i], exec_plans_dict,
                            tot_gpu_num, byte_per_gpu, cost_table,
                            uniq_model_plans_dict, model_plan_meta_data, sufficient_gpu_mem_ratio_dict)
                        if len(cand_modelset_plans) == 0:

                            success = False
                    if success:
                        valid_additional_allocs_list.append(additional_allocs)


                possible_mapping_info.append([placement, min_allocs, valid_additional_allocs_list])


            end_enu = time.perf_counter()
            global Plan_Group_Enu_Time, Plan_Group_Enu_num
            Plan_Group_Enu_Time += (end_enu - begin_enu)
            Plan_Group_Enu_num += (sum([len(vs[2]) for vs in possible_mapping_info]))


            print(f"len(possible_mapping_info): {len(possible_mapping_info)}, all additional allocs number: {sum([len(vs[2]) for vs in possible_mapping_info])}")


            level0_model_ids = self.all_level_model_ids[0]


            plans_to_query_cost = list(uniq_model_plans_dict.values())


            print(f"len(plans_to_query_cost): {len(plans_to_query_cost)}")
            if len(plans_to_query_cost) > 0:
                print(f"sufficient_gpu_mem_ratio_dict: {sufficient_gpu_mem_ratio_dict}, infer_args.max_seq_num: {plans_to_query_cost[0].infer_args.max_seq_num}")


            begin_query = time.perf_counter()

            batch_num = (len(plans_to_query_cost) + 1000 - 1) // 1000

            for batch_i in range(batch_num):
                uniq_plan_group_objs = [MyExecPlanGroup(
                    [plan], cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for plan in plans_to_query_cost[batch_i*1000:(batch_i+1)*1000]]
                for exec_plan in plans_to_query_cost[batch_i*1000:(batch_i+1)*1000]:

                    exec_plan._wait_for_remote_fake_scheduling()


            end_query = time.perf_counter()
            global Cost_Query_Time
            Cost_Query_Time += (end_query - begin_query)


            best_model_plans_dict = dict()
            _auto_parallel_given_model_ids(model_plan_meta_data, best_model_plans_dict, self.all_level_model_ids[0])

            print(f"best_model_plans_dict: {best_model_plans_dict}")


            model_chain_meta_dict = None
            if len(self.all_level_model_ids) > 1:

                uniq_model_chains, model_chain_meta_dict = _get_uniq_level1_models_and_start_times(possible_mapping_info, model_plan_meta_data, best_model_plans_dict, stage_policy, cost_table)


                uniq_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for plan_group in uniq_model_chains.values()]
                for plan_group in uniq_model_chains.values():

                    plan_group[-1]._wait_for_remote_fake_scheduling()


            best_mapping = _get_best_device_mapping(possible_mapping_info, best_model_plans_dict, model_chain_meta_dict, stage_policy)
            return best_mapping


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")

        stage_policy = search_method_baseline[len('hybridflow_'):]
        return get_best_device_mapping(stage_policy=stage_policy)


    def get_candidate_plan_groups_HybridFlow(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
        """


        def _get_placements(collection, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict, tot_gpu_num):
            if len(collection) == 1:
                yield [tuple(collection)]
                return
            first = collection[0]
            for smaller in _get_placements(collection[1:], fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict, tot_gpu_num):

                for n, subset in enumerate(smaller):


                    cand_modelset = tuple([first]) + subset
                    if not _GPU_mem_is_enough(
                            [self.model_dict[model_id] for model_id in cand_modelset], [fully_connected_gpu_unit]*len(cand_modelset),
                            tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict)[0]:
                        continue

                    yield smaller[:n] + [cand_modelset] + smaller[n+1:]


                if len(smaller) >= tot_gpu_num:
                    continue


                yield [tuple([first])] + smaller


        def _GPU_mem_is_enough(
                models: List[MyModelInfor], num_workers: List[int],
                tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict):
            '''
            check whether using tp=num_workers[i] for each model models[i], the gpu memory is large enough to hold the models together (colocation).
            Return: (1) whether the GPU memory is enough, (2) the total param_byte_per_comp_gpu per comp GPU
            '''


            KV_cache_demands = list()
            tot_mem_demands = list()
            for model, num_worker in zip(models, num_workers):

                if (model.model_path, num_worker) not in _ENGINE_ARGS_LIST:
                    _ENGINE_ARGS_LIST[(model.model_path, num_worker)] = get_engin_args(
                        model_path=model.model_path, tensor_parallel_size=num_worker)
                (model_config, cache_config, parallel_config, scheduler_config,
                    device_config, lora_config) = _ENGINE_ARGS_LIST[(model.model_path, num_worker)]

                infer_args = InferenceArgs(scheduler_config, cache_config)


                gpu_cache_byte_per_block = get_gpu_cache_byte_per_block(cache_config, model_config, parallel_config)
                KV_cache_demands.append(gpu_cache_byte_per_block * ((model_config.max_model_len + cache_config.block_size - 1) // cache_config.block_size))


                if (model.model_id, num_worker) not in sufficient_gpu_mem_ratio_dict:

                    sufficient_gpu_mem_ratio_dict[(model.model_id, num_worker)] = (KV_cache_demands[-1] * (infer_args.max_seq_num+1)) / byte_per_gpu


                param_byte_per_layer, extra_byte = \
                    get_per_layer_and_extra_param_and_buffer_byte(model, num_worker)


                for wld_degree in [2]:


                    param_byte_per_comp_gpu = float('inf')
                    if (model.model_path, num_worker) in peak_mem_dict:
                        param_byte_per_comp_gpu = peak_mem_dict[(model.model_path, num_worker)]


                    tot_mem_demands.append(param_byte_per_comp_gpu)


            mem_per_comp_gpu = 0.9

            tot_mem_demands = np.asarray(KV_cache_demands) + np.asarray(tot_mem_demands)

            if mem_per_comp_gpu * byte_per_gpu < max(tot_mem_demands):

                return False, max(tot_mem_demands)
            else:
                return True, max(tot_mem_demands)


        def _get_min_alloc(placement, tot_gpu_num, fully_connected_gpu_unit, byte_per_gpu, cost_table, sufficient_gpu_mem_ratio_dict, min_alloc_cache):


            import math
            min_allocs = list()
            if len(placement) > tot_gpu_num:
                return None
            for model_ids in placement:
                min_alloc = None


                if tuple(model_ids) in min_alloc_cache:
                    min_alloc = min_alloc_cache[tuple(model_ids)]
                else:

                    modelset = [self.model_dict[model_id] for model_id in model_ids]

                    for i in range(int(math.log(tot_gpu_num, 2)+1)):
                        num_worker = 2**i


                        if num_worker > fully_connected_gpu_unit:
                            break

                        if _GPU_mem_is_enough(
                            modelset, [num_worker]*len(modelset),
                            tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict)[0]:
                            min_alloc  = num_worker
                            break
                        else:
                            continue

                    min_alloc_cache[tuple(model_ids)] = min_alloc

                if min_alloc == None:

                    return None
                else:
                    min_allocs.append(min_alloc)

            if sum(min_allocs) > tot_gpu_num:
                return None

            return min_allocs


        def enum_allocs(tot_gpu_num, min_allocs):

            def backtrack(current_assignment, remaining_machines, additional_allocs_list, modelset_num, ori_remaining_gpu_num, min_allocs):

                if len(current_assignment) == modelset_num:

                    if sum(current_assignment) <= ori_remaining_gpu_num:
                        additional_allocs_list.append(current_assignment.copy())
                    return


                max_machines_for_current = remaining_machines

                for machines in range(max_machines_for_current + 1):
                    if (machines % min_allocs[len(current_assignment)]) != 0:

                        continue
                    current_assignment.append(machines)
                    backtrack(current_assignment, remaining_machines - machines, additional_allocs_list, modelset_num, ori_remaining_gpu_num, min_allocs)
                    current_assignment.pop()

            additional_allocs_list = []
            remaining_gpu_num = tot_gpu_num - sum(min_allocs)
            modelset_num = len(min_allocs)
            backtrack([], remaining_gpu_num, additional_allocs_list, modelset_num, remaining_gpu_num, min_allocs)

            return additional_allocs_list


        def _get_modelset_exec_plan_groups(
            modelset, min_alloc, additional_alloc, exec_plans_dict,
            tot_gpu_num, byte_per_gpu, cost_table,
            uniq_model_plans_dict, model_plan_meta_data, sufficient_gpu_mem_ratio_dict,
        ) -> List[List[MyExecPlan]]:
            import itertools
            import math


            if modelset[0] in model_plan_meta_data:
                if tuple(modelset) in model_plan_meta_data[modelset[0]]:
                    if (min_alloc + additional_alloc) in model_plan_meta_data[modelset[0]][tuple(modelset)]:
                        return model_plan_meta_data[modelset[0]][tuple(modelset)][min_alloc + additional_alloc]


            for model_id in modelset:
                if model_id not in model_plan_meta_data:
                    model_plan_meta_data[model_id] = dict()
                if tuple(modelset) not in model_plan_meta_data[model_id]:
                    model_plan_meta_data[model_id][tuple(modelset)] = dict()
                if (min_alloc + additional_alloc) not in model_plan_meta_data[model_id][tuple(modelset)]:
                    model_plan_meta_data[model_id][tuple(modelset)][min_alloc + additional_alloc] = list()


            valid_exec_plans_list = list()
            for model_id in modelset:
                valid_exec_plans = list()
                for plan in exec_plans_dict[model_id]:
                    if (get_tot_worker_num([plan]) == (min_alloc + additional_alloc)):

                        valid_exec_plans.append(plan)
                valid_exec_plans_list.append(valid_exec_plans)


            cartesian_product = list(itertools.product(*valid_exec_plans_list))

            models = [self.model_dict[model_id] for model_id in modelset]

            cand_modelset_plans = list()
            for plan_group in cartesian_product:
                num_workers = [plan.num_worker for plan in plan_group]
                is_enough, tot_param_byte_per_comp_gpu = _GPU_mem_is_enough(
                    models, num_workers,
                    tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict)

                if is_enough:

                    new_plan_group = list()
                    for plan in plan_group:


                        new_mem_per_comp_gpu = plan.mem_per_comp_gpu


                        key = (plan.model.model_id, plan.get_key(), new_mem_per_comp_gpu)
                        if key not in uniq_model_plans_dict:

                            new_plan = MyExecPlan(plan.model,
                                    plan.num_worker, plan.wld_degree, plan.cache_gpu_num, new_mem_per_comp_gpu, plan.dp_size,
                                    plan.param_byte_per_comp_gpu, plan.param_byte_per_cache_gpu,
                                    plan.gpu_cache_byte_per_block, plan.infer_args, tot_gpu_mem_byte=byte_per_gpu)

                            if isinstance(plan.model, MyFusedModelInfor):
                                new_plan = MyVerticalFusedExecPlan(plan.model, new_plan)

                            uniq_model_plans_dict[key] = new_plan
                        else:
                            new_plan = uniq_model_plans_dict[key]

                        new_plan_group.append(new_plan)


                        model_plan_meta_data[plan.model.model_id][tuple(modelset)][min_alloc + additional_alloc].append(new_plan)

                    cand_modelset_plans.append(new_plan_group)

            return cand_modelset_plans


        def _auto_parallel_given_model_ids(same_gpunum_plans, best_model_plans_dict, model_ids) -> None:
            """
                为model_ids中的model_id在uniq_model_plans_dict的所有分类中选择最佳plan，并相应地存到best_moedl_plans_dict中。
            """
            for model_id in model_ids:
                best_model_plans_dict[model_id] = dict()


                for gpu_num, plans in same_gpunum_plans[model_id].items():
                    if len(plans) > 0:
                        best_plan = min(plans, key=lambda plan: max(plan.total_latency_list) + plan.extra_cost)
                        best_model_plans_dict[model_id][gpu_num] = (best_plan, max(best_plan.total_latency_list) + best_plan.extra_cost)


        def _get_stage1_start_time(stage_policy, model_order_meta, best_model_plans_dict, placement, min_allocs, additional_allocs) -> float:
            if stage_policy == 'model_parallel':
                return 0
            target_labels = None
            if stage_policy == 'summary_first':
                target_labels = [-1]
            else:

                target_labels = [-1, 0]

            finish_times = list()
            for modelset, min_alloc, additional_alloc in zip(placement, min_allocs, additional_allocs):
                tot_latency = 0
                for model_id in modelset:
                    if model_order_meta[model_id] in target_labels:


                        _, latency = best_model_plans_dict[model_id][min_alloc + additional_alloc]
                        tot_latency += latency
                finish_times.append(tot_latency)
            return max(finish_times)


        def _get_uniq_level1_models_and_start_times(possible_mapping_info, same_gpunum_plans, best_model_plans_dict, stage_policy, cost_table, target_eval_model_id, model_chain_meta_dict):
            """
                Collect level-1 model plans and their start times for each device mapping.
            """

            def _get_model_start_time(model_in_set_info, model_id, best_model_plans_dict, placement, min_allocs, additional_allocs):
                modelset_i, pre_model_ids = model_in_set_info[model_id]
                start_time = 0
                for pre_model_id in pre_model_ids:

                    _, latency = best_model_plans_dict[pre_model_id][min_allocs[modelset_i] + additional_allocs[modelset_i]]
                    start_time += latency
                return start_time


            def _get_stage1_model_start_time(model_in_set_info, model_id, best_model_plans_dict, placement, min_allocs, additional_allocs, stage_policy, stage1_start_time, model_order_meta):

                target_labels = None
                if stage_policy == 'model_parallel':
                    target_labels = []
                elif stage_policy == 'summary_first':
                    target_labels = [-1]
                else:

                    target_labels = [-1, 0]

                modelset_i, pre_model_ids = model_in_set_info[model_id]
                start_time = stage1_start_time

                pre_tot_latency = 0
                reach_first_stage1_model = False
                for pre_model_id in pre_model_ids:

                    plan, latency = best_model_plans_dict[pre_model_id][min_allocs[modelset_i] + additional_allocs[modelset_i]]
                    if model_order_meta[pre_model_id] in target_labels:
                        pre_tot_latency += latency
                        continue
                    if not reach_first_stage1_model:
                        reach_first_stage1_model = True

                        start_time = max(plan.extra_cost + pre_tot_latency, start_time) - plan.extra_cost

                    start_time += latency

                return start_time, reach_first_stage1_model, pre_tot_latency


            model_ids = self.all_level_model_ids[1]
            model_order_meta = np.asarray([0]*(max(self.model_dict)+1))
            for model_id in model_ids:
                model_order_meta[self.model_dict[model_id].input_model_ids] = -1

            model_order_meta[model_ids] = 1


            uniq_model_chains = dict()


            for placement_i, (placement, min_allocs, additional_allocs_list) in enumerate(possible_mapping_info):
                model_in_set_info = dict()
                if placement_i not in model_chain_meta_dict:
                    model_chain_meta_dict[placement_i] = dict()
                for modelset_i, modelset in enumerate(placement):


                    if (model_order_meta[np.asarray(modelset)] != 0).any():

                        sorted_modelset = sorted(modelset, key=lambda model_id: (model_order_meta[model_id], model_id))
                        for order, model_id in enumerate(sorted_modelset):
                            model_in_set_info[model_id] = (modelset_i, sorted_modelset[:order])


                for additional_allocs_i, additional_allocs in enumerate(additional_allocs_list):


                    model_chain_meta = dict()

                    for model_id in [target_eval_model_id]:
                        model_chain_meta[model_id] = list()


                        model_chain = list()
                        start_times = list()
                        for inp_model_id in self.model_dict[model_id].input_model_ids:
                            modelset_i, pre_model_ids = model_in_set_info[inp_model_id]

                            model_chain.append(best_model_plans_dict[inp_model_id][min_allocs[modelset_i] + additional_allocs[modelset_i]][0])

                            start_time = _get_model_start_time(model_in_set_info, inp_model_id, best_model_plans_dict, placement, min_allocs, additional_allocs)
                            start_times.append(start_time)


                        stage1_start_time = _get_stage1_start_time(stage_policy, model_order_meta, best_model_plans_dict, placement, min_allocs, additional_allocs)


                        start_time, reach_first_stage1_model, pre_tot_latency = _get_stage1_model_start_time(model_in_set_info, model_id, best_model_plans_dict, placement, min_allocs, additional_allocs, stage_policy, stage1_start_time, model_order_meta)


                        additional_extra_time = start_time - max(start_times)


                        modelset_i, pre_model_ids = model_in_set_info[model_id]


                        cand_plans = same_gpunum_plans[model_id][min_allocs[modelset_i] + additional_allocs[modelset_i]]
                        for plan in cand_plans:


                            if not reach_first_stage1_model:

                                extra_cost = cost_table.get_prepare_cost(plan.model.model_name, plan.get_key_single_dp_worker())
                                additional_extra_time = max(extra_cost + pre_tot_latency, start_time) - extra_cost - max(start_times)


                            key = (model_chain[0].get_key(), plan.get_key(), additional_extra_time)
                            if key in uniq_model_chains:
                                model_chain_meta[model_id].append( uniq_model_chains[key][-1] )
                                continue


                            new_plan = plan.copy_the_plan()
                            new_plan.additional_extra_time = additional_extra_time

                            uniq_model_chains[ key ] = model_chain + [new_plan]
                            model_chain_meta[model_id].append( new_plan )


                    if additional_allocs_i not in model_chain_meta_dict[placement_i]:
                        model_chain_meta_dict[placement_i][additional_allocs_i] = dict()
                    model_chain_meta_dict[placement_i][additional_allocs_i].update(model_chain_meta)

            return uniq_model_chains, model_chain_meta_dict


        def _get_best_device_mapping(possible_mapping_info, best_model_plans_dict, model_chain_meta_dict, stage_policy):
            """
                计算每个device mapping方案对应的最佳cost。
            """

            def _get_modelset_time(
                modelset_i, sorted_modelset,
                best_model_plans_dict, model_chain_meta_dict,
                placement_i, placement, min_allocs, additional_allocs_i, additional_allocs,
                stage_policy, stage1_start_time, model_order_meta, plan_dict):

                target_labels = None
                if stage_policy == 'model_parallel':
                    target_labels = []
                elif stage_policy == 'summary_first':
                    target_labels = [-1]
                else:

                    target_labels = [-1, 0]

                start_time = stage1_start_time

                pre_tot_latency = 0
                reach_first_stage1_model = False
                for model_id in sorted_modelset:
                    best_plan, best_latency = None, None

                    if model_id in best_model_plans_dict:


                        best_plan, best_latency = best_model_plans_dict[model_id][min_allocs[modelset_i] + additional_allocs[modelset_i]]

                        plan_dict[model_id] = best_plan
                    else:

                        best_plan = min(model_chain_meta_dict[placement_i][additional_allocs_i][model_id], key=lambda plan: max(plan.total_latency_list) + plan.extra_cost)
                        best_latency = max(best_plan.total_latency_list) + best_plan.extra_cost

                        plan_dict[model_id] = best_plan


                    if model_order_meta[model_id] in target_labels:
                        pre_tot_latency += best_latency
                        continue
                    if not reach_first_stage1_model:
                        reach_first_stage1_model = True

                        start_time = max(best_plan.extra_cost + pre_tot_latency, start_time) - best_plan.extra_cost

                    start_time += best_latency

                if not reach_first_stage1_model:
                    start_time = pre_tot_latency

                return start_time


            best_cost = float('inf')
            best_mapping = None


            model_ids = []
            if len(self.all_level_model_ids) > 1:
                model_ids = self.all_level_model_ids[1]
            model_order_meta = np.asarray([0]*(max(self.model_dict)+1))
            for model_id in model_ids:
                model_order_meta[np.asarray(self.model_dict[model_id].input_model_ids)] = -1

            model_order_meta[model_ids] = 1


            for placement_i, (placement, min_allocs, additional_allocs_list) in enumerate(possible_mapping_info):
                for additional_allocs_i, additional_allocs in enumerate(additional_allocs_list):


                    stage1_start_time = _get_stage1_start_time(stage_policy, model_order_meta, best_model_plans_dict, placement, min_allocs, additional_allocs)

                    modelset_latencys = list()
                    plan_dict = dict()
                    for modelset_i, modelset in enumerate(placement):


                        sorted_modelset = sorted(modelset, key=lambda model_id: (model_order_meta[model_id], model_id))

                        modelset_latency = _get_modelset_time(
                            modelset_i, sorted_modelset,
                            best_model_plans_dict, model_chain_meta_dict,
                            placement_i, placement, min_allocs, additional_allocs_i, additional_allocs,
                            stage_policy, stage1_start_time, model_order_meta, plan_dict)


                        modelset_latencys.append(modelset_latency)
                    placement_latency = max(modelset_latencys)
                    if placement_latency < best_cost:
                        best_cost = placement_latency
                        best_mapping = [placement, np.asarray(min_allocs)+np.asarray(additional_allocs), plan_dict]


            sorted_placement = list()
            for modelset in best_mapping[0]:
                sorted_modelset = sorted(modelset, key=lambda model_id: (model_order_meta[model_id], model_id))
                sorted_placement.append(sorted_modelset)


            plan_dict = best_mapping[-1]
            stage_num = max([len(modelset) for modelset in sorted_placement])
            plan_group_seq = list()
            for stage_i in range(stage_num):

                plan_group = [plan_dict[modelset[stage_i]] if (stage_i < len(modelset)) else plan_dict[modelset[-1]] for modelset in sorted_placement]
                plan_group = MyDummyExecPlanGroup(plan_group)
                plan_group_seq.append(plan_group)


            best_mapping[0] = sorted_placement
            best_mapping[-1] = plan_group_seq
            return best_cost, best_mapping


        def get_best_device_mapping(stage_policy:str='model_parallel'):
            """
                Find the best device mapping.
                Input:
                    stage_policy: specify whether split the dependent model chains to different stages.
                                'model_parallel': allow model level pipeline parallelism like our method.
                                'summary_first': for chain summary & mixed application, put the summary model in the first stage alone.
                                'evaluator_last': for chain summary & mixed application, put the evaluator model in the last stage alone.
            """

            model_ids = list(self.model_dict.keys())
            sufficient_gpu_mem_ratio_dict = dict()
            all_placements = _get_placements(model_ids, fully_connected_gpu_unit, sufficient_gpu_mem_ratio_dict, tot_gpu_num)
            best_cost = float('inf')
            best_mapping = None


            exec_plans_dict = dict()
            for model_id, model in self.model_dict.items():
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
                exec_plans_dict[model_id] = exec_plans


            same_gpunum_plans = dict()
            for model_id, exec_plans in exec_plans_dict.items():
                same_gpunum_plans[model_id] = dict()
                for exec_plan in exec_plans:
                    gpu_num = get_tot_worker_num([exec_plan])
                    if gpu_num not in same_gpunum_plans[model_id]:
                        same_gpunum_plans[model_id][gpu_num] = list()
                    same_gpunum_plans[model_id][gpu_num].append(exec_plan)


            possible_mapping_info = list()
            uniq_model_plans_dict = dict()
            model_plan_meta_data = dict()

            min_alloc_cache = dict()


            import time
            begin_enu = time.perf_counter()

            for placement in all_placements:
                min_allocs = _get_min_alloc(placement, tot_gpu_num, fully_connected_gpu_unit, byte_per_gpu, cost_table, sufficient_gpu_mem_ratio_dict, min_alloc_cache)
                if min_allocs == None:

                    continue

                additional_allocs_list = enum_allocs(tot_gpu_num, min_allocs)


                valid_additional_allocs_list = additional_allocs_list


                possible_mapping_info.append([placement, min_allocs, valid_additional_allocs_list])


            end_enu = time.perf_counter()
            global Plan_Group_Enu_Time, Plan_Group_Enu_num
            Plan_Group_Enu_Time += (end_enu - begin_enu)
            Plan_Group_Enu_num += (sum([len(vs[2]) for vs in possible_mapping_info]))


            print(f"len(possible_mapping_info): {len(possible_mapping_info)}, all additional allocs number: {sum([len(vs[2]) for vs in possible_mapping_info])}", flush=True)


            level0_model_ids = self.all_level_model_ids[0]


            plans_to_query_cost = list(itertools.chain(*exec_plans_dict.values()))


            print(f"len(plans_to_query_cost): {len(plans_to_query_cost)}", flush=True)
            if len(plans_to_query_cost) > 0:
                print(f"sufficient_gpu_mem_ratio_dict: {sufficient_gpu_mem_ratio_dict}, infer_args.max_seq_num: {plans_to_query_cost[0].infer_args.max_seq_num}")


            begin_query = time.perf_counter()

            batch_num = (len(plans_to_query_cost) + 1000 - 1) // 1000

            for batch_i in range(batch_num):
                uniq_plan_group_objs = [MyExecPlanGroup(
                    [plan], cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for plan in plans_to_query_cost[batch_i*1000:(batch_i+1)*1000]]
                for exec_plan in plans_to_query_cost[batch_i*1000:(batch_i+1)*1000]:

                    exec_plan._wait_for_remote_fake_scheduling()


            end_query = time.perf_counter()
            global Cost_Query_Time
            Cost_Query_Time += (end_query - begin_query)


            best_model_plans_dict = dict()

            _auto_parallel_given_model_ids(same_gpunum_plans, best_model_plans_dict, self.all_level_model_ids[0])

            print(f"best_model_plans_dict: {best_model_plans_dict}")


            model_chain_meta_dict = dict()
            if len(self.all_level_model_ids) > 1:


                for target_eval_model_id in sorted(self.all_level_model_ids[1]):
                    print(f"target_eval_model_id: {target_eval_model_id}")


                    uniq_model_chains, model_chain_meta_dict = _get_uniq_level1_models_and_start_times(possible_mapping_info, same_gpunum_plans, best_model_plans_dict, stage_policy, cost_table, target_eval_model_id, model_chain_meta_dict)


                    uniq_plan_group_objs = [MyExecPlanGroup(
                        plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                        ) for plan_group in uniq_model_chains.values()]
                    for plan_group in uniq_model_chains.values():

                        plan_group[-1]._wait_for_remote_fake_scheduling()


            best_mapping = _get_best_device_mapping(possible_mapping_info, best_model_plans_dict, model_chain_meta_dict, stage_policy)
            return best_mapping


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")

        stage_policy = search_method_baseline[len('hybridflow_'):]
        return get_best_device_mapping(stage_policy=stage_policy)


    def get_candidate_plan_groups_saturn_LP(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
        """


        def init_presolved(model_id2_order, task_list, tot_gpu_num):
            """
                给LP模型整理一个initial presolved solution 避免LP solver很难找到合法解。
                需要准备： interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan
                sta = [[[t.value() for t in g] for g in n]
                    for n in start_time_array]

                tga = [[[g.value() for g in n] for n in t]
                    for t in task_gpu_occupancy_array]

                bss = [[s.value() for s in t] for t in binary_strategy_selection_array]

                bna = [[n.value() for n in t] for t in binary_node_selection_array]

                boa = [[t_prime.value() for t_prime in t] for t in before_or_after]
            """


            sorted_model_ids = sorted(self.model_dict.keys(), key=lambda model_id: len(self.model_dict[model_id].input_model_ids))
            model_num = len(sorted_model_ids)


            tga = [[[1 for g in range(tot_gpu_num)] for n in range(1)]
                    for t in range(model_num)]
            bna = [[1 for n in range(1)] for t in range(model_num)]

            boa = [[0 for t_prime in range(model_num)] for t in range(model_num)]
            for i, t1 in enumerate(sorted_model_ids):
                for t2 in sorted_model_ids[i+1:]:
                    boa[t1][t2] = 1

            sta = [[[None for t in range(model_num)] for g in range(tot_gpu_num)]
                for n in range(1)]
            sta = [None]*model_num
            start = 0
            bss = [[] for t in range(model_num)]
            for model_id in sorted_model_ids:

                task_idx = model_id2_order[model_id]
                strats = task_list[task_idx]
                for strat_idx, strat in enumerate(strats):
                    if strat[0] == tot_gpu_num:

                        sta[task_idx] = start
                        start = int(start + strat[1] + 1)
                        bss[task_idx] = [0]*len(strats)
                        bss[task_idx][strat_idx] = 1
                        break
            sta = [[sta for g in range(tot_gpu_num)]
                for n in range(1)]

            return sta, tga, bss, bna, boa, None


        def _get_presolved(model_id2_order, task_list):
            '''
            sta = [[[t.value() for t in g] for g in n]
                for n in start_time_array]

            tga = [[[g.value() for g in n] for n in t]
                for t in task_gpu_occupancy_array]

            bss = [[s.value() for s in t] for t in binary_strategy_selection_array]

            bna = [[n.value() for n in t] for t in binary_node_selection_array]

            boa = [[t_prime.value() for t_prime in t] for t in before_or_after]
            '''

            global _Presolved_LP

            if _Presolved_LP != None:
                interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan, last_model_id2_order, last_task_list = _Presolved_LP


                new2old_task_idx = dict()
                for model_id in model_id2_order:
                    new2old_task_idx[model_id2_order[model_id]] = last_model_id2_order[model_id]
                old_task_idxs_needed = [new2old_task_idx[t1] for t1 in sorted(new2old_task_idx)]
                interval_sta = [[[g[idx] for idx in old_task_idxs_needed] for g in n] for n in interval_sta]
                interval_tga = [interval_tga[idx] for idx in old_task_idxs_needed]
                interval_bss = [interval_bss[idx] for idx in old_task_idxs_needed]
                for idx, t in enumerate(interval_bss):
                    old_strat_idx_mapping = {(strat[-1].get_key(), strat[-2]): s_idx for s_idx, strat in enumerate(last_task_list[new2old_task_idx[idx]])}
                    new_t = [t[old_strat_idx_mapping[(strat[-1].get_key(), strat[-2])]] if (strat[-1].get_key(), strat[-2]) in old_strat_idx_mapping else 0 for strat in task_list[idx]]
                    interval_bss[idx] = new_t
                return interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan
            else:
                return None


        def _get_complete_non_preemptive_schedule(each_gpu_remaining_model_info, plan_group_list, task_list, selection, model_id2_order, tot_gpu_num):
            """
            这个方法以每个model的最晚完成时间为分割点，检查每个分割点正在运行的model，以及每个分割点可以启动的model（如果当前该model的gpu上没有别的正在运行的model或者完全没有model）
            selection存储的内容格式为： selection[task_idx] = [strat_idx, selected_gpus, max(start_times), task_list[task_idx][strat_idx]]
            """

            start_time_dict = {task_idx: selection[task_idx][2] for task_idx in selection}
            plan_dict = {task_idx: selection[task_idx][-1][-1] for task_idx in selection}
            end_time_dict = {task_idx: (start_time_dict[task_idx] + max(plan.total_latency_list) + plan.extra_cost) for task_idx, plan in plan_dict.items()}

            print(f"start_time_dict: {start_time_dict}")
            print(f"end_time_dict: {end_time_dict}")
            print(f"each_gpu_remaining_model_info: {each_gpu_remaining_model_info}")


            for model in self.model_dict.values():
                if len(model.input_model_ids) > 0:
                    inp_task_idx = model_id2_order[model.input_model_ids[0]]
                    end_time_dict[model_id2_order[model.model_id]] = max(end_time_dict[model_id2_order[model.model_id]], end_time_dict[inp_task_idx])


            print(f"updated end_time_dict: {end_time_dict}")


            running_plan_group = plan_group_list[-1]
            running_task_idxs = [model_id2_order[plan.model.model_id] for plan in running_plan_group]


            min_end_time = 0
            if len(running_task_idxs) > 0:
                min_end_time = min([end_time_dict[task_idx] for task_idx in running_task_idxs])


            print(f"running_task_idxs: {running_task_idxs}, min_end_time: {min_end_time}")


            gpu_is_busy = {gpu_idx: False for gpu_idx in range(tot_gpu_num)}
            for task_idx in running_task_idxs:
                selected_gpus = selection[task_idx][1]
                if end_time_dict[task_idx] > min_end_time:

                    for gpu_idx in selected_gpus:
                        gpu_is_busy[gpu_idx] = True
                else:

                    for gpu_idx in selected_gpus:
                        gpu_is_busy[gpu_idx] = False


            print(f"gpu_is_busy: {gpu_is_busy}")


            alive_task_idxs = list()
            for task_idx in end_time_dict:
                if (start_time_dict[task_idx] <= min_end_time) and (end_time_dict[task_idx] > min_end_time):

                    alive_task_idxs.append(task_idx)
                    print(f"task_idx: {task_idx} is running")
                elif end_time_dict[task_idx] <= min_end_time:

                    print(f"task_idx: {task_idx} is finished")
                    continue
                else:


                    if task_idx in running_task_idxs:
                        alive_task_idxs.append(task_idx)
                        print(f"task_idx: {task_idx} has been started in advance")
                        continue

                    selected_gpus = selection[task_idx][1]
                    runnable = True
                    for gpu_idx in selected_gpus:

                        if each_gpu_remaining_model_info[gpu_idx][0] != task_idx:
                            runnable = False
                            print(f"task_idx: {task_idx} is not the first waiting model")
                            break

                        if gpu_is_busy[gpu_idx]:
                            runnable = False
                            print(f"task_idx: {task_idx} gpu_idx {gpu_idx} is busy")
                            break
                    if runnable:
                        alive_task_idxs.append(task_idx)
                        print(f"task_idx: {task_idx} can be started in advance")


            print(f"alive_task_idxs: {alive_task_idxs}\n")


            plan_group = [plan_dict[task_idx] for task_idx in alive_task_idxs]
            for task_idx in alive_task_idxs:
                selected_gpus = selection[task_idx][1]
                for gpu_idx in selected_gpus:
                    if (len(each_gpu_remaining_model_info[gpu_idx]) > 0) and (each_gpu_remaining_model_info[gpu_idx][0] == task_idx):

                        each_gpu_remaining_model_info[gpu_idx] = each_gpu_remaining_model_info[gpu_idx][1:]

            plan_group_list.append(plan_group)


        def get_complete_non_preemptive_schedule(gpu_model_order, task_list, selection, model_id2_order, tot_gpu_num):
            each_gpu_remaining_model_info = dict()
            for gpu_idx, gpu in enumerate(gpu_model_order):
                if len(gpu) == 0:
                    each_gpu_remaining_model_info[gpu_idx] = list()
                else:
                    sorted_task_infos = sorted(gpu, key=lambda task_info: task_info[1])
                    each_gpu_remaining_model_info[gpu_idx] = [task_info[0] for task_info in sorted_task_infos]

            plan_group_list = [[]]
            while True:
                _get_complete_non_preemptive_schedule(each_gpu_remaining_model_info, plan_group_list, task_list, selection, model_id2_order, tot_gpu_num)
                if len(plan_group_list[-1]) == 0:
                    break

            best_plan_group_seq = list()
            for plan_group in plan_group_list:
                if len(plan_group) > 0:
                    best_plan_group_seq.append(MyDummyExecPlanGroup(plan_group))
            return best_plan_group_seq


        def get_best_device_mapping():
            """
                Find the best device mapping.
                Input:
                    stage_policy: specify whether split the dependent model chains to different stages.
                                'model_parallel': allow model level pipeline parallelism like our method.
                                'summary_first': for chain summary & mixed application, put the summary model in the first stage alone.
                                'evaluator_last': for chain summary & mixed application, put the evaluator model in the last stage alone.
            """

            global _Presolved_LP


            exec_plans_dict = dict()
            for model_ids in self.all_level_model_ids:
                for model_id in model_ids:
                    model = self.model_dict[model_id]
                    exec_plans = get_possible_exec_plans(
                        model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
                    exec_plans_dict[model_id] = exec_plans


            uniq_plan_group_objs = [MyExecPlanGroup(
                [plan], cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status={k:True for k in base_model_finish_status}
                ) for plans in exec_plans_dict.values() for plan in plans]

            for plans in exec_plans_dict.values():
                for exec_plan in plans:

                    exec_plan._wait_for_remote_fake_scheduling()


            model_id2_order = dict()
            model_order2_id = dict()
            order = 0
            for model_id in exec_plans_dict:
                model_id2_order[model_id] = order
                model_order2_id[order] = [model_id]
                order += 1


            print(f"model_id2_order: {model_id2_order}")
            print(f"exec_plans_dict: {exec_plans_dict}")


            model_chains = list()
            for model_id in exec_plans_dict:


                if (len(self.model_dict[model_id].input_model_ids) > 0) and (self.model_dict[model_id].input_model_ids[0] in model_id2_order):
                    chain = list(self.model_dict[model_id].input_model_ids) + [model_id]
                    chain = [model_id2_order[i] for i in chain]
                    model_chains.append(chain)


            task_list = list()
            for model_id, exec_plans in exec_plans_dict.items():
                strategies = list()
                for plan in exec_plans:

                    strat = [get_tot_worker_num([plan]), max(plan.total_latency_list) + plan.extra_cost, plan.extra_cost > 0, plan]
                    strategies.append(strat)
                    if plan.extra_cost == 0:

                        extra_cost = cost_table.get_prepare_cost(plan.model.model_name, plan.get_key_single_dp_worker())
                        strat = [get_tot_worker_num([plan]), max(plan.total_latency_list) + extra_cost, extra_cost > 0, plan]
                        strategies.append(strat)
                task_list.append(strategies)


            import saturn_lp

            timeout=5*60
            task_list_for_lp = [[strat[:-1]+[None] for strat in strategies] for strategies in task_list]


            presolved = _get_presolved(model_id2_order, task_list)
            interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan = saturn_lp.solve(model_chains, task_list_for_lp, presolved=presolved, gurobi=True, threads=os.cpu_count() // 1, interval=1000, timeout=timeout)


            print(f"model_chains: {model_chains}")
            print(f"task_list_for_lp: {task_list_for_lp}")
            print(f"interval_sta: {interval_sta}")
            print(f"interval_tga: {interval_tga}")
            print(f"interval_bss: {interval_bss}")
            print(f"interval_bna: {interval_bna}")
            print(f"interval_boa: {interval_boa}")
            print(f"saved_makespan: {saved_makespan}")


            _Presolved_LP = interval_sta, interval_tga, interval_bss, interval_bna, interval_boa, saved_makespan, model_id2_order, task_list

            assert interval_bna[0][0]!=None


            selection = dict()
            gpu_model_order = [[] for i in range(tot_gpu_num)]
            for task_idx, task_res in enumerate(interval_tga):
                start_times = list()
                selected_gpus = list()
                for node_idx, node in enumerate(task_res):
                    for gpu_idx, gpu_selected in enumerate(node):
                        if gpu_selected > 0.5:
                            start_times.append(interval_sta[node_idx][gpu_idx][task_idx])
                            gpu_model_order[gpu_idx].append((task_idx, start_times[-1]))
                            selected_gpus.append(gpu_idx)
                print(f"start_times: {start_times}")
                if not (np.asarray(start_times) == start_times[0]).all():
                    print(f"Wrong LP result!, start_times: {start_times}")
                for strat_idx, strat_selected in enumerate(interval_bss[task_idx]):
                    if strat_selected > 0.5:
                        selection[task_idx] = [strat_idx, selected_gpus, max(start_times), task_list[task_idx][strat_idx]]
                        break

            to_print = {task_idx: (select_info[1], select_info[2], select_info[3][-1].get_key()) for task_idx, select_info in selection.items()}
            print(f"selection: {to_print}")


            if 'no_preemption' in search_method_baseline:
                return get_complete_non_preemptive_schedule(gpu_model_order, task_list, selection, model_id2_order, tot_gpu_num)


            plan_group = list()

            each_gpu_first_model = dict()
            for gpu_idx, gpu in enumerate(gpu_model_order):
                if len(gpu) == 0:
                    continue
                task_idx, start_time = sorted(gpu, key=lambda task_info: task_info[1])[0]


                each_gpu_first_model[gpu_idx] = task_idx


            for task_idx in selection:
                selected_gpus = selection[task_idx][1]
                if (np.asarray([each_gpu_first_model[gpu_idx] for gpu_idx in selected_gpus]) == task_idx).all():

                    strat_idx = selection[task_idx][0]
                    plan = task_list[task_idx][strat_idx][-1]
                    new_plan = plan.copy_the_plan()
                    plan_group.append(new_plan)


            new_plan_group = list()
            plan_group_model_ids = [plan.model.model_id for plan in plan_group]
            for plan in plan_group:
                to_add = True
                for model_id in plan.model.input_model_ids:
                    if (model_id not in plan_group_model_ids) and (not self.model_dict[model_id].is_finished()):
                        to_add = False
                        break
                if to_add:
                   new_plan_group.append(plan)
            plan_group = new_plan_group


            level0_plans = list()
            for plan in plan_group:
                if len(plan.model.input_model_ids) == 0:
                    level0_plans.append(plan)
            if len(level0_plans) > 0:
                uniq_plan_group_obj = MyExecPlanGroup(
                    level0_plans, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    )
            for exec_plan in level0_plans:

                exec_plan._wait_for_remote_fake_scheduling()


            plan_groups = [plan_group]
            plan_groups = [MyExecPlanGroup(
                plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
            for plan_group in plan_groups:
                plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)

            return plan_groups


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")

        return get_best_device_mapping()


    def get_candidate_plan_groups_LightRule(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
        """

        def _get_throughput(token_per_secs_dict, plan):

            N, tp = get_tot_worker_num([plan]), plan.num_worker

            return token_per_secs_dict[(plan.model.model_name, tp, tp)][1] * (N/tp)


        def _get_throughput_wrong(token_per_secs_dict, plan, plan_dict, avg_outlen_dict):


            N, tp = get_tot_worker_num([plan]), plan.num_worker

            output_gen_speed = token_per_secs_dict[(plan.model.model_name, tp, tp)][1] * (N/tp)


            if len(plan.model.input_model_ids) == 0:
                return output_gen_speed

            inp_plan = plan_dict[plan.model.input_model_ids[0]]
            if inp_plan.model.model_id not in plan_dict:
                return output_gen_speed

            N, tp = get_tot_worker_num([inp_plan]), inp_plan.num_worker
            input_gen_speed = token_per_secs_dict[(inp_plan.model.model_name, tp, tp)][1] * (N/tp)
            inp_model_avg_outlen = avg_outlen_dict[inp_plan.model.model_id]
            avg_outlen = avg_outlen_dict[plan.model.model_id]
            output_demand_speed = input_gen_speed / inp_model_avg_outlen * avg_outlen
            return min(output_demand_speed, output_gen_speed)


        def _comp_single_model_cost(
            token_per_secs_dict, model_id, plan_dict, last_stage_plan_dict, workload_dict,
            cost_dict, extra_cost_dict
        ):
            """
                考虑model dependency导致的计算速度其实可能并不是模型原本可以达到的计算速度。
            """
            plan = plan_dict[model_id]
            extra_cost = 0
            if (model_id not in last_stage_plan_dict) or (last_stage_plan_dict[model_id]!=plan):
                extra_cost = cost_table.get_prepare_cost(plan.model.model_name, plan.get_key_single_dp_worker())
            extra_cost_dict[model_id] = extra_cost
            cost_dict[model_id] = workload_dict[model_id] / _get_throughput(token_per_secs_dict, plan) + extra_cost

            if len(plan.model.input_model_ids) == 0:
                return


            inp_model_id = plan.model.input_model_ids[0]
            if inp_model_id not in plan_dict:
                return


            inp_model_finish_time = cost_dict[inp_model_id]
            if cost_dict[model_id] <= inp_model_finish_time:

                cost_dict[model_id] = inp_model_finish_time


        def comp_model_costs(plan_dict, last_stage_plan_dict, workload_dict, token_per_secs_dict):
            cost_dict = dict()
            extra_cost_dict = dict()


            sorted_model_ids = sorted(plan_dict.keys(), key=lambda model_id: len(self.model_dict[model_id].input_model_ids))


            for model_id in sorted_model_ids:


                _comp_single_model_cost(
                    token_per_secs_dict, model_id, plan_dict, last_stage_plan_dict, workload_dict,
                    cost_dict, extra_cost_dict
                )

            return cost_dict, extra_cost_dict


        def _update_single_model_workload(
            model_id, min_cost, cost_dict, extra_cost_dict, workload_dict, token_per_secs_dict, plan_dict, last_stage_plan_dict, avg_outlen_dict
        ):
            """
                考虑model dependency导致的计算速度其实可能并不是模型原本可以达到的计算速度。
            """
            plan = plan_dict[model_id]
            if min_cost > extra_cost_dict[model_id]:
                workload_dict[model_id] = workload_dict[model_id] - (min_cost - extra_cost_dict[model_id]) * _get_throughput(token_per_secs_dict, plan)
                workload_dict[model_id] = max(workload_dict[model_id], 0)

                if cost_dict[model_id] == min_cost:
                    workload_dict[model_id] = 0

                if (len(plan.model.input_model_ids) == 0) or (plan.model.input_model_ids[0] not in plan_dict):

                    return
                else:


                    inp_model_id = plan.model.input_model_ids[0]

                    workload_not_ready = workload_dict[inp_model_id] / avg_outlen_dict[inp_model_id] * avg_outlen_dict[model_id]
                    if workload_dict[model_id] <= workload_not_ready:

                        workload_dict[model_id] = workload_not_ready


        def update_workload_dict(workload_dict, token_per_secs_dict, plan_dict, last_stage_plan_dict, avg_outlen_dict):


            cost_dict, extra_cost_dict = comp_model_costs(plan_dict, last_stage_plan_dict, workload_dict, token_per_secs_dict)


            min_cost = min(cost_dict.values())
            print(f"update_workload_dict cost_dict: {cost_dict}")
            print(f"update_workload_dict extra_cost_dict: {extra_cost_dict}")
            print(f"update_workload_dict min_cost: {min_cost}")


            sorted_model_ids = sorted(plan_dict.keys(), key=lambda model_id: len(self.model_dict[model_id].input_model_ids))

            for model_id in sorted_model_ids:
                _update_single_model_workload(
                    model_id, min_cost, cost_dict, extra_cost_dict, workload_dict, token_per_secs_dict, plan_dict, last_stage_plan_dict, avg_outlen_dict
                )


        def _select_best_plan_to_decrease_max_latency(
            plan_group, model_costs, exec_plans_dict, remaining_gpu_num, allocation, last_stage_plan_dict, workload_dict, token_per_secs_dict
        ):
            """
                这个算法的逻辑：选择能让当前stage的最大完成时间减少的exec plan，如果有多个plan满足条件，则选择GPU数最小的plan，如果仍然有多个plan满足条件，则选择平均每GPU的下降量最大的。
                返回True 如果能成功找到plan来降低所有model的最大完成时间，否则返回False。
            """

            ori_model_costs = model_costs
            print(f"\nori_model_costs: {ori_model_costs}")

            max_cost = max(model_costs.values())


            model_ids_to_check = [model_id for model_id in model_costs if model_costs[model_id] == max_cost]
            print(f"max cost model ids: {model_ids_to_check}")
            related_model_ids = list()
            for model_id in model_ids_to_check:

                related_model_ids.extend(self.model_dict[model_id].input_model_ids)

                for model in self.model_dict.values():
                    if model_id in model.input_model_ids:
                        related_model_ids.append(model.model_id)
            model_ids_to_check.extend(related_model_ids)
            model_ids_to_check = set(model_ids_to_check)

            model_ids_to_check = [model_id for model_id in model_ids_to_check if workload_dict[model_id] > 0]
            print(f"model_ids_to_check: {model_ids_to_check}")


            plan_costs = dict()
            for model_id in model_ids_to_check:
                max_gpu_num = remaining_gpu_num + allocation[model_id][0]
                possible_plans = [plan for plan in exec_plans_dict[model_id] \
                    if (get_tot_worker_num([plan]) > allocation[model_id][0]) and (get_tot_worker_num([plan]) <= max_gpu_num) ]

                original_plan = plan_group[model_id]
                for plan in possible_plans:
                    plan_group[model_id] = plan
                    model_costs, _ = comp_model_costs(plan_group, last_stage_plan_dict, workload_dict, token_per_secs_dict)
                    plan_costs[plan] = (max(model_costs.values()), model_costs[model_id])
                plan_group[model_id] = original_plan


            print(f"all plans costs: {[(plan.model.model_id, plan.get_key(), c) for plan, c in plan_costs.items()]}")


            new_plan = None
            better_plans = [plan for plan in plan_costs if plan_costs[plan][0] < max_cost]
            if len(better_plans) == 0:


                better_plans = [plan for plan in plan_costs if plan_costs[plan][1] < ori_model_costs[plan.model.model_id]]
                if len(better_plans) == 0:

                    return False, None
                better_plans = sorted(better_plans, key=lambda plan: (get_tot_worker_num([plan]) - allocation[plan.model.model_id][0], -(ori_model_costs[plan.model.model_id]-plan_costs[plan][1])))

                print(f"NO SINGLE PLAN CAN REDUCE MAX COST, TRY TO REDUCE MODEL'S LATENCY \n")


                new_plan = better_plans[0]
            else:


                better_plans = sorted(better_plans, key=lambda plan: ((max_cost - plan_costs[plan][0])/(get_tot_worker_num([plan]) - allocation[plan.model.model_id][0]), get_tot_worker_num([plan]) - allocation[plan.model.model_id][0] ), reverse=True)

                print(f"FIND A BETTER PLAN \n")

                new_plan = better_plans[0]


            plan_group[new_plan.model.model_id] = new_plan
            model_costs, _ = comp_model_costs(plan_group, last_stage_plan_dict, workload_dict, token_per_secs_dict)
            remaining_gpu_num = remaining_gpu_num + allocation[new_plan.model.model_id][0] - get_tot_worker_num([new_plan])
            allocation[new_plan.model.model_id] = (get_tot_worker_num([new_plan]), new_plan.num_worker)

            return True, (plan_group, model_costs, remaining_gpu_num, allocation)


        def _preemptive_scheduling(last_stage_plan_dict, workload_dict, waiting_models, exec_plans_dict, token_per_secs_dict, min_gpu_num_dict, avg_outlen_dict):
            """
                根据上一阶段的plan group，剩余未schedule的model，各个任务剩余的workload，选择这一阶段的plan group。
            """


            tot_gpu_num_demand = sum([ min_gpu_num_dict[self.model_dict[model_id].model_name][0] for model_id in workload_dict if workload_dict[model_id] > 0])

            if tot_gpu_num_demand >= tot_gpu_num:


                remaining_models = list()
                plan_group = {model_id: plan for model_id, plan in last_stage_plan_dict.items() if workload_dict[model_id] > 0 }
                gpu_num = sum([get_tot_worker_num([plan]) for plan in plan_group.values()])
                for model in waiting_models:

                    for model_id in model.input_model_ids:
                        if model_id not in plan_group:
                            continue

                    selected_plan = sorted(exec_plans_dict[model.model_id], key=lambda plan: get_tot_worker_num([plan]))[0]
                    if gpu_num + get_tot_worker_num([selected_plan]) <= tot_gpu_num:
                        plan_group[model.model_id] = (selected_plan)
                        gpu_num = gpu_num + get_tot_worker_num([selected_plan])
                    else:
                        remaining_models.append(model)
                        continue

                print(f"plan_group: {[(model_id, plan.get_key()) for model_id, plan in plan_group.items()]}")
                update_workload_dict(workload_dict, token_per_secs_dict, plan_group, last_stage_plan_dict, avg_outlen_dict)
                return plan_group, remaining_models

            else:

                plan_group = {model_id: plan for model_id, plan in last_stage_plan_dict.items() if workload_dict[model_id] > 0 }
                allocation = {model_id:(get_tot_worker_num([plan]), plan.num_worker) for model_id, plan in plan_group.items()}
                gpu_num = tot_gpu_num - sum([gpu_num for gpu_num, _ in allocation.values()])
                if len(last_stage_plan_dict) == 0:

                    allocation = {model_id:min_gpu_num_dict[model.model_name] for model_id, model in self.model_dict.items()}
                    gpu_num = tot_gpu_num - sum([gpu_num for gpu_num, _ in allocation.values()])
                    for model_id, exec_plans in exec_plans_dict.items():
                        plan_group[model_id] = [plan for plan in exec_plans if (get_tot_worker_num([plan]), plan.num_worker) == allocation[model_id]][0]

                model_costs, _ = comp_model_costs(plan_group, last_stage_plan_dict, workload_dict, token_per_secs_dict)


                plan_extra_costs = dict()
                for model_id in model_costs:
                    plan_extra_costs.update({plan: (cost_table.get_prepare_cost(plan.model.model_name, plan.get_key_single_dp_worker()) \
                                                        if ((model_id not in last_stage_plan_dict) or (plan != last_stage_plan_dict[model_id])) \
                                                        else 0\
                                                    ) for plan in exec_plans_dict[model_id]})

                print(f"allocation: {allocation}")

                while gpu_num > 0:

                    success, new_state = _select_best_plan_to_decrease_max_latency(
                        plan_group, model_costs, exec_plans_dict, gpu_num, allocation, last_stage_plan_dict, workload_dict, token_per_secs_dict
                    )


                    if not success:

                        print(f"ALL MODELS FAILED TO HAVE A BETTER PLAN\n")
                        break
                    else:
                        plan_group, model_costs, gpu_num, allocation = new_state


                print(f"plan_group: {[(model_id, plan.get_key()) for model_id, plan in plan_group.items()]}")
                update_workload_dict(workload_dict, token_per_secs_dict, plan_group, last_stage_plan_dict, avg_outlen_dict)

                return plan_group, waiting_models


        def preemptive_schedule(workload_dict, waiting_models, exec_plans_dict, token_per_secs_dict, min_gpu_num_dict, avg_outlen_dict):
            last_stage_plan_dict = dict()
            plan_group_list = list()
            while max(workload_dict.values()) > 0:

                print(f"workload_dict: {workload_dict}", flush=True)
                plan_group, waiting_models = _preemptive_scheduling(last_stage_plan_dict, workload_dict, waiting_models, exec_plans_dict, token_per_secs_dict, min_gpu_num_dict, avg_outlen_dict)
                plan_group_list.append(MyDummyExecPlanGroup(plan_group.values()))
                last_stage_plan_dict = plan_group
            return plan_group_list


        def non_preemptive_schedule(workload_dict, exec_plans_dict, token_per_secs_dict, sorted_models_list, min_gpu_num_dict):


            min_tot_gpu_demand = sum([min_gpu_num_dict[model.model_name][0] for model in self.model_dict.values()])


            if min_tot_gpu_demand >= tot_gpu_num:


                plan_group_list = list()
                for level_i, sorted_models in enumerate(sorted_models_list):
                    models_to_check = sorted_models
                    while True:
                        gpu_num = 0
                        remaining_models = list()
                        plan_group = list()
                        for model in models_to_check:
                            selected_plan = sorted(exec_plans_dict[model.model_id], key=lambda plan: get_tot_worker_num([plan]))[0]
                            if gpu_num + get_tot_worker_num([selected_plan]) <= tot_gpu_num:
                                plan_group.append(selected_plan)
                                gpu_num = gpu_num + get_tot_worker_num([selected_plan])
                            else:
                                remaining_models.append(model)
                                continue
                        plan_group_list.append(plan_group)
                        models_to_check = remaining_models
                        if len(models_to_check) == 0:
                            break

                    if level_i == 0:
                        for plan_group in plan_group_list[2:]:
                            plan_group_list[1] = plan_group_list[1].extend(plan_group)
                        plan_group_list = plan_group_list[:2]

                plan_group_seq = [MyDummyExecPlanGroup(plan_group) for plan_group in plan_group_list]
                return plan_group_seq
            else:

                allocation = {model_id:min_gpu_num_dict[model.model_name] for model_id, model in self.model_dict.items()}
                gpu_num = tot_gpu_num - sum([gpu_num for gpu_num, _ in allocation.values()])


                plan_group = dict()
                for model_id, exec_plans in exec_plans_dict.items():
                    plan_group[model_id] = [plan for plan in exec_plans if (get_tot_worker_num([plan]), plan.num_worker) == allocation[model_id]][0]

                model_costs, _ = comp_model_costs(plan_group, dict(), workload_dict, token_per_secs_dict)
                while gpu_num > 0:

                    success, new_state = _select_best_plan_to_decrease_max_latency(
                        plan_group, model_costs, exec_plans_dict, gpu_num, allocation, dict(), workload_dict, token_per_secs_dict
                    )


                    if not success:

                        break
                    else:
                        plan_group, model_costs, gpu_num, allocation = new_state

                plan_group_seq = [MyDummyExecPlanGroup(plan_group.values())]
                return plan_group_seq


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")


        exec_plans_dict = dict()
        for model in self.model_dict.values():
            exec_plans = get_possible_exec_plans(
                model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)
            exec_plans_dict[model.model_id] = exec_plans


        avg_outlens_dict={'Llama-2-70b-chat-hf': 401.2403923948438, 'Mixtral-8x7B-Instruct-v0.1': 325.5201172667553, 'WizardLM-13B-V1.2': 340.9273346056888, 'CodeLlama-34b-Instruct-hf': 302.73938106799994, 'vicuna-13b-v1.5': 207.9670307443602, 'Mistral-7B-Instruct-v0.2': 533.8308049504117, 'oasst-sft-4-pythia-12b-epoch-3.5': 705.3162484956827, 'alpaca-13b': 266.8701405682041, 'baize-v2-13b': 1623.3929272967703, 'koala-13B-HF': 225.58350931773063, 'dolly-v2-12b': 1716.3403786009605, 'mpt-7b-chat': 449.972644200943, 'chatglm3-6b': 205.10255479022274, 'stablelm-tuned-alpha-7b': 789.427071985797}


        new_avg_outlens_dict = dict()
        for model in self.model_dict.values():

            outlens = model.get_inp_out_seqlens()[1]
            if isinstance(model, MyFusedModelInfor):
                outlens = outlens[0]
            assert (np.asarray(outlens) == outlens[0]).all(), f"Wrong output sampler! np.asarray(outlens): {np.asarray(outlens)}"
            new_avg_outlens_dict[model.model_id] = outlens[0]
        avg_outlens_dict = new_avg_outlens_dict


        token_per_secs_dict = {('Llama-2-70b-chat-hf', 2, 2): (595.9931687140544, 1235.253351431285), ('Mixtral-8x7B-Instruct-v0.1', 2, 2): (2559.3048069074266, 3819.206623180357), ('WizardLM-13B-V1.2', 1, 1): (1673.232649404706, 2751.0772892532855), ('WizardLM-13B-V1.2', 2, 2): (2084.776973024239, 4138.184383703365), ('CodeLlama-34b-Instruct-hf', 1, 1): (746.346169370902, 1571.9112590596335), ('CodeLlama-34b-Instruct-hf', 2, 2): (1653.3652655836356, 2729.882927916327), ('vicuna-13b-v1.5', 1, 1): (1905.9135872427203, 3170.798786737927), ('vicuna-13b-v1.5', 2, 2): (2883.4503765427344, 4444.338965808569), ('Mistral-7B-Instruct-v0.2', 1, 1): (482.47911950293536, 4276.864786439816), ('Mistral-7B-Instruct-v0.2', 2, 2): (576.7413513555978, 5279.408925577375), ('oasst-sft-4-pythia-12b-epoch-3.5', 1, 1): (1313.046359718209, 3812.303960918927), ('oasst-sft-4-pythia-12b-epoch-3.5', 2, 2): (2273.3657472524237, 4160.999603238219), ('alpaca-13b', 1, 1): (1616.4765969524747, 2696.6362832506293), ('alpaca-13b', 2, 2): (2513.7964247514865, 4534.108172987039), ('baize-v2-13b', 1, 1): (1001.6609234946691, 2862.37158955556), ('baize-v2-13b', 2, 2): (1795.8825019371268, 3994.7681215057532), ('koala-13B-HF', 1, 1): (1642.3956482117685, 3042.0269338847784), ('koala-13B-HF', 2, 2): (2372.6042185545193, 4523.656888614792), ('dolly-v2-12b', 1, 1): (1092.043138533935, 3307.079023032004), ('dolly-v2-12b', 2, 2): (1922.536183804213, 4239.770575908282), ('mpt-7b-chat', 1, 1): (2713.9114252105705, 4845.3604927518945), ('mpt-7b-chat', 2, 2): (3619.9402204202506, 5710.853801138684), ('chatglm3-6b', 1, 1): (3580.7919879216374, 5825.0825433811815), ('chatglm3-6b', 2, 2): (5158.406650637974, 6907.910132009481), ('stablelm-tuned-alpha-7b', 1, 1): (1768.1366465944936, 4999.847168223587), ('stablelm-tuned-alpha-7b', 2, 2): (2649.589762622431, 5120.828078955421)}

        min_gpu_num_dict = {'Llama-2-70b-chat-hf': (2, 2), 'Mixtral-8x7B-Instruct-v0.1': (2, 2), 'WizardLM-13B-V1.2': (1, 1), 'CodeLlama-34b-Instruct-hf': (1, 1), 'vicuna-13b-v1.5': (1, 1), 'Mistral-7B-Instruct-v0.2': (1, 1), 'oasst-sft-4-pythia-12b-epoch-3.5': (1, 1), 'alpaca-13b': (1, 1), 'baize-v2-13b': (1, 1), 'koala-13B-HF': (1, 1), 'dolly-v2-12b': (1, 1), 'mpt-7b-chat': (1, 1), 'chatglm3-6b': (1, 1), 'stablelm-tuned-alpha-7b': (1, 1)}

        min_gpu_costs = dict()
        workload_dict = dict()

        for model_id, model in self.model_dict.items():
            min_gpu, tp_size = min_gpu_num_dict[model.model_name]
            inp_req_num = model.ori_tot_inp_num
            if isinstance(model, MyFusedModelInfor):
                inp_req_num = sum(inp_req_num)
            workload_dict[model_id] = avg_outlens_dict[model_id] * inp_req_num
            min_gpu_plan = [plan for plan in exec_plans_dict[model_id] if (get_tot_worker_num([plan]), plan.num_worker) == (min_gpu, tp_size)][0]
            min_gpu_cost = workload_dict[model_id] / _get_throughput(token_per_secs_dict, min_gpu_plan) + cost_table.get_prepare_cost(model.model_name, min_gpu_plan.get_key_single_dp_worker())
            print(f"_get_throughput(token_per_secs_dict, min_gpu_plan): {_get_throughput(token_per_secs_dict, min_gpu_plan)}")
            min_gpu_costs[model_id] = min_gpu_cost


        print(f"workload_dict: {workload_dict}")
        print(f"min_gpu_costs: {min_gpu_costs}", flush=True)
        print(f"avg_outlens_dict: {avg_outlens_dict}")
        print(f"model_names: {[(model_id, model.model_name) for model_id, model in self.model_dict.items()]}")


        sorted_models_list = list()
        waiting_models = list()
        level_i = 0
        for level_i in range(len(self.all_level_model_ids)):
            models = self.get_models_at_given_level(level_i)
            sorted_models_list.append(sorted(models, key=lambda model: min_gpu_costs[model.model_id], reverse=True))
            waiting_models.extend(sorted_models_list[-1])


        if 'no_preemption' in search_method_baseline:
            return non_preemptive_schedule(workload_dict, exec_plans_dict, token_per_secs_dict, sorted_models_list, min_gpu_num_dict)
        else:
            return preemptive_schedule(workload_dict, waiting_models, exec_plans_dict, token_per_secs_dict, min_gpu_num_dict, avg_outlens_dict)


    def get_candidate_plan_groups_LightRule_our_cost_model(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
        """

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


            for model_ids in model_ids_of_each_layer:


                plans_to_wait: List[MyExecPlan] = [plan for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]


                import time
                begin_query = time.perf_counter()


                uniq_sub_plan_group_objs = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status
                    ) for model_id in model_ids for plan, plan_group in new_uniq_sub_plan_groups[model_id]]
                for exec_plan in plans_to_wait:
                    if PRINT_LOGS:
                        print(f"waiting for exec plan: {(exec_plan.model.get_base_model_ids(), exec_plan.get_key())}")
                    exec_plan._wait_for_remote_fake_scheduling()


                end_query = time.perf_counter()
                global Cost_Query_Time
                Cost_Query_Time += (end_query-begin_query)


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


            model_min_gpu_dict = dict()

            for plan_group, exec_plan in plan_groups_to_check:
                if exec_plan.model.model_id in model_min_gpu_dict:
                    if get_tot_worker_num([exec_plan]) >= model_min_gpu_dict[exec_plan.model.model_id]:
                        continue
                model_min_gpu_dict[exec_plan.model.model_id] = get_tot_worker_num([exec_plan])

                for plan in plan_group:
                    if plan.model.model_id == exec_plan.model.model_id:
                        min_gpu_latency_dict[exec_plan.model.model_id] = max(plan.total_latency_list) + plan.extra_cost

            return min_gpu_latency_dict


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


        def _get_best_new_group(
                plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]], cand_plan_group: List[MyExecPlan]):
            """
                This function provides a fast method to estimate the throughput of the plan group.
                NOTE: this function assumes we have the fake scheduling results of all plans in the group.
            """


            original_gpu_nums = {plan.model.model_id: get_tot_worker_num([plan]) for plan in cand_plan_group}
            original_tot_gpu_num = sum(original_gpu_nums.values())
            original_max_cost = None
            original_latency_dict = None
            original_cand_group = None


            plan_group_costs = dict()
            for _, plan_group in plan_groups_to_check:

                plan_group = tuple(plan_group)


                assert False not in [exec_plan.total_latency_list[0]!=None for exec_plan in plan_group], [(exec_plan.model.get_base_model_ids(), exec_plan.get_key(), exec_plan.total_latency_list) for exec_plan in plan_group]

                latencys = {exec_plan.model.model_id: (max(exec_plan.total_latency_list) + exec_plan.extra_cost, get_tot_worker_num([exec_plan])) for exec_plan in plan_group}


                print(f"latencys: {latencys}")

                max_latency = max([latency for latency, gpu_num in latencys.values()])
                tot_gpu_num = sum([gpu_num for latency, gpu_num in latencys.values()])

                if tot_gpu_num > original_tot_gpu_num:


                    for model_id, (latency, gpu_num) in latencys.items():
                        if gpu_num > original_gpu_nums[model_id]:

                            plan_group_costs[plan_group] = (max_latency, tot_gpu_num, model_id, latency, gpu_num - original_gpu_nums[model_id])
                            print(f"new plan for model: {model_id}")
                            break
                    assert plan_group in plan_group_costs, f"plan_group {plan_group} not in plan_group_costs"
                else:
                    assert tot_gpu_num == original_tot_gpu_num, f"tot_gpu_num: {tot_gpu_num}, original_tot_gpu_num: {original_tot_gpu_num}, wrong candidate plan group: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group]}"
                    original_max_cost = max_latency
                    original_latency_dict = latencys
                    original_cand_group = plan_group


            new_plan_group = None
            better_groups = [(plan_group, plan_group_costs[plan_group]) for plan_group in plan_group_costs if plan_group_costs[plan_group][0] < original_max_cost]
            if len(better_groups) == 0:


                better_groups = [(plan_group, delta_gpu_num, -(original_latency_dict[model_id][0] - latency)) for plan_group, (max_latency, tot_gpu_num, model_id, latency, delta_gpu_num) in plan_group_costs.items() if latency < original_latency_dict[model_id][0]]
                if len(better_groups) == 0:

                    return False, original_cand_group

                better_groups = sorted(better_groups, key=lambda group: (group[1], group[2]))

                print(f"NO SINGLE PLAN CAN REDUCE MAX COST, TRY TO REDUCE MODEL'S LATENCY \n")


                new_plan_group = better_groups[0][0]
            else:


                better_groups = sorted(better_groups, key=lambda group: ((original_max_cost - group[1][0])/group[1][-1], group[1][-1] ), reverse=True)

                print(f"FIND A BETTER PLAN \n")

                new_plan_group = better_groups[0][0]


            return True, new_plan_group


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
                    1. Apply early validity checks to execution plans.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    all candidate execution plans must be checked individually.
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
                return False, cand_plan_group


            success, best_new_plan_group = _get_best_new_group(
                plan_groups_to_check, cand_plan_group)

            print(f"the best_plan_group we found a round: {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in best_new_plan_group]}")

            return success, best_new_plan_group


        def _preemptive_scheduling(exec_plans_dict, uniq_exec_plan_mapping,
        check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num):
            """
                根据上一阶段的plan group，剩余未schedule的model，各个任务剩余的workload，选择这一阶段的plan group。
                last_stage_plan_dict：直接从上一个iteration的搜索传入的input获得。
                waiting_models：根据第一次构建的顺序固定下来，在调用本函数之前就准备好。waiting models可以包含已经被schedule但是还没有结束的model。
            """


            tot_gpu_num_demand = sum([ min([get_tot_worker_num([plan]) for plan in exec_plans]) for exec_plans in exec_plans_dict.values()])


            cand_plan_group_dict = {last_plan.model.model_id: [plan for plan in exec_plans_dict[last_plan.model.model_id] if plan.get_key() == last_plan.get_key()][0] for last_plan in last_stage_exec_plans if not last_plan.model.is_finished()}


            print(f"cand_plan_group_dict from last stage: {[(model_id, plan.get_key()) for model_id, plan in cand_plan_group_dict.items()]}")

            if tot_gpu_num_demand >= tot_gpu_num:


                plan_group = cand_plan_group_dict
                gpu_num = sum([get_tot_worker_num([plan]) for plan in plan_group.values()])


                waiting_models = [self.model_dict[model_id] for model_id in exec_plans_dict if model_id not in cand_plan_group_dict]
                waiting_models = sorted(waiting_models, key=lambda model: model.check_order)
                assert waiting_models[-1].check_order < int(1e9)

                for model in waiting_models:

                    if model.model_id in plan_group:

                        continue


                    for model_id in model.input_model_ids:
                        if model_id not in plan_group:
                            continue

                    selected_plan = sorted(exec_plans_dict[model.model_id], key=lambda plan: get_tot_worker_num([plan]))[0]
                    if gpu_num + get_tot_worker_num([selected_plan]) <= tot_gpu_num:
                        plan_group[model.model_id] = (selected_plan)
                        gpu_num = gpu_num + get_tot_worker_num([selected_plan])
                    else:

                        continue

                print(f"plan_group: {[(model_id, plan.get_key()) for model_id, plan in plan_group.items()]}")


                return list(plan_group.values())

            else:


                for model_id in exec_plans_dict:
                    if model_id in cand_plan_group_dict:

                        continue


                    cand_plan_group_dict[model_id] = sorted(exec_plans_dict[model_id], key=lambda plan: get_tot_worker_num([plan]))[0]


                print(f"before greedy allocation, cand_plan_group_dict: {[(model_id, plan.get_key()) for model_id, plan in cand_plan_group_dict.items()]}")


                while get_tot_worker_num(list(cand_plan_group_dict.values())) < tot_gpu_num:


                    runnable_exec_plans_list = [[plan for plan in exec_plans if get_tot_worker_num([plan]) >= get_tot_worker_num([cand_plan_group_dict[model_id]])] for model_id, exec_plans in exec_plans_dict.items()]


                    success, best_plan_group = _get_best_runnable_exec_plan(
                        uniq_exec_plan_mapping,
                        list(cand_plan_group_dict.values()), runnable_exec_plans_list,

                        check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


                    if not success:

                        print(f"ALL MODELS FAILED TO HAVE A BETTER PLAN\n")
                        break


                    cand_plan_group_dict = {plan.model.model_id: plan for plan in best_plan_group}


                print(f"plan_group: {[(model_id, plan.get_key()) for model_id, plan in cand_plan_group_dict.items()]}")


                return list(cand_plan_group_dict.values())


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")


        exec_plans_dict = dict()

        for model_ids in self.all_level_model_ids:
            for model_id in model_ids:
                model = self.model_dict[model_id]
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)


                exec_plans_dict[model_id] = exec_plans


        print(f"exec_plans_dict: {[[(exec_plan.model.model_id, exec_plan.get_key()) for exec_plan in exec_plans] for exec_plans in exec_plans_dict.values()]}")


        best_plan_group = []


        global _MODEL_CHECK_ORDER
        if _MODEL_CHECK_ORDER == 0:


            min_gpu_exec_plans = [sorted(exec_plans, key=lambda plan: get_tot_worker_num([plan]))[:1] for exec_plans in exec_plans_dict.values()]

            min_gpu_latency_dict: Dict[int, float] = _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                best_plan_group,
                min_gpu_exec_plans,
                check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


            min_gpu_costs = min_gpu_latency_dict


            print(f"min_gpu_costs: {min_gpu_costs}", flush=True)

            print(f"model_names: {[(model_id, model.model_name) for model_id, model in self.model_dict.items()]}")


            sorted_models_list = list()
            waiting_models = list()
            level_i = 0
            for level_i in range(len(self.all_level_model_ids)):
                models = self.get_models_at_given_level(level_i)
                sorted_models_list.append(sorted(models, key=lambda model: min_gpu_costs[model.model_id], reverse=True))
                waiting_models.extend(sorted_models_list[-1])


            for model in waiting_models:
                assert model.check_order == int(1e9)
                model.check_order = _MODEL_CHECK_ORDER
                _MODEL_CHECK_ORDER += 1


        if 'no_preemption' in search_method_baseline:

            assert False, "we have not fully tested the non-preemption version, so we do not open this option for now"
            return non_preemptive_schedule(workload_dict, exec_plans_dict, token_per_secs_dict, sorted_models_list, min_gpu_num_dict)
        else:


            plan_group = _preemptive_scheduling(exec_plans_dict, uniq_exec_plan_mapping,
        check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


            old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
            plan_group = _get_correct_tmp_plan_group(
                uniq_exec_plan_mapping,
                plan_group[:-1],
                plan_group[-1],
            )


            _run_exec_cost_estimation_on_uniq_sub_plan_groups(
                uniq_exec_plan_mapping,
                old_uniq_exec_plan_mapping_keys,
                cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status
            )

            plan_groups = [plan_group]
            plan_groups = [MyExecPlanGroup(
                plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
            for plan_group in plan_groups:
                plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)

            return plan_groups


    def get_candidate_plan_groups_saturn_optimus_dynamic_random(
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
            LPT stands for ``Longest Processing Time``.
            Algorithm:
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
                Algorithm:
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




        def _get_delta_score_by_makespan(
                plan_groups_to_check: List[Tuple[int, List[MyExecPlan]]], cand_plan_group: List[MyExecPlan]
            ) -> Tuple[List[Tuple[float, int]], List[Tuple[int, List[MyExecPlan]]]]:
            """
                Algorithm:
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


            pruned_plan_groups_to_check = list()
            plan_group_score = list()

            for gpu_num, plan_group in plan_groups_to_check:


                infos = _get_model_latency_to_complete(plan_group)

                model_id, comp_cost, extra_cost = [(model_id, comp_cost, extra_cost) for model_id, (comp_cost, extra_cost, gpu_count) in infos.items() if model_id not in base_infos or base_infos[model_id][2]!=gpu_count][0]

                plan_group_score.append( ((base_infos[model_id][0]+base_infos[model_id][1] - (comp_cost+extra_cost))/gpu_num, gpu_num)  )
                pruned_plan_groups_to_check.append( (gpu_num, plan_group) )


            return plan_group_score, pruned_plan_groups_to_check


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
                    1. Apply early validity checks to execution plans.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    all candidate execution plans must be checked individually.
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


            plan_group_throughputs, plan_groups_to_check = _get_delta_score_by_makespan(
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


        def _gen_cand_exec_plans_using_more_gpus(unfinished_model_num, cand_models: List[MyModelInfor], gpu_nums_for_each_model: Dict[int, int]):
            exec_plans_list = list()
            last_stage_plan_dict = {plan.model.model_id:plan.get_key() for plan in last_stage_exec_plans}
            for model in cand_models:
                exec_plans = get_possible_exec_plans(
                    model, tot_gpu_num, byte_per_gpu, cost_table, gen_execplans_baseline, sort_input=sort_input, fully_connected_gpu_unit=fully_connected_gpu_unit)


                if 'no_preemption' in search_method_baseline:
                    if model.model_id in last_stage_plan_dict:
                        exec_plans = [plan for plan in exec_plans if plan.get_key() == last_stage_plan_dict[plan.model.model_id]]


                if unfinished_model_num >= tot_gpu_num:
                    exec_plans = [sorted(exec_plans, key=lambda _: get_tot_worker_num([_]))[0]]


                if model.model_id not in gpu_nums_for_each_model:
                    gpu_nums_for_each_model[model.model_id] = 0

                exec_plans = [plan for plan in exec_plans if get_tot_worker_num([plan]) > gpu_nums_for_each_model[model.model_id]]


                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")
            return exec_plans_list


        def _sample_cand_models(cand_models, tot_cand_models):
            last_stage_models = [plan.model.model_id for plan in last_stage_exec_plans]
            models_to_keep = list()
            models_to_sample = list()
            for model in cand_models:
                if model.model_id in last_stage_models:
                    models_to_keep.append(model)
                else:
                    models_to_sample.append(model)

            import random
            sampled = random.sample(models_to_sample, min(len(models_to_sample), tot_gpu_num-len(tot_cand_models)-len(models_to_keep)))
            return models_to_keep + sampled


        def _select_exec_plans_in_model_order(best_plan_group, cand_models, exec_plans_list):
            exec_plans_dict = {exec_plans[0].model.model_id: min(exec_plans, key=lambda plan: get_tot_worker_num([plan])) for exec_plans in exec_plans_list}
            selected = best_plan_group
            selected_models = list()
            for model in cand_models:


                if get_tot_worker_num(selected+[exec_plans_dict[model.model_id]]) > tot_gpu_num:
                    return selected, selected_models
                else:
                    selected = selected+[exec_plans_dict[model.model_id]]
                    selected_models.append(model)
            return selected, selected_models


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


        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")

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


            exec_plans_list = _gen_cand_exec_plans_using_more_gpus(unfinished_model_num, cand_models, gpu_nums_for_each_model)


            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)


            _get_exec_plan_latency(
                uniq_exec_plan_mapping,
                best_plan_group,
                exec_plans_list,
                check_gap, sort_input, last_stage_exec_plans, cost_table, tot_gpu_num)


            cand_models = _sample_cand_models(cand_models, tot_cand_models)
            best_plan_group, selected_models = _select_exec_plans_in_model_order(best_plan_group, cand_models, exec_plans_list)

            gpu_nums_for_each_model.update({plan.model.model_id:get_tot_worker_num([plan]) for plan in best_plan_group})
            print(f"len(selected_models): {len(selected_models)}, len(cand_models): {len(cand_models)}, get_tot_worker_num(best_plan_group): {get_tot_worker_num(best_plan_group)}")
            best_plan_group = _get_correct_tmp_plan_group(
                    uniq_exec_plan_mapping,
                    best_plan_group,
                    best_plan_group[-1],
            )
            if (len(selected_models) < len(cand_models)) or (get_tot_worker_num(best_plan_group) == tot_gpu_num):

                plan_groups = [best_plan_group]

                time1 = time.perf_counter()
                plan_groups = [MyExecPlanGroup(
                    plan_group, cost_table, last_stage_exec_plans, check_gap, sort_input, base_model_finish_status=base_model_finish_status) for plan_group in plan_groups]
                for plan_group in plan_groups:
                    plan_group.wait_remote_fake_scheduling_to_compute_infer_stage_data(cost_table=cost_table)
                print(f"time finish get costs: {time.perf_counter() - time1}")


                print(f"time_analysis: {time_analysis}")

                return plan_groups


            tot_cand_models.extend(selected_models)
            cand_models = list()


            should_stop = False
            while((len(cand_models) > 0) or is_last_level):


                print(f"tot_cand_models: {[model.model_id for model in tot_cand_models]}")


                exec_plans_list = _gen_cand_exec_plans_using_more_gpus(unfinished_model_num, tot_cand_models, gpu_nums_for_each_model)


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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
            LPT stands for ``Longest Processing Time``.
            Algorithm:
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
                    1. Apply early validity checks to execution plans.
                    2. for models which may have inp models in the same stage, e.g., a horizontally fused model (we can check this by ``models_have_in_level_dependency``), 
                    all candidate execution plans must be checked individually.
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
            Supports unfinished-model loading penalties and non-preemptive scheduling.
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


        def _sample_cand_exec_plans_to_speedup(runnable_exec_plans_list, cand_plan_group):
            last_stage_models = [plan.model.model_id for plan in last_stage_exec_plans]
            exec_plans_to_keep = list()
            exec_plans_to_sample = list()
            for exec_plans in runnable_exec_plans_list:
                if exec_plans[0].model.model_id in last_stage_models:
                    exec_plans_to_keep.append(exec_plans)
                else:
                    exec_plans_to_sample.append(exec_plans)

            import random
            sample_size = int(os.getenv('SAMPLE_SIZE', f'{tot_gpu_num}'))


            sampled = random.sample(exec_plans_to_sample, min(len(exec_plans_to_sample), sample_size-len(cand_plan_group)-len(exec_plans_to_keep)))
            return exec_plans_to_keep + sampled


        def _keep_no_preemption_plan_groups_only(plan_groups, runnable_exec_plans_list):
            last_stage_models = [plan.model.model_id for plan in last_stage_exec_plans]
            cand_model_ids = [exec_plans[0].model.model_id for exec_plans in runnable_exec_plans_list]
            model_ids_to_keep = set(last_stage_models).intersection(cand_model_ids)
            ret = list()
            for plan_group in plan_groups:
                if len(plan_group) < len(model_ids_to_keep):
                    continue
                model_ids = set([plan.model.model_id for plan in plan_group])
                if model_ids_to_keep.issubset(model_ids):
                    ret.append(plan_group)
            return ret


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

        unfinished_model_num = sum([len(_) for _ in self.all_level_model_ids])
        print(f"unfinished_model_num: {unfinished_model_num}")

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


                if 'sampleModels' in search_method_baseline:
                    if unfinished_model_num >= tot_gpu_num:
                        exec_plans = [sorted(exec_plans, key=lambda _: get_tot_worker_num([_]))[0]]


                exec_plans_list.append(exec_plans)
                print(f"model finished? {model.is_finished()}, model_id: {model.get_base_model_ids()}, can exec_plans: {[str(plan) for plan in exec_plans]}")


            if gen_execplans_baseline == 'naive':
                exec_plans_list = _wait_for_possible_exec_plans_latency_naive_baseline(exec_plans_list, cost_table)


            time_analysis[0] += (time.perf_counter() - time1)


            for cand_plan_group in new_plan_groups:


                time1 = time.perf_counter()


                runnable_exec_plans_list = self.get_runnable_plans_from_cand_plans(cand_plan_group, cand_models, exec_plans_list)


                if 'sampleModels' in search_method_baseline:
                    runnable_exec_plans_list = _sample_cand_exec_plans_to_speedup(runnable_exec_plans_list, cand_plan_group)
                    print(f"number of cand models after sample: {len(runnable_exec_plans_list)}")


                time_analysis[1] += (time.perf_counter() - time1)
                time1 = time.perf_counter()


                good_runnable_exec_plan_keys_list = [[exec_plan.get_key() for exec_plan in exec_plans] for exec_plans in runnable_exec_plans_list]


                time_analysis[2] += (time.perf_counter() - time1)
                time1 = time.perf_counter()


                plan_groups = [cand_plan_group]
                old_uniq_exec_plan_mapping_keys = list(uniq_exec_plan_mapping.keys())
                _append_exec_plan(plan_groups, runnable_exec_plans_list, 0, tot_gpu_num, byte_per_gpu, uniq_exec_plan_mapping, good_runnable_exec_plan_keys_list,
                                  self.out_edge_dict)


                if 'no_preemption' in search_method_baseline:
                    plan_groups = plan_groups[:1] + _keep_no_preemption_plan_groups_only(plan_groups[1:], runnable_exec_plans_list)


                time_analysis[3] += (time.perf_counter() - time1)
                time1 = time.perf_counter()


                if len(plan_groups) == 1:


                    if visit_model_level+1 >= len(self.all_level_model_ids):
                        if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_groups[0], tot_gpu_num
                            ):
                            tot_plan_groups.extend(plan_groups)
                else:


                    if (visit_model_level+1 >= len(self.all_level_model_ids)) and ('no_preemption' not in search_method_baseline):

                        plan_groups = [plan_groups[0]]+[plan_group for plan_group in plan_groups[1:] if not _directly_discard(
                            gen_execplans_baseline, not_finished_base_model_num, plan_group, tot_gpu_num
                            )]


                    time1 = time.perf_counter()


                    plan_groups = plan_groups[:1] + [plan_group for plan_group in plan_groups[1:] if _has_in_stage_inp_models(plan_group)]


                    time_analysis[4] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()


                    time_analysis[5] += (time.perf_counter() - time1)
                    time1 = time.perf_counter()


                    good_plan_groups = plan_groups[1:]
                    all_plan_group_list.extend(good_plan_groups)


                    tmp_new_plan_groups.extend(good_plan_groups)


                    time_analysis[6] += (time.perf_counter() - time1)


            new_plan_groups = tmp_new_plan_groups
            if len(new_plan_groups) == 0:
                break


            if 'sampleModels' in search_method_baseline:
                model_num_considered = sum([len(_) for _ in self.all_level_model_ids[:visit_model_level+1]])
                print(f"model_num_considered: {model_num_considered}")
                sample_size = int(os.getenv('SAMPLE_SIZE', f'{tot_gpu_num}'))


                if model_num_considered >= sample_size:

                    break


        plan_groups = all_plan_group_list

        print(f"\nfinal candidates before pruning:\n")
        print(f"\nnumber of final candidates before pruning: {len(plan_groups)}\n")


        max_model_num = max([len(plan_group) for plan_group in plan_groups])


        max_gpu_num = max([get_tot_worker_num(plan_group) for plan_group in plan_groups])
        plan_groups = [plan_group for plan_group in plan_groups if (len(plan_group) == max_model_num) \
                       and (get_tot_worker_num(plan_group)==max_gpu_num)]

        print(f"\nfinal candidates:\n")
        print(f"\nnumber of final candidates: {len(plan_groups)}\n")


        gpu_nums_list = [ [get_tot_worker_num([plan]) for plan in plan_group] for plan_group in plan_groups]
        min_gpu_num_difference = min([max(gpu_nums)-min(gpu_nums) for gpu_nums in gpu_nums_list])
        plan_groups = [plan_group for gpu_nums, plan_group in zip(gpu_nums_list, plan_groups) if max(gpu_nums)-min(gpu_nums) == min_gpu_num_difference]
        print(f"\nfinal candidates: min_gpu_num_difference: {min_gpu_num_difference}\n")
        print(f"\nnumber of final candidates after min_gpu_num_difference pruning: {len(plan_groups)}\n")


        time1 = time.perf_counter()


        if 'randomCost' in search_method_baseline:
            import random
            plan_groups = [random.choice(plan_groups)]


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
        elif search_method_baseline in ['min_gpu_useAll', 'min_gpu_useAll_no_preemption', 'min_gpu_useAll_sampleModels', 'min_gpu_useAll_no_preemption_randomCost']:
            return self.get_candidate_plan_groups_min_gpu_use_all_gpus(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ['greedy_saturn_LPT', 'greedy_saturn_LPT_no_preemption', 'greedy_saturn_LPT_no_preemption_randomCost']:
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
        elif search_method_baseline in ['saturn_optimus_dynamic_random', 'saturn_optimus_dynamic_random_no_preemption']:
            return self.get_candidate_plan_groups_saturn_optimus_dynamic_random(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline == 'oracle':
            return self.get_candidate_plan_groups_oracle(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline == 'hybridflow_model_parallel_old':


            return self.get_candidate_plan_groups_HybridFlow_old(
                gen_execplans_baseline,
                'hybridflow_model_parallel',
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ['hybridflow_model_parallel', 'hybridflow_summary_first', 'hybridflow_evaluator_last']:
            return self.get_candidate_plan_groups_HybridFlow(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ['saturnLP', 'saturnLP_no_preemption']:
            return self.get_candidate_plan_groups_saturn_LP(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ["LightRule", "LightRule_no_preemption"]:
            return self.get_candidate_plan_groups_LightRule(
                gen_execplans_baseline,
                search_method_baseline,
                check_gap, sort_input,
                last_stage_exec_plans, cost_table,
                tot_gpu_num, byte_per_gpu,
                top_k,
                fully_connected_gpu_unit)
        elif search_method_baseline in ['LightRule_our_cost_model']:
            return self.get_candidate_plan_groups_LightRule_our_cost_model(
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

    global USE_PEAK_MEM

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

            if USE_PEAK_MEM:


                param_byte_per_comp_gpu = float('inf')
                if (model.model_path, num_worker) in peak_mem_dict:
                    param_byte_per_comp_gpu = peak_mem_dict[(model.model_path, num_worker)]
                else:
                    continue


            if True:
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

    global USE_PEAK_MEM

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
            if USE_PEAK_MEM:


                param_byte_per_comp_gpu = float('inf')
                if (model.model_path, num_worker) in peak_mem_dict:
                    param_byte_per_comp_gpu = peak_mem_dict[(model.model_path, num_worker)]
                else:
                    continue


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
    elif baseline == 'max_gpu':
        ret = _get_possible_exec_plans(model, tot_gpu_num, byte_per_gpu, cost_table, fully_connected_gpu_unit)
        print(f"gen max gpu exec plans------")
        return [i for i in ret if get_tot_worker_num([i]) == tot_gpu_num]
    else:

        return _get_possible_exec_plans_naive_baseline(model, tot_gpu_num, byte_per_gpu, cost_table, sort_input, fully_connected_gpu_unit)


def select_best_gpus_for_cache_request(request: np.ndarray, resources: np.ndarray, requests: List[List[int]]):
    '''
    Input: 
        request: list of ints: each int is the cache mem required on a cache gpu.
        resources: the available gpu mems on each candidate cache gpu.
        requests: the remaining cache mem requests.
    NOTE: we deal with all cache requirement for a model at a time.
    Rank feasible placements by the remaining-memory fragmentation score.
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


def _get_path_key(plan_group:List[MyExecPlan], exec_plan):
    """Return the dependency key and related plans within the current stage group."""
    inp_model_ids = sorted(exec_plan.model.input_model_ids)
    key = list()
    key_plans = list()
    for inp in inp_model_ids:
        for plan in plan_group:
            if (inp == plan.model.model_id) or (inp in plan.model.get_base_model_ids()):


                # Chain models are already fused; no additional fusion check is needed.
                key.append(( plan.model.model_id, plan.get_key() ))

                key_plans.append(plan)
                break


    # Chain models are already fused; no additional fusion check is needed.
    key.append(( exec_plan.model.model_id, exec_plan.get_key() ))

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
            2. model_ids_fused: the base model ids of the fused model being extended.
            3. fused_model_inp_base_model_ids: the input base model ids of the fused model being extended.
    """
    cond1 = (len(to_fuse_inp_base_model_ids) == 1) and \
        (to_fuse_inp_base_model_ids[0] == model_ids_fused[-1])


    cond2 = sorted(to_fuse_inp_base_model_ids) == sorted(fused_model_inp_base_model_ids+model_ids_fused)
    return (cond1 or cond2)


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


def _update_uniq_exec_plan_mapping(
        plan_group: List[MyExecPlan], plan: MyExecPlan, uniq_exec_plan_mapping
    ) -> MyExecPlan:
    """
        plan_group is a list of execution plans.
        plan is an execution plan to be appended.
        Check whether the plan will need to be a new uniq plan object after being appended.
        If so, update the uniq_exec_plan_mapping; otherwise, use the plan stored in the uniq_exec_plan_mapping.
    """


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


    if len(affected_model_ids) == 0:
        return plan_group


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


            # Chain fusion is resolved before this search-space enumeration.


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


    if 'hybridflow' in search_method_baseline:
        return plan_groups


    if 'LightRule' in search_method_baseline and search_method_baseline != 'LightRule_our_cost_model':
        return plan_groups


    if search_method_baseline == 'saturnLP_no_preemption':
        return plan_groups


    print(f"in get_one_stage_exec_plans_sorted: the plan groups we generated: ")
    for plan_group in plan_groups:
        print(f"{len(plan_group)}, {[(plan.model.get_base_model_ids(), plan.get_key()) for plan in plan_group.exec_plans], plan_group.get_throughput()}")


    if search_method_baseline=='oracle':
        return plan_groups


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


def get_total_model_flops(model_list: List[MyModelInfor], cost_table: CostTable):
    flops = 0
    for model in model_list:
        assert not isinstance(model, MyFusedModelInfor)
        i = fake_scheduling.comp_flops_from_seqlens(
            model.inp_lens, model.out_lens, only_decode=False, cost_table=cost_table,
            model_path=model.model_path, trust_remote_code=model.trust_remote_code, revision=model.revision)
        flops += i
    return flops


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
    '''

    global _MAX_SEQ_NUM, _CHECKED_SEQ_NUM


    if (search_method_baseline!='oracle') and (_CHECKED_SEQ_NUM > _MAX_SEQ_NUM):
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


    if search_method_baseline!='oracle':

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


    if 'hybridflow' in search_method_baseline:
        best_group_seq.hybridflow_result = plan_groups
        best_group_seq.set_plan_group_and_time(
                plan_groups[1][-1], None, None)
        return


    if 'LightRule' in search_method_baseline and search_method_baseline != 'LightRule_our_cost_model':
        best_group_seq.set_plan_group_and_time(
                plan_groups, None, None)
        return


    if 'saturnLP_no_preemption' in search_method_baseline:
        best_group_seq.set_plan_group_and_time(
                plan_groups, None, None)
        return


    if search_method_baseline!='oracle':
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

        # Check vertical fusion with the preceding stage before evaluating this group.

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


        if search_method_baseline!='oracle':
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
    '''
    print(f"search_method_baseline: {search_method_baseline}")


    _get_best_model_schedule(
        gen_execplans_baseline, search_method_baseline,
        check_gap, sort_input,
        cost_table,
        model_sys,
        curr_group_seq,
        best_group_seq,
        uniq_model_states,
        gpu_name, tot_gpu_num, byte_per_gpu, top_k, fully_connected_gpu_unit)


def get_engin_args(model_path, tensor_parallel_size):
    backend = "ours"
    model = model_path
    if os.environ.get('MIL_CPU_SCHEDULING') != '1':
        from artifact.mil_artifact.model_runtime import resolve_model_snapshot
        model = resolve_model_snapshot(model_path)
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
                The output-request mapping also supports reuse of an upstream output by multiple downstream requests.
    '''


    import time
    begin_sample = time.perf_counter()

    out_lens_dict = {i: outlen_generators[i](
            i, model_path[model_path.find('/')+1:], inp_lens_dict[i]) for i, model_path in enumerate(model_paths)}


    end_sample = time.perf_counter()
    print(f"outlen gen time: {end_sample - begin_sample}")


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
        machine_name='lccpu'):
    """
        NOTE: ``inp_generator``, ``inp_merger``, ``outlen_generator`` are 3 functions about model inp/out lens.
    """


    sample_size = os.getenv('SAMPLE_SIZE', f'{tot_gpu_num}')
    print(f"SAMPLE_SIZE if will sample models: {sample_size}")


    global USE_PEAK_MEM

    if search_method_baseline == 'oracle':
        USE_PEAK_MEM = True


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
    if machine_name == 'lccpu':
        cost_table: CostTable = get_my_cost_table_directly.cost_table
    elif machine_name == 'zxcpu':
        cost_table: CostTable = get_my_cost_table_directly_zxcpu.cost_table

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


    global Cost_Query_num, Cost_Query_Time, Plan_Group_Enu_num, Plan_Group_Enu_Time, Plan_Group_num_After_Pruning, Select_Best_Time
    print(f"Cost_Query_Time, Cost_Query_num, Plan_Group_Enu_Time, Plan_Group_Enu_num, Plan_Group_num_After_Pruning, Select_Best_Time: {Cost_Query_Time, Cost_Query_num, Plan_Group_Enu_Time, Plan_Group_Enu_num, Plan_Group_num_After_Pruning, Select_Best_Time}")


    if 'hybridflow' in search_method_baseline:
        time2 = time.perf_counter()
        print(f"Total search time: {time2 - time1}")

        print(f"best_group_seq: {[[str(exec_plan) for exec_plan in group.exec_plans] for group in best_group_seq.plan_group_seq]}")
        print(f"other hybridflow res infor: {best_group_seq.hybridflow_result}")
        return best_group_seq


    for plan_group in best_group_seq.plan_group_seq:
        for exec_plan in plan_group.exec_plans:
            exec_plan.load_cost_just_for_refer = cost_table.get_prepare_cost(exec_plan.model.model_name, exec_plan.get_key_single_dp_worker())


    if ('LightRule' in search_method_baseline and search_method_baseline != 'LightRule_our_cost_model') or (search_method_baseline == 'saturnLP_no_preemption'):
        time2 = time.perf_counter()
        print(f"Total search time: {time2 - time1}")
        print(f"Total time for preparation before search: {time_before_search - time1}")
        print(f"Best group seq: {[[str(exec_plan) for exec_plan in group.exec_plans] for group in best_group_seq.plan_group_seq]}")

        print(f"_CHECKED_SEQ_NUM: {_CHECKED_SEQ_NUM}")
        return best_group_seq


    time2 = time.perf_counter()
    print(f"Total search time: {time2 - time1}")
    print(f"Total time for preparation before search: {time_before_search - time1}")
    print(f"Best group seq: {best_group_seq.get_str_using_model_ids()}")
    print(f"Best group seq throughputs: {best_group_seq.get_stage_throughputs()}")
    print(f"_CHECKED_SEQ_NUM: {_CHECKED_SEQ_NUM}")

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
