"""
This file contains functions to schedule multiple models together on a given set of GPUs on the same machine. 
"""







from concurrent.futures import ProcessPoolExecutor, wait
import asyncio
from multiprocessing import Array, Event


from vllm.core.multimodel_scheduler import SHARED_CONTECT, LLM_COMMUNICATOR, MyManager
from vllm.sampling_params import SamplingParams
import benchmark_throughput

import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import itertools

from search_exec_plans import MyExecPlan, MyExecPlanGroupSeq, MyModelInfor, get_best_model_schedule, get_dependent_exec_plans_for_each_plan 
import output_length_sampler

from collections import defaultdict



import traceback
import argparse


class MyExecPlanState:
    """Record the state of an exec plan"""
    def __init__(self, 
        exec_plan: MyExecPlan, 
        launched: bool,
        stage_i: int,
        last_exec_plan_for_the_model: bool,
        need_prepare_infer_env: bool,
    ) -> None:
        self.exec_plan = exec_plan
        self.stage_i = stage_i
        self.last_exec_plan_for_the_model = last_exec_plan_for_the_model
        self.need_prepare_infer_env = need_prepare_infer_env
        
        
        self.comp_gpus: List[int] = list()

        
        self.launched = launched
    
    def set_comp_gpus(self, comp_gpus: List[int]):
        self.comp_gpus = list(comp_gpus)
    def get_comp_gpus(self):
        return self.comp_gpus

    def __str__(self) ->str:
        return f'{str(self.exec_plan)}, launched:{self.launched}, stage_i:{self.stage_i}, model_id: {self.exec_plan.model.model_id}, comp_gpus: {self.comp_gpus}'






class InferenceArgs:
    """Arguments for vLLM single model inference."""
    def __init__(self, 
        model:str="huggyllama/llama-7b", 
        num_prompts: int = 1000,
        dataset: str = "ShareGPT_V3_unfiltered_cleaned_split.json",
        ignore_eos: bool = False, 
        fixed_output_len: int = None,
        
    ) -> None:
        self.backend: str = "vllm"
        self.dataset: str = dataset
        self.input_len: int = None
        self.output_len: int = fixed_output_len
        self.model: str = model
        self.tokenizer: str = None
        self.quantization = None
        self.tensor_parallel_size: int = 1
        self.n: int = 1
        self.use_beam_search: bool = False
        
        self.num_prompts: int = num_prompts 
        self.seed: int = 0
        self.hf_max_batch_size: int = None
        self.trust_remote_code: bool = True
        self.max_model_len: int = None
        self.dtype: str = 'auto'
        self.enforce_eager: bool = True
        self.kv_cache_dtype: str = "auto"
        self.device: str = "cuda"

        
        self.weight_load_degree: str = '16'
        self.gpu_use_ratio: float = 0.9
        
        self.temperature: float = 1.0
        
        self.ignore_eos: bool = ignore_eos

        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.dataset is None:
            assert self.input_len is not None
            assert self.output_len is not None
        else:
            assert self.input_len is None

        if self.backend == "vllm":
            if self.hf_max_batch_size is not None:
                raise ValueError("HF max batch size is only for HF backend.")
        elif self.backend == "hf":
            if self.hf_max_batch_size is None:
                raise ValueError("HF max batch size is required for HF backend.")
            if self.quantization is not None:
                raise ValueError("Quantization is only for vLLM backend.")
        elif self.backend == "mii":
            if self.dtype != "auto":
                raise ValueError("dtype must be auto for MII backend.")
            if self.n != 1:
                raise ValueError("n must be 1 for MII backend.")
            if self.use_beam_search:
                raise ValueError("Beam search is not supported for MII backend.")
            if self.quantization is not None:
                raise ValueError("Quantization is only for vLLM backend.")
            if self.hf_max_batch_size is not None:
                raise ValueError("HF max batch size is only for HF backend.")
            if self.tokenizer != self.model:
                raise ValueError("Tokenizer must be the same as the model for MII "
                                "backend.")








def start_a_model_inference_child_process(
        communicator: LLM_COMMUNICATOR, use_vllm: bool, gpus: str, shared_id: int, model: str = "huggyllama/llama-7b", 
        return_str=True, req_num=None):
    try:
        print(f"in running start_a_model_inference_child_process")
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        
        os.environ['USE_VLLM']='False'
        os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'
        if use_vllm:
            os.environ['USE_VLLM']='True'
            os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False'
        
        
        
        from huggingface_hub import snapshot_download
        local_model_path = snapshot_download(model)
        
        args = InferenceArgs(local_model_path, req_num)

        
        
        
        SHARED_CONTECT.shared_id = shared_id
        SHARED_CONTECT.communicator = communicator
        SHARED_CONTECT.return_str = return_str
        SHARED_CONTECT.tot_req_num_remained = req_num
        print(f"SHARED_CONTECT.shared_id: {SHARED_CONTECT.shared_id}")
        print(f"SHARED_CONTECT.tot_req_num_remained: {SHARED_CONTECT.tot_req_num_remained}")
        
        benchmark_throughput.main(args)
        print(f"MODEL PROCESS ENDS: shared_id: {SHARED_CONTECT.shared_id}", flush=True)
    except Exception as e:
        print(f"Exception in running benchmark_throughput.main(): {e}")
        print(traceback.format_exc())






def start_a_model_inference(
        communicator: LLM_COMMUNICATOR, use_vllm: bool, gpus: str, model_id: int, model: str = "huggyllama/llama-7b", 
        return_str=True, req_num=None):
    
    print(f"in running start_a_model_inference")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            print(f"in running start_a_model_inference 1")
            executor.submit(start_a_model_inference_child_process, communicator, use_vllm, gpus, model_id, 
                            model, return_str, req_num)
        except Exception as e:
            print(f"Exception in running start_a_model_inference: {e}")
            print(traceback.format_exc())




def get_exec_settings_from_exec_plans(
        exec_plan: MyExecPlan, available_gpus: List[int], tot_gpu_num: int, gpu_order_we_set: List[int]):
    """
        Get the exec setting to store in the SHARED_CONTECT later, based on the given exec_plan.
    """

    
    
    
    tp, gpu_ratio, wldeg, cache_gpu_num, dp_size = exec_plan.get_key()
    gpu_list = available_gpus + [i for i in range(tot_gpu_num) if i not in available_gpus]

    
    if max(gpu_list) > tot_gpu_num-1:
        print(f"available_gpus:{available_gpus}, tot_gpu_num: {tot_gpu_num}, [i for i in range(tot_gpu_num) if i not in available_gpus]: {[i for i in range(tot_gpu_num) if i not in available_gpus]}")
        assert False

    
    print(f"gpu_order_we_set: {gpu_order_we_set}, gpu_list: {gpu_list}")
    gpu_list = [gpu_order_we_set[i] for i in gpu_list]
    print(f"gpu_list to set: {gpu_list}", flush=True)


    
    
    new_setting = [tp, int(gpu_ratio*10), wldeg, dp_size] + gpu_list
    return new_setting





def get_model_path_list() -> List[str]:
    model_paths = [
                'lmsys/vicuna-13b-v1.5',
                'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
                'chavinlo/alpaca-13b',
                'project-baize/baize-v2-13b',
                'TheBloke/koala-13B-HF',
                'databricks/dolly-v2-12b',
                'mosaicml/mpt-7b-chat',
                ]
    
    
    
    return model_paths



def query_use_vllm(model_path: str) -> bool:
    return True
    setting_dict = {
        'NousResearch/Llama-2-7b-hf': False, 
        'NousResearch/Llama-2-7b-chat-hf': False,
        'NousResearch/Llama-2-13b-hf': False,
        'NousResearch/Llama-2-70b-hf': False,
        'THUDM/chatglm3-6b': True,
        'EleutherAI/gpt-j-6b': True, 
        'EleutherAI/gpt-neox-20b': True,
        'baichuan-inc/Baichuan2-13B-Chat': True,
        'baichuan-inc/Baichuan-7B': True,
        'mistralai/Mixtral-8x7B-v0.1': True,
        
        
        'lmsys/vicuna-13b-v1.5': True,
        'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5': True,
        'chavinlo/alpaca-13b': True,
        'project-baize/baize-v2-13b': True,
        'TheBloke/koala-13B-HF': True,
        'databricks/dolly-v2-12b': True,
        'mosaicml/mpt-7b-chat': True,   
        
        'meta-llama/Llama-2-70b-chat-hf': True,   
        'mistralai/Mixtral-8x7B-Instruct-v0.1': True,   
        'WizardLMTeam/WizardLM-13B-V1.2': True,   
        'meta-llama/CodeLlama-34b-Instruct-hf': True,   
        'mistralai/Mistral-7B-Instruct-v0.2': True,   
    }
    return setting_dict[model_path]






def prepare_exec_plan_states(
        plan_group_seq: MyExecPlanGroupSeq
        )->List[List[MyExecPlanState]]:
    '''
        Prepare the corresponding MyExecPlanState object for each exec plan.
        Set the ``last_exec_plan_for_the_model`` and ``need_prepare_infer_env`` attributes for each exec plan.
        Output:
            List of exec plan state objects for each stage.
    '''
    plan_group_list = plan_group_seq.plan_group_seq
    plan_state_group_list: List[List[MyExecPlanState]] = [[] for i in range(len(plan_group_list))]
    checked_model_ids: List[int] = list()

    for stage_i in range(len(plan_group_list)-1, -1, -1):
        plan_group = plan_group_list[stage_i]
        plan_state_group: List[MyExecPlanState] = plan_state_group_list[stage_i]
        if stage_i == 0:
            
            for exec_plan in plan_group.exec_plans:
                last_exec_plan_for_the_model = False
                if exec_plan.model.model_id not in checked_model_ids:
                    last_exec_plan_for_the_model = True
                    checked_model_ids.append(exec_plan.model.model_id)

                plan_state_group.append(MyExecPlanState(
                    exec_plan, launched=False, stage_i=stage_i, 
                    last_exec_plan_for_the_model=last_exec_plan_for_the_model, 
                    need_prepare_infer_env=False))

        else:
            
            last_stage_exec_plans = plan_group_seq.plan_group_seq[stage_i-1].exec_plans
            last_stage_exec_plans_info = [
                (exec_plan.model.model_id, exec_plan.get_key()) for exec_plan in last_stage_exec_plans
            ]
            
            for exec_plan in plan_group.exec_plans:

                
                

                need_prepare_infer_env = True
                if (exec_plan.model.model_id, exec_plan.get_key()) in last_stage_exec_plans_info:
                    
                    need_prepare_infer_env = False
                
                last_exec_plan_for_the_model = False
                if exec_plan.model.model_id not in checked_model_ids:
                    last_exec_plan_for_the_model = True
                    checked_model_ids.append(exec_plan.model.model_id)
                
                plan_state_group.append(MyExecPlanState(
                    exec_plan, launched=False, stage_i=stage_i, 
                    last_exec_plan_for_the_model=last_exec_plan_for_the_model, 
                    need_prepare_infer_env=need_prepare_infer_env))   

    return plan_state_group_list




def _get_model_sys_structure_from_selected_plan_group_seq(
        plan_state_group_list: List[List[MyExecPlanState]], 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]], 
        out_edge_dict: Dict[int, List[int]],
) -> Tuple[Dict[int, int], Dict[int, MyModelInfor], Dict[int, List[int]], Dict[int, List[int]]]:
    """
        Input:
            1. in_edge_dict_with_dummy_inp_nodes: the in edge dict of the initial model system with dummy inp nodes.
            2. out_edge_dict: the out edge dict of the initial model system without dummy inp nodes.
    """


    print(f"in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"out_edge_dict: {out_edge_dict}")


    model_id_shared_id_mapping: Dict[int, int] = dict()
    shared_id: int = 0

    
    model_dict: Dict[int, MyModelInfor] = dict()
    for plan_state_group in plan_state_group_list:
        for plan_state in plan_state_group:
            model_id = plan_state.exec_plan.model.model_id
            model_dict[model_id] = plan_state.exec_plan.model
            if model_id not in model_id_shared_id_mapping:
                model_id_shared_id_mapping[model_id] = shared_id
                shared_id += 1

    
    node_mapping: Dict[int, int] = dict()
    for model_id, model in model_dict.items():
        for ori in model.get_base_model_ids():
            node_mapping[ori] = model_id
    dummy_model_ids = np.concatenate(list(in_edge_dict_with_dummy_inp_nodes.values()))
    dummy_model_ids = dummy_model_ids[dummy_model_ids<0]
    for model_id in dummy_model_ids:
        node_mapping[model_id] = model_id

    
    new_in_edge_dict_with_dummy_inp_nodes = defaultdict(list)
    for k, vs in in_edge_dict_with_dummy_inp_nodes.items():
        new_in_edge_dict_with_dummy_inp_nodes[node_mapping[k]].extend([node_mapping[v] for v in vs])

    
    new_out_edge_dict = defaultdict(list)
    for k, vs in out_edge_dict.items():
        new_out_edge_dict[node_mapping[k]].extend([node_mapping[v] for v in vs])
    

    return model_id_shared_id_mapping, model_dict, new_in_edge_dict_with_dummy_inp_nodes, new_out_edge_dict




def search_best_scheduling(
        test_cases:List[str], versions: List[str], max_token_nums: List[str], specify_outlens: List[bool],
        
        gen_execplans_baseline:str,
        search_method_baseline:str,
        model_paths: List[str], 
        
        out_edge_dict: Dict[int, List[int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, 
        inp_seq_ids_dict, 
        out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]],
        inp_req_ids: Dict[int, Dict[int, List[int]]], 
        inp_req_from_which_model_which_out_reqs: Dict[int, Dict[int, Dict[int, int]]],
        
        independent_srcs: Dict[int, bool],
        
        prompt_templates_lens: Dict[int, int],
        
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num: int = 4,
        max_group_seq_num: int = 100,
        top_k: int=100,
        similar_threshold: float=0.1,
        fully_connected_gpu_unit: int=4,
        machine_name:str='machine1',
    )->List[List[MyExecPlanState]]:
    
    
    
    
    
    
    print(f"test_cases, versions, max_token_nums, specify_outlens: {test_cases, versions, max_token_nums, specify_outlens}")
    funcs = [_get_req_len_funcs(
        test_case=test_case, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen) \
        for test_case, version, max_token_num, specify_outlen in zip(test_cases, versions, max_token_nums, specify_outlens)]
    inp_generators = [_[0] for _ in funcs]
    inp_mergers = [_[1] for _ in funcs]
    outlen_generators = [_[2] for _ in funcs]

    
    
    


    
    
    
    
    best_group_seq = get_best_model_schedule(
        search_method_baseline,
        gen_execplans_baseline,
        check_gap,
        sort_input,
        model_paths, 
        num_prompts,
        inp_seq_ids_dict,
        out_req_id_mapping,
        inp_req_ids, 
        inp_req_from_which_model_which_out_reqs,
        independent_srcs,
        
        inp_generators,
        inp_mergers,
        outlen_generators,
        
        
        
        
        prompt_templates_lens,
        out_edge_dict,
        sample_config=(1, 1, -1, 0),
        trust_remote_code=True, revision=None,
        gpu_name=gpu_name, tot_gpu_num = tot_gpu_num, byte_per_gpu=byte_per_gpu, 
        data_byte=2,
        max_group_seq_num=max_group_seq_num,
        top_k=top_k,
        similar_threshold=similar_threshold,
        fully_connected_gpu_unit=fully_connected_gpu_unit,
        machine_name=machine_name,
    )


    
    plan_state_group_list = prepare_exec_plan_states(best_group_seq)
    
    return plan_state_group_list





def initialize_SHARED_CONTECT_not_support_fused_models(
        tot_gpu_num: int,
        model_paths: List[str], 
        check_gap: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        gpu_order_we_set: List[int],
    ) -> Tuple[List[MyExecPlanState], int, List[MyExecPlanState]]:
    '''
        Update: (1) SHARED_CONTECT events, shared_finish_status, shared_setting
                (2) call SHARED_CONTECT.start_specific_models()
        Output: (1) launched_exec_plan_states; (2) new target stage i; (3) candidate_exec_plan_states
        NOTE:
            1. this version does not support the case where there are fused models in the model system.
    '''

    import ctypes
    
    SHARED_CONTECT.set_execution_plan_size(tot_gpu_num)
    counter = Array('i', [0 for i in range(len(model_paths)*SHARED_CONTECT.execution_plan_size)]) 
    
    SHARED_CONTECT.events = [Event() for _ in range(2+len(model_paths))]
    
    
    SHARED_CONTECT.started_status = [Event() for _ in range(len(model_paths))]
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(len(model_paths))])
    
    check_out_gaps = Array('i', [int(1e9)]*len(model_paths)) 
    SHARED_CONTECT.check_out_gaps = check_out_gaps
    SHARED_CONTECT.check_in_gap = check_gap

    
    
    available_gpus: List[int] = list(range(tot_gpu_num))
    launched_exec_plan_states: List[MyExecPlanState] = plan_state_group_list[0]
    for exec_plan_state in launched_exec_plan_states:
        
        exec_plan = exec_plan_state.exec_plan
        
        
        
        exec_plan_state.set_comp_gpus(available_gpus[:exec_plan.num_worker*exec_plan.dp_size])
        
        setting = get_exec_settings_from_exec_plans(
            exec_plan=exec_plan, available_gpus=available_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
        SHARED_CONTECT.set_execution_plan(setting, model_ids=[exec_plan.model.model_id])
        
        
        if exec_plan.model.model_id not in model_driver_worker_gpu_i:
            model_driver_worker_gpu_i[exec_plan.model.model_id] = available_gpus[0]

        
        
        
        available_gpus = available_gpus[exec_plan.num_worker*exec_plan.dp_size:]

        exec_plan_state.launched = True
        
        

    new_target_stage_i: int = 1
    candidate_exec_plan_states: List[MyExecPlanState] = []
    if len(plan_state_group_list)>1:
        candidate_exec_plan_states = plan_state_group_list[1]

    return launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states





def initialize_SHARED_CONTECT(
        tot_gpu_num: int,
        
        check_gap: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        gpu_order_we_set: List[int],
        model_id_shared_id_mapping: Dict[int, int],
        new_out_edge_dict: Dict[int, List[int]],
        sampling_args_dict: Dict[int, Tuple[bool, int, int]],
        seq_outlen_dict: Dict[int, Dict[int,int]],
        fully_connected_gpu_unit: int,
    ) -> Tuple[List[MyExecPlanState], int, List[MyExecPlanState]]:
    '''
        Update: (1) SHARED_CONTECT events, shared_finish_status, shared_setting
                (2) call SHARED_CONTECT.start_specific_models()
        Output: (1) launched_exec_plan_states; (2) new target stage i; (3) candidate_exec_plan_states
    '''

    import ctypes

    new_model_num = len(model_id_shared_id_mapping)
    
    SHARED_CONTECT.set_execution_plan_size(tot_gpu_num)
    counter = Array('i', [0 for i in range(new_model_num*SHARED_CONTECT.execution_plan_size)]) 
    
    SHARED_CONTECT.events = [Event() for _ in range(2+new_model_num)]
    
    
    SHARED_CONTECT.started_status = [Event() for _ in range(new_model_num)]
    SHARED_CONTECT.shared_setting = counter
    SHARED_CONTECT.shared_finish_status = Array(ctypes.c_bool, [False for i in range(new_model_num)])
    
    check_out_gaps = Array('i', [int(1e9)]*new_model_num) 
    SHARED_CONTECT.check_out_gaps = check_out_gaps
    SHARED_CONTECT.check_in_gap = check_gap
    SHARED_CONTECT.sampling_args_dict = sampling_args_dict
    SHARED_CONTECT.seq_outlen_dict = seq_outlen_dict

    
    
    available_gpus: List[int] = list(range(tot_gpu_num))
    launched_exec_plan_states: List[MyExecPlanState] = plan_state_group_list[0]
    
    
    
    launched_exec_plan_states = sorted(launched_exec_plan_states, key=lambda plan_state: (plan_state.exec_plan.num_worker, (plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size)%fully_connected_gpu_unit == 0, plan_state.exec_plan.dp_size), reverse=True)
    for exec_plan_state in launched_exec_plan_states:
        
        exec_plan = exec_plan_state.exec_plan
        
        
        
        exec_plan_state.set_comp_gpus(available_gpus[:exec_plan.num_worker*exec_plan.dp_size])
        
        setting = get_exec_settings_from_exec_plans(
            exec_plan=exec_plan, available_gpus=available_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
        
        
        SHARED_CONTECT.set_execution_plan(setting, shared_ids=[model_id_shared_id_mapping[exec_plan.model.model_id]])
        
        
        if exec_plan.model.model_id not in model_driver_worker_gpu_i:
            model_driver_worker_gpu_i[exec_plan.model.model_id] = available_gpus[0]

        
        
        
        available_gpus = available_gpus[exec_plan.num_worker*exec_plan.dp_size:]

        exec_plan_state.launched = True
        
        

    new_target_stage_i: int = 1
    candidate_exec_plan_states: List[MyExecPlanState] = []
    if len(plan_state_group_list)>1:
        candidate_exec_plan_states = plan_state_group_list[1]



    
    set_check_in_out_gap(
        curr_stage_plan_states=launched_exec_plan_states, check_gap=check_gap, new_out_edge_dict=new_out_edge_dict,
        model_id_shared_id_mapping=model_id_shared_id_mapping)

    return launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states







def get_the_next_round_exec_plan_schedule_deprecated(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], List[int], List[MyExecPlanState], int]:
    '''
        Output: 
            (1) the updated launched_exec_plan_states (i.e., running exec plan states);
            (2) the updated candidate_exec_plan_states;
            (3) the models to stop;
            (4) the new exec plans to launch;
            (5) the new target stage i;
    '''

    to_launch: List[MyExecPlanState] = list()
    to_launch_model_ids: List[int] = list()
    
    
    
    
    cand_to_launch_list: List[List[MyExecPlanState]] = [list(), list()]
    for plan_state in launched_exec_plan_states:
        
        if SHARED_CONTECT.query_finish_status(plan_state.exec_plan.model.model_id):
            continue

        if plan_state.stage_i < target_stage_i:
            if plan_state.last_exec_plan_for_the_model:
                to_launch.append(plan_state)
                to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
                print(f"to_launch add 0: {str(plan_state)}")
            else:
                
                cand_to_launch_list[0].append(plan_state)
        else:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            print(f"to_launch add 1: {str(plan_state)}")
    
    
    for plan_state in candidate_exec_plan_states:
        
        if SHARED_CONTECT.query_finish_status(plan_state.exec_plan.model.model_id):
            plan_state.launched = True
            continue

        if not plan_state.need_prepare_infer_env:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            print(f"to_launch add 2: {str(plan_state)}")
        else:
            
            cand_to_launch_list[1].append(plan_state)
    
    print(f"to_launch 1: {[str(i) for i in to_launch]}")
    print(f"cand_to_launch_list 1: {[[str(i) for i in cand_to_launch] for cand_to_launch in cand_to_launch_list]}")


    
    occupied_gpus: List[int] = list()
    for plan_state in to_launch:
        occupied_gpus.extend(SHARED_CONTECT.get_comp_gpus(plan_state.exec_plan.model.model_id))
    available_gpus = [i for i in range(tot_gpu_num) if i not in occupied_gpus]

    
    cand_to_launch = sorted(cand_to_launch, key=lambda i: (i.stage_i, i.exec_plan.num_worker), reverse=True)
    
    
    new_launch: List[MyExecPlanState] = list()
    model_ids_to_stop: List[int] = list()
    new_candidate_exec_plan_states: List[MyExecPlanState] = list()
    for plan_state in cand_to_launch:
        if plan_state.exec_plan.model.model_id in to_launch_model_ids:
            
            
            continue

        tp_size = plan_state.exec_plan.num_worker
        if tp_size <= len(available_gpus):
            
            to_launch.append(plan_state)
            print(f"to_launch add 3: {str(plan_state)}")
            if plan_state.launched:
                
                comp_gpus = SHARED_CONTECT.get_comp_gpus(plan_state.exec_plan.model.model_id)
                available_gpus = [i for i in available_gpus if i not in comp_gpus]
            else:
                plan_state.launched = True
                if plan_state.exec_plan.model.model_id not in model_driver_worker_gpu_i:
                    
                    plan_state.set_comp_gpus(available_gpus[:tp_size])
                    model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id] = available_gpus[0]
                    available_gpus = available_gpus[tp_size:]
                else:
                    
                    driver_gpu_i = model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id]
                    assert driver_gpu_i in available_gpus, f"The driver gpu is not available: {driver_gpu_i, available_gpus}"
                    
                    available_gpus = [i for i in available_gpus if i != driver_gpu_i]
                    comp_gpus = [driver_gpu_i]+available_gpus[:tp_size-1]
                    plan_state.set_comp_gpus(comp_gpus)
                    available_gpus = available_gpus[tp_size-1:]
                
                new_launch.append(plan_state)
        else:
            
            print(f"cannot run")
            if plan_state.launched:
                print(f"plan launced: {str(plan_state)}")
                model_ids_to_stop.append(plan_state.exec_plan.model.model_id)
            else:
                print(f"add to new candidate: {str(plan_state)}")
                new_candidate_exec_plan_states.append(plan_state)


    new_target_stage_i = target_stage_i
    if len(new_candidate_exec_plan_states) == 0:
        new_target_stage_i = target_stage_i + 1
        if new_target_stage_i < len(plan_state_group_list):
            new_candidate_exec_plan_states = plan_state_group_list[new_target_stage_i]
    
    return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i








def _get_the_first_plan_state_when_sorted_by_gpu_num_and_topology(
        plan_state_list: List[MyExecPlanState]
    )->List[MyExecPlanState]:
    """
        Sort the given list of plan states by the topology order and their gpu numbers.
        Policy:
            if a plan state A depends on plan state B, then A must be checked later than B.
        Output:
            1. the first plan state to consider
            2. the remaining plan states to be considered
    """
    
    model_ids = [plan_state.exec_plan.model.model_id for plan_state in plan_state_list]
    roots = [plan_state for plan_state in plan_state_list if len(set(plan_state.exec_plan.model.input_model_ids).intersection(model_ids)) == 0]
    roots = sorted(roots, key=lambda plan_state: (plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size) , reverse=True)
    remaining_plan_states = [plan_state for plan_state in plan_state_list if plan_state != roots[0]]
    return roots[0], remaining_plan_states









def _try_to_load_exec_plans(
        cands_to_launch: List[MyExecPlanState],
        to_launch: List[MyExecPlanState],
        
        
        new_candidate_exec_plan_states: List[MyExecPlanState],
        model_driver_worker_gpu_i: Dict[int,int],
        available_gpus: List[int],
        to_launch_model_ids: List[int]
        ) -> List[int]:
    '''
        Get the exec plans to launch from the given candidates.
        Update: to_launch, to_launch_model_ids, model_driver_worker_gpu_i;
                new_launch,
                model_ids_to_stop, new_candidate_exec_plan_states;
                set the comp gpus of the to-launch plans;
        Output: available_gpus
        NOTE:
            since we support model-level pipeline: we do not simply sort the plan states by their gpu number, but also consider their dependency,
            therefore, we call ``_get_the_first_plan_state_when_sorted_by_gpu_num_and_topology`` to get the next plan state to consider every time.
    '''
    
    
    while len(cands_to_launch) > 0:
        plan_state, cands_to_launch = _get_the_first_plan_state_when_sorted_by_gpu_num_and_topology(cands_to_launch)


        model_id = plan_state.exec_plan.model.model_id
        if model_id in to_launch_model_ids:
            
            
            continue

        
        
        
        gpu_num = plan_state.exec_plan.num_worker * plan_state.exec_plan.dp_size
        if gpu_num <= len(available_gpus):
            
            to_launch.append(plan_state)
            to_launch_model_ids.append(model_id)
            print(f"to_launch add 3: {str(plan_state)}")
            if plan_state.launched:
                
                
                comp_gpus = plan_state.get_comp_gpus()
                available_gpus = [i for i in available_gpus if i not in comp_gpus]
            else:
                plan_state.launched = True
                
                
                
                

                plan_state.set_comp_gpus(available_gpus[:gpu_num])
                model_driver_worker_gpu_i[plan_state.exec_plan.model.model_id] = available_gpus[0]
                available_gpus = available_gpus[gpu_num:]   

                
                
                
                
                
                
                
                
                
                
                    
                
                
                
                
                
                
        else:
            
            print(f"cannot run")
            if plan_state.launched:
                print(f"plan launced: {str(plan_state)}")
                
            else:
                print(f"add to new candidate: {str(plan_state)}")
                new_candidate_exec_plan_states.append(plan_state)

    return available_gpus










def _has_model_finished(
        plan_state_group_list:List[List[MyExecPlanState]],
        stage_i: int,
        model_id_shared_id_mapping: Dict[int, int],
    )-> bool:
    finished = [i for i in plan_state_group_list[stage_i] \
                if SHARED_CONTECT.query_finish_status(model_id_shared_id_mapping[i.exec_plan.model.model_id])]
    return len(finished) > 0

def _get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        model_id_shared_id_mapping: Dict[int, int],
    
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], int]:
    '''
        Output: 
            (1) the updated launched_exec_plan_states (i.e., running exec plan states);
            (2) the updated candidate_exec_plan_states;
            (3) the models to stop;
            (4) the new exec plans to launch;
            (5) the new target stage i;
    '''

    to_launch: List[MyExecPlanState] = list()
    to_launch_model_ids: List[int] = list()

    launched_plan_gpus = {(i.exec_plan.model.model_id, tuple(i.exec_plan.get_key())):i.get_comp_gpus() for i in launched_exec_plan_states}
    
    
    
    
    cand_to_launch_list: List[List[MyExecPlanState]] = [list(), list()]
    for plan_state in launched_exec_plan_states:
        
        if SHARED_CONTECT.query_finish_status(model_id_shared_id_mapping[plan_state.exec_plan.model.model_id]):
            continue

        if plan_state.stage_i < target_stage_i:
            if plan_state.last_exec_plan_for_the_model:
                to_launch.append(plan_state)
                to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
                print(f"to_launch add 0: {str(plan_state)}")
            else:
                
                cand_to_launch_list[0].append(plan_state)
        else:
            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            print(f"to_launch add 1: {str(plan_state)}")
    
    
    for plan_state in candidate_exec_plan_states:
        
        if SHARED_CONTECT.query_finish_status(model_id_shared_id_mapping[plan_state.exec_plan.model.model_id]):
            plan_state.launched = True
            continue

        if not plan_state.need_prepare_infer_env:
            plan_state.launched = True

            to_launch.append(plan_state)
            to_launch_model_ids.append(plan_state.exec_plan.model.model_id)
            plan_state.set_comp_gpus(launched_plan_gpus[(plan_state.exec_plan.model.model_id, tuple(plan_state.exec_plan.get_key()))])

            print(f"to_launch add 2: {str(plan_state)}")
        else:
            
            cand_to_launch_list[1].append(plan_state)
    
    print(f"to_launch 1: {[str(i) for i in to_launch]}")
    print(f"cand_to_launch_list 1: {[[str(i) for i in cand_to_launch] for cand_to_launch in cand_to_launch_list]}")


    
    occupied_gpus: List[int] = list()
    for plan_state in to_launch:
        
        occupied_gpus.extend(plan_state.get_comp_gpus())
    available_gpus = [i for i in range(tot_gpu_num) if i not in occupied_gpus]

    new_launch: List[MyExecPlanState] = list()
    
    model_ids_to_stop: List[int] = list()
    
    new_candidate_exec_plan_states: List[MyExecPlanState] = list()


    
    
    

    
    

    cand_to_launch = sorted(cand_to_launch_list[1], key=lambda i: i.exec_plan.num_worker, reverse=True)
    available_gpus = _try_to_load_exec_plans(cand_to_launch, to_launch, 
        new_candidate_exec_plan_states, model_driver_worker_gpu_i, available_gpus, to_launch_model_ids)

    new_target_stage_i = target_stage_i
    if len(new_candidate_exec_plan_states)>0:
        
        cand_to_launch = sorted(cand_to_launch_list[0], key=lambda i: i.exec_plan.num_worker, reverse=True)
        available_gpus = _try_to_load_exec_plans(cand_to_launch, to_launch, 
            new_candidate_exec_plan_states, model_driver_worker_gpu_i, available_gpus, to_launch_model_ids)        
    else:
        
        new_target_stage_i = target_stage_i + 1
        if new_target_stage_i < len(plan_state_group_list):
            
            new_candidate_exec_plan_states = plan_state_group_list[new_target_stage_i]
            if _has_model_finished(plan_state_group_list, target_stage_i, model_id_shared_id_mapping):
                
                to_launch, new_candidate_exec_plan_states, new_target_stage_i = \
                    _get_the_next_round_exec_plan_schedule(
                    to_launch, new_candidate_exec_plan_states,
                    new_target_stage_i,
                    tot_gpu_num,
                    plan_state_group_list,
                    model_driver_worker_gpu_i,
                    model_id_shared_id_mapping,
                )
    
    
    return to_launch, new_candidate_exec_plan_states, new_target_stage_i




def _adjust_comp_gpus_for_current_launched_exec_plans(
        launched_exec_plan_states: List[MyExecPlanState],
        new_launch: List[MyExecPlanState],
        old_launched: List[MyExecPlanState],
        tot_gpu_num: int,
        fully_connected_gpu_unit: int)->List[MyExecPlanState]:
    """
        INPUT:
            fully_connected_gpu_unit: the number of gpus that are fully connected, 
                e.g., 2 if 1 gpu is only connected with 1 other gpu with NV-links;
                      4 if 1 gpu is connected by 3 other gpus with NV-links.
            old_launched: last round launched exec plans.
        OUTPUT: 
            the plan_states that will need reload model weights.
    """
    launched_comp_gpu_dict = {(i.exec_plan.model.model_id,i.exec_plan.get_key()): i.get_comp_gpus() for i in old_launched}

    launched_exec_plan_states = sorted(launched_exec_plan_states, key=lambda plan_state: plan_state.exec_plan.num_worker)
    cand_gpu_groups = np.arange(tot_gpu_num).reshape((-1, fully_connected_gpu_unit))
    
    cost_to_clean_models = np.asarray([0]*cand_gpu_groups.shape[0])
    plan_state_to_reassign_gpus: List[MyExecPlanState] = list()
    occupied_gpus = np.asarray([False]*tot_gpu_num)
    
    
    
    for plan_state in launched_exec_plan_states:
        if plan_state not in new_launch:
            
            
            gpus = launched_comp_gpu_dict[(plan_state.exec_plan.model.model_id, plan_state.exec_plan.get_key())]
            
            plan_state.set_comp_gpus(gpus)

            gpus = np.asarray(gpus)[:plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size]
            occupied_gpus[gpus] = True
            print(f"model_id: {plan_state.exec_plan.model.model_id}, gpus: {gpus}", flush=True)
            gpus, counts = np.unique(gpus // fully_connected_gpu_unit, return_counts=True)
            if plan_state.exec_plan.num_worker >= fully_connected_gpu_unit:
                
                
                assert (counts == fully_connected_gpu_unit).all()
                cost_to_clean_models[gpus] = 1e9
                print(f"keep gpu assignment: model_id: {plan_state.exec_plan.model.model_id}", flush=True)
            else:
                
                
                cost_to_clean_models[gpus] = cost_to_clean_models[gpus] + plan_state.exec_plan.load_cost_just_for_refer
                plan_state_to_reassign_gpus.append(plan_state)
        else:
            plan_state_to_reassign_gpus.append(plan_state)
    
    
    sorted_gpu_group_ids = np.argsort(cost_to_clean_models)
    sorted_gpu_group_ids = sorted_gpu_group_ids[ cost_to_clean_models[sorted_gpu_group_ids]<1e9 ]
    
    for group_i in sorted_gpu_group_ids:
        cand_gpu_groups[group_i] = sorted(cand_gpu_groups[group_i], key=lambda gpu_i: occupied_gpus[gpu_i])

    cand_gpus = np.concatenate(cand_gpu_groups[sorted_gpu_group_ids])
    extra_new_launch = list()

    
    
    
    plan_state_to_reassign_gpus = sorted(
        plan_state_to_reassign_gpus, 
        key=lambda plan_state: (plan_state.exec_plan.num_worker, plan_state not in new_launch, 
        (plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size)%fully_connected_gpu_unit == 0,
        plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size), reverse=True)

    for plan_state in plan_state_to_reassign_gpus:
        comp_gpu_num = plan_state.exec_plan.num_worker*plan_state.exec_plan.dp_size
        if plan_state.exec_plan.num_worker >= fully_connected_gpu_unit:
            gpus = cand_gpus[:comp_gpu_num]
            cand_gpus = cand_gpus[comp_gpu_num:]
            plan_state.set_comp_gpus(gpus)
            if plan_state not in new_launch:
                extra_new_launch.append(plan_state)
            print(f"model_id: {plan_state.exec_plan.model.model_id}, reassign gpus: {gpus}, cand_gpus: {cand_gpus}", flush=True)
        else:
            if plan_state not in new_launch:
                
                gpus = plan_state.get_comp_gpus()[:comp_gpu_num]
                if set(gpus).issubset(cand_gpus):
                    
                    num_worker = plan_state.exec_plan.num_worker
                    dp_size = plan_state.exec_plan.dp_size
                    gpu_for_dps = [gpus[dp_i*num_worker:(dp_i+1)*num_worker] for dp_i in range(dp_size)]
                    if False not in [(min(i) // fully_connected_gpu_unit) == (max(i) // fully_connected_gpu_unit) for i in gpu_for_dps]:
                        
                        cand_gpus = [_ for _ in cand_gpus if _ not in gpus]
                        print(f"model_id: {plan_state.exec_plan.model.model_id}, keep gpus: {gpus}, cand_gpus: {cand_gpus}", flush=True)
                        continue
                extra_new_launch.append(plan_state)
                
            gpus = cand_gpus[:comp_gpu_num]
            cand_gpus = cand_gpus[comp_gpu_num:]
            plan_state.set_comp_gpus(gpus)
            print(f"model_id: {plan_state.exec_plan.model.model_id}, reassign gpus: {gpus}, cand_gpus: {cand_gpus}", flush=True)
    
    
    return extra_new_launch


        


def get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states: List[MyExecPlanState], candidate_exec_plan_states: List[MyExecPlanState],
        target_stage_i: int,
        tot_gpu_num: int,
        plan_state_group_list:List[List[MyExecPlanState]],
        model_driver_worker_gpu_i: Dict[int,int],
        model_id_shared_id_mapping: Dict[int, int],
        fully_connected_gpu_unit: int,
    )->Tuple[List[MyExecPlanState], List[MyExecPlanState], List[int], List[MyExecPlanState], int]:

    
    to_launch, new_candidate_exec_plan_states, new_target_stage_i = \
        _get_the_next_round_exec_plan_schedule(
        launched_exec_plan_states, candidate_exec_plan_states,
        target_stage_i,
        tot_gpu_num,
        plan_state_group_list,
        model_driver_worker_gpu_i,
        model_id_shared_id_mapping,
    )

    
    launched_exec_plans = [(i.exec_plan.model.model_id,i.exec_plan.get_key()) for i in launched_exec_plan_states]
    to_launch_exec_plans = [(i.exec_plan.model.model_id,i.exec_plan.get_key()) for i in to_launch]
    
    new_launch = [i for i in to_launch if (i.exec_plan.model.model_id,i.exec_plan.get_key()) not in launched_exec_plans]
    model_ids_to_stop = [i.exec_plan.model.model_id for i in launched_exec_plan_states 
                         if (i.exec_plan.model.model_id,i.exec_plan.get_key()) not in to_launch_exec_plans]

    print(f"next round to_launch: {[str(plan_state) for plan_state in to_launch]}")


    
    extra_new_launch = _adjust_comp_gpus_for_current_launched_exec_plans(
        to_launch, new_launch, launched_exec_plan_states, tot_gpu_num, fully_connected_gpu_unit)
    new_launch = new_launch + extra_new_launch
    model_ids_to_stop = model_ids_to_stop + [i.exec_plan.model.model_id for i in extra_new_launch]

    print(f"ORI launched_exec_plan_states: {[str(plan_state) for plan_state in launched_exec_plan_states]}")
    print(f"extra_new_launch: {[str(plan_state) for plan_state in extra_new_launch]}")
    print(f"model_ids_to_stop: {model_ids_to_stop}")

    return to_launch, new_candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i




def start_exec_plans(
        new_launch: List[MyExecPlanState], tot_gpu_num: int, gpu_order_we_set: List[int],
        model_id_shared_id_mapping: Dict[int, int]):
    try:
        for exec_plan_state in new_launch:
            
            exec_plan = exec_plan_state.exec_plan
            assert len(exec_plan_state.comp_gpus) == (exec_plan.num_worker * exec_plan.dp_size)
            
            print(f"before call get_exec_settings_from_exec_plans: available_gpus: {exec_plan_state.comp_gpus}, tot_gpu_num: {tot_gpu_num}, gpu_order_we_set: {gpu_order_we_set}")

            setting = get_exec_settings_from_exec_plans(
                exec_plan=exec_plan, available_gpus=exec_plan_state.comp_gpus, tot_gpu_num=tot_gpu_num, gpu_order_we_set=gpu_order_we_set)
            shared_id = model_id_shared_id_mapping[exec_plan.model.model_id]
            SHARED_CONTECT.set_execution_plan(setting, shared_ids=[shared_id])

            exec_plan_state.launched = True
            SHARED_CONTECT.start_specific_models([shared_id])
    except Exception as e:
        print(f"Exception in start_exec_plans: {e}")
        print(f"exec plan comp gpus: {[exec_plan_state.comp_gpus for exec_plan_state in new_launch]}, tot_gpu_num: {tot_gpu_num}, gpu_order_we_set: {gpu_order_we_set}")
        print(f"tot_gpu_num: {tot_gpu_num}, gpu_order_we_set: {gpu_order_we_set}")
        print(traceback.format_exc())
        assert False






def get_out_edge_dict_from_in_edge_dict_with_inp_nodes(
        in_edge_dict:Dict[int, List[int]]):
    """
        NOTE: we use negative model_ids to represent dummy inp nodes.
    """
    out_edge_dict = defaultdict(list)
    for tgt, srcs in in_edge_dict.items():
        
        for src in srcs:
            if src >= 0:
                out_edge_dict[src].append(tgt)
    return out_edge_dict




def _get_dummy_requests():
    import json
    with open(f"./my_dummy_requests/my_dummy_requests.json", 'r') as f:
        dataset = json.load(f)
    return dataset









def _init_dummy_requests(
        inp_lens: List[int],
        sampled_inps: Union[List[List[int]], List[str]]=None,
        out_lens_dict: Dict[int, List[int]]=None):
    import json
    if sampled_inps == None:
        with open(f"./my_dummy_requests/my_dummy_requests.json", 'w') as f:
            requests = ["hi" * (input_len - 1) for input_len in inp_lens]
            json.dump(requests, f)    
    else:
        with open(f"./my_dummy_requests/my_dummy_requests.json", 'w') as f:
            
            
            assert sampled_inps != None
            json.dump(sampled_inps, f)
    
    if out_lens_dict != None:
        with open(f"./my_dummy_requests/my_dummy_requests_outlen.json", 'w') as f:
            json.dump(out_lens_dict, f)


def init_prompts_for_the_model_system(
        communicator: LLM_COMMUNICATOR,
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]], 
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]], 
        num_prompts: int, 
        inp_seq_ids_dict: Dict[int, List[int]],
        model_path:str):
    """
        Sample input dataset for the model system.
        INPUT:
            node_dataset_chunk_mapping: {model_id: (dataset_name, chunk_id, chunk_size)}
            independent_srcs: stores whether each model's different input sources are independent, or need to be concatenated to be an input.
        NOTE: we also set the total number of requests each model needs to do inference for.
        OUTPUT:
            req_num_dict: the number of req to answer for each model.
        Modify:
            call communicator.add_seqs to add the input requests to the corresponding dummy models.
    """

    
    args = InferenceArgs(model=model_path, num_prompts=num_prompts)

    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    datasets = set([v[0] for v in node_dataset_chunk_mapping.values()])
    dataset_dict = dict()
    print(f"datasets: {datasets}")
    for dataset in datasets:
        if dataset == None:
            requests = _get_dummy_requests()
            inp_prompts = [(i, req) for i, req in enumerate(requests)]
            dataset_dict[dataset] = inp_prompts
            
            print(f"prompt_lens: {[len(req) for _, req in dataset_dict[None]]}")
            
            continue
        requests = benchmark_throughput.sample_requests(
            dataset, args.num_prompts, tokenizer,args.output_len, random_seed=args.seed)
        inp_prompts = [(i, req[0]) for i, req in enumerate(requests)]
        dataset_dict[dataset] = inp_prompts

    req_num_dict = defaultdict(int)
    prompts_dict = dict()
    for model_id, (dataset, chunk_id, chunk_size) in node_dataset_chunk_mapping.items():
        inp_prompts = dataset_dict[dataset]
        to_add = inp_prompts
        

        print(f"model_id, (dataset, chunk_id, chunk_size): {model_id, (dataset, chunk_id, chunk_size)} prompt_lens: {[len(req) for _, req in dataset_dict[dataset]]}")

        
        
        

        
        to_add = [(req_i, chunks[chunk_id]) if chunk_id < len(chunks) else None for (req_i, chunks) in inp_prompts]


        
        
        
        
        
        
        

        
        if model_id not in inp_seq_ids_dict:
            assert False
            prompts_dict[model_id] = to_add
            req_num_dict[model_id] = len(to_add)
        else:
            prompts_dict[model_id] = [to_add[i] for i in inp_seq_ids_dict[model_id]] 
            req_num_dict[model_id] = len(inp_seq_ids_dict[model_id]) 
    

    print(f"req_num_dict: {req_num_dict}", flush=True)
    

    
    tot_node_num = len(req_num_dict) + len(in_edge_dict_with_dummy_inp_nodes)
    visited = list(req_num_dict.keys())
    
    print(f"tot_node_num: {tot_node_num}, visited: {visited}", flush=True)

    while len(visited) < tot_node_num:
        for tgt, srcs in in_edge_dict_with_dummy_inp_nodes.items():
            if tgt in visited:
                continue
            print(f"tgt, srcs: {tgt, srcs}", flush=True)
            if set(srcs).issubset(visited):
                
                
                
                
                req_num_dict[tgt] = len(inp_seq_ids_dict[tgt])
                visited.append(tgt)
                print(f"visited: {visited}", flush=True)

    
    ungened_out_req_nums = req_num_dict.copy()

    
    
    for model_id in node_dataset_chunk_mapping:
        req_num_dict[model_id] = 0
    communicator.init_unavailable_req_nums_and_ungened_out_req_nums(req_num_dict, ungened_out_req_nums)

    
    for model_id, to_add in prompts_dict.items():

        
        
        print(f"model_id: {model_id}")
        
        if not isinstance(to_add[0][1], str):
            assert False
            to_add = [(req_i, tokenizer.decode(token_ids)) for req_i, token_ids in to_add]


        
        to_add = [(i, j, -1) for i, j in to_add]

        

        communicator.add_seqs(model_id, to_add)

    return req_num_dict
        
    




def get_return_str_list_version(out_edge_dict: Dict[int, List[int]], model_id: int, model_paths: List[str])->bool:
    outs = out_edge_dict[model_id]
    tgt = model_paths[model_id]
    return (True in [tgt != model_paths[out] for out in outs])


def get_return_str(
        new_out_edge_dict: Dict[int, List[int]], model_id: int, model_path_dict: Dict[int, str],
        
        
        )->bool:
    """
        NOTE: 1. we also need to consider the other inputs of the output nodes of model_id, because different input sources 
            may need to be concatenated. 
            Currently, we return str if two input sources need to be concat, even if they use the same tokenizer.
            
            2. each base model has its own ``return_str``. [not implemented]
    """
    return True
    outs = new_out_edge_dict[model_id]
    tgt = model_path_dict[model_id]
    return (True in [tgt != model_path_dict[out] for out in outs])


def set_check_in_out_gap_not_support_fused_model(
        curr_stage_plan_states: List[MyExecPlanState], 
        check_gap: int, out_edge_dict: Dict[int, List[int]]):
    """
        If a model has no input model in this stage, check_in_gap is 1e9;
        If a model has no output model in this stage, check_out_gap is 1e9.
        OUTPUT:
            SHARED_CONTECT.check_in (deleted), SHARED_CONTECT.check_in_gap, SHARED_CONTECT.check_out_gap
        NOTE:
            all the models share the same fixed SHARED_CONTECT.check_in_gap;
            different models have different SHARED_CONTECT.check_out_gap which may change over stages
    """

    model_ids = [plan_state.exec_plan.model.model_id for plan_state in curr_stage_plan_states]
    for plan_state in curr_stage_plan_states:
        model_id = plan_state.exec_plan.model.model_id
        out_model_ids = out_edge_dict[model_id]
        if len(set(out_model_ids).intersection(model_ids)) > 0:
            SHARED_CONTECT.check_out_gaps[model_id] = check_gap
        else:
            SHARED_CONTECT.check_out_gaps[model_id] = int(1e9)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        





def set_check_in_out_gap(
        curr_stage_plan_states: List[MyExecPlanState], 
        check_gap: int, new_out_edge_dict: Dict[int, List[int]],
        model_id_shared_id_mapping: Dict[int, int]):
    """
        If a model has no input model in this stage, check_in_gap is 1e9;
        If a model has no output model in this stage, check_out_gap is 1e9.
        OUTPUT:
            SHARED_CONTECT.check_in (deleted), SHARED_CONTECT.check_in_gap, SHARED_CONTECT.check_out_gap
        NOTE:
            all the models share the same fixed SHARED_CONTECT.check_in_gap;
            different models have different SHARED_CONTECT.check_out_gap which may change over stages
            1. support fused models in the model system.
    """

    model_ids = [plan_state.exec_plan.model.model_id for plan_state in curr_stage_plan_states]
    for plan_state in curr_stage_plan_states:
        model_id = plan_state.exec_plan.model.model_id
        out_model_ids = new_out_edge_dict[model_id]
        shared_id = model_id_shared_id_mapping[model_id]
        if len(set(out_model_ids).intersection(model_ids)) > 0:
            SHARED_CONTECT.check_out_gaps[shared_id] = check_gap
        else:
            SHARED_CONTECT.check_out_gaps[shared_id] = int(1e9)



    



def _search_best_scheduling_with_another_process(
        test_cases:List[str], versions: List[str], max_token_nums: List[str], specify_outlens: List[bool],
        
        gen_execplans_baseline,
        search_method_baseline,
        model_paths, 
        
        out_edge_dict,
        check_gap, sort_input,
        num_prompts, 
        inp_seq_ids_dict, 
        out_req_id_mapping, inp_req_ids, 
        inp_req_from_which_model_which_out_reqs,
        
        independent_srcs,
        
        prompt_templates_lens, 
        
        gpu_name,
        byte_per_gpu,
        tot_gpu_num, 
        max_group_seq_num,
        top_k,
        similar_threshold,
        fully_connected_gpu_unit,
        machine_name,
):
    print(f"in running _search_best_scheduling_with_another_process")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(
                search_best_scheduling, 
                    test_cases, versions, max_token_nums, specify_outlens,
                    gen_execplans_baseline,
                    search_method_baseline,
                    model_paths, 
                    
                    out_edge_dict,
                    check_gap, sort_input,
                    num_prompts, 
                    inp_seq_ids_dict, 
                    out_req_id_mapping, inp_req_ids, 
                    inp_req_from_which_model_which_out_reqs,
                    
                    independent_srcs,
                    
                    prompt_templates_lens, 
                    
                    gpu_name,
                    byte_per_gpu,
                    tot_gpu_num, 
                    max_group_seq_num,
                    top_k,
                    similar_threshold,
                    fully_connected_gpu_unit,
                    machine_name)
            done, not_done = wait([future])
            plan_state_group_list = list(done)[0].result()
            return plan_state_group_list

        except Exception as e:
            print(f"Exception in running start_a_model_inference: {e}")
            print(traceback.format_exc())



async def main_with_preemption(
        main_args,
        test_cases:List[str], versions: List[str], max_token_nums: List[str], specify_outlens: List[bool],
        
        model_paths:List[str],
        gen_execplans_baseline:str,
        search_method_baseline:str,
        
        in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]],
        node_dataset_chunk_mapping: Dict[int, Tuple[str, int, int]],
        check_gap: int, sort_input: bool,
        num_prompts: int, 
        sampling_args_dict: Dict[int, SamplingParams],
        seq_outlen_dict: Dict[int, Dict[int,int]],
        
        inp_seq_ids_dict, 
        inp_req_ids, 
        inp_req_from_which_model_which_out_reqs,
        
        out_req_id_mapping, new_out_req_part_num, independent_srcs,
        prompt_template_args: Dict[int, Tuple],
        
        inp_generator, inp_merger, outlen_generator,
        prompt_templates_lens, 
        
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num:int = 4,
        max_group_seq_num: float = float('inf'),
        top_k: float = float('inf'),
        similar_threshold: float=0.1,
        fully_connected_gpu_unit: int = 4,
        machine_name: str='machine1',
):
    
    print(f"fully_connected_gpu_unit: {fully_connected_gpu_unit}")

    import os
    os.environ['RUN_MULTI_MODEL'] = 'True'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'False'
    os.environ['COLLECT_TIME_LOG'] = 'False' 
    os.environ['MY_SORT_INPS'] = 'True' if sort_input else 'False'
    os.environ['GET_INP_FROM_COMMUNICATOR'] = 'True' 

    print(f"os.environ['CUDA_VISIBLE_DEVICES'] in main_with_preemption: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    gpu_order_we_set = None
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        gpu_order_we_set = list(range(tot_gpu_num))
    else:
        gpu_order_we_set = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(gpu_order_we_set) == tot_gpu_num, f"gpu_order_we_set: {gpu_order_we_set} should contain {tot_gpu_num} cards"


    model_driver_worker_gpu_i: Dict[int, int] = dict()

    
    

    loop = asyncio.get_running_loop()
    tasks = []

    
    
    
    
    out_edge_dict = get_out_edge_dict_from_in_edge_dict_with_inp_nodes(in_edge_dict_with_dummy_inp_nodes)
    

    
    plan_state_group_list:List[List[MyExecPlanState]] = _search_best_scheduling_with_another_process(
        test_cases, versions, max_token_nums, specify_outlens,
        gen_execplans_baseline,
        search_method_baseline,
        model_paths, 
        
        out_edge_dict,
        check_gap, sort_input,
        num_prompts, 
        inp_seq_ids_dict, 
        out_req_id_mapping, inp_req_ids, 
        inp_req_from_which_model_which_out_reqs,
        
        independent_srcs,
        
        prompt_templates_lens, 
        
        gpu_name,
        byte_per_gpu,
        tot_gpu_num = tot_gpu_num, 
        max_group_seq_num = max_group_seq_num,
        top_k = top_k,
        similar_threshold=similar_threshold,
        fully_connected_gpu_unit=fully_connected_gpu_unit,
        machine_name=machine_name)
    

    
    
    

    
    model_id_shared_id_mapping, model_dict, new_in_edge_dict_with_dummy_inp_nodes, new_out_edge_dict = \
        _get_model_sys_structure_from_selected_plan_group_seq(
            plan_state_group_list, in_edge_dict_with_dummy_inp_nodes, out_edge_dict,)

    
    new_model_num = len(model_id_shared_id_mapping)
    new_model_path_dict = {model_id: model.model_path for model_id, model in model_dict.items()}

    print(f"\nnew_in_edge_dict_with_dummy_inp_nodes: {new_in_edge_dict_with_dummy_inp_nodes}")
    print(f"new_out_edge_dict: {new_out_edge_dict}")
    print(f"model_id_shared_id_mapping: {model_id_shared_id_mapping}")
    print(f"new_model_path_dict: {new_model_path_dict}")

    print("\n\n\n\n\nfinish searching!\n\n\n\n\n", flush=True)
    
    
    

    launched_exec_plan_states, new_target_stage_i, candidate_exec_plan_states = initialize_SHARED_CONTECT(
        tot_gpu_num=tot_gpu_num, 
        check_gap=check_gap,
        plan_state_group_list=plan_state_group_list,
        model_driver_worker_gpu_i=model_driver_worker_gpu_i, 
        gpu_order_we_set=gpu_order_we_set,
        model_id_shared_id_mapping=model_id_shared_id_mapping,
        new_out_edge_dict=new_out_edge_dict,
        sampling_args_dict=sampling_args_dict,
        seq_outlen_dict=seq_outlen_dict,
        fully_connected_gpu_unit=fully_connected_gpu_unit)
    first_stage_model_ids = [exec_plan_state.exec_plan.model.model_id for exec_plan_state in launched_exec_plan_states]




    print(f"\nTIMESTAMP 1: {time.perf_counter()}\n")
    time_lists: List[float] = list()

    with MyManager() as manager:

        print(f"\nTIMESTAMP 2: {time.perf_counter()}\n")

        shared_id_2_base_model_ids_dict = {model_id_shared_id_mapping[model_id]:model.get_base_model_ids() for model_id, model in model_dict.items()}
        communicator: LLM_COMMUNICATOR = manager.Communicator(
            new_model_num, 
            in_edge_dict_with_dummy_inp_nodes,
            shared_id_2_base_model_ids_dict,
            inp_req_ids, inp_req_from_which_model_which_out_reqs, out_req_id_mapping, new_out_req_part_num, independent_srcs,
            prompt_template_args,
            )

        
        
        
        base_req_num_dict = init_prompts_for_the_model_system(communicator, node_dataset_chunk_mapping, in_edge_dict_with_dummy_inp_nodes,
                                                         num_prompts, inp_seq_ids_dict, model_path=model_paths[0])

        print(f"base_req_num_dict: {base_req_num_dict}")


        print(f"\nTIMESTAMP 3: {time.perf_counter()}\n")

        
        
        with ProcessPoolExecutor(max_workers=new_model_num) as executor:

            print(f"\nTIMESTAMP 4: {time.perf_counter()}\n")

            
            
            
            
            
            
            
            
            
            
            
            
            


            task_dict = dict() 

            
            for model_id, model_path in new_model_path_dict.items():
                print(f"init process for {model_id, model_path}")
                shared_id = model_id_shared_id_mapping[model_id]
                tot_req_num = sum([base_req_num_dict[base_model_id] for base_model_id in model_dict[model_id].get_base_model_ids()])
                tasks.append(
                    loop.run_in_executor(
                        executor, start_a_model_inference, 
                        communicator, query_use_vllm(model_path), ','.join([str(i) for i in gpu_order_we_set]), shared_id, model_path, 
                        get_return_str(new_out_edge_dict=new_out_edge_dict, model_id=model_id, model_path_dict=new_model_path_dict),
                        tot_req_num,
                    )        
                )

                task_dict[model_id] = tasks[-1]



            


            print(f"\nTIMESTAMP 5: {time.perf_counter()}\n")


            
            
            SHARED_CONTECT.wait_all_models_to_finish_preparation_before_init_LLM(shared_ids=range(new_model_num))
            
            
            print(f"[model_id_shared_id_mapping[_] for _ in first_stage_model_ids]: {[model_id_shared_id_mapping[_] for _ in first_stage_model_ids]}")
            SHARED_CONTECT.start_specific_models([model_id_shared_id_mapping[_] for _ in first_stage_model_ids])


            start = time.perf_counter()
            print(f"Outer iter start time ---abs: {start}")
            time_lists.append(start)

            pending_list = tasks
            model_schedule_iter = 0
            while len(pending_list) > 0:
                
                print(f"a new iteration==================", flush=True)
                print(f"MAIN PROCESS: {[str(plan_state) for plan_state in launched_exec_plan_states]}", flush=True)

                done_list, pending_list = await asyncio.wait(pending_list, return_when=asyncio.FIRST_COMPLETED)

                
                start_waiting = time.perf_counter()
                print(f"MAIN PROCESS: total time to launch processes (just the value of iter 0 is useful) {model_schedule_iter}: {start_waiting-start}s ---abs: {start_waiting}", flush=True)
                time_lists.append(start_waiting)


                
                


                
                try:
                    launched_exec_plan_states, candidate_exec_plan_states, model_ids_to_stop, new_launch, new_target_stage_i = \
                        get_the_next_round_exec_plan_schedule(
                            launched_exec_plan_states, candidate_exec_plan_states,
                            new_target_stage_i,
                            tot_gpu_num, plan_state_group_list,
                            model_driver_worker_gpu_i,
                            model_id_shared_id_mapping,
                            fully_connected_gpu_unit,
                        )
                except Exception as e:
                    print(f"Exception in running benchmark_throughput.main(): {e}")
                    print(traceback.format_exc())
                
                print(f"new_launch: {new_launch}")
                print(f"model_ids_to_stop: {model_ids_to_stop}, --abs: {time.perf_counter()}")
                
                
                


                
                
                SHARED_CONTECT.stop_specific_models([model_id_shared_id_mapping[_] for _ in model_ids_to_stop])

                
                print(len(done_list), len(pending_list))
                print(f"MAIN PROCESS: next iter plans: {[str(plan_state) for plan_state in launched_exec_plan_states]}")
                print(f"MAIN PROCESS: model_ids_to_stop: {model_ids_to_stop}")
                print(f"MAIN PROCESS: new_launch: {[str(plan_state) for plan_state in new_launch]}")
                print(f"MAIN PROCESS: candidate_exec_plan_states: {[str(plan_state) for plan_state in candidate_exec_plan_states]}")
                
                for task in done_list:
                    await task

                
                
                
                
                
                model_ids_to_stop = [model_id for model_id in model_ids_to_stop if not task_dict[model_id].done()]
                print(f"new model_ids_to_stop: {model_ids_to_stop}, --abs: {time.perf_counter()}")

                SHARED_CONTECT.wait_all_models_to_finish_prepare_for_reschedule([model_id_shared_id_mapping[_] for _ in model_ids_to_stop])
                
                
                set_check_in_out_gap(
                    curr_stage_plan_states=launched_exec_plan_states, check_gap=check_gap, new_out_edge_dict=new_out_edge_dict,
                    model_id_shared_id_mapping=model_id_shared_id_mapping)
                start_exec_plans(new_launch, tot_gpu_num, gpu_order_we_set=gpu_order_we_set,
                                 model_id_shared_id_mapping=model_id_shared_id_mapping)


                
                end_waiting = time.perf_counter()
                print(f"MAIN PROCESS: total waiting time in iter {model_schedule_iter}: {end_waiting-start_waiting}s ---abs: {end_waiting}")
                model_schedule_iter += 1

            end = time.perf_counter()
            print(f"total running time: {end-start}s ---abs: {end}")

            time_lists.append(end)
            print(f"all stage times: timestamps: {time_lists}")
            print(f"all stage times: time lengths: {np.diff(time_lists).tolist()}")


            output_len_file_name = get_output_len_file_name(main_args)
            with open(output_len_file_name, 'w') as f:
                outlen_dict = communicator.get_all_model_outputs()
                import json
                json.dump(outlen_dict, f)
                print(f"\n\n\n\n\n\noutlen_dict={outlen_dict}\n")



            non_computation_ranges = communicator.get_non_compute_ranges()
            print(f"\n\n\n\n\n\nnon_computation_ranges={dict(non_computation_ranges)}\n")







def get_output_len_file_name(args):
    """
        This method returns the output len file name for a given exp setting.
    """
    
    
    
    
    
    
    
    
    

    file_name = str()
    if args.test_case == 'chain-summary':
        summarize_model_setting='vicuna-13b-v1.5'
        if args.summarize_model == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            summarize_model_setting='Mixtral-8x7B-Instruct-v0.1'
        
        file_name = f'./test_end2end_schedule/outlen_files/test-booookscore_{summarize_model_setting}_{args.evaluator_num}eval_maxlen_{args.max_token_num}_{args.reqnum}_ind{args.test_id}.json'
    
    elif args.test_case == 'general':
        file_name = f'./test_end2end_schedule/outlen_files/test-llm-blender_maxlen_{args.max_token_num}_{args.reqnum}_ind{args.test_id}.json'
    
    elif args.test_case == 'router':
        outlen_file_name_setting='maxlen_4096'
        if args.specify_outlen:
            outlen_file_name_setting='setOutlen'
        file_name = f'./test_end2end_schedule/outlen_files/test-router_{args.router_question_version}_{outlen_file_name_setting}_{args.reqnum}_{args.router_replicate_num}_ind{args.test_id}.json'


    elif args.test_case == 'mixed':
        summarize_model_setting='vicuna-13b-v1.5'
        if args.summarize_model == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            summarize_model_setting='Mixtral-8x7B-Instruct-v0.1'
        
        file_name = f'./test_end2end_schedule/outlen_files/test-mixed_{summarize_model_setting}_{args.evaluator_num}eval_maxlen_{args.max_token_num}_{args.reqnum}_maxlen_{args.max_token_num_mixed_blender}_{args.reqnum_mixed_blender}_ind{args.test_id}.json'

    return file_name






   
    
        
    
    
    
    
    
    
    
    
    


    
    
    
    


    
    

    
    
    
    
    
    

    

    
    
    

    
    
    
    
    
    
            
    
    


    
    
    
    

    
    



















def get_tot_latency_from_log(filename: str):
    tot = 0
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            if 'proc_output_time:' in line:
                pos = line.find('proc_output_time:')
                v = float(line[pos+len('proc_output_time:'):])
                tot += v 
    return tot





def _get_document_prompts(
        dataset_path: str, model_path: str, num_requests: int
        ) -> List[List[int]]:
    """
        NOTE: 
            1. Currently, we do not sort the input chunks in this case.
            2. The dataset in this function only has prompts, i.e., no output texts.
        Output:
            1. list of prompt token ids
    """
    from benchmark_throughput import get_dataset
    import random
    dataset = get_dataset(dataset_path=dataset_path)

    
    args = InferenceArgs(model=model_path, num_prompts=1)

    random.seed(args.seed)

    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    
    prompts = dataset
    prompt_token_ids = tokenizer(prompts).input_ids

    tokenized_dataset = prompt_token_ids

    
    filtered_dataset: List[List[int]] = tokenized_dataset

    
    
    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))

    
    


    print(f"tot_tokens: {sum([len(x) for x in sampled_requests])}, tot_context_lens: {sum([(len(x)-1)*len(x)/2 for x in sampled_requests])}")


    return sampled_requests





def get_token_num(model_path: str, seqs: List[str]) -> List[int]:
    args = InferenceArgs(model=model_path, num_prompts=1)
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    token_ids = tokenizer(seqs).input_ids
    token_nums = [len(tokens) for tokens in token_ids]
    return token_nums










def get_inplens(req_num: int, model_id: int, model_path: str, inp_seq_ids: List[int]):
    import json
    inp_lens = list()
    with open("./my_dummy_requests/my_dummy_requests.json", 'r') as file:
        prompts = json.load(file)
        prompts = [prompts[seq_id][0] for seq_id in inp_seq_ids]
        inp_lens = get_token_num(model_path, prompts)
        
        
        
        
        
        
        
    
    
    print(f"len(inp_lens):{len(inp_lens)}, inp_seq_ids:{inp_seq_ids}")
    return list(inp_lens)
    




def get_template_chain_summary():
    
    init_template = '''Below is the beginning part of a story:

---

{}

---

We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot. Write a summary for the excerpt provided above, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, and motivations. You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary. The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and chronological narrative. Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go. The summary could include multiple paragraphs.

Summary:'''
    
    
    intermediate_template = '''Below is a segment from a story:

---

{}

---

Below is a summary of the story up until this point:

---

{}

---

We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot. You are required to update the summary to incorporate any new vital information in the current excerpt. This information may relate to key events, backgrounds, settings, characters, their objectives, and motivations. You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary. The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and chronological narrative. Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go. The updated summary could include multiple paragraphs.

Updated summary:'''

    
    evaluate_template = ''' Is the summary easy to understand and free of grammatical errors? Format your response in JSON, containing a 'yes' or 'no' decision, along with justifications.

Summary: {}'''

    return init_template, intermediate_template, evaluate_template




























        









    









def get_inplens_chain_summary(req_num: int, model_id: int, model_path: str, inp_seq_ids: List[int]):
    """
        We do not consider the template length in the input length computation here.
    """
    import json
    inp_lens = list()
    with open("./my_dummy_requests/my_dummy_requests.json", 'r') as file:
        
        prompts = json.load(file)

        max_chunk_num = len(prompts[0])
        if model_id < max_chunk_num:
            
            prompts = [prompts[seq_id][model_id] for seq_id in inp_seq_ids]
        else:
            
            
            
            prompts = ['']*len(inp_seq_ids)
        
        
        inp_lens = get_token_num(model_path, prompts)
        
        
        
        
        
        
        
    
    
    print(f"len(inp_lens):{len(inp_lens)}, inp_seq_ids:{inp_seq_ids}")
    return inp_lens
    
    return [2048 for i in inp_lens]



































    






















    






def get_outlens_router_bench(model_id: int, model_name: str, inp_lens: List[int]):
    import json
    out_lens = list()
    with open('./my_dummy_requests/my_dummy_requests_outlen.json', 'r') as f:
        outlen_dict = json.load(f)
        out_lens = outlen_dict[str(model_id)]
        

    print(f"len(out_lens):{len(out_lens)}")
    assert len(out_lens) == len(inp_lens), f"model_id: {model_id}, inp_lens: {inp_lens}, out_lens: {out_lens}"
    return out_lens





def _get_req_len_funcs(test_case:str, version:str, max_token_num: int, specify_outlen: bool):
    inp_generator, inp_merger, outlen_generator = None, None, None
    if test_case == 'router':
        inp_generator = get_inplens 
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] 
        outlen_generator = None
        if max_token_num != None:
            outlen_generator = lambda model_id, model_name, inp_lens: np.minimum(max_token_num, output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens))
        else:
            outlen_generator = lambda model_id, model_name, inp_lens: output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens)
        
        
        
        
        if specify_outlen:
            
            
            outlen_generator = get_outlens_router_bench

    elif test_case == 'general':
        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] 
        
        outlen_generator = lambda model_id, model_name, inp_lens: output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens)
        if max_token_num != None:
            outlen_generator = lambda model_id, model_name, inp_lens: np.minimum(max_token_num, output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens))
    
        if specify_outlen:
            
            
            outlen_generator = get_outlens_router_bench
    
    elif test_case == 'map-reduce':
        chunk_size = 512
        fixed_output_size = 50
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] 
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))
    elif test_case == 'chain-summary':
        inp_generator = get_inplens_chain_summary
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] 
        
        outlen_generator = lambda model_id, model_name, inp_lens: output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens)
        if max_token_num != None:
            outlen_generator = lambda model_id, model_name, inp_lens: np.minimum(max_token_num, output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens))


        
        def _execpted_outlen_generator(model_id: int, model_name: str, inp_lens: List[int]):
            
            repeat_num = 100
            repeated_inp_lens = np.repeat(inp_lens, repeat_num)
            func = lambda model_id, model_name, inp_lens: output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens)
            if max_token_num != None:
                func = lambda model_id, model_name, inp_lens: np.minimum(max_token_num, output_length_sampler.sample_out_len_for_given_model(model_name, inp_lens))

            repeated_out_lens = func(model_id, model_name, repeated_inp_lens)
            out_lens = repeated_out_lens.reshape((-1, repeat_num))
            out_lens = np.mean(out_lens, axis=1).astype(int)
            return out_lens
        outlen_generator = _execpted_outlen_generator



        if specify_outlen:
            
            
            outlen_generator = get_outlens_router_bench

        
        
        
        
        
        
        
        

    return inp_generator, inp_merger, outlen_generator















def get_output_lens_from_existing_file(args, model_paths, inp_seq_ids_dict)->Tuple[Dict[int, List[int]], Dict[int, Dict[int, int]]]:
    """
        This method reads output lengths for each model and each req from the existing file if any.
    """
    file_name = get_output_len_file_name(args)
    import json
    out_lens_dict = dict()
    seq_outlen_dict = dict() 
    try:
        with open(file_name, 'r') as f:
            seq_outlen_dict = json.load(f)
            
            converted = dict()
            for i, vs in seq_outlen_dict.items():
                converted[int(i)] = {int(req_id):outlen for req_id, outlen in vs.items()}
            seq_outlen_dict = converted
        
        
        for i in seq_outlen_dict:
            out_lens_dict[i] = [seq_outlen_dict[i][req_id] for req_id in inp_seq_ids_dict[i]]
    except FileNotFoundError as e:
        print(f"Exception in getting output lengths from existing file: \n{e}")
        print(f"seq_outlen_dict: {seq_outlen_dict}")
        print(f"out_lens_dict: {out_lens_dict}")

    return out_lens_dict, seq_outlen_dict







def _get_router_bench_data(main_args, version:str, max_token_num:int, specify_outlen: bool, reqnum: int):
    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    
    inp_req_ids = dict()
    inp_req_from_which_model_which_out_reqs = dict()
    
    
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    
    independent_srcs = dict()
    prompt_template_args: Dict[int, Tuple] = None
    sampling_args_dict = dict()

    
    prompt_templates_lens = dict()

    
    model_paths = [
        'meta-llama/Llama-2-70b-chat-hf',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'WizardLMTeam/WizardLM-13B-V1.2',
        'meta-llama/CodeLlama-34b-Instruct-hf',
        'mistralai/Mistral-7B-Instruct-v0.2',     
    ]

    
    in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}

    prompt_template_args = dict()
    for i, model_path in enumerate(model_paths):
        args = InferenceArgs(model=model_path)
        prompt_template_args[i] = (args.tokenizer, args.trust_remote_code)



    
    import json
    prompt_dict = None
    
    dataset_type = 'multiple_choice_question'
    dataset_type = 'not_multiple_choice_question'
    dataset_type = version
    
    with open(f'./router_bench_{dataset_type}_dataset_with_responses.json', 'r') as f:
        prompt_dict = json.load(f)
    
    tot_inp_prompts = list()

    inp_seq_ids_dict = dict()
    req_num = 0
    out_lens_dict = dict()
    seq_outlen_dict = dict() 
    for i, model_path in enumerate(model_paths):
        data = prompt_dict[model_path]
        prompts = data
        seq_outlen_dict[i] = dict()
        if isinstance(data[0], list):
            
            prompts = [record[0] for record in data]
            outs = [record[1] for record in data]
            
            outlens = get_token_num(model_path, outs)

            
            prompts = [prompt for prompt in prompts for rep_i in range(main_args.router_replicate_num)]
            outs = [out for out in outs for rep_i in range(main_args.router_replicate_num)]
            outlens = [outlen for outlen in outlens for rep_i in range(main_args.router_replicate_num)]

            model_name = model_path[model_path.find('/')+1:]
            out_lens_dict[i] = outlens 
            seq_outlen_dict[i] = {req_id:out_len for req_id, out_len in \
                 zip(range(req_num, req_num + len(prompts)), outlens)}
        
        inp_seq_ids_dict[i] = list(range(req_num, req_num + len(prompts)))
        inp_seq_ids_dict[-(i+1)] = list(range(req_num, req_num + len(prompts)))
        req_num+=len(prompts)
        tot_inp_prompts.extend(prompts)


    
    specify_outlen_in_inference: bool = specify_outlen
    if (specify_outlen == False) and (main_args.test_case!='mixed'):
        tmp_out_lens_dict, tmp_seq_outlen_dict = get_output_lens_from_existing_file(main_args, model_paths, inp_seq_ids_dict)
        if len(tmp_out_lens_dict) > 0:
            
            out_lens_dict = tmp_out_lens_dict
            seq_outlen_dict = tmp_seq_outlen_dict
            specify_outlen_in_inference = True
            print(f"out_lens_dict: {out_lens_dict}")
            print(f"seq_outlen_dict: {seq_outlen_dict}")



    print(f"new inp_seq_ids_dict: ")
    for i, v in inp_seq_ids_dict.items():
        print(f"model {i}, ratio: {len(v)/req_num}")


    
    
    
    
    
    
    max_tokens = 4096 
    max_tokens = max_token_num
    inp_generator, inp_merger, outlen_generator = _get_req_len_funcs(
        'router', dataset_type, max_tokens, specify_outlen)
    
    node_dataset_chunk_mapping = {-(i+1): (None, 0, -1) \
                                    for i in range(len(model_paths))}


    
    tot_inp_prompts = [[_] for _ in tot_inp_prompts] 
    _init_dummy_requests(None, tot_inp_prompts, out_lens_dict)


    
    independent_srcs = {i:False for i in range(len(model_paths))}
    
    
    
    sampling_args2 = {                    
        "n":1,
        
        "temperature":1.0, 
        "top_p":1.0,
        "use_beam_search":False,
        "ignore_eos":False if not specify_outlen_in_inference else True, 
        "max_tokens":max_tokens, 
        }
    sampling_args_dict = {base_model_id:{0: SamplingParams(**sampling_args2)} for base_model_id in range(len(model_paths))}

    if specify_outlen_in_inference:
        
        get_sampling_args = lambda _max_tokens: {                  
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False if not specify_outlen_in_inference else True, 
            "max_tokens":_max_tokens, 
            }
        sampling_args_dict = {base_model_id: {req_id: SamplingParams(**get_sampling_args(outlen)) for req_id, outlen in vs.items()}\
                            for base_model_id, vs in seq_outlen_dict.items()}


    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
    print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")
    print(f"\nsampling_args_dict: {sampling_args_dict}\n")
    check_gap = 16
    sort_input = True


    
    check_gap = 16*100

    specify_outlens = [specify_outlen]*len(model_paths)
    if main_args.outlen_known:
        specify_outlens = [True]*len(model_paths)

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict, \
                    ['router']*len(model_paths), [version]*len(model_paths), [max_token_num]*len(model_paths), specify_outlens 









def _get_llm_blender_data(main_args, version:str, max_token_num:int, specify_outlen: bool, reqnum: int):
    """
        Get the real data from MixInstruct of LLM-Blender.
    """

    print(f"running in _get_llm_blender_data")


    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    
    inp_req_ids = dict()
    inp_req_from_which_model_which_out_reqs = dict()
    
    
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    
    independent_srcs = dict()
    prompt_template_args: Dict[int, Tuple] = None
    sampling_args_dict = dict()

    
    prompt_templates_lens = dict()

    
    model_paths = [
        'lmsys/vicuna-13b-v1.5',
        'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
        'chavinlo/alpaca-13b',
        'project-baize/baize-v2-13b',
        'TheBloke/koala-13B-HF',
        'databricks/dolly-v2-12b',
        'mosaicml/mpt-7b-chat',
        'THUDM/chatglm3-6b',
        'stabilityai/stablelm-tuned-alpha-7b',
    ]

    
    in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}

    prompt_template_args = dict()
    for i, model_path in enumerate(model_paths):
        args = InferenceArgs(model=model_path)
        prompt_template_args[i] = (args.tokenizer, args.trust_remote_code)



    
    import json
    tot_inp_prompts = list()
    inp_seq_ids_dict = dict()
    req_num = 10000
    req_num = reqnum
    out_lens_dict = dict() 
    seq_outlen_dict = dict() 

    
    
    '''
    models_to_keep = {
        'vicuna-13b-1.1':'eachadea/vicuna-13b-1.1', 
        'stablelm-tuned-alpha-7b':'stabilityai/stablelm-tuned-alpha-7b', 
        'koala-7B-HF':'TheBloke/koala-7B-HF', 
        'dolly-v2-12b':'databricks/dolly-v2-12b', 
        'chatglm-6b':'THUDM/chatglm3-6b', 
        'oasst-sft-4-pythia-12b-epoch-3.5':'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', 
        'llama-7b-hf-baize-lora-bf16':'huggyllama/llama-7b',  
        'mpt-7b-instruct':'mosaicml/mpt-7b-instruct', 
        'alpaca-native':'chavinlo/alpaca-native',
    }
    '''
    
    tot_outs = dict()
    with open(f'./train_data_prepared.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            record = json.loads(line)
            prompt = record['instruction'] + record['input']
            tot_inp_prompts.append(prompt)
            
            for cand in record['candidates']:
                if cand['model'] not in tot_outs:
                    tot_outs[cand['model']] = list()
                tot_outs[cand['model']].append(cand['text'])

    
    '''
    from transformers import AutoTokenizer
    for model, outs in tot_outs.items():
        if model not in models_to_keep:
            continue
        tokenizer = AutoTokenizer.from_pretrained(
            models_to_keep[model], trust_remote_code=True)
        outs_token_ids = tokenizer(outs).input_ids
        out_lens_dict[model] = [len(response) for response in outs_token_ids]


    max([max(outlens) for outlens in out_lens_dict.values()]) 
    [sum(outlens)/len(outlens) for outlens in out_lens_dict.values()] 

    '''

    
    import random
    random.seed(0)
    tot_inp_prompts = random.sample(tot_inp_prompts, min(req_num, len(tot_inp_prompts)))

    print(f"len(tot_inp_prompts): {len(tot_inp_prompts)}")

    inp_seq_ids_dict = {i: list(range(len(tot_inp_prompts))) for i in range(len(model_paths))}
    inp_seq_ids_dict.update({-(i+1): list(range(len(tot_inp_prompts))) for i in range(len(model_paths))})


    assert specify_outlen == False
    specify_outlen_in_inference: bool = specify_outlen
    
    if (specify_outlen == False) and (main_args.test_case!='mixed'):
        tmp_out_lens_dict, tmp_seq_outlen_dict = get_output_lens_from_existing_file(main_args, model_paths, inp_seq_ids_dict)
        if len(tmp_out_lens_dict) > 0:
            
            out_lens_dict = tmp_out_lens_dict
            seq_outlen_dict = tmp_seq_outlen_dict
            specify_outlen_in_inference = True
            print(f"out_lens_dict: {out_lens_dict}")
            print(f"seq_outlen_dict: {seq_outlen_dict}")




    
    
    
    
    
    
    max_tokens = 4096 
    max_tokens = max_token_num
    
    dataset_type = None
    inp_generator, inp_merger, outlen_generator = _get_req_len_funcs(
        'general', dataset_type, max_tokens, specify_outlen)
    
    node_dataset_chunk_mapping = {-(i+1): (None, 0, -1) \
                                    for i in range(len(model_paths))}


    
    tot_inp_prompts = [[_] for _ in tot_inp_prompts] 
    _init_dummy_requests(None, tot_inp_prompts, out_lens_dict)


    
    independent_srcs = {i:False for i in range(len(model_paths))}
    
    
    
    sampling_args2 = {                    
        "n":1,
        
        "temperature":1.0, 
        "top_p":1.0,
        "use_beam_search":False,
        "ignore_eos":False if not specify_outlen_in_inference else True, 
        "max_tokens":max_tokens, 
        }
    sampling_args_dict = {base_model_id:{0: SamplingParams(**sampling_args2)} for base_model_id in range(len(model_paths))}

    if specify_outlen_in_inference:
        
        get_sampling_args = lambda _max_tokens: {                  
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False if not specify_outlen_in_inference else True, 
            "max_tokens":_max_tokens, 
            }
        sampling_args_dict = {base_model_id: {req_id: SamplingParams(**get_sampling_args(outlen)) for req_id, outlen in vs.items()}\
                            for base_model_id, vs in seq_outlen_dict.items()}


    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
    print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")
    print(f"\nsampling_args_dict: {sampling_args_dict}\n")

    check_gap = 16
    sort_input = True


    
    check_gap = 16*100

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict, \
                    ['general']*len(model_paths), [version]*len(model_paths), [max_token_num]*len(model_paths), [specify_outlen]*len(model_paths)










def _get_booookscore_data(main_args, version:str, max_token_num:int, specify_outlen: bool, reqnum: int, 
                          evaluator_num: int, summarize_model: str, evaluator_model: str):
    """
        Get the real data from the documents chunked by BooookScore.
        Include the data from Arxiv, booookscore, and booksum.
        NOTE:
            the prompt template will not be considered when we prepare the input, 
            i.e., (1) comp inp len for search and (2) add inp req to the dummy inp nodes for scheduling.
    """

    print(f"running in _get_booookscore_data")


    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    
    inp_req_ids = dict()
    inp_req_from_which_model_which_out_reqs = dict()
    
    
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    
    independent_srcs = dict()
    prompt_template_args: Dict[int, Tuple] = None
    sampling_args_dict = dict()

    
    prompt_templates_lens = dict()


    
    tot_inp_prompts = list()
    inp_seq_ids_dict = dict()
    
    req_num = 50 
    req_num = reqnum
    out_lens_dict = dict() 
    seq_outlen_dict = dict() 


    
    data_files = [
        'BooookScore/all_books_chunked_2048.pkl', 
        
        'BooookScore/booksum_data_chunked_2048.pkl'
        ]
    
    print(f"data_files: {data_files}")

    chunk_lists = list() 
    import pickle
    for data_file in data_files:
        with open(data_file, 'rb') as file:
            data = pickle.load(file)
            chunk_lists.extend(list(data.values()))
    
    tot_inp_prompts = chunk_lists

    
    import random
    random.seed(0)
    req_num = min(req_num, len(tot_inp_prompts))
    tot_inp_prompts = random.sample(tot_inp_prompts, req_num)
    tot_inp_prompts = sorted(tot_inp_prompts, key=lambda chunks: len(chunks), reverse=True)

    print(f"len(tot_inp_prompts): {len(tot_inp_prompts)}")


    
    
    evaluate_item_num = 10
    evaluate_item_num = evaluator_num

    max_chunk_num = len(tot_inp_prompts[0])
    chunk_nums = np.asarray([len(chunks) for chunks in tot_inp_prompts])
    print(f"chunk_nums: {list(chunk_nums)}")
    inp_seq_ids_dict = {i: list(range(sum(chunk_nums > i))) for i in range(max_chunk_num)}
    for i in range(max_chunk_num):
        inp_seq_ids_dict[-(i+1)] = inp_seq_ids_dict[i].copy()
    inp_seq_ids_dict[max_chunk_num] = list(range(req_num*evaluate_item_num))


    
    

    inp_req_from_which_model_which_out_reqs = {max_chunk_num: dict()}
    for i in range(max_chunk_num-1):
        inp_req_from_which_model_which_out_reqs[max_chunk_num][i] = \
            {out_i*evaluate_item_num+inp_i: out_i \
                for inp_i in range(evaluate_item_num) \
                    for out_i in sorted(set(inp_seq_ids_dict[i])-set(inp_seq_ids_dict[i+1]))}
    inp_req_from_which_model_which_out_reqs[max_chunk_num][max_chunk_num-1] = \
            {out_i*evaluate_item_num+inp_i: out_i \
                for inp_i in range(evaluate_item_num) \
                    for out_i in inp_seq_ids_dict[max_chunk_num-1]}


    in_edge_dict_with_dummy_inp_nodes = {0:[-1]}
    in_edge_dict_with_dummy_inp_nodes.update({i:[-(i+1), i-1] for i in range(1, max_chunk_num)})
    in_edge_dict_with_dummy_inp_nodes[max_chunk_num] = list(range(max_chunk_num)) 

    
    summ_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    
    
    evaluate_model = 'meta-llama/Llama-2-70b-chat-hf'
    

    summ_model = summarize_model
    evaluate_model = evaluator_model
    
    
    init_template, intermediate_template, evaluate_template = get_template_chain_summary()
    prompt_templates = [init_template] + [intermediate_template]*(max_chunk_num-1) + [evaluate_template]
    prompt_templates_lens = get_token_num(summ_model, prompt_templates[:-1]) + \
        get_token_num(evaluate_model, prompt_templates[-1:])
    prompt_templates_lens = {model_id: l for model_id, l in enumerate(prompt_templates_lens)}
    
    
    
    
    
    prompt_template_args = dict()
    args = InferenceArgs(model=summ_model)
    for i in range(max_chunk_num):
        if i == 0:
            prompt_template_args[i] = (args.tokenizer, args.trust_remote_code, prompt_templates[i])
        else:
            prompt_template_args[i] = (args.tokenizer, args.trust_remote_code, prompt_templates[i], (-(i+1), i-1))
    args = InferenceArgs(model=evaluate_model)
    prompt_template_args[max_chunk_num] = (args.tokenizer, args.trust_remote_code, prompt_templates[max_chunk_num])
    
    
    
    model_paths = [summ_model]*max_chunk_num + [evaluate_model]



    assert specify_outlen == False
    specify_outlen_in_inference: bool = specify_outlen
    
    if (specify_outlen == False) and (main_args.test_case!='mixed'):
        tmp_out_lens_dict, tmp_seq_outlen_dict = get_output_lens_from_existing_file(main_args, model_paths, inp_seq_ids_dict)
        if len(tmp_out_lens_dict) > 0:
            
            out_lens_dict = tmp_out_lens_dict
            seq_outlen_dict = tmp_seq_outlen_dict
            specify_outlen_in_inference = True
            print(f"out_lens_dict: {out_lens_dict}")
            print(f"seq_outlen_dict: {seq_outlen_dict}")








    
    max_tokens = 4096 
    max_tokens = max_token_num 
    
    dataset_type = None
    inp_generator, inp_merger, outlen_generator = _get_req_len_funcs(
        'chain-summary', dataset_type, max_tokens, specify_outlen)
    
    
    node_dataset_chunk_mapping = {-(i+1): (None, i, -1) \
                                    for i in range(max_chunk_num)}




    
    _init_dummy_requests(None, tot_inp_prompts, out_lens_dict)


    
    independent_srcs = {i:False for i in range(max_chunk_num)}
    independent_srcs[max_chunk_num] = True
    
    sampling_args2 = {                    
        "n":1,
        
        "temperature":1.0, 
        "top_p":1.0,
        "use_beam_search":False,
        "ignore_eos":False if not specify_outlen_in_inference else True, 
        "max_tokens":max_tokens, 

        
        
        }
    sampling_args_dict = {base_model_id:{0: SamplingParams(**sampling_args2)} for base_model_id in range(len(model_paths))}

    if specify_outlen_in_inference:
        
        get_sampling_args = lambda _max_tokens: {                  
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False if not specify_outlen_in_inference else True, 
            "max_tokens":_max_tokens, 
            }
        sampling_args_dict = {base_model_id: {req_id: SamplingParams(**get_sampling_args(outlen)) for req_id, outlen in vs.items()}\
                            for base_model_id, vs in seq_outlen_dict.items()}


    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreq_num: {req_num}, evaluate_item_num: {evaluate_item_num}")
    print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
    print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")
    print(f"\nreal inp_req_ids: {inp_req_ids}\n")
    print(f"\nreal inp_req_from_which_model_which_out_reqs: {inp_req_from_which_model_which_out_reqs}\n")
    print(f"\nreal independent_srcs: {independent_srcs}\n")
    print(f"\nsampling_args_dict: {sampling_args_dict}\n")

    check_gap = 16
    sort_input = True


    
    check_gap = 16*100

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict, \
                    ['chain-summary']*len(model_paths), [version]*len(model_paths), [max_token_num]*len(model_paths), [specify_outlen]*len(model_paths)









































def _get_mixed_application_data(main_args, version:str, max_token_num:int, specify_outlen: bool, reqnum: int,
                          evaluator_num: int, summarize_model: str, evaluator_model: str, ):
    """
        Get the application data if we want to run more than 1 application together.
    """

    print(f"running in _get_mixed_application_data")


    
    model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict, \
                    test_cases1, versions1, max_token_nums1, specify_outlens1 = \
                    _get_booookscore_data(main_args, version, max_token_num, specify_outlen, reqnum,
                            evaluator_num, summarize_model, evaluator_model)
                    
                    


    
    import json
    out_lens_dict = dict()
    with open(f"./my_dummy_requests/my_dummy_requests.json", 'r') as f:
        requests = json.load(f)
    try:
        with open(f"./my_dummy_requests/my_dummy_requests_outlen.json", 'r') as f:
            _out_lens_dict = json.load(f)
            out_lens_dict.update(_out_lens_dict)
    except Exception as e:
        print(f"chain summary output len is not specified.")
    


    
    model_paths_2, check_gap_2, sort_input_2, in_edge_dict_with_dummy_inp_nodes_2, \
        req_num_2, inp_seq_ids_dict_2, inp_generator_2, inp_merger_2, outlen_generator_2, \
        prompt_templates_lens_2, node_dataset_chunk_mapping_2, \
        inp_req_ids_2, inp_req_from_which_model_which_out_reqs_2, \
            out_req_id_mapping_2, new_out_req_part_num_2, independent_srcs_2, prompt_template_args_2, \
                sampling_args_dict_2, seq_outlen_dict_2, \
                    test_cases2, versions2, max_token_nums2, specify_outlens2 = \
                    _get_llm_blender_data(main_args, None, max_token_num=main_args.max_token_num_mixed_blender, specify_outlen=False, reqnum=main_args.reqnum_mixed_blender)
                    


    with open(f"./my_dummy_requests/my_dummy_requests.json", 'r') as f:
        requests_2 = json.load(f)
        requests.extend(requests_2)
    try:
        with open(f"./my_dummy_requests/my_dummy_requests_outlen.json", 'r') as f:
            _out_lens_dict = json.load(f)
            out_lens_dict.update(_out_lens_dict)
    except Exception as e:
        print(f"blender output len is not specified.")
    


    
    

    
    app_model_nums = [len(model_paths), len(model_paths_2)]
    app_req_nums = [req_num, req_num_2]

    model_paths.extend(model_paths_2)

    in_edge_dict_with_dummy_inp_nodes.update({i+app_model_nums[0]:[-(i+app_model_nums[0]+1)] for i in in_edge_dict_with_dummy_inp_nodes_2})
    req_num = sum(app_req_nums)
    
    for i, seq_ids in inp_seq_ids_dict_2.items():
        if i >= 0:
            inp_seq_ids_dict[i+app_model_nums[0]] = np.asarray(seq_ids)+app_req_nums[0]
        else:
            inp_seq_ids_dict[i-app_model_nums[0]] = np.asarray(seq_ids)+app_req_nums[0]
    
    inp_generator_list = [inp_generator]*app_model_nums[0]+[inp_generator_2]*app_model_nums[1]
    inp_merger_list = [inp_merger]*app_model_nums[0]+[inp_merger_2]*app_model_nums[1]
    outlen_generator_list = [outlen_generator]*app_model_nums[0]+[outlen_generator_2]*app_model_nums[1]

    prompt_templates_lens.update({i+app_model_nums[0]:l for i, l in prompt_templates_lens_2.items()})
    node_dataset_chunk_mapping.update({i-app_model_nums[0]:j for i, j in node_dataset_chunk_mapping_2.items()})

    
    
    
    
    independent_srcs.update({i+app_model_nums[0]:j for i, j in independent_srcs_2.items()})
    prompt_template_args.update({i+app_model_nums[0]:j for i, j in prompt_template_args_2.items()})


    
    
    
    
    
    test_cases = test_cases1 + test_cases2
    versions = versions1 + versions2
    max_token_nums = max_token_nums1 + max_token_nums2
    specify_outlens = specify_outlens1 + specify_outlens2

    
    if main_args.outlen_known:
        specify_outlens = [True]*len(specify_outlens)

    
    
    

    for i, vs in seq_outlen_dict_2.items():
        seq_outlen_dict[i+app_model_nums[0]] = {j+app_req_nums[0]:k for j, k in vs.items()}
        
    assert specify_outlen == False
    specify_outlen_in_inference: bool = specify_outlen
    
    if specify_outlen == False:
        tmp_out_lens_dict, tmp_seq_outlen_dict = get_output_lens_from_existing_file(main_args, model_paths, inp_seq_ids_dict)
        
        
        if main_args.outlen_known:
            assert len(tmp_out_lens_dict) > 0
        
        if len(tmp_out_lens_dict) > 0:
            
            out_lens_dict = tmp_out_lens_dict
            seq_outlen_dict = tmp_seq_outlen_dict
            specify_outlen_in_inference = True
            print(f"out_lens_dict: {out_lens_dict}")
            print(f"seq_outlen_dict: {seq_outlen_dict}")

    
    _init_dummy_requests(None, requests, out_lens_dict)


    sampling_args2_chain_summary = {                    
        "n":1,
        
        "temperature":1.0, 
        "top_p":1.0,
        "use_beam_search":False,
        "ignore_eos":False if not specify_outlen_in_inference else True, 
        "max_tokens":max_token_num, 
        }
    sampling_args2_blender = {                    
        "n":1,
        
        "temperature":1.0, 
        "top_p":1.0,
        "use_beam_search":False,
        "ignore_eos":False if not specify_outlen_in_inference else True, 
        "max_tokens":main_args.max_token_num_mixed_blender, 
        }
    sampling_args_dict = {base_model_id:{0: SamplingParams(**sampling_args2_chain_summary)} for base_model_id in range(app_model_nums[0])}
    sampling_args_dict.update({base_model_id:{0: SamplingParams(**sampling_args2_blender)} for base_model_id in range(app_model_nums[0], app_model_nums[0]+app_model_nums[1])})

    if specify_outlen_in_inference:
        
        get_sampling_args = lambda _max_tokens: {                  
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False if not specify_outlen_in_inference else True, 
            "max_tokens":_max_tokens, 
            }
        sampling_args_dict = {base_model_id: {req_id: SamplingParams(**get_sampling_args(outlen)) for req_id, outlen in vs.items()}\
                            for base_model_id, vs in seq_outlen_dict.items()}




    print(f"\nreal model_paths: {model_paths}")
    print(f"\nreq_num: {req_num}, evaluate_item_num: {evaluator_num}")
    print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
    print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
    print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")
    print(f"\nreal inp_req_ids: {inp_req_ids}\n")
    print(f"\nreal inp_req_from_which_model_which_out_reqs: {inp_req_from_which_model_which_out_reqs}\n")
    print(f"\nreal independent_srcs: {independent_srcs}\n")
    
    print(f"\nreal seq_outlen_dict: {seq_outlen_dict}\n")
    print(f"\nreal prompt_templates_lens: {prompt_templates_lens}\n")




    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator_list, inp_merger_list, outlen_generator_list, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict, \
                    test_cases, versions, max_token_nums, specify_outlens

















def _get_schedule_setting_with_real_data(
        main_args,
        test_case: str, version:str, max_token_num:int, specify_outlen: bool,
        ratio_seed:int, ratio_set:int, reqnum: int,
        evaluator_num:int, summarize_model:str, evaluator_model:str):





    
    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    
    inp_req_ids = dict()
    inp_req_from_which_model_which_out_reqs = dict()
    
    
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    
    independent_srcs = dict()
    prompt_template_args: Dict[int, Tuple] = None
    sampling_args_dict = dict()

    
    prompt_templates_lens = list()


    if test_case == 'router':
        return _get_router_bench_data(main_args, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen, reqnum=reqnum)
        return _get_router_bench_data(version=version_router, max_token_num=max_token_num_router, specify_outlen=specify_outlen_router, reqnum=reqnum_router)
    elif (test_case == 'general'):
        return _get_llm_blender_data(main_args, version=None, max_token_num=max_token_num, specify_outlen=specify_outlen, reqnum=reqnum)
        return _get_llm_blender_data(version=None, max_token_num=max_token_num_general, specify_outlen=specify_outlen_general, reqnum=reqnum_general)
        
        model_paths = get_model_path_list()
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}
        req_num = 10000
        inp_seq_ids_dict = {i: list(range(req_num)) for i in range(len(model_paths))}
        if test_case == 'router':
            ratios = np.arange(1, len(model_paths)+1)
            if ratio_set == 2:
                ratios = np.asarray([2**i for i in range(len(model_paths)//2+1) for j in range(2)][:len(model_paths)])
            rng = np.random.default_rng(seed=ratio_seed)
            rng.shuffle(ratios)

            ratios = ratios/sum(ratios)
            cumnums = np.cumsum(np.concatenate(([0], (ratios*req_num).astype(int))))
            cumnums[-1] = req_num
            rand_seq_ids = np.arange(req_num)
            rng = np.random.default_rng(seed=0)
            rng.shuffle(rand_seq_ids)
            print(f"ratios: {ratios}, cumnums: {cumnums}, rand_seq_ids: {rand_seq_ids}")
            inp_seq_ids_dict = {i:sorted(rand_seq_ids[cumnums[i]:cumnums[i+1]]) for i in range(len(model_paths))}
            inp_seq_ids_dict.update({-(i+1):inp_seq_ids_dict[i] for i in range(len(model_paths))})
            print(f"new inp_seq_ids_dict: ")
            for i, v in inp_seq_ids_dict.items():
                print(f"model {i}, ratio: {ratios[i]} : {v}")

        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] 
        outlen_generator = output_length_sampler.sample_out_len_for_given_model
        node_dataset_chunk_mapping = {-(i+1): (None, 0, -1) \
                                      for i in range(len(model_paths))}



        
        
        args = InferenceArgs(model='NousResearch/Llama-2-7b-hf', num_prompts=req_num)
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
        requests = benchmark_throughput.sample_requests(
            "ShareGPT_V3_unfiltered_cleaned_split.json", args.num_prompts, tokenizer, args.output_len, 
            random_seed=args.seed)
        inp_prompts = [req[0] for req in requests]
        
        _init_dummy_requests(None, inp_prompts)



        independent_srcs = {i:False for i in range(len(model_paths))}

        sampling_args2 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, 
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args2) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
        print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")

    elif test_case == 'map-reduce':
        
        req_num = 10
        chunk_size = 512
        fixed_output_size = 50
        model_paths = ['NousResearch/Llama-2-13b-hf'] * 2
        
        in_edge_dict_with_dummy_inp_nodes = {0: [-1], 1:[0]}
        
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] 
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))
        node_dataset_chunk_mapping = {-1: (None, 0, chunk_size)}


        
        dataset_path = 'train-00000-of-00001-b334c773bce22cb2.parquet'
        sampled_inps: List[List[int]] = _get_document_prompts(dataset_path=dataset_path, model_path=model_paths[0], num_requests=req_num)
        inp_lens = np.asarray([len(prompt_token_ids) for prompt_token_ids in sampled_inps])
        req_num = min(req_num, len(inp_lens))

        
        out_req_id_mapping = {0: dict()}
        tot_req_num = 0
        inp_seq_ids_dict = {1:[]}
        sampled_inp_chunks = list()
        for i, inp_len in enumerate(inp_lens):
            chunk_num = (inp_len+chunk_size-1)//chunk_size
            out_req_id_mapping[0].update({chunk_i+tot_req_num:(i, chunk_i) for chunk_i in range(chunk_num) })
            tot_req_num += chunk_num
            inp_seq_ids_dict[1].append(tot_req_num-1)
            sampled_inp_chunks.extend([sampled_inps[i][chunk_i*chunk_size:(chunk_i+1)*chunk_size] for chunk_i in range(chunk_num)])

        inp_seq_ids_dict.update({0:list(out_req_id_mapping[0].keys())})
        


        new_out_req_part_num = { 0: { i:(inp_len+chunk_size-1)//chunk_size for i, inp_len in enumerate(inp_lens)} }
        independent_srcs = {i:False for i in range(len(model_paths))}

        
        _init_dummy_requests([chunk_size]*tot_req_num, sampled_inp_chunks)

        req_num = tot_req_num

        sampling_args1 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, 
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, 
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")



    elif test_case == 'chain-summary':
        return _get_booookscore_data(main_args, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen, reqnum=reqnum,
                                     evaluator_num=evaluator_num, summarize_model=summarize_model, evaluator_model=evaluator_model)
        return _get_booookscore_data(version=version_chainsumm, max_token_num=max_token_num_chainsumm, specify_outlen=specify_outlen_chainsumm, reqnum=reqnum_chainsumm,
                                     evaluator_num=evaluator_num, summarize_model=summarize_model, evaluator_model=evaluator_model)
        
        req_num = 1000
        chunk_size = 2048 
        fixed_output_size = 900 

        
        dataset_path = 'train-00000-of-00001-b334c773bce22cb2.parquet'
        model_path = 'NousResearch/Llama-2-13b-hf'
        
        sampled_inps: List[List[int]] = _get_document_prompts(dataset_path=dataset_path, model_path=model_path, num_requests=req_num)
        
        sampled_inps = sorted(sampled_inps, key=lambda i: len(i), reverse=True)
        inp_lens = np.asarray([len(prompt_token_ids) for prompt_token_ids in sampled_inps])
        req_num = min(req_num, len(inp_lens))

        print(f"inp_lens: {inp_lens}")

        max_length = max(inp_lens)

        print(f"max chunk num: {(max_length + chunk_size - 1) // chunk_size}")

        
        model_paths = [model_path] * ((max_length + chunk_size - 1) // chunk_size)
        print(f"model_paths: {model_paths}")
        
        
        
        in_edge_dict_with_dummy_inp_nodes = {0: [-1]}
        in_edge_dict_with_dummy_inp_nodes.update({i:[-(i+1)] + [i-1] for i in range(1, len(model_paths))})

        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] 
        outlen_generator = lambda model_name, inplens: np.asarray([fixed_output_size]*len(inplens))
        
        node_dataset_chunk_mapping = {-(i+1): (None, i, chunk_size)\
                                      for i in range(len(model_paths))}
        
        inp_seq_ids_dict = defaultdict(list)
        
        
        
        
        
        inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*i)))) for i in range(len(model_paths))})
        print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")
        

        
        
        model_paths.append('NousResearch/Llama-2-70b-hf')
        in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = list(range(len(model_paths)-1)) 
        
        
        
        
        inp_seq_ids_dict[len(model_paths)-1] = inp_seq_ids_dict[0]

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")

        
        
        
        
        inp_req_ids = {len(model_paths)-1: {i:sorted(set(inp_seq_ids_dict[i])-set(inp_seq_ids_dict[i+1])) for i in range(len(model_paths)-2)}}
        inp_req_ids[len(model_paths)-1][len(model_paths)-2] = inp_seq_ids_dict[len(model_paths)-2]
        independent_srcs[len(model_paths)-1] = True

        print(f"\nreal inp_req_ids: {inp_req_ids}\n")
        print(f"\nreal independent_srcs: {independent_srcs}\n")

        
        
        _init_dummy_requests(inp_lens, sampled_inps)
        sampling_args1 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, 
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, 
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths)-1)}
        sampling_args_dict.update({len(model_paths)-1:SamplingParams(**sampling_args1)})

    elif test_case == 'mixed':
        return _get_mixed_application_data(main_args, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen, reqnum=reqnum,
                                     evaluator_num=evaluator_num, summarize_model=summarize_model, evaluator_model=evaluator_model)
        
        
        
        


    
    
    
    
    check_gap = 16
    sort_input = True

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict






def get_schedule_setting(
        main_args,
        test_case:str, version:str, max_token_num:int, specify_outlen: bool,
        use_real_dataset:bool, ratio_seed:int, ratio_set:int, reqnum: int,
        evaluator_num:int, summarize_model:str, evaluator_model:str):






    if use_real_dataset:
        return _get_schedule_setting_with_real_data(
            main_args,
            test_case=test_case, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen,
            ratio_seed=ratio_seed, ratio_set=ratio_set, reqnum=reqnum,
            evaluator_num=evaluator_num, summarize_model=summarize_model, evaluator_model=evaluator_model)
        
        
        
        
        
        
    


    in_edge_dict_with_dummy_inp_nodes, inp_generator, inp_merger, outlen_generator, node_dataset_chunk_mapping = \
        None, None, None, None, None

    req_num = None
    inp_seq_ids_dict = None
    model_paths = None

    
    inp_req_ids = dict()
    inp_req_from_which_model_which_out_reqs = dict()
    
    
    out_req_id_mapping = dict()
    new_out_req_part_num = dict()
    
    
    independent_srcs = dict()
    prompt_template_args: Dict[int, Tuple] = None
    sampling_args_dict = dict()

    
    prompt_templates_lens = list()


    if test_case == 'general':
        
        model_paths = get_model_path_list()
        in_edge_dict_with_dummy_inp_nodes = {i:[-(i+1)] for i in range(len(model_paths))}
        req_num = 1000
        inp_seq_ids_dict = {i: list(range(req_num)) for i in range(len(model_paths))}
        inp_generator = get_inplens
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*inp_lists)] 
        outlen_generator = output_length_sampler.sample_out_len_for_given_model
        node_dataset_chunk_mapping = {-(i+1): ("ShareGPT_V3_unfiltered_cleaned_split.json", 0, -1) \
                                      for i in range(len(model_paths))}


        
        
        
        
        
        
        
        

        independent_srcs = {i:False for i in range(len(model_paths))}


        
        
        

        

        
        
        
        
        
        
        
        
        sampling_args2 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, 
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args2) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")
        print(f"node_dataset_chunk_mapping: {node_dataset_chunk_mapping}")

    elif test_case == 'map-reduce':
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


        
        req_num = 10
        chunk_size = 512
        model_paths = ['NousResearch/Llama-2-13b-hf'] * 2
        
        in_edge_dict_with_dummy_inp_nodes = {0: [-1], 1:[0]}
        
        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [512]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists[1:]))] 
        outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        node_dataset_chunk_mapping = {-1: (None, 0, chunk_size)}

        inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))

        
        out_req_id_mapping = {0: dict()}
        tot_req_num = 0
        inp_seq_ids_dict = {1:[]}
        for i, inp_len in enumerate(inp_lens):
            chunk_num = (inp_len+chunk_size-1)//chunk_size
            out_req_id_mapping[0].update({chunk_i+tot_req_num:(i, chunk_i) for chunk_i in range(chunk_num) })
            tot_req_num += chunk_num
            inp_seq_ids_dict[1].append(tot_req_num-1)

        inp_seq_ids_dict.update({0:list(out_req_id_mapping[0].keys())})
        


        new_out_req_part_num = { 0: { i:(inp_len+chunk_size-1)//chunk_size for i, inp_len in enumerate(inp_lens)} }
        independent_srcs = {i:False for i in range(len(model_paths))}

        
        _init_dummy_requests([chunk_size]*tot_req_num)

        req_num = tot_req_num

        sampling_args1 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, 
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, 
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths))}

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")


    elif test_case == 'chain-summary':
        
        
        
        
        
        
        
        
        
        
        
        
        


        req_num = 10
        chunk_size = 512
        max_length = chunk_size*50 
        model_paths = ['NousResearch/Llama-2-13b-hf'] * (max_length // chunk_size)
        print(f"model_paths: {model_paths}")
        
        
        
        in_edge_dict_with_dummy_inp_nodes = {0: [-1]}
        in_edge_dict_with_dummy_inp_nodes.update({i:[-(i+1)] + [i-1] for i in range(1, len(model_paths))})

        inp_generator = lambda req_num, model_path, inp_seq_ids_dict: [chunk_size]*req_num
        inp_merger = lambda inp_lists: [sum(i) for i in zip(*(inp_lists))] 
        outlen_generator = lambda model_name, inplens: np.asarray([50]*len(inplens))
        
        node_dataset_chunk_mapping = {-(i+1): (None, i, chunk_size)\
                                      for i in range(len(model_paths))}
        
        inp_seq_ids_dict = defaultdict(list)
        
        
        
        
        inp_lens = np.asarray([20*chunk_size]*int(0.8*req_num)+[50*chunk_size]*int(0.2*req_num))
        inp_seq_ids_dict.update({i:list(range(sum(inp_lens>(chunk_size*i)))) for i in range(len(model_paths))})
        print(f"inp_seq_ids_dict: {inp_seq_ids_dict}")
        

        
        model_paths.append('NousResearch/Llama-2-7b-hf')
        
        in_edge_dict_with_dummy_inp_nodes[len(model_paths)-1] = [19, 49]
        
        
        inp_seq_ids_dict[len(model_paths)-1] = sorted(set(inp_seq_ids_dict[len(model_paths)-2] + inp_seq_ids_dict[len(model_paths)-3]))

        print(f"\nreal model_paths: {model_paths}")
        print(f"\nreal in_edge_dict_with_dummy_inp_nodes: {in_edge_dict_with_dummy_inp_nodes}")
        print(f"\nreal inp_seq_ids_dict: {inp_seq_ids_dict}\n")

        
        
        inp_req_ids = dict()
        independent_srcs = {i:False for i in range(len(model_paths))}
        independent_srcs[len(model_paths)-1] = True

        
        _init_dummy_requests(inp_lens)
        sampling_args1 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":True, 
            "max_tokens":50}
        sampling_args2 = {                    
            "n":1,
            
            "temperature":1.0, 
            "top_p":1.0,
            "use_beam_search":False,
            "ignore_eos":False, 
            "max_tokens":int(1e9)}
        sampling_args_dict = {base_model_id:SamplingParams(**sampling_args1) for base_model_id in range(len(model_paths)-1)}
        sampling_args_dict.update({len(model_paths)-1:SamplingParams(**sampling_args1)})



    
    
    
    
    check_gap = 16
    sort_input = True

    return model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        req_num, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
        prompt_templates_lens, node_dataset_chunk_mapping, \
        inp_req_ids, inp_req_from_which_model_which_out_reqs, \
            out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                sampling_args_dict, seq_outlen_dict




if __name__ == "__main__":
    print(f"start: --abs: {time.perf_counter()}")
    

    parser = argparse.ArgumentParser(description="args of end 2 end test")
    parser.add_argument("--gen-execplans-baseline",
                        type=str,
                        choices=["ours", "naive", "max_gpu"],
                        default="ours")
    

    parser.add_argument("--search_method_baseline",
                        type=str,
                        
                        default="ours")


    parser.add_argument("--test-case",
                        type=str,
                        choices=["general", "map-reduce", "chain-summary", "router", "mixed"],
                        default="router")
    
    parser.add_argument("--ratio-seed",
                        type=int)    
    
    parser.add_argument("--ratio-set",
                        type=int)    
    

    parser.add_argument("--reqnum",
                        type=int)


    parser.add_argument("--router_replicate_num",
                        
                        type=int,
                        default=1)


    parser.add_argument("--router_question_version",
                        type=str, 
                        choices=['multiple_choice_question', 'not_multiple_choice_question'],
                        default='multiple_choice_question')
    

    parser.add_argument("--max_token_num",
                        type=int,
                        default=900)


    parser.add_argument('--specify_outlen', 
                        action='store_true')
    
    parser.add_argument('--outlen_known', 
                        action='store_true')

    parser.add_argument("--gpu_name",
                        type=str,
                        default='A100-80G')


    parser.add_argument("--byte_per_gpu",
                        type=int,
                        default=80*(1024**3))
    

    parser.add_argument("--tot_gpu_num",
                        type=int,
                        default=8)


    
    parser.add_argument("--max_group_seq_num",
                        type=int,
                        default=1)


    parser.add_argument("--top_k",
                        type=int,
                        default=20)

    parser.add_argument("--similar_threshold",
                        type=float,
                        default=0.2)
    

    parser.add_argument("--fully_connected_gpu_unit",
                        type=int,
                        default=2)
    

    parser.add_argument("--machine_name",
                        type=str,
                        choices=['machine1', 'machine2'],
                        default='machine2')
    

    parser.add_argument("--evaluator_num",
                        type=int,
                        default=5)    


    parser.add_argument("--summarize_model",
                        type=str,
                        
                        default='lmsys/vicuna-13b-v1.5')
    

    parser.add_argument("--evaluator_model",
                        type=str,
                        
                        default='meta-llama/Llama-2-70b-chat-hf')


    
    parser.add_argument("--reqnum_mixed_blender",
                        type=int)
    

    parser.add_argument("--max_token_num_mixed_blender",
                        type=int)


    
    parser.add_argument("--test_id",
                        type=int,
                        default=0)
    

    args = parser.parse_args()


    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


    
    
    gen_execplans_baseline = 'ours' 
    search_method_baseline = 'ours' 
    test_case = 'router' 
    version = 'multiple_choice_question' 
    max_token_num = 900 
    specify_outlen = False 

    gen_execplans_baseline = args.gen_execplans_baseline
    search_method_baseline = args.search_method_baseline
    test_case = args.test_case
    ratio_seed = args.ratio_seed
    ratio_set = args.ratio_set
    reqnum = args.reqnum
    version = args.router_question_version
    max_token_num = args.max_token_num
    specify_outlen = args.specify_outlen



    model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        num_prompts, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
            prompt_templates_lens, node_dataset_chunk_mapping, \
                inp_req_ids, inp_req_from_which_model_which_out_reqs, \
                    out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                        sampling_args_dict, seq_outlen_dict, \
                            test_case, version, max_token_num, specify_outlen = \
        get_schedule_setting(args, test_case=test_case, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen,
                             use_real_dataset=True, ratio_seed=ratio_seed, ratio_set=ratio_set, reqnum=reqnum,
                             evaluator_num=args.evaluator_num, summarize_model=args.summarize_model, evaluator_model=args.evaluator_model)

    
    test_cases, versions, max_token_nums, specify_outlens = [], [], [], []
    if isinstance(test_case, List):
        test_cases, versions, max_token_nums, specify_outlens = test_case, version, max_token_num, specify_outlen
    else:
        test_cases, versions, max_token_nums, specify_outlens = [test_case], [version], [max_token_num], [specify_outlen]

    asyncio.run(main_with_preemption(
        args,
        test_cases=test_cases, versions=versions, max_token_nums=max_token_nums, specify_outlens=specify_outlens,
        model_paths=model_paths,
        gen_execplans_baseline=gen_execplans_baseline,
        search_method_baseline=search_method_baseline,
        
        
        in_edge_dict_with_dummy_inp_nodes=in_edge_dict_with_dummy_inp_nodes,
        node_dataset_chunk_mapping=node_dataset_chunk_mapping,
        check_gap=check_gap, sort_input=sort_input,
        num_prompts=num_prompts, 
        
        sampling_args_dict=sampling_args_dict,
        seq_outlen_dict=seq_outlen_dict,
        
        inp_seq_ids_dict=inp_seq_ids_dict, 
        inp_req_ids=inp_req_ids, 
        inp_req_from_which_model_which_out_reqs=inp_req_from_which_model_which_out_reqs,
        
        out_req_id_mapping=out_req_id_mapping, 
        new_out_req_part_num=new_out_req_part_num, independent_srcs=independent_srcs,
        prompt_template_args=prompt_template_args,
        inp_generator=inp_generator, inp_merger=inp_merger, outlen_generator=outlen_generator,
        prompt_templates_lens=prompt_templates_lens, 
        
        gpu_name=args.gpu_name,  
        byte_per_gpu=args.byte_per_gpu,  
        tot_gpu_num=args.tot_gpu_num,  
        max_group_seq_num=args.max_group_seq_num,  
        top_k=args.top_k,  
        similar_threshold=args.similar_threshold,  
        
        fully_connected_gpu_unit=args.fully_connected_gpu_unit,  
        machine_name=args.machine_name,  
        ))
    


    
    
    
    
    
    
    
    
    
    
    
    


    

    
    
