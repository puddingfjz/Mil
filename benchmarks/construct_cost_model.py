'''
This file contains the code for us to get the cost model we need. 
Our cost model is based on loop-up tables.
Copy contents from benchmark_throughput.py
'''




import os
# temporarily comment this "os.environ['CUDA_VISIBLE_DEVICES']='2,3,0,1'" as we will set it before the running command
# os.environ['CUDA_VISIBLE_DEVICES']='2,3,0,1' # '2,3' # '3,0,1,2' # should be set before initialize cuda in torch
os.environ['USE_VLLM']='True'
# os.environ['TOT_GPU_NUM'] = '4' # should be consistent with os.environ['CUDA_VISIBLE_DEVICES']
# os.environ['WEIGHT_LOAD_DEGREE'] = '16' # now will set it in command
# os.environ['CHANGE_KV_LAYOUT'] = 'True' # whether the KV layout is changed
os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False' # whether we will dynamically increase the on-card layer weights


os.environ['RUN_MULTI_MODEL'] = 'False' # whether this model is running in a multi-model environment
os.environ['SOFT_RESCHEDULE'] = 'False' # whether to reinitialize LLMs directly or update the current LLM (i.e., soft reschedule)
os.environ['NO_PREEMPT'] = 'True' # allow model preemption or not
# about scheduling
os.environ['SORT_REQS'] = 'True' # whether to sort the requests according to their output lengths, default is False
os.environ['COLLECT_TIME_LOG'] = 'True' # whether to collect time log (related to flops metadata collection)

def environs_are_correct():
    if os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] == 'True':
        assert (os.environ['USE_VLLM'] == 'False')

# we first check the os environ variables are correct
environs_are_correct()
    









import argparse
import json
import random
import time
from typing import List, Optional, Tuple, Dict

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from tqdm import tqdm



from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


print(f'executing benchmark_throughput.py')





def get_dataset(dataset_path: str):
    if dataset_path == 'ShareGPT_V3_unfiltered_cleaned_split.json':
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [(data["conversations"][0]["value"],
                    data["conversations"][1]["value"]) for data in dataset]
        return dataset
    elif dataset_path == 'no_robot.parquet':
        # deal with other dataset
        import pyarrow.parquet as pq
        dataset = list()
        for fname in ['no_robot_train.parquet', 'no_robot_test.parquet']:
            a = pq.read_table(fname)
            a = a.to_pylist()
            dataset.extend([(data['messages'][0]['content'],
                             data['messages'][1]['content']) for data in a])
        return dataset
          





def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    # with open(dataset_path) as f:
    #     dataset = json.load(f)
    # # Filter out the conversations with less than 2 turns.
    # dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # # Only keep the first two turns of each conversation.
    # dataset = [(data["conversations"][0]["value"],
    #             data["conversations"][1]["value"]) for data in dataset]


    dataset = get_dataset(dataset_path)

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    # make sample size be ``min(num_requests, len(filtered_dataset))''
    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))

    if os.environ['SORT_REQS'] == 'True':
        sampled_requests = sorted(sampled_requests, key=lambda x: x[1], reverse=True)


    print(f"tot_tokens: {sum([x[1]+x[2] for x in sampled_requests])}, tot_context_lens: {sum([(x[1]+x[2]-1)*(x[1]+x[2])/2 for x in sampled_requests])}")

    return sampled_requests





def get_prompt_iter_worklods(llm, iter_workloads):
    '''
        Get the iter_workloads whose per_iter execution latencies we are going to collect for prompt stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    block_size = llm.llm_engine.cache_config.block_size
    max_token_num = len(kv_caches[0][0]) * block_size
    prompt_limit = min(llm.llm_engine.scheduler_config.max_model_len,
                    llm.llm_engine.scheduler_config.max_num_batched_tokens)


    # prepare iter workloads
    max_token_num = min(max_token_num, max_num_batched_tokens)
    is_prompt = True
    for seq_num in range(1, max_num_seqs+1):
    # for seq_num in [400]:
        max_max_seqlen = (max_token_num + seq_num - 1) // seq_num
        count = 0
        for max_seqlen in range(max_max_seqlen, 0, -1):
        # for tot_token_num in [max_token_num]:
            tot_token_num = min(seq_num * max_seqlen, max_token_num)
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num >= prompt_limit:
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)
            count += 1
            if count >= 2:
                break
        # collect for the smallest tot_token_num
        for max_seqlen in range(1, max_max_seqlen+1):
        # for tot_token_num in [max_token_num]:
            tot_token_num = min(seq_num * max_seqlen, max_token_num)
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num >= prompt_limit:
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)
            break     





def get_decode_iter_workloads(llm, iter_workloads):
    '''
        Get the iter_workloads whose per_iter execution latencies we are going to collect for decoding stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    block_size = llm.llm_engine.cache_config.block_size
    max_token_num = len(kv_caches[0][0]) * block_size
    max_model_len = llm.llm_engine.model_config.max_model_len

    # prepare iter workloads
    is_prompt = False
    for seq_num in range(1, max_num_seqs+1):
    # for seq_num in [199]:
        # for tot_token_num in range(max_token_num, 107, -108):
        for tot_token_num in range(max_token_num, 0, -128):
        # for tot_token_num in range(seq_num, max_token_num+1, 128):
        # for tot_token_num in range((max_token_num - seq_num)//128*128+seq_num, 0, -128):
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num >= max_model_len:
                # there must be a seq of length larger than max_model_len
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)
            break
        # collect for the smallest tot_token_num
        for tot_token_num in range(seq_num, max_token_num+1, 128):
        # for tot_token_num in range(seq_num, (max_token_num - seq_num)//128*128+seq_num+1, 128):
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num >= max_model_len:
                # there must be a seq of length larger than max_model_len
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)
            break




def get_interesting_points(start_rng, end_rng, profile_num, piecewise_cost_model_build_mode):
    '''
        The interesting points are 1/2, 3/4, 7/8, ... points in the range [start_rng, end_rng].
    '''
    ret = [start_rng]

    # if not piecewise linear function, but simple linear function, do not need to sample multiple interesting points
    if not piecewise_cost_model_build_mode:
        return ret
    
    while True:
        v = (start_rng+end_rng)//2
        if v == start_rng:
            break
        if v < (end_rng-profile_num+1):
            ret.append(v)
        start_rng = v
    return sorted(ret)



def get_max_token_num(
        max_num_seqs, tot_blk_num, block_size, is_prompt, prompt_max_num_batched_tokens, 
        max_seqlen):
    # 1. we first compute the max_tot_token_num and max_seq_num for the given ``model'' given the ``gpu mem utilization ratio''.
    # we need to comp max_num_batched_tokens, considering the available tot blk num
    max_num_seqs = min(max_num_seqs, tot_blk_num) # if the KV cache is too small that we cannot run max_num_seqs together
    max_num_batched_tokens = ((tot_blk_num//max_num_seqs)*block_size)*max_num_seqs + (tot_blk_num%max_num_seqs)

    if is_prompt:
        max_num_batched_tokens = min(max_num_batched_tokens, prompt_max_num_batched_tokens)
        # max_num_batched_tokens = 2048
        # max_num_seqs = 2
    # else:
    #     max_num_batched_tokens = min(max_num_batched_tokens, 112394)

    return min(max_num_batched_tokens, max_num_seqs*max_seqlen)



def get_prompt_iter_worklods_sample_and_prepInp(llm, iter_workloads, piecewise_cost_model_build_mode):
    '''
        NOTE: this function is used for collecting cost data for sampling and preparing input.
        
        Get the iter_workloads whose per_iter execution latencies we are going to collect for prompt stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    tot_blk_num = len(kv_caches[0][0])
    block_size = llm.llm_engine.cache_config.block_size
    # max_token_num = len(kv_caches[0][0]) * block_size
    prompt_limit = min(llm.llm_engine.scheduler_config.max_model_len,
                    llm.llm_engine.scheduler_config.max_num_batched_tokens)


    profile_num = 2

    # prepare iter workloads
    # max_token_num = min(max_token_num, max_num_batched_tokens)
    is_prompt = True
    for seq_num in range(1, min(max_num_seqs, tot_blk_num)+1):
    # for seq_num in [400]:
        # max_max_seqlen = (max_token_num + seq_num - 1) // seq_num
        max_token_num = get_max_token_num(
            seq_num, tot_blk_num, block_size, is_prompt, max_num_batched_tokens, prompt_limit)
        count = 0
        # for max_seqlen in range(max_max_seqlen, 0, -1):
        for tot_token_num in range(max_token_num, 0, -1): #-128):
        # for tot_token_num in [max_token_num]:
            # tot_token_num = min(seq_num * max_seqlen, max_token_num)
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num > prompt_limit:
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)
            iter_workloads['corr_context_tot_len'].append(max_token_num)
            count += 1
            if count >= profile_num:
                break
        # count = 0
        # collect for the smallest tot_token_num
        # for max_seqlen in range(1, max_max_seqlen+1):
        # for tot_token_num in [max_token_num]:
        for sample_point in get_interesting_points(seq_num, max_token_num, profile_num, piecewise_cost_model_build_mode):
            count = 0
            for tot_token_num in range(sample_point, max_token_num+1): #-128):
                # tot_token_num = min(seq_num * max_seqlen, max_token_num)
                if seq_num > tot_token_num:
                    continue
                if (tot_token_num + seq_num - 1) // seq_num > prompt_limit:
                    continue
                iter_workloads['is_prompt'].append(is_prompt)
                iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
                iter_workloads['set_max_num_seqs'].append(seq_num)
                iter_workloads['corr_context_tot_len'].append(sample_point)
                count += 1
                if count >= profile_num:
                    break





def get_decode_iter_workloads_sample_and_prepInp(llm, iter_workloads, piecewise_cost_model_build_mode):
    '''
        NOTE: this function is used for collecting cost data for sampling and preparing input.
        
        Get the iter_workloads whose per_iter execution latencies we are going to collect for decoding stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    tot_blk_num = len(kv_caches[0][0])
    block_size = llm.llm_engine.cache_config.block_size
    # max_token_num = len(kv_caches[0][0]) * block_size
    max_model_len = llm.llm_engine.model_config.max_model_len

    profile_num = 2

    # prepare iter workloads
    is_prompt = False
    for seq_num in range(1, min(max_num_seqs, tot_blk_num)+1):
    # for seq_num in [199]:
        max_token_num = get_max_token_num(
            seq_num, tot_blk_num, block_size, is_prompt, max_num_batched_tokens, max_model_len)
        # for tot_token_num in range(max_token_num, 107, -108):
        count = 0
        for tot_token_num in range(max_token_num, 0, -1): #-128):
        # for tot_token_num in range(seq_num, max_token_num+1, 128):
        # for tot_token_num in range((max_token_num - seq_num)//128*128+seq_num, 0, -128):
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num > max_model_len:
                # there must be a seq of length larger than max_model_len
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)
            iter_workloads['corr_context_tot_len'].append(max_token_num)
            # break
            count += 1
            if count >= profile_num:
                break
        # count = 0
        # # collect for the smallest tot_token_num
        for sample_point in get_interesting_points(seq_num, max_token_num, profile_num, piecewise_cost_model_build_mode):
            count = 0
            for tot_token_num in range(sample_point, max_token_num+1): #, 128):
            # for tot_token_num in range(seq_num, (max_token_num - seq_num)//128*128+seq_num+1, 128):
                if seq_num > tot_token_num:
                    continue
                if (tot_token_num + seq_num - 1) // seq_num > max_model_len:
                    # there must be a seq of length larger than max_model_len
                    continue
                iter_workloads['is_prompt'].append(is_prompt)
                iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
                iter_workloads['set_max_num_seqs'].append(seq_num)
                iter_workloads['corr_context_tot_len'].append(sample_point)
                count += 1
                if count >= profile_num:
                    break






def get_prompt_iter_workloads_to_verify(llm, iter_workloads):
    '''
        NOTE: this function is used for collecting cost data to verify our linear cost model assumption.
        
        Get the iter_workloads whose per_iter execution latencies we are going to collect for prompt stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    max_num_batched_tokens = llm.llm_engine.scheduler_config.max_num_batched_tokens
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    block_size = llm.llm_engine.cache_config.block_size
    max_token_num = len(kv_caches[0][0]) * block_size
    prompt_limit = min(llm.llm_engine.scheduler_config.max_model_len,
                    llm.llm_engine.scheduler_config.max_num_batched_tokens)


    # prepare iter workloads
    max_token_num = min(max_token_num, max_num_batched_tokens)
    is_prompt = True
    for seq_num in range(1, max_num_seqs+1, 50):
    # for seq_num in [400]:
        # max_max_seqlen = (max_token_num + seq_num - 1) // seq_num
        count = 0
        # for max_seqlen in range(max_max_seqlen, 0, -1):
        for tot_token_num in range(max_token_num, 0, -128):
        # for tot_token_num in [max_token_num]:
            # tot_token_num = min(seq_num * max_seqlen, max_token_num)
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num >= prompt_limit:
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)






def get_decode_iter_workloads_to_verify(llm, iter_workloads):
    '''
        NOTE: this function is used for collecting cost data to verify our linear cost model assumption.
        
        Get the iter_workloads whose per_iter execution latencies we are going to collect for decoding stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    block_size = llm.llm_engine.cache_config.block_size
    max_token_num = len(kv_caches[0][0]) * block_size
    max_model_len = llm.llm_engine.model_config.max_model_len

    # prepare iter workloads
    is_prompt = False
    for seq_num in range(1, max_num_seqs+1, 50):
    # for seq_num in [199]:
        # for tot_token_num in range(max_token_num, 107, -108):
        count = 0
        for tot_token_num in range(max_token_num, 0, -128):
        # for tot_token_num in range(seq_num, max_token_num+1, 128):
        # for tot_token_num in range((max_token_num - seq_num)//128*128+seq_num, 0, -128):
            if seq_num > tot_token_num:
                continue
            if (tot_token_num + seq_num - 1) // seq_num >= max_model_len:
                # there must be a seq of length larger than max_model_len
                continue
            iter_workloads['is_prompt'].append(is_prompt)
            iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
            iter_workloads['set_max_num_seqs'].append(seq_num)








def get_set_seqlens_list(set_seqlens_list: List[Tuple[bool, List[int]]]):
    set_seqlens_list.append((False, [16*i for i in range(1, 120)]))
    set_seqlens_list.append((False, [16*i for i in range(1, 120)]))
    set_seqlens_list.append((False, [1024] + [1] * 511))
    set_seqlens_list.append((False, [1024] + [1] * 511))
    set_seqlens_list.append((False, [4096] + [1] * 511))
    set_seqlens_list.append((False, [4096] + [1] * 511))
    # set_seqlens_list.append((True, [16*i for i in range(1, 120)]))
    # set_seqlens_list.append((True, [16*i for i in range(1, 120)]))
    # set_seqlens_list.append((True, [1024] + [1] * 3))
    # set_seqlens_list.append((True, [1024] + [1] * 3))
    # set_seqlens_list.append((False, [16*i for i in range(1, 120)]))



def get_decode_iter_workloads_for_prepInp(llm, iter_workloads, set_seqlens_list: List[Tuple[bool, List[int]]]):
    '''
        NOTE: this function is used for collecting cost data for preparing input in decoding stages.
        
        Get the iter_workloads whose per_iter execution latencies we are going to collect for decoding stages.
        Input:
            llm: LLM. The LLM object.
        Output:
            Update iter_workloads --> {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
            Also update seqlens_list.
    '''
    # get the basic information of exec environment
    max_num_seqs = llm.llm_engine.scheduler_config.max_num_seqs
    kv_caches = llm.llm_engine.driver_worker.gpu_cache
    block_size = llm.llm_engine.cache_config.block_size
    max_token_num = len(kv_caches[0][0]) * block_size
    max_model_len = llm.llm_engine.model_config.max_model_len
    max_block_num = len(kv_caches[0][0])

    profile_num = 2

    # prepare iter workloads
    is_prompt = False
    for seq_num in range(1, min(max_num_seqs, max_block_num)+1):
    # for seq_num in [199]:
        # for tot_token_num in range(max_token_num, 107, -108):
        count = 0
        max_seqlen = (max_block_num-(seq_num-1)) * block_size
        max_seqlen = min(max_seqlen, max_model_len)
        for seqlen in range(max_seqlen, 0, -1):
            set_seqlens_list.append((False, [seqlen]+[1]*(seq_num-1)))
            count += 1
            if count >= profile_num:
                break
        # count = 0
        # # collect for the smallest tot_token_num
        # for tot_token_num in range(seq_num, max_token_num+1): #, 128):
        # # for tot_token_num in range(seq_num, (max_token_num - seq_num)//128*128+seq_num+1, 128):
        #     if seq_num > tot_token_num:
        #         continue
        #     if (tot_token_num + seq_num - 1) // seq_num > max_model_len:
        #         # there must be a seq of length larger than max_model_len
        #         continue
        #     iter_workloads['is_prompt'].append(is_prompt)
        #     iter_workloads['set_max_num_batched_tokens'].append(tot_token_num)
        #     iter_workloads['set_max_num_seqs'].append(seq_num)
        #     count += 1
        #     if count >= profile_num:
        #         break











def run_vllm_for_cost_model(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    quantization: Optional[str],
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
    dtype: str,
    max_model_len: Optional[int],
    enforce_eager: bool,
    kv_cache_dtype: str,
    device: str,
    

    gpu_memory_utilization: float,
    temperature: float,
) -> float:
    from vllm import LLM, SamplingParams
    llm = LLM(
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

        # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
        # max_num_seqs=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=512,
        max_paddings=512,
    )


    sampling_params_dict = {
        'n':n, 
        # change to greedy sampling to check correctness.
        'temperature': (0.0 if use_beam_search else temperature), # 0 or 1e-6 (greedy), #1.0
        'top_p': 1.0,
        'use_beam_search': use_beam_search,
        # 'ignore_eos': False, # True (original),
        # 'max_tokens': llm.llm_engine.model_config.max_model_len-_ # 4096-_  # output_len, #TODO test when using max tokens,

        # controls the procedure of sampling
        # 'presence_penalty': 0.1,
        # 'frequency_penalty': 0.1,
        # 'repetition_penalty': 0.9,
        # 'top_p': 0.8, #1.0,
        # 'top_k': -1,
        # 'min_p': 0.1, #0.0
        }

    iter_workloads = {k:list() for k in ['is_prompt', 'set_max_num_batched_tokens', 'set_max_num_seqs', 'corr_context_tot_len']}
    set_seqlens_list = list()
    piecewise_cost_model_build_mode = True if os.environ['WEIGHT_LOAD_DEGREE'] != '2' else False

    # get the iter workloads
    # get_decode_iter_workloads(llm=llm, iter_workloads=iter_workloads)
    # get_prompt_iter_worklods(llm=llm, iter_workloads=iter_workloads)
    # get_set_seqlens_list(set_seqlens_list=set_seqlens_list)
    # 
    # get the iter workloads [formally]
    get_decode_iter_workloads_sample_and_prepInp(
        llm=llm, iter_workloads=iter_workloads, piecewise_cost_model_build_mode=piecewise_cost_model_build_mode)
    get_prompt_iter_worklods_sample_and_prepInp(
        llm=llm, iter_workloads=iter_workloads, piecewise_cost_model_build_mode=piecewise_cost_model_build_mode)
    get_decode_iter_workloads_for_prepInp(llm=llm, iter_workloads=iter_workloads,set_seqlens_list=set_seqlens_list)

    # get_prompt_iter_workloads_to_verify(llm=llm, iter_workloads=iter_workloads)
    # get_decode_iter_workloads_to_verify(llm=llm, iter_workloads=iter_workloads)

    # FIXME(woosuk): Do not use internal method.
    torch.cuda.cudart().cudaProfilerStart()
    start_time = time.perf_counter()
    llm._profile_per_iter_latency(
        sampling_params_dict=sampling_params_dict, iter_workloads=iter_workloads, 
        set_seqlens_list=set_seqlens_list, piecewise_cost_model_build_mode=piecewise_cost_model_build_mode)
    # try:
    #     llm._profile_per_iter_latency(
    #         sampling_params_dict=sampling_params_dict, iter_workloads=iter_workloads, 
    #         set_seqlens_list=set_seqlens_list)
    # except Exception as e:
    #     print(e)
    end_time = time.perf_counter()
    print(f"TOTAL TIME OF _profile_per_iter_latency: {end_time-start_time} (s)")
    torch.cuda.cudart().cudaProfilerStop()


    # print the throughput results
    my_throughput_logger = llm.llm_engine.driver_worker.model_runner.my_throughput_logger
    my_throughput_logger.cal_throughput()
    my_throughput_logger.print_by_record()

    return 1






def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) +
                    max(max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt",
                              padding=True).input_ids

        print(f"do sample: {not use_beam_search}")
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample= False, #not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        gened_strs = tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)


        print(f"output_lens: {[(prompt_len, output_len, req_output.shape, prompt_len+output_len) for req_output, gend_str in zip (llm_outputs, gened_strs)]}") 
        for str1, str2 in zip(batch, gened_strs):
            print('Q---------------------------------------')
            print(str1)
            print('A---------------------------------------')
            print(str2[len(str1):])        

        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import pipeline
    llm = pipeline(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _ in requests]

    start = time.perf_counter()
    llm(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    from huggingface_hub import login
    login(token='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')

    print(args)

    # For Profiling
    start_main = time.perf_counter()


    # deal with extra parameters
    os.environ['WEIGHT_LOAD_DEGREE'] = args.weight_load_degree
    if args.backend == "ours":
        os.environ['USE_VLLM'] = 'False'
        os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'



    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)

    # For Profiling
    print(f"finish get tokenizer ---abs: {time.perf_counter()}")

    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len)
                    for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)


    # For Profiling
    print(f"finish request sampling ---abs: {time.perf_counter()}")


    if args.backend in ["vllm", "ours"]:
        elapsed_time = run_vllm_for_cost_model(requests, args.model, args.tokenizer,
                                args.quantization, args.tensor_parallel_size,
                                args.seed, args.n, args.use_beam_search,
                                args.trust_remote_code, args.dtype,
                                args.max_model_len, args.enforce_eager,
                                args.kv_cache_dtype, args.device,
                                # add more control
                                args.gpu_use_ratio,
                                args.temperature
                                )
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size,
                              args.trust_remote_code)
    elif args.backend == "mii":
        elapsed_time = run_mii(requests, args.model, args.tensor_parallel_size,
                               args.output_len)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s",
        # flush print
          flush=True)
    
    # For Profiling
    end_main = time.perf_counter()
    print(f"TOT TIME TO RUN MAIN(): {end_main - start_main}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii", "ours"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    


    # deal with extra parameters
    parser.add_argument(
        "--weight-load-degree", "-wldegree", 
        type=str,
        default="16",
        help='weight load degree when cache model weights on other gpus.')


    parser.add_argument(
        "--gpu-use-ratio", "-gpuratio", 
        type=float,
        default="0.9",
        help='gpu utilization ratio.')    

    parser.add_argument(
        "--temperature", 
        type=float,
        default="1.0",
        help='temperature.')    



    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    if args.backend in ["vllm", "ours"]:
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError("Tokenizer must be the same as the model for MII "
                             "backend.")
    main(args)

