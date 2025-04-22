"""
This file does fake scheduling based on the given output lengths.
The output lengths can be obtained by sampling following the output length distribution 
we obtain from experiment results (on the no-robot dataset).
"""


from typing import List, Optional, Tuple, Dict
import numpy as np
from my_per_iter_latency_estimator import CostTable
from vllm.engine.metrics import MyThroughputLogger


def remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size):
    '''
        Output:
            Modify running_seqs.
            Return: running_seqs_num, unfinished_reqnum, token_num_stored
    '''
    indices = running_seqs[2][:running_seqs_num].nonzero()[0]
    new_running_seqs_num = len(indices)
    running_seqs[0][:new_running_seqs_num] = running_seqs[0][indices]
    running_seqs[1][:new_running_seqs_num] = running_seqs[1][indices]
    running_seqs[2][:new_running_seqs_num] = running_seqs[2][indices]
    new_unfinished_reqnum = unfinished_reqnum - (running_seqs_num - new_running_seqs_num)
    
    new_block_num_used = sum((running_seqs[1][:new_running_seqs_num] - 1 + block_size - 1) // block_size)
    return new_running_seqs_num, new_unfinished_reqnum, new_block_num_used




def kill_seqs_for_more_cache_space(
        running_seqs, 
        max_block_num, block_size, running_seqs_num, 
        inp_lens, out_lens, seq_ids, pointer):
    '''
        Kill seqs to release cache slots.
        Output:
            Update running_seqs.
            Return: new_block_num_used, running_seqs_num, inp_lens, out_lens, pointer.
    '''
    ori_running_seqs_num = running_seqs_num
    
    
    
    

    
    
    
    
    
    
    

    
    
    new_block_nums = (running_seqs[1][:running_seqs_num] + block_size - 1) // block_size
    approx_tot_block_nums = (running_seqs[1][:running_seqs_num] - 1 + block_size - 1) // block_size + 1
    approx_tot_block_nums[1:] = approx_tot_block_nums[1:] + np.cumsum(new_block_nums)[:-1]
    if max(approx_tot_block_nums) > max_block_num:
        
        running_seqs_num = np.nonzero(approx_tot_block_nums > max_block_num)[0][0]
    

    
    
    
    new_block_num_used = sum((running_seqs[1][:running_seqs_num] - 1 + block_size - 1) // block_size)

    
    killed_seqs_num = ori_running_seqs_num - running_seqs_num
    seq_ids[pointer-killed_seqs_num : pointer] = running_seqs[0][running_seqs_num:ori_running_seqs_num]
    inp_lens[pointer-killed_seqs_num : pointer] = running_seqs[1][running_seqs_num:ori_running_seqs_num]
    out_lens[pointer-killed_seqs_num : pointer] = running_seqs[2][running_seqs_num:ori_running_seqs_num]
    pointer -= killed_seqs_num
    return new_block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer




def _add_one_infer_rng(rngs, start, end):
    if len(rngs) == 0:
        rngs.extend([start, end])
    else:
        if (rngs[-1] + 1) == start:
            
            rngs[-1] = end
        else:
            rngs.extend([start, end])


def _store_infer_state(start, end, infer_progress, seq_ids):
    
    for seq_i in seq_ids:
        
        _add_one_infer_rng(infer_progress[seq_i], start, end)



def update_prefill_logs_NO_max_infer_step_num_limit(
        prefill_logs: List[Tuple[int, int, int, int]], 
        prompt_lens, max_num_batched_tokens, 
        prompt_ids, infer_progress, tot_iter_num) -> int:
    '''
        Get the steps to complete the prefill stages for the given prompt_lens.
        INPUT:
            tot_iter_num: the current total iteration number.
        Output:
            1. Add new tuples to prefill_logs, 
            each tuple is (seqnum, tot_token_num, attention_sum, max_seqlen) of a prefill step.
            2. store per iter infer information in infer_progress, i.e., which prompt ids are involved for each iter.
            NOTE:
                we only consider the ``max_num_batched_tokens'' constraint here.
                the ``max_seq_num'' and `` cache space'' are not considered 
                because they are considered in ``fake_FCFS_schedule''.
            NOTE:
                all prompts are padded to the max len.
            NOTE:
                in this version, we generate all the prefill steps required to start all the given prompts.
    '''
    def get_prefill_step(seqs):
        seqnum = len(seqs)
        max_seqlen = max(seqs)
        tot_token_num = sum(seqs)
        attention_sum = sum([(1+si)*si for si in seqs])
        return [seqnum, tot_token_num, attention_sum, max_seqlen]
    
    if len(prompt_lens)==0:
        
        return tot_iter_num
    
    seqs = list()
    seq_ids = list()
    for i, seq_id in zip(prompt_lens, prompt_ids):
        if max(seqs+[i]) * (len(seqs)+1) <= max_num_batched_tokens:
            
            seqs.append(i)
            seq_ids.append(seq_id)
        else:
            prefill_logs.append(get_prefill_step(seqs))
            _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
            tot_iter_num += 1
            
            seqs = [i]
            seq_ids = [seq_id]
    
    
    prefill_logs.append(get_prefill_step(seqs))
    _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
    tot_iter_num += 1
    
    return tot_iter_num




def update_prefill_logs(
        prefill_logs: List[Tuple[int, int, int, int]], 
        
        
        
        valid_prefill_logs: List[Tuple[int, int]],
        
        prompt_lens, max_num_batched_tokens, 
        prompt_ids, infer_progress, tot_iter_num, 
        
        need_query_available_requests: bool, 
        check_gap: int,
        last_iter_seqs: List[int], 
        last_iter_seq_ids: List[int],
        must_record_first_step: bool,
        ) -> Tuple[List[int], List[int], int]:
    '''
        Get the steps to complete the prefill stages for the given prompt_lens.
        INPUT:
            tot_iter_num: the current total iteration number.
            last_iter_seqs/last_iter_seq_ids: the prompts from last iter which is remained because we need to check 
                if there is newly available input requests.
        Output:
            1. Add new tuples to prefill_logs, 
            each tuple is (seqnum, tot_token_num, attention_sum, max_seqlen) of a prefill step.
            2. store per iter infer information in infer_progress, i.e., which prompt ids are involved for each iter.
            NOTE:
                we only consider the ``max_num_batched_tokens'' constraint here.
                the ``max_seq_num'' and `` cache space'' are not considered 
                because they are considered in ``fake_FCFS_schedule''.
            NOTE:
                all prompts are padded to the max len.
            NOTE:
                in this version, we will stop the prefill step generation when 
                    (1) ``need_query_available_requests`` is True, i.e., with the last step, there will be no waiting
                    request in the current waiting list.
                    (2) the step % check_gap == 0
            NOTE:
                if a seq has been run before, then we treat this seq as in decode phase when computing flops metadata (because they have been started in previous iterations); otherwise, we treat it as in prefill phase
    '''
    def get_prefill_step(seqs, seq_ids):
        seqnum = len(seqs)
        max_seqlen = max(seqs)
        tot_token_num = sum(seqs)
        attention_sum = sum([(1+si)*si for si in seqs])

        
        seq_array_not_started = np.asarray([seqlen for seqlen, seq_id in zip(seqs, seq_ids) if len(infer_progress[seq_id])==0 ])
        valid_seqnum = seqnum + sum(seq_array_not_started-1)
        valid_tot_token_num = tot_token_num + sum(seq_array_not_started*(seq_array_not_started-1))/2

        return [seqnum, tot_token_num, attention_sum, max_seqlen], [valid_seqnum, valid_tot_token_num]
    
    
    

    
    if (len(prompt_lens)==0) and (len(last_iter_seqs)==0):
        
        return list(), list(), tot_iter_num
    

    
    
    

    
    
    seqs = last_iter_seqs
    seq_ids = last_iter_seq_ids
    for i, seq_id in zip(prompt_lens, prompt_ids):
        
        

        if max(seqs+[i]) * (len(seqs)+1) <= max_num_batched_tokens:
            
            

            
            seqs.append(i)
            seq_ids.append(seq_id)
        else:
            
            

            
            log, valid_log = get_prefill_step(seqs, seq_ids)
            prefill_logs.append(log)
            valid_prefill_logs.append(valid_log)
            
            
            

            _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
            tot_iter_num += 1
            
            seqs = [i]
            seq_ids = [seq_id]

            
            must_record_first_step = False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    log, valid_log = get_prefill_step(seqs, seq_ids)
    prefill_logs.append(log)
    valid_prefill_logs.append(valid_log)
    
    
    

    _store_infer_state(tot_iter_num, tot_iter_num, infer_progress, seq_ids)
    tot_iter_num += 1
    
    return list(), list(), tot_iter_num




def _fake_FCFS_schedule_NO_continuous_model_level_pipeline(
        inp_lens: List[int], out_lens: List[int], 
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int):
    '''
        Do the fake scheduling using the first-come-first-serve policy.
        inp_lens: the input lengths of the requests.
        out_lens: the output lengths of the requests.
        max_seq_num: the maximum number of requests running concurrently.
        max_cache_slot_num: the maximum number of tokens whose KV cache can be stored at the same time.
        There is only two constraints when trying to add a running request:
            (1) max_seq_num; (2) max_cache_slot_num (consider watermark=0.01).
        NOTE:
            (1) We ignore the block size here to make the fake schedule faster.
                --> it seems there will be a lot of request kill when block size is 1, 
                --> so we HAVE TO CONSIDER block size!
            (2) For prefill stage, we also consider  
                ``max_num_batched_tokens'' and TODO ``scheduler_config.max_paddings''. [Try this first]
            (3) When killing seqs, consider if there is an extra block for each sequence.
    '''
    def has_enough_cache(block_num_used, new_token_num, consider_watermark=False):
        
        new_block_num = (new_token_num + block_size - 1) // block_size
        if consider_watermark:
            watermark_blocks = 0.01 * max_block_num
            return max_block_num - block_num_used - new_block_num >= watermark_blocks
        return (block_num_used + new_block_num) <= max_block_num
    def add_block_num_used(block_num_used, new_token_num):
        new_block_num = (new_token_num + block_size - 1) // block_size
        block_num_used = block_num_used + new_block_num
        return block_num_used
    def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
        
        

        
        
        iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
        
        
        
        extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
        
        extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
        
        block_num_left = (max_block_num - block_num_used) % running_seqs_num
        
        
        extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
        iter_num += extra_iter_num
        
        
        
        

        
        iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        

        return iter_num
    def get_tot_token_num(running_seqs_num, running_seqs):
        return sum(running_seqs[1][:running_seqs_num])
    def get_max_seqlen(running_seqs_num, running_seqs):
        return max(running_seqs[1][:running_seqs_num])
    

    
    ori_inplens = inp_lens.copy()
    ori_outlens = out_lens.copy()

    unfinished_reqnum = len(inp_lens)
    running_seqs = np.zeros((3, max_seq_num), dtype=np.int32) 
    seq_ids = list(range(len(inp_lens)))
    pointer = 0 
    running_seqs_num = 0
    
    block_num_used = 0
    logs = list()
    prefill_logs = list() 

    
    
    
    valid_prefill_logs: List[Tuple[int, int]] = list()
    
    
    infer_progress = list([] for _ in range(len(inp_lens)))
    
    is_prefill_steps: List[bool] = list()
    tot_iter_num: int = 0 



    
    import time
    
    time_start = time.perf_counter()




    while unfinished_reqnum:
        
        new_prompt_lens: List[int] = list()
        new_prompt_ids: List[int] = list()
        while (pointer < len(inp_lens)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
            
            
            if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
                running_seqs[0][running_seqs_num] = seq_ids[pointer] 
                running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
                running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


                new_prompt_lens.append(inp_lens[pointer])
                new_prompt_ids.append(seq_ids[pointer])
                if running_seqs[2][running_seqs_num] == 0:
                    
                    unfinished_reqnum -= 1
                    pointer += 1
                    continue


                
                block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
                pointer += 1
                running_seqs_num += 1
            else:
                break


        
        
        
        new_prompt_lens = np.asarray(new_prompt_lens)
        new_prompt_ids = np.asarray(new_prompt_ids)

        

        
        
        
        _, _, tot_iter_num = update_prefill_logs(prefill_logs, 
                            valid_prefill_logs,
                            new_prompt_lens, max_num_batched_tokens,
                            new_prompt_ids, infer_progress, tot_iter_num,
                            need_query_available_requests=False, check_gap=1,
                            last_iter_seqs=list(), last_iter_seq_ids=list(),
                            must_record_first_step=False)
        is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))
      
        
        if unfinished_reqnum == 0:
            
            assert running_seqs_num == 0
            break

        
        
        block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
            kill_seqs_for_more_cache_space(
                running_seqs, 
                max_block_num, block_size, running_seqs_num, 
                inp_lens, out_lens, seq_ids, pointer)

        
        
        
        
        iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
        
        
        
        
        tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
        curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
        logs.extend([(running_seqs_num, 
                      tot_token_num + running_seqs_num*i,
                      tot_token_num + running_seqs_num*i,
                      curr_max_seqlen + i) \
                     for i in range(iter_num)])
        
        
        

        _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
        is_prefill_steps.extend([False]*iter_num)
        tot_iter_num += iter_num

        
        
        running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
        running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

        
        running_seqs_num, unfinished_reqnum, block_num_used = \
            remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)
        
        
    
    
    
    
    
    
        
    
    print(f"tot schedule time: {time.perf_counter() - time_start}")


    assert tot_iter_num == (len(logs) + len(prefill_logs)), (tot_iter_num, len(logs), len(prefill_logs), ori_inplens, ori_outlens, max_seq_num, max_block_num, max_num_batched_tokens, block_size) 
    return logs, prefill_logs, is_prefill_steps, infer_progress, valid_prefill_logs 




def _check_new_input_requests(
        sort_input: bool, 
        
        seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],
        time_when_checking: float,
        
        pointer: int,
        running_seqs_num: int,
        ) -> Tuple[List[int], List[int], List[int], List[float], float]:
    """
        This function checks whether there are new input requests and update the input request array if any.
        INPUT:
            sort_input: controls whether we need to sort the waiting input list every time we add some new inputs.
            pointer: seq_ids[pointer:] are the sequences currently received and in the waiting list.
        UPDATE:
            seq_ids, inp_lens, out_lens, arrive_times, time_when_checking.
        NOTE: 
            1. if we want to sort the input request list, we need to reorder seq_ids, inp_lens, out_lens accordingly 
            (we do not change the seq_id of any input request).
            2. when there is no new request at ``time_when_checking'' but there are input requests we need to wait, 
            we need to update ``time_when_checking'' to the latest time we can receive a new input request.
    """
    if len(seq_ids) == len(inp_lens):
        
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking
    
    if (running_seqs_num > 0) and (arrive_times[len(seq_ids)] > time_when_checking):
        
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking
    
    if arrive_times[len(seq_ids)] > time_when_checking:
        
        
        time_when_checking = arrive_times[len(seq_ids)]


    seq_id = None
    new_inp_end = None
    
    for seq_id in range(len(seq_ids), len(inp_lens)):
        if arrive_times[seq_id] > time_when_checking:
            new_inp_end = seq_id
            break
    
    if new_inp_end == None:
        
        new_inp_end = len(inp_lens)
    
    waiting_inp_lens = inp_lens[pointer:new_inp_end]
    seq_ids = np.concatenate((seq_ids, (range(pointer, new_inp_end))))

    
    
    
    
    

    if sort_input:
        order = np.argsort(-waiting_inp_lens, kind='stable')
        seq_ids[pointer:] = seq_ids[pointer:][order]
        inp_lens[pointer:new_inp_end] = inp_lens[pointer:new_inp_end][order]
        out_lens[pointer:new_inp_end] =out_lens[pointer:new_inp_end][order]
        arrive_times[pointer:new_inp_end] =arrive_times[pointer:new_inp_end][order]
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking
    else:
        
        return seq_ids, inp_lens, out_lens, arrive_times, time_when_checking











def _update_seq_info_with_known_arrive_time(
        time_when_check: float,
        running_seq_ids: List[int],
        pointer: int,
        
        ref_seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],        
        
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],
        
        infer_progress, full_infer_progress, fixed_ref_seq_ids_list,
        ):
    """
        This function update the info of the seqs whose arrive time become known every time there is 
        new output generated.
        NOTE:
            1. the seqs in ref_seq_ids_list, ..., and fixed_ref_seq_ids_list, are sorted by the seq ids. --> !! no such requirement !!
            2. we need to sort the ready seqs by their arrive times.
    """
    
    
    



    for i in range(len(ref_seq_ids_list)):

        
        running_or_pending_seq_ids = np.concatenate((running_seq_ids, ref_seq_ids[pointer:]))


        
        seq_ids_to_add = np.asarray(sorted(set(ref_seq_ids_list[i]).difference(running_or_pending_seq_ids)), dtype=np.int64)
        
        inds = np.searchsorted(ref_seq_ids_list[i], seq_ids_to_add)
        inp_lens_to_add = inp_lens_list[i][inds]
        out_lens_to_add = out_lens_list[i][inds]
        arrive_times_to_add = np.maximum(arrive_times_list[i][inds], time_when_check)
        
        
        _inds = np.searchsorted(fixed_ref_seq_ids_list[i], seq_ids_to_add)
        for ind, seq_id in zip(_inds, seq_ids_to_add):
            infer_progress[seq_id] = full_infer_progress[i+1][ind]
        
        
        ref_seq_ids = np.concatenate((ref_seq_ids, seq_ids_to_add))
        inp_lens = np.concatenate((inp_lens, inp_lens_to_add))
        out_lens = np.concatenate((out_lens, out_lens_to_add))
        arrive_times = np.concatenate((arrive_times, arrive_times_to_add))

        
        order = np.argsort(arrive_times[pointer:], kind='stable')
        ref_seq_ids[pointer:] = ref_seq_ids[pointer:][order]
        inp_lens[pointer:] = inp_lens[pointer:][order]
        out_lens[pointer:] = out_lens[pointer:][order]
        arrive_times[pointer:] = arrive_times[pointer:][order]

        
        remaining_inds = set(range(len(ref_seq_ids_list[i]))).difference(inds)
        remaining_inds = sorted(remaining_inds)
        
        ref_seq_ids_list[i] = ref_seq_ids_list[i][remaining_inds]
        inp_lens_list[i] = inp_lens_list[i][remaining_inds]
        out_lens_list[i] = out_lens_list[i][remaining_inds]
        arrive_times_list[i] = arrive_times_list[i][remaining_inds]

        
        
        
        
        
        

    return ref_seq_ids, inp_lens, out_lens, arrive_times, \
        ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list
    





def _update_seq_info_with_known_arrive_time_fast_version(
        time_when_check: float,
        running_seq_ids: List[int],
        pointer: int,
        
        ref_seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],        
        
        
        unknown_seq_info: Dict[int, List[Tuple[int, int, float, List[int]]]],
        
        infer_progress,
        ):
    """
        This function update the info of the seqs whose arrive time become known every time there is 
        new output generated.
        NOTE:
            1. the seqs in ref_seq_ids_list, ..., and fixed_ref_seq_ids_list, are sorted by the seq ids. --> !! no such requirement !!
            2. we need to sort the ready seqs by their arrive times.
    """
    
    
    


    

    
    running_or_pending_seq_ids = np.concatenate((running_seq_ids, ref_seq_ids[pointer:]))

    
    seq_ids_to_add = np.asarray(sorted(set(unknown_seq_info.keys()).difference(running_or_pending_seq_ids)), dtype=np.int64)
    infos_to_add = [unknown_seq_info[seq_id][0] for seq_id in seq_ids_to_add]
    inp_lens_to_add = [info[0] for info in infos_to_add]
    out_lens_to_add = [info[1] for info in infos_to_add]
    arrive_times_to_add = np.maximum([info[2] for info in infos_to_add], time_when_check) 


    
    for seq_id in seq_ids_to_add:
        infer_progress[seq_id] = unknown_seq_info[seq_id][0][3]


    
    ref_seq_ids = np.concatenate((ref_seq_ids, seq_ids_to_add))
    inp_lens = np.concatenate((inp_lens, inp_lens_to_add))
    out_lens = np.concatenate((out_lens, out_lens_to_add))
    arrive_times = np.concatenate((arrive_times, arrive_times_to_add))

    
    order = np.argsort(arrive_times[pointer:], kind='stable')
    ref_seq_ids[pointer:] = ref_seq_ids[pointer:][order]
    inp_lens[pointer:] = inp_lens[pointer:][order]
    out_lens[pointer:] = out_lens[pointer:][order]
    arrive_times[pointer:] = arrive_times[pointer:][order]


    
    for seq_id in seq_ids_to_add:
        unknown_seq_info[seq_id] = unknown_seq_info[seq_id][1:]
        if len(unknown_seq_info[seq_id]) == 0:
            del unknown_seq_info[seq_id]



    return ref_seq_ids, inp_lens, out_lens, arrive_times
    
















def _check_new_input_requests_support_vertical_fuse(
        sort_input: bool, 
        
        seq_ids: List[int],
        
        ref_seq_ids: List[int],
        inp_lens: List[int],
        out_lens: List[int],
        arrive_times: List[float],
        time_when_checking: float,
        
        pointer: int,
        running_seqs_num: int,
        
        ) -> Tuple[List[int], List[int], List[int], List[float], float]:
    """
        This function checks whether there are new input requests and update the input request array if any.
        INPUT:
            sort_input: controls whether we need to sort the waiting input list every time we add some new inputs.
            pointer: seq_ids[pointer:] are the sequences currently received and in the waiting list.
        UPDATE:
            seq_ids, inp_lens, out_lens, arrive_times, time_when_checking,
            ref_seq_ids; ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list
        NOTE: 
            1. if we want to sort the input request list, we need to reorder seq_ids, inp_lens, out_lens accordingly 
            (we do not change the seq_id of any input request).
            2. when there is no new request at ``time_when_checking'' but there are input requests we need to wait, 
            we need to update ``time_when_checking'' to the latest time we can receive a new input request.
            3. this version supports model vertical fusion.
                ``ref_seq_ids``, ``inp_lens``, ``out_lens``, ``arrive_times`` store the info of seqs whose arrive times are known
                ``ref_seq_ids_list``, ``inp_lens_list``, ``out_lens_list``, ``arrive_times_list`` 
                    store the info of seqs of each model which are fused together
                Both of these two groups of variables will be modified in this method.
    """
    
    

    if len(seq_ids) == len(inp_lens):
        
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking
    
    if (running_seqs_num > 0) and (arrive_times[len(seq_ids)] > time_when_checking):
        
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking
    
    if arrive_times[len(seq_ids)] > time_when_checking:
        
        
        time_when_checking = arrive_times[len(seq_ids)]


    seq_id = None
    new_inp_end = None
    
    for seq_id in range(len(seq_ids), len(inp_lens)):
        if arrive_times[seq_id] > time_when_checking:
            new_inp_end = seq_id
            break
    
    if new_inp_end == None:
        
        new_inp_end = len(inp_lens)
    
    waiting_inp_lens = inp_lens[pointer:new_inp_end]
    
    
    seq_ids = np.concatenate((seq_ids, ref_seq_ids[pointer:new_inp_end]))

    
    
    
    
    

    if sort_input:
        order = np.argsort(-waiting_inp_lens, kind='stable')
        seq_ids[pointer:] = seq_ids[pointer:][order]
        inp_lens[pointer:new_inp_end] = inp_lens[pointer:new_inp_end][order]
        out_lens[pointer:new_inp_end] =out_lens[pointer:new_inp_end][order]
        arrive_times[pointer:new_inp_end] =arrive_times[pointer:new_inp_end][order]
        
        ref_seq_ids[pointer:new_inp_end] =ref_seq_ids[pointer:new_inp_end][order]
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking
    else:
        
        return seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, time_when_checking









































        










































































            






















        
























































        


















def _fake_FCFS_schedule_continuous_model_level_pipeline(
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    '''
        Do the fake scheduling using the first-come-first-serve policy.
        inp_lens: the input lengths of the requests.
        out_lens: the output lengths of the requests.
        max_seq_num: the maximum number of requests running concurrently.
        max_cache_slot_num: the maximum number of tokens whose KV cache can be stored at the same time.
        cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}

        Output: [not only output fake scheduling logs, but also output latency]
            cumsum_latencys, is_prefill_steps, infer_progress

        There is only two constraints when trying to add a running request:
            (1) max_seq_num; (2) max_cache_slot_num (consider watermark=0.01).
        NOTE:
            (1) We ignore the block size here to make the fake schedule faster.
                --> it seems there will be a lot of request kill when block size is 1, 
                --> so we HAVE TO CONSIDER block size!
            (2) For prefill stage, we also consider  
                ``max_num_batched_tokens'' and TODO ``scheduler_config.max_paddings''. [Try this first]
            (3) When killing seqs, consider if there is an extra block for each sequence.

        NOTE: for continuous model-level pipeline, e.g., we may have model A -> model B, but A, B run in 
            the same execution stage.
            In this function, we will check whether there is new input requests every k (``check_gap'') inference steps,
            according to ``arrive_times''.
            If yes, we will add the new requests into the waiting list; 
            else, we do nothing but keep doing inference.
        
        This function runs K (i.e., check_gap) step fake scheduling starting from the given inference progress.
        NOTE:
            1. if before we finish the K inference steps we run out of requests, we stop the inference process 
            and turn to waiting more available input requests.
            2. in the current code, ``sort_input`` performs differently from the version that we must query every 
            check_gap steps.
    '''
    def has_enough_cache(block_num_used, new_token_num, consider_watermark=False):
        
        new_block_num = (new_token_num + block_size - 1) // block_size
        if consider_watermark:
            watermark_blocks = 0.01 * max_block_num
            return max_block_num - block_num_used - new_block_num >= watermark_blocks
        return (block_num_used + new_block_num) <= max_block_num
    def add_block_num_used(block_num_used, new_token_num):
        new_block_num = (new_token_num + block_size - 1) // block_size
        block_num_used = block_num_used + new_block_num
        return block_num_used
    def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
        
        

        
        
        iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
        
        
        
        extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
        
        extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
        
        block_num_left = (max_block_num - block_num_used) % running_seqs_num
        
        
        extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
        iter_num += extra_iter_num
        
        
        
        

        
        iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        

        return iter_num
    def get_tot_token_num(running_seqs_num, running_seqs):
        return sum(running_seqs[1][:running_seqs_num])
    def get_max_seqlen(running_seqs_num, running_seqs):
        return max(running_seqs[1][:running_seqs_num])
    

    
    inp_lens = np.asarray(inp_lens)
    out_lens = np.asarray(out_lens)
    arrive_times = np.asarray(arrive_times)


    
    ori_inplens = inp_lens.copy()
    ori_outlens = out_lens.copy()

    unfinished_reqnum = len(inp_lens)
    running_seqs = np.zeros((3, max_seq_num), dtype=np.int32) 
    
    
    seq_ids = np.asarray(list(), dtype=np.int64)
    pointer = 0 
    running_seqs_num = 0
    
    block_num_used = 0
    logs = list()
    prefill_logs = list() 

    
    
    
    valid_prefill_logs: List[Tuple[int, int]] = list()
    
    
    infer_progress = list([] for _ in range(len(inp_lens)))
    
    is_prefill_steps: List[bool] = list()
    tot_iter_num: int = 0 

    
    last_iter_seqs = list()
    last_iter_seq_ids = list()
    need_query_available_requests = True
    must_record_first_step = False
    tot_inference_time = 0

    
    cumsum_latencys: List[float] = np.asarray(list())



    while unfinished_reqnum:

        
        
        
        
        
        
        
        if ((running_seqs_num == 0) and (pointer==len(seq_ids))) or \
            (need_query_available_requests and (tot_iter_num % check_gap == 0)):

            

            
            
            
            
            
            

            seq_ids, inp_lens, out_lens, arrive_times, tot_inference_time = _check_new_input_requests(
                sort_input, seq_ids, inp_lens, out_lens, arrive_times, tot_inference_time, pointer, running_seqs_num)
        
            
            
            
            
            
            
            
            
            
            
            


        
        new_prompt_lens: List[int] = list()
        new_prompt_ids: List[int] = list()
        while (pointer < len(seq_ids)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
            
            
            if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
                running_seqs[0][running_seqs_num] = seq_ids[pointer] 
                running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
                running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


                new_prompt_lens.append(inp_lens[pointer])
                new_prompt_ids.append(seq_ids[pointer])
                if running_seqs[2][running_seqs_num] == 0:
                    
                    unfinished_reqnum -= 1
                    pointer += 1
                    continue


                
                block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
                pointer += 1
                running_seqs_num += 1
            else:
                break



        
        if (len(seq_ids)<len(inp_lens)) and (pointer == len(seq_ids)) \
            and has_enough_cache(block_num_used,1,consider_watermark=True) \
                and (running_seqs_num < max_seq_num):
            
            
            need_query_available_requests = True
        else:
            need_query_available_requests = False


        
        
        
        
        
        


        
        
        
        new_prompt_lens = np.asarray(new_prompt_lens)
        new_prompt_ids = np.asarray(new_prompt_ids)
        ori_prefill_logs_num = len(prefill_logs)

        
        

        
        
        
        
        
        last_iter_seqs, last_iter_seq_ids, tot_iter_num = update_prefill_logs(
                            prefill_logs, 
                            valid_prefill_logs, 
                            new_prompt_lens, max_num_batched_tokens,
                            new_prompt_ids, infer_progress, tot_iter_num, 
                            need_query_available_requests, check_gap, last_iter_seqs, last_iter_seq_ids, 
                            must_record_first_step)
        is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))


        
        

        
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=prefill_logs[ori_prefill_logs_num:], decode_logs=list(), **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(prefill_latencys)+tot_inference_time))
        if len(cumsum_latencys) > 0:
            tot_inference_time = cumsum_latencys[-1]
        
        
        
        
        

        
        if len(last_iter_seqs) > 0:
            must_record_first_step = True
            continue
        elif len(new_prompt_ids) > 0:
            
            must_record_first_step = False

        if running_seqs_num == 0:
            
            continue

        
        
        
        
        
        
        
        

        block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
            kill_seqs_for_more_cache_space(
                running_seqs, 
                max_block_num, block_size, running_seqs_num, 
                inp_lens, out_lens, seq_ids, pointer)

        

        
        
        
        
        iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
        
        

        
        if pointer < len(seq_ids):
            need_query_available_requests = False

        
        if need_query_available_requests:
            iter_num = min(iter_num, (tot_iter_num + must_record_first_step + check_gap - 1) // check_gap * check_gap \
                - tot_iter_num)
            must_record_first_step = False
            
            if iter_num == 0:
                must_record_first_step = True
                continue

        
        
        
        
        
        tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
        curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
        logs.extend([(running_seqs_num, 
                      tot_token_num + running_seqs_num*i,
                      tot_token_num + running_seqs_num*i,
                      curr_max_seqlen + i) \
                     for i in range(iter_num)])

        
        

                     
        _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
        is_prefill_steps.extend([False]*iter_num)
        tot_iter_num += iter_num

        
        
        running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
        running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

        
        running_seqs_num, unfinished_reqnum, block_num_used = \
            remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)

        
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=list(), decode_logs=logs[-iter_num:], **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(decode_latencys)+tot_inference_time))
        tot_inference_time = cumsum_latencys[-1]

        
        
        

        
        

        
        
    
    
    
    
    
    
    assert tot_iter_num == (len(logs) + len(prefill_logs)), (tot_iter_num, len(logs), len(prefill_logs), ori_inplens, ori_outlens, max_seq_num, max_block_num, max_num_batched_tokens, block_size) 
    
    return cumsum_latencys, is_prefill_steps, infer_progress, prefill_logs, logs, valid_prefill_logs 
















def _fake_FCFS_schedule_continuous_model_level_pipeline_vertical_fuse(
        
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], ref_seq_ids: List[int],
        
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],        
        
        check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    '''
        Do the fake scheduling using the first-come-first-serve policy.
        inp_lens: the input lengths of the requests.
        out_lens: the output lengths of the requests.
        max_seq_num: the maximum number of requests running concurrently.
        max_cache_slot_num: the maximum number of tokens whose KV cache can be stored at the same time.
        cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}

        Output: [not only output fake scheduling logs, but also output latency]
            cumsum_latencys, is_prefill_steps, infer_progress

        There is only two constraints when trying to add a running request:
            (1) max_seq_num; (2) max_cache_slot_num (consider watermark=0.01).
        NOTE:
            (1) We ignore the block size here to make the fake schedule faster.
                --> it seems there will be a lot of request kill when block size is 1, 
                --> so we HAVE TO CONSIDER block size!
            (2) For prefill stage, we also consider  
                ``max_num_batched_tokens'' and TODO ``scheduler_config.max_paddings''. [Try this first]
            (3) When killing seqs, consider if there is an extra block for each sequence.

        NOTE: for continuous model-level pipeline, e.g., we may have model A -> model B, but A, B run in 
            the same execution stage.
            In this function, we will check whether there is new input requests every k (``check_gap'') inference steps,
            according to ``arrive_times''.
            If yes, we will add the new requests into the waiting list; 
            else, we do nothing but keep doing inference.
        
        This function runs K (i.e., check_gap) step fake scheduling starting from the given inference progress.
        NOTE:
            1. if before we finish the K inference steps we run out of requests, we stop the inference process 
            and turn to waiting more available input requests.
            2. in the current code, ``sort_input`` performs differently from the version that we must query every 
            check_gap steps.
            3. this version supports the vertical fusion of models.
    '''



    def has_enough_cache(block_num_used, new_token_num, consider_watermark=False):
        
        new_block_num = (new_token_num + block_size - 1) // block_size
        if consider_watermark:
            watermark_blocks = 0.01 * max_block_num
            return max_block_num - block_num_used - new_block_num >= watermark_blocks
        return (block_num_used + new_block_num) <= max_block_num
    def add_block_num_used(block_num_used, new_token_num):
        new_block_num = (new_token_num + block_size - 1) // block_size
        block_num_used = block_num_used + new_block_num
        return block_num_used
    def get_max_iter_num(block_num_used, running_seqs_num, running_seqs):
        
        

        
        
        iter_num = ((max_block_num - block_num_used) // running_seqs_num) * block_size
        
        
        
        extra_iter_nums = ((-running_seqs[1][:running_seqs_num] + 1) % block_size) + 1
        
        extra_iter_nums, counts = np.unique(extra_iter_nums, return_counts=True)
        
        block_num_left = (max_block_num - block_num_used) % running_seqs_num
        
        
        extra_iter_num = extra_iter_nums[np.nonzero(np.cumsum(counts) > block_num_left)[0][0]]-1
        iter_num += extra_iter_num
        
        
        
        

        
        iter_num = min(min(running_seqs[2][:running_seqs_num]), iter_num)

        

        return iter_num
    def get_tot_token_num(running_seqs_num, running_seqs):
        return sum(running_seqs[1][:running_seqs_num])
    def get_max_seqlen(running_seqs_num, running_seqs):
        return max(running_seqs[1][:running_seqs_num])
    


    
    
    
    
    
    
    
    

    inp_lens_list = [np.asarray(_.copy()) for _ in inp_lens_list]
    out_lens_list = [np.asarray(_.copy()) for _ in out_lens_list]
    arrive_times_list = [np.asarray(_.copy()) for _ in arrive_times_list]
    fixed_ref_seq_ids_list = ref_seq_ids_list
    ref_seq_ids_list = [np.asarray(_.copy(), dtype=np.int64) for _ in ref_seq_ids_list]

    
    ref_seq_ids = np.asarray(ref_seq_ids, dtype=np.int64)
    inp_lens = np.asarray(inp_lens.copy())
    out_lens = np.asarray(out_lens.copy())
    arrive_times = np.asarray(arrive_times.copy())

    
    
    
    

    
    curr_model_level_id = 1

    
    
    
    
    


    
    ori_inplens = inp_lens.copy()
    ori_outlens = out_lens.copy()

    
    
    
    req_nums = [len(inp_lens)]+[len(_inp_lens) for _inp_lens in inp_lens_list]
    unfinished_reqnum = sum(req_nums)
    
    running_seqs = np.zeros((3, max_seq_num), dtype=np.int32) 
    
    
    seq_ids = np.asarray(list(), dtype=np.int64)
    pointer = 0 
    running_seqs_num = 0
    
    block_num_used = 0
    logs = list()
    prefill_logs = list() 

    
    valid_prefill_logs: List[Tuple[int, int]] = list()
    
    
    
    full_infer_progress = list([[[] for _ in range(req_num)] for req_num in req_nums])
    
    
    infer_progress = {seq_id:full_infer_progress[0][i] for i, seq_id in enumerate(ref_seq_ids)}
    
    is_prefill_steps: List[bool] = list()
    tot_iter_num: int = 0 


    
    uniq_seq_ids = sorted(set(np.concatenate(ref_seq_ids_list)))
    unknown_seq_info: Dict[int, List[Tuple[int, int, float, List[int]]]] = {seq_id:list() for seq_id in uniq_seq_ids}
    for _seq_ids, _inp_lens, _out_lens, _arrive_times, _infer_progresses \
        in zip(ref_seq_ids_list, inp_lens_list, out_lens_list, arrive_times_list, full_infer_progress[1:]):
        for _seq_id, _inp_len, _out_len, _arrive_time, _infer_progress \
            in zip(_seq_ids, _inp_lens, _out_lens, _arrive_times, _infer_progresses):
            unknown_seq_info[_seq_id].append((_inp_len, _out_len, _arrive_time, _infer_progress))


    
    last_iter_seqs = list()
    last_iter_seq_ids = list()
    need_query_available_requests = True
    must_record_first_step = False
    tot_inference_time = 0

    
    cumsum_latencys: List[float] = np.asarray(list())

    
    finished_seq_ids = list()


    
    import time
    
    time_start = time.perf_counter()

    while unfinished_reqnum:

        
        
        


        
        
        
        
        
        
        
        if ((running_seqs_num == 0) and (pointer==len(seq_ids))) or \
            (need_query_available_requests and (tot_iter_num % check_gap == 0)):

            

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            ref_seq_ids, inp_lens, out_lens, arrive_times = \
                _update_seq_info_with_known_arrive_time_fast_version(
                    tot_inference_time, running_seqs[0][:running_seqs_num], pointer,
                    ref_seq_ids, inp_lens, out_lens, arrive_times,      
                    
                    
                    unknown_seq_info,
                    infer_progress,
                    )


            finished_seq_ids = list()

            

            seq_ids, inp_lens, out_lens, arrive_times, ref_seq_ids, tot_inference_time = \
                _check_new_input_requests_support_vertical_fuse(
                    sort_input, seq_ids, ref_seq_ids, inp_lens, out_lens, arrive_times, 
                    tot_inference_time, pointer, running_seqs_num)
            
            
        
            
            
            
            
            
            
            
            
            
            
            
            
            


        
        new_prompt_lens: List[int] = list()
        new_prompt_ids: List[int] = list()
        while (pointer < len(seq_ids)) and has_enough_cache(block_num_used,1,consider_watermark=True) and (running_seqs_num < max_seq_num):
            
            
            if has_enough_cache(block_num_used, inp_lens[pointer],consider_watermark=True):
                running_seqs[0][running_seqs_num] = seq_ids[pointer] 
                running_seqs[1][running_seqs_num] = inp_lens[pointer] + 1
                running_seqs[2][running_seqs_num] = out_lens[pointer] - 1


                new_prompt_lens.append(inp_lens[pointer])
                new_prompt_ids.append(seq_ids[pointer])
                if running_seqs[2][running_seqs_num] == 0:
                    
                    unfinished_reqnum -= 1
                    pointer += 1
                    continue


                
                block_num_used = add_block_num_used(block_num_used, inp_lens[pointer])
                pointer += 1
                running_seqs_num += 1
            else:
                break



        
        
        
        if (len(seq_ids)<sum(req_nums)) \
            and (pointer == len(seq_ids)) \
            and has_enough_cache(block_num_used,1,consider_watermark=True) \
                and (running_seqs_num < max_seq_num) \
                    and (len(seq_ids) < len(ref_seq_ids)):
            
            
            
            need_query_available_requests = True
        else:
            need_query_available_requests = False


        
        
        
        
        
        


        
        
        
        new_prompt_lens = np.asarray(new_prompt_lens)
        new_prompt_ids = np.asarray(new_prompt_ids)
        ori_prefill_logs_num = len(prefill_logs)

        
        

        
        
        
        
        
        ori_last_iter_seq_ids = last_iter_seq_ids.copy()
        last_iter_seqs, last_iter_seq_ids, tot_iter_num = update_prefill_logs(prefill_logs, 
                            valid_prefill_logs,                                                  
                            new_prompt_lens, max_num_batched_tokens,
                            new_prompt_ids, infer_progress, tot_iter_num, 
                            need_query_available_requests, check_gap, last_iter_seqs, last_iter_seq_ids, 
                            must_record_first_step)
        is_prefill_steps.extend([True]*(tot_iter_num - len(is_prefill_steps)))

        
        

        
        run_prompt_ids = np.setdiff1d(
            np.concatenate((new_prompt_ids, ori_last_iter_seq_ids)), last_iter_seq_ids, 
            assume_unique=True)
        finished_seq_ids.extend(
            np.setdiff1d(run_prompt_ids, running_seqs[0][:running_seqs_num], assume_unique=True))
        


        
        

        
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=prefill_logs[ori_prefill_logs_num:], decode_logs=list(), **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(prefill_latencys)+tot_inference_time))
        if len(cumsum_latencys) > 0:
            tot_inference_time = cumsum_latencys[-1]
        
        
        
        
        

        
        if len(last_iter_seqs) > 0:
            must_record_first_step = True
            
            continue
        elif len(new_prompt_ids) > 0:
            
            must_record_first_step = False

        if running_seqs_num == 0:
            
            
            continue

        
        
        
        
        
        
        
        

        block_num_used, running_seqs_num, inp_lens, out_lens, seq_ids, pointer = \
            kill_seqs_for_more_cache_space(
                running_seqs, 
                max_block_num, block_size, running_seqs_num, 
                inp_lens, out_lens, seq_ids, pointer)

        

        
        
        
        
        iter_num = get_max_iter_num(block_num_used, running_seqs_num, running_seqs)
        
        

        
        if pointer < len(seq_ids):
            need_query_available_requests = False

        
        if need_query_available_requests:
            iter_num = min(iter_num, (tot_iter_num + must_record_first_step + check_gap - 1) // check_gap * check_gap \
                - tot_iter_num)
            must_record_first_step = False
            
            if iter_num == 0:
                must_record_first_step = True
                
                continue

        
        
        
        
        
        tot_token_num = get_tot_token_num(running_seqs_num, running_seqs)
        curr_max_seqlen = get_max_seqlen(running_seqs_num, running_seqs)
        logs.extend([(running_seqs_num, 
                      tot_token_num + running_seqs_num*i,
                      tot_token_num + running_seqs_num*i,
                      curr_max_seqlen + i) \
                     for i in range(iter_num)])
        _store_infer_state(tot_iter_num, tot_iter_num+iter_num-1, infer_progress, running_seqs[0][:running_seqs_num])
        is_prefill_steps.extend([False]*iter_num)
        tot_iter_num += iter_num

        
        

        
        
        running_seqs[1][:running_seqs_num] = running_seqs[1][:running_seqs_num] + iter_num
        running_seqs[2][:running_seqs_num] = running_seqs[2][:running_seqs_num] - iter_num

        
        finished_seq_ids.extend(running_seqs[0][:running_seqs_num][running_seqs[2][:running_seqs_num]==0])

        
        running_seqs_num, unfinished_reqnum, block_num_used = \
            remove_finished_seqs(running_seqs, running_seqs_num, unfinished_reqnum, block_size)

        
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=list(), decode_logs=logs[-iter_num:], **cost_estimate_args)

        cumsum_latencys = np.concatenate((cumsum_latencys, np.cumsum(decode_latencys)+tot_inference_time))
        tot_inference_time = cumsum_latencys[-1]

        
        
        

        
        
        
        

        
        
    
    
    
    
    
    
    
    print(f"tot schedule time: {time.perf_counter() - time_start}")
    
    assert tot_iter_num == (len(logs) + len(prefill_logs)), (tot_iter_num, len(logs), len(prefill_logs), ori_inplens, ori_outlens, max_seq_num, max_block_num, max_num_batched_tokens, block_size) 
    
    return cumsum_latencys, is_prefill_steps, full_infer_progress, prefill_logs, logs, valid_prefill_logs












def get_finish_times(cumsum_latencys: List[float], infer_progress: List[List[int]]):
    """
        We use a list to store the continuous inference iteration ranges for each sequence.
            E.g., 
                [start1, end1, start2, end2, ...] --> for iter_i with 
                    start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
                the seq attend the corresponding iteration steps.
    """
    
    
    last_iters = [rng[-1] for rng in infer_progress]
    finish_times = cumsum_latencys[last_iters]
    return finish_times



def get_finish_times_from_rng_infos(
        cumsum_latencys: List[float], 
        cum_rng_nums: List[int], rng_ends: List[int],
        ):
    """
        We use a list to store the continuous inference iteration ranges for each sequence.
            E.g., 
                [start1, end1, start2, end2, ...] --> for iter_i with 
                    start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
                the seq attend the corresponding iteration steps.
    """
    last_iters = rng_ends[cum_rng_nums[1:] - 1]
    finish_times = cumsum_latencys[last_iters]
    return finish_times





def fake_FCFS_schedule(
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    """
        This function calls ``_fake_FCFS_schedule_NO_continuous_model_level_pipeline`` or
        ``_fake_FCFS_schedule_continuous_model_level_pipeline`` depending on whether arrive_times is empty.

        Input:
            cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}
        
        Output: 
            cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times, throughput_till_each_iter

        NOTE: finish_times: the finish times of each request.
    """

    if len(inp_lens) == 0:
        return [], [0], [], [], [], [], []

    if (len(arrive_times) == 0) or (max(arrive_times)<0):

        
        
        
        
        
        
        
        
        

        
        
        
            
        inp_lens = np.asarray(inp_lens)
        out_lens = np.asarray(out_lens)

        
        decode_logs, prefill_logs, is_prefill_steps, infer_progress, valid_prefill_logs =  \
            _fake_FCFS_schedule_NO_continuous_model_level_pipeline(
                inp_lens=inp_lens, out_lens=out_lens,
                max_seq_num=max_seq_num, max_block_num=max_block_num, max_num_batched_tokens=max_num_batched_tokens,
                block_size=block_size)
        
        
        tot_latency, prefill_latencys, decode_latencys = \
            _estimate_prefill_and_decode_cost_from_predicted_logs(
                prefill_logs=prefill_logs, decode_logs=decode_logs, **cost_estimate_args)

        
        (cumsum_latencys, cum_rng_nums, rng_starts, rng_ends) = \
            get_cumLatency_inferRng_info(
                    decode_latencys, prefill_latencys, 
                    is_prefill_steps, infer_progress)

        
        finish_times = get_finish_times(cumsum_latencys, infer_progress)


        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        

        throughput_till_each_iter = comp_throughput_for_each_iteration_given_logs(
            prefill_logs=prefill_logs, valid_prefill_logs=valid_prefill_logs, decode_logs=decode_logs, is_prefill_steps=is_prefill_steps,
            cumsum_latencys=cumsum_latencys, 
            extra_cost=0, 
            cost_table=cost_estimate_args["cost_table"], model_path=cost_estimate_args["model_name"], 
            trust_remote_code=cost_estimate_args["trust_remote_code"], revision=cost_estimate_args["revision"])

        


        return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times, throughput_till_each_iter
    else:

        print(f"Has input exec plans in this stage")
        
        
        

        
        cumsum_latencys, is_prefill_steps, infer_progress, prefill_logs, decode_logs, valid_prefill_logs = _fake_FCFS_schedule_continuous_model_level_pipeline(
            inp_lens=inp_lens,out_lens=out_lens, arrive_times=arrive_times, check_gap=check_gap,
            max_seq_num=max_seq_num, max_block_num=max_block_num, max_num_batched_tokens=max_num_batched_tokens,
            block_size=block_size, sort_input=sort_input, cost_estimate_args=cost_estimate_args)

        

        
        cum_rng_nums, rng_starts, rng_ends = _get_inferRng_info(infer_progress)

        
        finish_times = get_finish_times(cumsum_latencys, infer_progress)

        
        
        
        
        
        
        

        
        
        
        
        
        
        

        throughput_till_each_iter = comp_throughput_for_each_iteration_given_logs(
            prefill_logs=prefill_logs, valid_prefill_logs=valid_prefill_logs, decode_logs=decode_logs, is_prefill_steps=is_prefill_steps,
            cumsum_latencys=cumsum_latencys, 
            extra_cost=0, 
            cost_table=cost_estimate_args["cost_table"], model_path=cost_estimate_args["model_name"], 
            trust_remote_code=cost_estimate_args["trust_remote_code"], revision=cost_estimate_args["revision"])

        return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times, throughput_till_each_iter










def fake_FCFS_schedule_vertical_fuse(
        inp_lens: List[int], out_lens: List[int], arrive_times: List[float], ref_seq_ids: List[int],
        
        ref_seq_ids_list: List[List[int]],
        inp_lens_list: List[List[int]],
        out_lens_list: List[List[int]],
        arrive_times_list: List[List[int]],        
        
        check_gap: int,
        max_seq_num: int, max_block_num: int, max_num_batched_tokens: int, 
        block_size: int,
        sort_input: bool,
        cost_estimate_args,
        ):
    """
        This function calls ``_fake_FCFS_schedule_NO_continuous_model_level_pipeline`` or
        ``_fake_FCFS_schedule_continuous_model_level_pipeline`` depending on whether arrive_times is empty.

        Input:
            cost_estimate_args: {"cost_table"=cost_table, "model_name"=model_name, "exec_plan"=exec_plan, "sample_config"=sample_config, 
                "trust_remote_code"=trust_remote_code, "revision"=revision}
        
        Output: 
            cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times, throughput_till_each_iter

        NOTE: 
            1. finish_times: the finish times of each request.
            2. support the vertical fusion of models.
    """

    
    
    
    

    
    
    

    
    
    
    if (len(inp_lens) + sum([len(_) for _ in inp_lens_list])) == 0:
        model_num = 1+len(inp_lens_list)
        return [], [[0] for _ in range(model_num)], [[] for _ in range(model_num)], [[] for _ in range(model_num)], [], [[] for _ in range(model_num)], []

    if len(arrive_times) == 0:
        arrive_times = np.asarray([-1]*len(inp_lens))
    
    for i in range(len(arrive_times_list)):
        if len(arrive_times_list[i]) == 0:
            arrive_times_list[i] = np.asarray([-1]*len(inp_lens_list[i]))

    
    
    
    
    
    
    

    
    cumsum_latencys, is_prefill_steps, full_infer_progress, prefill_logs, decode_logs, valid_prefill_logs = \
        _fake_FCFS_schedule_continuous_model_level_pipeline_vertical_fuse(
            
            inp_lens, out_lens, arrive_times, ref_seq_ids,
            
            ref_seq_ids_list,
            inp_lens_list,
            out_lens_list,
            arrive_times_list,      
            
            check_gap,
            max_seq_num, max_block_num, max_num_batched_tokens,
            block_size,
            sort_input,
            cost_estimate_args,
            )

    

    cum_rng_nums_list, rng_starts_list, rng_ends_list, finish_times_list = list(), list(), list(), list()
    for infer_progress in full_infer_progress:
        if len(infer_progress) == 0:
            cum_rng_nums_list.append(np.asarray([0], dtype=np.int64))
            rng_starts_list.append(np.asarray([], dtype=np.int64))
            rng_ends_list.append(np.asarray([], dtype=np.int64))
            finish_times_list.append(np.asarray([]))
            continue

        
        cum_rng_nums, rng_starts, rng_ends = _get_inferRng_info(infer_progress)
        cum_rng_nums_list.append(cum_rng_nums)
        rng_starts_list.append(rng_starts)
        rng_ends_list.append(rng_ends)

        
        
        finish_times = get_finish_times(cumsum_latencys, infer_progress)
        finish_times_list.append(finish_times)


    
    
    
    
    
    
    
    
    
    

    throughput_till_each_iter = comp_throughput_for_each_iteration_given_logs(
        prefill_logs=prefill_logs, valid_prefill_logs=valid_prefill_logs, decode_logs=decode_logs, is_prefill_steps=is_prefill_steps,
        cumsum_latencys=cumsum_latencys, 
        extra_cost=0, 
        cost_table=cost_estimate_args["cost_table"], model_path=cost_estimate_args["model_name"], 
        trust_remote_code=cost_estimate_args["trust_remote_code"], revision=cost_estimate_args["revision"])   


    return cumsum_latencys, cum_rng_nums_list, rng_starts_list, rng_ends_list, is_prefill_steps, finish_times_list, throughput_till_each_iter











def _update_fake_FCFS_schedule_metadata(
        old_inp_lens: List[int], 
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int],
        is_prefill_steps: List[bool], 
        throughput_till_each_iter: List[float],
        max_num_batched_tokens: int, stop_iter_i: int,
        cost_table: CostTable, 
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Compute the fake FCFS scheduling metadata restart from the iter ``stop_iter_i+1'' based on the given metadata.
        NOTE: we restart all running seqs at iter ``stop_iter_i'' and not finished after iter ``stop_iter_i'' ends.
        We use prefill steps to recover their seq lens after iter ``stop_iter_i'' ends.
            (1) We do not consider the ``watermark'' constraint in the prefill stage.
        NOTE:
            this function assumes there are running seqs after iter ``stop_iter_i'' ends.
    '''

    
    
    
    
    
    

    
    new_rng_starts = rng_starts - (stop_iter_i + 1)
    new_rng_ends = rng_ends - (stop_iter_i + 1)
    
    cum_alive_rng_nums = np.cumsum(np.concatenate(([0], (new_rng_ends >= 0))))
    alive_rng_nums = (cum_alive_rng_nums[cum_rng_nums[1:]] - cum_alive_rng_nums[cum_rng_nums[:-1]])
    
    alive_cum_rng_nums = np.cumsum(np.concatenate(([0], alive_rng_nums)))
    
    new_rng_starts = np.maximum(new_rng_starts[new_rng_ends >= 0], 0)
    new_rng_ends = new_rng_ends[new_rng_ends >= 0]

    
    
    
    
    

    
    start_from_prefillstep = np.asarray(is_prefill_steps)[new_rng_starts[alive_cum_rng_nums[:-1][alive_rng_nums>0]] + (stop_iter_i+1)]
    
    
    
    
    
    running_seq_ids = np.nonzero(alive_rng_nums>0)[0][ np.nonzero(start_from_prefillstep==False)[0] ]

    
    
    

    


    prefill_logs = list() 
    
    infer_progress = list([] for _ in range(len(old_inp_lens)))
    
    tot_iter_num: int = 0 

    tmp_throughput_till_each_iter: List[float] = list()

    
    valid_prefill_logs: List[Tuple[int, int]] = list()


    
    
    finished_lens = np.cumsum(np.concatenate(
        ([0], ((stop_iter_i >= rng_starts) * (np.minimum(stop_iter_i, rng_ends) - rng_starts + 1)))
        ))
    finished_lens = finished_lens[cum_rng_nums[1:]] - finished_lens[cum_rng_nums[:-1]]

    
    
    

    
    
    new_prompt_lens = np.asarray(old_inp_lens)[running_seq_ids] + finished_lens[running_seq_ids]
    
    new_prompt_ids = running_seq_ids
    
    
    
    _, _, tot_iter_num = update_prefill_logs(prefill_logs, 
                        valid_prefill_logs, 
                        new_prompt_lens, max_num_batched_tokens,
                        new_prompt_ids, infer_progress, tot_iter_num,
                        need_query_available_requests=False, check_gap=1,
                        last_iter_seqs=list(), last_iter_seq_ids=list(),
                        must_record_first_step=False)

    tmp_is_prefill_steps = np.concatenate(([True]*len(prefill_logs), is_prefill_steps[stop_iter_i+1:]))
    

    
    
    
    
    
    prefill_latencys = estimate_cost_from_predicted_logs(
        prefill_logs, cost_table=cost_table, is_prompt=True,
        model_name=model_name, exec_plan=exec_plan, sample_config=sample_config, 
        trust_remote_code=trust_remote_code, revision=revision)
    tmp_cum_latencys = cumsum_latencys[stop_iter_i+1:] - cumsum_latencys[stop_iter_i]
    cum_prefill_latencys = np.cumsum(prefill_latencys)
    tmp_cum_latencys = np.concatenate((cum_prefill_latencys, (sum(prefill_latencys)+tmp_cum_latencys)))

    
    tmp_throughput_till_each_iter = comp_throughput_for_each_iteration_given_logs(
        prefill_logs=prefill_logs, valid_prefill_logs=valid_prefill_logs, decode_logs=[], is_prefill_steps=tmp_is_prefill_steps[:len(prefill_logs)],
        cumsum_latencys=cum_prefill_latencys, 
        extra_cost=0, 
        cost_table=cost_table, model_path=model_name, 
        trust_remote_code=trust_remote_code, revision=revision)
    remaining_cum_flops = throughput_till_each_iter[stop_iter_i:]*cumsum_latencys[stop_iter_i:]
    remaining_cum_flops = remaining_cum_flops[1:] - remaining_cum_flops[0]
    remaining_throughput_till_each_iter = None
    if len(cum_prefill_latencys) == 0:
        remaining_throughput_till_each_iter = (remaining_cum_flops) / tmp_cum_latencys[len(cum_prefill_latencys):]        
    else:
        remaining_throughput_till_each_iter = (tmp_throughput_till_each_iter[-1]*cum_prefill_latencys[-1] + remaining_cum_flops) / tmp_cum_latencys[len(cum_prefill_latencys):]
    tmp_throughput_till_each_iter = np.concatenate( (tmp_throughput_till_each_iter, remaining_throughput_till_each_iter) )

    
    

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    running_seq_first_rng_ids = alive_cum_rng_nums[running_seq_ids]
    first_decode_iter_is = new_rng_starts[running_seq_first_rng_ids].copy()
    new_rng_starts[running_seq_first_rng_ids] += 1
    
    
    
    invalid_cum_rng_nums = np.cumsum(np.concatenate(([0], new_rng_starts > new_rng_ends)))
    reduce_rng_nums = invalid_cum_rng_nums[alive_cum_rng_nums[1:]] - invalid_cum_rng_nums[alive_cum_rng_nums[:-1]]
    alive_cum_rng_nums[1:] -= np.cumsum(reduce_rng_nums)

    keep_rng_ids = (new_rng_starts <= new_rng_ends)
    new_rng_starts = new_rng_starts[keep_rng_ids]
    new_rng_ends = new_rng_ends[keep_rng_ids]

    
    


    assert len(new_rng_starts) == alive_cum_rng_nums[-1]

    
    
    
    
    
    


    
    
    
    
    first_decode_iter_is = np.unique(first_decode_iter_is)
    
    
    for first_decode_iter_i in first_decode_iter_is:
        
        if not ((first_decode_iter_i>=new_rng_starts)*(first_decode_iter_i<=new_rng_ends)).any():
            

            
            

            new_rng_starts[new_rng_starts>=first_decode_iter_i] -= 1
            new_rng_ends[new_rng_ends>=first_decode_iter_i] -= 1
            
            tmp_cum_latencys[first_decode_iter_i+len(prefill_latencys):-1] = \
                tmp_cum_latencys[first_decode_iter_i+len(prefill_latencys)+1:]
            tmp_cum_latencys = tmp_cum_latencys[:-1]

            tmp_throughput_till_each_iter[first_decode_iter_i+len(prefill_latencys):-1] = \
                tmp_throughput_till_each_iter[first_decode_iter_i+len(prefill_latencys)+1:]
            tmp_throughput_till_each_iter = tmp_throughput_till_each_iter[:-1]
            
            tmp_is_prefill_steps[first_decode_iter_i+len(prefill_latencys):-1] = \
                tmp_is_prefill_steps[first_decode_iter_i+len(prefill_latencys)+1:]
            tmp_is_prefill_steps = tmp_is_prefill_steps[:-1]

            


    
    
    
    
    
    
    
    
    
    new_rng_starts += len(prefill_logs)
    new_rng_ends += len(prefill_logs)

    

    
    need_add_rng = np.full((len(old_inp_lens)), False)
    need_add_rng[new_prompt_ids] = True

    alive_prompt_ids = new_prompt_ids[np.diff(alive_cum_rng_nums)[new_prompt_ids]>0]
    alive_prompt_first_rng_ids = alive_cum_rng_nums[new_prompt_ids][np.diff(alive_cum_rng_nums)[new_prompt_ids]>0]
    prefill_rng_ends = np.asarray([infer_progress[seq_i][1] for seq_i in alive_prompt_ids])
    
    
    
    
    
    


    need_add_rng[alive_prompt_ids] = new_rng_starts[alive_prompt_first_rng_ids] > (prefill_rng_ends + 1)
    old_alive_cum_rng_nums = alive_cum_rng_nums.copy()
    
    alive_cum_rng_nums[1:] += np.cumsum(need_add_rng)

    tmp_rng_starts = np.empty([len(new_rng_starts)+sum(need_add_rng)], dtype=new_rng_starts.dtype)
    tmp_rng_ends = np.empty_like(tmp_rng_starts)
    
    
    
    

    

    for seq_i, add_rng in zip(range(len(old_inp_lens)), need_add_rng):
        i = alive_cum_rng_nums[seq_i]
        j = alive_cum_rng_nums[seq_i+1]
        if i==j:
            
            continue
        old_i = old_alive_cum_rng_nums[seq_i]
        old_j = old_alive_cum_rng_nums[seq_i+1]

        
        
        

        if add_rng:
            start, end = infer_progress[seq_i]
            tmp_rng_starts[i+1:j] = new_rng_starts[old_i:old_j]
            tmp_rng_ends[i+1:j] = new_rng_ends[old_i:old_j]
            tmp_rng_starts[i] = start
            tmp_rng_ends[i] = end
        else:
            
            tmp_rng_starts[i:j] = new_rng_starts[old_i:old_j]
            tmp_rng_ends[i:j] = new_rng_ends[old_i:old_j]
            if len(infer_progress[seq_i])>0:
                start, end = infer_progress[seq_i]
                tmp_rng_starts[i] = start




    
    
    
    alive_cum_rng_nums, valid_indices = np.unique(alive_cum_rng_nums, return_index=True)
    
    valid_indices = valid_indices[1:] - 1

    

    
    
    
    
    
    
    
    assert len(tmp_rng_starts) == len(tmp_rng_ends)
    assert len(tmp_rng_starts) == len(tmp_rng_ends)
    assert len(tmp_cum_latencys) == len(tmp_is_prefill_steps)
    assert max(tmp_rng_ends) == len(tmp_is_prefill_steps)-1
    assert alive_cum_rng_nums[-1] == len(tmp_rng_starts)
    return tmp_cum_latencys, alive_cum_rng_nums, tmp_rng_starts, tmp_rng_ends, tmp_is_prefill_steps, valid_indices, tmp_throughput_till_each_iter



def update_fake_FCFS_schedule_metadata(
        old_inp_lens: List[int], 
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int],
        is_prefill_steps: List[bool],
        throughput_till_each_iter: List[float],
        max_num_batched_tokens: int, stop_iter_i: int,
        cost_table: CostTable, 
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Compute the fake FCFS scheduling metadata restart from the iter ``stop_iter_i+1'' based on the given metadata.
        NOTE: we restart all running seqs at iter ``stop_iter_i'' and not finished after iter ``stop_iter_i'' ends.
        We use prefill steps to recover their seq lens after iter ``stop_iter_i'' ends.
            (1) We do not consider the ``watermark'' constraint in the prefill stage.
        NOTE:
            this function assumes there are running seqs after iter ``stop_iter_i'' ends.
    '''
    
    if (len(cumsum_latencys) == 0) or (stop_iter_i == (len(cumsum_latencys)-1)):
        
        return np.asarray([]), np.asarray([0]), np.asarray([]), np.asarray([]), np.asarray([]), \
            np.asarray([]), np.asarray([]), np.asarray([])


    cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, valid_indices, throughput_till_each_iter = \
        _update_fake_FCFS_schedule_metadata(
            old_inp_lens, 
            cumsum_latencys, cum_rng_nums, rng_starts, rng_ends,
            is_prefill_steps,
            throughput_till_each_iter, 
            max_num_batched_tokens, stop_iter_i,
            cost_table, 
            model_name, exec_plan, sample_config, trust_remote_code, revision)
    
    
    finish_times = get_finish_times_from_rng_infos(cumsum_latencys, cum_rng_nums, rng_ends)
    return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends, is_prefill_steps, finish_times, valid_indices, throughput_till_each_iter













def plot_seq_curve(logs, tag:str):
    import matplotlib.pyplot as plt
    seq_nums = [i[0] for i in logs]
    fig, ax = plt.subplots()
    ax.plot(range(len(seq_nums)), seq_nums)
    ax.set(xlabel='iter', ylabel='seq_num',)
        
    ax.grid()
    fig.savefig(f"./test_sampler/seq_nums{'Llama-2-7b-hf'}_tp{1}_{tag}.png")
    plt.show()
    plt.close(fig)

def plot_cum_seqnum_curve(logs, tag:str):
    import matplotlib.pyplot as plt
    seq_nums = np.cumsum([i[0] for i in logs])
    fig, ax = plt.subplots()
    ax.plot(range(len(seq_nums)), seq_nums)
    ax.set(xlabel='iter', ylabel='seq_num',)
        
    ax.grid()
    fig.savefig(f"./test_sampler/cum_seq_nums{'Llama-2-7b-hf'}_tp{1}_{tag}.png")
    plt.show()
    plt.close(fig)



def plot_latency_curve(latencys, tag: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(len(latencys)), latencys)
    ax.set(xlabel='iter', ylabel='latency (s)',)
        
    ax.grid()
    fig.savefig(f"./test_sampler/latency_{'Llama-2-7b-hf'}_{tag}.png")
    plt.show()
    plt.close(fig)





def estimate_cost_from_predicted_logs(
        logs: List[Tuple[int, int]], cost_table: CostTable, is_prompt: bool,
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Estimate the costs of the iterations in the logs.
    '''
    seqnums = np.asarray([i[0] for i in logs])
    context_tot_lens = np.asarray([i[1] for i in logs])

    s = np.asarray([i[-1] for i in logs])
    
    
    
    
    
    

    latencys = cost_table.estimate_cost(
        seqnums, s, context_tot_lens, is_prompt, 
        model_name, exec_plan, sample_config, trust_remote_code, revision)
    return latencys



def _estimate_prefill_and_decode_cost_from_predicted_logs(
        prefill_logs: List[Tuple[int, int]], decode_logs: List[Tuple[int, int]],
        cost_table: CostTable,
        model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None
        ) -> Tuple[float, List[float], List[float]]:
    prefill_latencys = estimate_cost_from_predicted_logs(
        prefill_logs, cost_table=cost_table, is_prompt=True,
        model_name=model_name, exec_plan=exec_plan, sample_config=sample_config, 
        trust_remote_code=trust_remote_code, revision=revision)
    decode_latencys = estimate_cost_from_predicted_logs(
        decode_logs, cost_table=cost_table, is_prompt=False,
        model_name=model_name, exec_plan=exec_plan, sample_config=sample_config, 
        trust_remote_code=trust_remote_code, revision=revision)

    return sum(prefill_latencys)+sum(decode_latencys), prefill_latencys, decode_latencys




'''
We use a list to store the continuous inference iteration ranges for each sequence.
    E.g., 
        [start1, end1, start2, end2, ...] --> for iter_i with 
            start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
        the seq attend the corresponding iteration steps.
'''
def get_info_at_stop_time_slowVersion(
        decode_latencys: List[float], prefill_latencys: List[float], 
        is_prefill_steps: List[bool], infer_progress: List[List[int]], 
        stop_time: float):
    '''
        Get the seq statuses at the given stop time.
        Output:
            1. finished seq lengths;
            2. remaining seq lengths.
    '''    
    is_prefill_steps = np.asarray(is_prefill_steps)


    import time
    time1 = time.perf_counter()

    
    latencys = np.empty(len(decode_latencys)+len(prefill_latencys))
    latencys[is_prefill_steps==True] = prefill_latencys
    latencys[is_prefill_steps==False] = decode_latencys
    cumsum_latencys = np.cumsum(latencys)
    
    stop_iter_i = np.searchsorted(cumsum_latencys, stop_time, side='left')

    
    finished_lens = [0 for i in range(len(infer_progress))]
    remaining_lens = [0 for i in range(len(infer_progress))]
    for seq_i, rngs in enumerate(infer_progress):
        
        
        i = np.searchsorted(rngs, stop_iter_i, side='left')
        tmp_rngs = np.asarray(rngs).reshape((-1, 2))
        rng_lens = tmp_rngs[:,1]-tmp_rngs[:,0]+1
        
        if i%2 == 1:
            
            finished_lens[seq_i] = sum(rng_lens[:i//2]) + stop_iter_i - rngs[i-1] + 1
        else:
            
            finished_lens[seq_i] = sum(rng_lens[:i//2])
            if (i < len(rngs)) and (stop_iter_i == rngs[i]):
                finished_lens[seq_i] = finished_lens[seq_i] + 1
        
        remaining_lens[seq_i] = sum(rng_lens) - finished_lens[seq_i]


    time2 = time.perf_counter()
    print(f"time: {time2 - time1}")

    
    another_res = get_info_at_stop_time(
        decode_latencys, prefill_latencys, 
        is_prefill_steps, infer_progress, 
        stop_time)
    assert ((finished_lens==another_res[0]).all()) and ((remaining_lens==another_res[1]).all()), (finished_lens, remaining_lens, another_res)


    return finished_lens, remaining_lens



def _get_inferRng_info(infer_progress: List[List[int]]):
    '''
        Get the infer ranges information for each seq in the inference process.
    '''  
    
    rng_nums = np.asarray([len(rngs) for rngs in infer_progress])//2
    cum_rng_nums = np.cumsum(np.concatenate(([0], rng_nums)))
    concat_infer_progress = np.concatenate(infer_progress).reshape((-1, 2))
    rng_starts = concat_infer_progress[:,0]
    rng_ends = concat_infer_progress[:,1]
    
    return cum_rng_nums, rng_starts, rng_ends





def get_cumLatency_inferRng_info(
        decode_latencys: List[float], prefill_latencys: List[float], 
        is_prefill_steps: List[bool], infer_progress: List[List[int]]):
    '''
        Get the cumulative latencys and the infer ranges information for each seq in the inference process.
    '''  
    
    is_prefill_steps = np.asarray(is_prefill_steps)
    latencys = np.empty(len(decode_latencys)+len(prefill_latencys))
    latencys[is_prefill_steps==True] = prefill_latencys
    latencys[is_prefill_steps==False] = decode_latencys
    cumsum_latencys = np.cumsum(latencys)

    
    
    
    
    
    
    cum_rng_nums, rng_starts, rng_ends = _get_inferRng_info(infer_progress)
    return cumsum_latencys, cum_rng_nums, rng_starts, rng_ends




'''
We use a list to store the continuous inference iteration ranges for each sequence.
    E.g., 
        [start1, end1, start2, end2, ...] --> for iter_i with 
            start1 <= iter_i <= end1, or start2 <= iter_i <= end2, or ..., 
        the seq attend the corresponding iteration steps.
'''
def get_info_at_stop_time( 
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int], 
        stop_time: float, stop_iter_i: int):
    '''
        Get the seq statuses at the given stop time.
        Output:
            1. finished seq lengths;
            2. remaining seq lengths.
        NOTE: this is the fast version.
    '''    

    
    
    

    
    
    
    
    finished_lens = (stop_iter_i>=rng_starts) * (np.minimum(stop_iter_i, rng_ends)-rng_starts+1)
    finished_lens = np.cumsum(np.concatenate(([0], finished_lens)))
    finished_lens = finished_lens[cum_rng_nums[1:]] - finished_lens[cum_rng_nums[:-1]]
    
    
    
    
    


    return finished_lens












def comp_flops_from_seqlens(
        inp_lens: List[int], out_lens: List[int], only_decode, cost_table: CostTable, 
        model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    if only_decode:
        B_array = np.asarray([sum(out_lens)])
        s_array = np.asarray([1])
        inp_lens = np.asarray(inp_lens)
        out_lens = np.asarray(out_lens)
        context_tot_len_array = np.asarray([sum((2*inp_lens+out_lens-1)*out_lens/2)])
        tp_size=2
        flops = cost_table.comp_flops(
            tp_size,
            B_array, s_array, context_tot_len_array, is_prompt=False,
            model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
        return flops*tp_size
    else:
        
        inp_lens = np.asarray(inp_lens)
        out_lens = np.asarray(out_lens)
        B_array = np.asarray([sum(inp_lens+out_lens-1)])
        s_array = np.asarray([1])
        context_tot_len_array = np.asarray([sum((inp_lens+out_lens)*(inp_lens+out_lens-1)/2)])
        tp_size=2
        print(f"B_array: {B_array}, context_tot_len_array: {context_tot_len_array}")
        flops = cost_table.comp_flops(
            tp_size,
            B_array, s_array, context_tot_len_array, is_prompt=False,
            model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
        return flops*tp_size


def comp_valid_throughput_at_stop_time(
        inp_lens: List[int],
        finished_lens: List[int],
        stop_time: float, cost_table: CostTable,
        model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    '''
        Valid flops means we only consider the necessary flops in model computation 
        (not including 
            1. prepare input tensor and sampling, 
            2. as well as the waste flops due to padding or recomputation after kill).
            3. as well as the waste flops that may be due to tensor parallelism (e.g., kv_head_num < tp_size).
        Valid throughput is computed based on valid flops.
    '''

    
    
    
    
    
    
    
    
    
    
    inp_lens_array = np.asarray(inp_lens)
    finished_lens_array = np.asarray(finished_lens)
    
    
    tot_lens_array = (inp_lens_array + finished_lens_array - 1) * (finished_lens_array>0)
    
    
    
    
    
    
    
    
    
    
    


    flops = comp_flops_from_seqlens(
        inp_lens=tot_lens_array, out_lens=[1], only_decode=False, cost_table=cost_table, 
        model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)


    
    throughput = flops / stop_time
    print(f"flops: {flops}, stop_time: {stop_time}, throughput: {throughput}")

    return throughput










def comp_throughput_for_each_iteration(
        inp_lens: List[int],
        cumsum_latencys: List[float], cum_rng_nums: List[int], rng_starts: List[int], rng_ends: List[int], 
        extra_cost: float, 
        cost_table: CostTable, model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    """
        This function compute the throughput for each iteration.
    """

    
    import time
    time1 = time.perf_counter()

    print(f"len(cumsum_latencys): {len(cumsum_latencys)}, len(rng_starts): {len(rng_starts)}")

    
    
    B_array: List[int] = np.zeros((len(cumsum_latencys),), dtype=int)

    print(f"comp_throughput_for_each_iteration: (0.3/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()
    
    
    for start, end in zip(rng_starts, rng_ends):
        B_array[start:end+1] = B_array[start:end+1] + 1

    
    

    
    
    
    
    

    print(f"comp_throughput_for_each_iteration: (0.6/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()
    
    
    for inp_len, i in zip(inp_lens, cum_rng_nums[:-1]):
        B_array[rng_starts[i]] = B_array[rng_starts[i]] + (inp_len-1)

    print(f"comp_throughput_for_each_iteration: (0.9/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()
    
    
    B_array = np.cumsum(B_array)


    print(f"comp_throughput_for_each_iteration: (1/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()

    
    context_tot_len_array: List[int] = np.zeros((len(cumsum_latencys),), dtype=int)

    
    for ind in range(len(cum_rng_nums)-1):
        i, j = cum_rng_nums[ind], cum_rng_nums[ind+1]
        inp_len = inp_lens[ind]
        gened_len_start = 0
        for start, end in zip(rng_starts[i:j], rng_ends[i:j]):
            context_tot_len_array[start:end+1] = context_tot_len_array[start:end+1] + inp_len + np.arange(gened_len_start, gened_len_start+end+1-start)
            gened_len_start = gened_len_start + end+1-start
    
        
        context_tot_len_array[rng_starts[i]] = context_tot_len_array[rng_starts[i]] - inp_len + inp_len*(inp_len+1)/2

    
    context_tot_len_array = np.cumsum(context_tot_len_array)

    print(f"comp_throughput_for_each_iteration: (2/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()


    
    s_array = np.asarray([1])
    tp_size=2
    flops = cost_table.comp_flops(
        tp_size,
        B_array, s_array, context_tot_len_array, is_prompt=False,
        model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
    
    flops = flops*tp_size

    print(f"comp_throughput_for_each_iteration: (3/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()


    
    throughputs = (np.asarray(cumsum_latencys) + extra_cost) / flops
    return throughputs





def comp_throughput_for_each_iteration_given_metadata(
        delta_tot_lens: List[int], pre_comp_running_lens: List[int],
        cumsum_latencys: List[float], 
        extra_cost: float, 
        cost_table: CostTable, model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    """
        This function compute the throughput for each iteration.
    """

    
    import time
    time1 = time.perf_counter()
    
    
    B_array = np.cumsum(delta_tot_lens)


    print(f"comp_throughput_for_each_iteration: (1/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()

    
    context_tot_len_array = np.cumsum(pre_comp_running_lens)

    print(f"comp_throughput_for_each_iteration: (2/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()


    
    s_array = np.asarray([1])
    tp_size=2
    flops = cost_table.comp_flops(
        tp_size,
        B_array, s_array, context_tot_len_array, is_prompt=False,
        model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)[0]/1e12
    
    flops = flops*tp_size

    print(f"comp_throughput_for_each_iteration: (3/3): {time.perf_counter() - time1}")
    time1 = time.perf_counter()


    
    throughputs = (np.asarray(cumsum_latencys) + extra_cost) / flops
    return throughputs










def comp_throughput_for_each_iteration_given_logs(
        prefill_logs: List[Tuple[int, int, int, int]], valid_prefill_logs: List[Tuple[int, int]], 
        decode_logs: List[Tuple[int, int, int, int]], is_prefill_steps: List[bool],
        cumsum_latencys: List[float], 
        extra_cost: float, 
        cost_table: CostTable, model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
    """
        This function compute the throughput for each iteration.
    """

    
    
    
    


    

    is_prefill_steps = np.asarray(is_prefill_steps)


    
    import time
    time1 = time.perf_counter()

    
    

    
    
    
    seqnums = np.empty(len(prefill_logs)+len(decode_logs))
    
    
    seqnums[is_prefill_steps==True] = np.asarray([i[0] for i in valid_prefill_logs])
    seqnums[is_prefill_steps==False] = np.asarray([i[0] for i in decode_logs])
    cum_seqnums = np.cumsum(seqnums)

    context_tot_lens = np.empty(len(prefill_logs)+len(decode_logs))
    
    
    context_tot_lens[is_prefill_steps==True] = np.asarray([i[1] for i in valid_prefill_logs])
    context_tot_lens[is_prefill_steps==False] = np.asarray([i[1] for i in decode_logs])
    cum_context_tot_lens = np.cumsum(context_tot_lens)

    
    

    
    
    
    

    


    s_array = np.asarray([1])
    tp_size=2
    cum_flops = cost_table.comp_flops(
        tp_size,
        cum_seqnums, s_array, cum_context_tot_lens, is_prompt=False,
        model_path=model_path, trust_remote_code=trust_remote_code, revision=revision)/1e12
    
    cum_flops = cum_flops*tp_size

    

    
    
    
    
    
    
    
    


    
    

    
    
    

    

    

    
    throughputs = cum_flops / (np.asarray(cumsum_latencys) + extra_cost)

    print(f"comp_throughput_for_each_iteration: {time.perf_counter() - time1}")
    time1 = time.perf_counter()

    return throughputs   












