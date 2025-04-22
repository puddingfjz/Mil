""" 
This file is about llm inference workers to support data parallelism.
Each llm inference worker deals with a part of the requests. 
We support data parallelism + tensor parallelism.
Each worker process is the driver process is it further does tensor parallelism.
"""

from typing import List, Optional, Tuple
from collections import deque

import time
import os
from vllm.core.multimodel_scheduler import SHARED_CONTECT, LLM_COMMUNICATOR


import traceback

   
def do_inference(
        worker_i: int, 
        remaining_requests, 

        
        
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
        ignore_eos: bool,
        fixed_output_len: int
    ):

    from vllm import LLM, SamplingParams

    
    rescheduled_iter_num = 0
    
    
    
    SHARED_CONTECT.dp_id = worker_i

    os.environ['DP_WORKER_I'] = str(worker_i)
    

    
    gpus = os.environ['TOT_ORDERED_GPUS'].split(',')
    gpus = gpus[worker_i*tensor_parallel_size:] + gpus[:worker_i*tensor_parallel_size]
    gpus = ','.join([str(i) for i in gpus])
    os.environ['TOT_ORDERED_GPUS'] = gpus

    

    
    start_prepare_model = time.perf_counter()
    
    print(f"start do_inference: model id{SHARED_CONTECT.shared_id} dp_id {worker_i}: ---abs {start_prepare_model}")

    if (rescheduled_iter_num == 0) or (os.environ['SOFT_RESCHEDULE'] == 'False'):
        start_time_load_LLM = time.perf_counter()
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
            
            
            
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=512,
            max_paddings=512,
        )
        end_time_load_LLM = time.perf_counter()
        print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, total time to load LLM: {end_time_load_LLM - start_time_load_LLM}")
    else:
        
        start_time_load_LLM = time.perf_counter()
        llm.update_llm_engine(
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
            
            
            
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=512,
            max_paddings=512,
        )
        end_time_load_LLM = time.perf_counter()
        print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, total time to update LLM: {end_time_load_LLM - start_time_load_LLM}")


    
    
    
    
    if (len(remaining_requests)>0) and isinstance(remaining_requests[0], tuple):
        

        
        print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, max_model_len: {llm.llm_engine.model_config.max_model_len}")
        print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, temperature: {temperature}")
        print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, ignore_eos: {ignore_eos}")

        
        for prompt, _, output_len in remaining_requests:
            
            sampling_params = SamplingParams(
                n=n,
                
                temperature=0.0 if use_beam_search else temperature, 
                top_p=1.0,
                use_beam_search=use_beam_search,
                ignore_eos=ignore_eos, 
                max_tokens=output_len if ignore_eos else (llm.llm_engine.model_config.max_model_len-_),
                
            )
            
            llm._add_request(
                prompt=prompt,
                prompt_token_ids=None,
                sampling_params=sampling_params,
            )
    else:
        
        print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, directly using unfinished requests!")
        
        llm.llm_engine.scheduler.waiting = deque(remaining_requests)


    
    sampling_parameters = {                    
        "n":n,
        
        "temperature":0.0 if use_beam_search else temperature, 
        "top_p":1.0,
        "use_beam_search":use_beam_search,
        "ignore_eos":ignore_eos, 
        "max_tokens":fixed_output_len if ignore_eos else (llm.llm_engine.model_config.max_model_len)}


    
    end_prepare_model = time.perf_counter()
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, total time to prepare model: {end_prepare_model-start_prepare_model}s ---abs {end_prepare_model}", flush=True)



    
    SHARED_CONTECT.communicator.update_non_compute_ranges(
        [[start_prepare_model], [end_prepare_model]], SHARED_CONTECT.shared_id, SHARED_CONTECT.get_gpus_used())


    
    
    
    

    tmp_start = time.perf_counter()

    
    try:
        outputs = llm._run_engine(use_tqdm=True, sampling_parameters=sampling_parameters)
    except Exception as e:
        print(f"Exception in llm._run_engine: {e}, shared_id: {SHARED_CONTECT.shared_id}  worker i: {worker_i}")
        print(traceback.format_exc())
        

    end = time.perf_counter()
    

    print(f"shared_id: {SHARED_CONTECT.shared_id}  worker i: {worker_i}, this execution plan running time: {end - tmp_start}s ---abs {end}")
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, outputs:\n")
    
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, output_lens = {[[len(req_output.prompt_token_ids), len(completion_output.token_ids), -1] for req_output in outputs for completion_output in req_output.outputs]}")
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, tot_inp_lens = {sum([len(req_output.prompt_token_ids) for req_output in outputs])}")
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, tot_out_len = {sum([len(completion_output.token_ids) for req_output in outputs for completion_output in req_output.outputs])}")
    num_completion_output = len([completion_output for req_output in outputs for completion_output in req_output.outputs])
    max_out_len = None
    if num_completion_output > 0:
        max_out_len = max([len(completion_output.token_ids) for req_output in outputs for completion_output in req_output.outputs])
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, num if outputs: {num_completion_output}, max_out_len = {max_out_len}")
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, output_lens = {[[(req_output.request_id,), len(req_output.prompt_token_ids), len(completion_output.token_ids), -1] for req_output in outputs for completion_output in req_output.outputs]}")
    
    
    
    
    
    
    
    



    
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, deleting LLM-----------------", flush=True)
    
    
    
    

    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)
    print(f"{model} {SHARED_CONTECT.shared_id} One round finished!!!!!!!!!!")


    
    
    
    

    
    print(f"shared_id {SHARED_CONTECT.shared_id} worker i: {worker_i}, len of outputs and remainings before return: {len(SHARED_CONTECT.gened_outputs), len(SHARED_CONTECT.remaining_requests)}")
    
    return len(SHARED_CONTECT.gened_outputs), SHARED_CONTECT.remaining_requests
    

