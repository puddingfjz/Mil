" This file prepares the model size information we need when determining model exec plans. "

from concurrent.futures import ProcessPoolExecutor
import math

def print_parameters(model):
    named_parameters = model.named_parameters()
    for k, v in named_parameters:
        print(f"{k}: {v.shape}", flush=True)


def get_vLLM_LLM(model_path, tensor_parallel_size, init_times=[]):
    '''
        Input:
            init_times: List[float], stores the time to initialize an LLM engine.
    '''
    backend = "vllm"
    model = model_path
    tokenizer = None
    quantization = None
    # tensor_parallel_size = 1
    n = 1
    use_beam_search = False
    # num_prompts = 1000
    seed = 0
    hf_max_batch_size = None
    trust_remote_code = True
    max_model_len = None
    dtype = 'auto'
    enforce_eager = True
    kv_cache_dtype = "auto"
    device = "cuda"
    weight_load_degree = '2'
    gpu_use_ratio = 0.9
    # temperature = 1.0
   
    # copy the args check from vllm-----------------------------------------
    if tokenizer is None:
        tokenizer = model

    # set os environ variables----------------------------------------------
    import os
    # os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3' # '2,3' # '3,0,1,2' # should be set before initialize cuda in torch
    os.environ['USE_VLLM']='True'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False' # whether we will dynamically increase the on-card layer weights

    os.environ['RUN_MULTI_MODEL'] = 'False' # whether this model is running in a multi-model environment
    os.environ['SOFT_RESCHEDULE'] = 'False' # whether to reinitialize LLMs directly or update the current LLM (i.e., soft reschedule)
    os.environ['NO_PREEMPT'] = 'True' # allow model preemption or not
    # about scheduling
    os.environ['SORT_REQS'] = 'True' # whether to sort the requests according to their output lengths, default is False

    #  deal with extra parameters
    os.environ['WEIGHT_LOAD_DEGREE'] = weight_load_degree
    if backend == "ours":
        os.environ['USE_VLLM'] = 'False'
        os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'

    # get engine args
    from vllm import LLM

    import time
    start_time = time.perf_counter()

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
        # 
        # gpu_memory_utilization=0.5, #0.5689, #0.5, # 0.5373
        # max_num_seqs=2048,
        gpu_memory_utilization=gpu_use_ratio,
        max_num_seqs=512,
        max_paddings=512,
    )

    end_time = time.perf_counter()
    init_times.append(end_time - start_time)

    # print_parameters(llm.llm_engine.driver_worker.model_runner.model)
    
    return llm




# compute model parameter mem in bytes (about the weights of the decode layers)
def get_per_layer_and_extra_param_and_buffer_byte(
        model_path: str, tp_size: int):
    '''
        Input:
            comp_worker_num: the number of tensor parallel workers.

        NOTE: compute according to model.parameters() and model.buffers().
    '''
    # try:
    llm = get_vLLM_LLM(model_path=model_path, tensor_parallel_size=tp_size)
    named_parameters = llm.llm_engine.driver_worker.model_runner.model.named_parameters()
    named_buffers = llm.llm_engine.driver_worker.model_runner.model.named_buffers()
    per_layer = 0
    total = 0
    for k, v in named_parameters:
        if '.0.' in k:
            # get the layer 0
            per_layer += (v.numel()*v.element_size())
        total += (v.numel()*v.element_size())
        print(k, v.shape)

    for k, v in named_buffers:
        if '.0.' in k:
            # get the layer 0
            per_layer += (v.numel()*v.element_size())            
        total += (v.numel()*v.element_size())
        print(k, v.shape)
    
    layer_num = llm.llm_engine.model_config.get_num_layers(llm.llm_engine.parallel_config)
    extra = total - layer_num * per_layer
    # assert len(list(llm.llm_engine.driver_worker.model_runner.model.named_buffers())) == 1
    return per_layer, extra
    # except Exception as e:
    #     print(f"error: {e}")
    #     return None, None





def get_non_attention_flops_coefficient(
        model_path: str, tp_size: int):
    '''
        Get the coefficient for the non-attention flops.
        Total flops = coefficient * #token + attention_flops.
    '''
    llm = get_vLLM_LLM(model_path=model_path, tensor_parallel_size=tp_size)
    model = llm.llm_engine.driver_worker.model_runner.model
    named_parameters = model.named_parameters()
    coeff = 0
    for k, v in named_parameters:
        if ('.0.' in k) and (len(v.shape)>1):
            # get the parameter: (1) in layer 0 and (2) used for matrix multiplication
            coeff += (math.prod(v.shape))
    return coeff




# compute model parameter mem in bytes (about the weights of the decode layers)
def get_per_layer_and_extra_param_and_buffer_byte_AND_non_attention_flops_coefficient(
        model_path: str, tp_size: int):
    '''
        Input:
            comp_worker_num: the number of tensor parallel workers.

        NOTE: compute according to model.parameters() and model.buffers().

        Also get the coefficient for the non-attention flops.
        Total flops = coefficient * #token + attention_flops.
    '''
    # try:
    llm = get_vLLM_LLM(model_path=model_path, tensor_parallel_size=tp_size)
    named_parameters = llm.llm_engine.driver_worker.model_runner.model.named_parameters()
    named_buffers = llm.llm_engine.driver_worker.model_runner.model.named_buffers()
    per_layer = 0
    total = 0
    coeff = 0
    for k, v in named_parameters:
        if '.0.' in k:
            # get the layer 0
            per_layer += (v.numel()*v.element_size())
            print("per layer params")
        total += (v.numel()*v.element_size())
        print(k, v.shape)

        # comp flops coeffs
        if ('.0.' in k) and (len(v.shape)>1):
            # get the parameter: (1) in layer 0 and (2) used for matrix multiplication
            coeff += (math.prod(v.shape))
            print("flops coeffs")

    for k, v in named_buffers:
        # if '.0.' in k:
        #     # get the layer 0
        #     per_layer += (v.numel()*v.element_size())            
        #     print("per layer buffers")
        total += (v.numel()*v.element_size())
        print(k, v.shape)
    
    layer_num = llm.llm_engine.model_config.get_num_layers(llm.llm_engine.parallel_config)
    extra = total - layer_num * per_layer
    # assert len(list(llm.llm_engine.driver_worker.model_runner.model.named_buffers())) == 1
    return per_layer, extra, coeff
    # except Exception as e:
    #     print(f"error: {e}")
    #     return None, None




# collect model init cost for multiple rounds to ensure the model weights are stored in the RAM.
def get_model_init_cost(model_path: str, tp_size: int):
    init_times=list()
    get_vLLM_LLM(model_path, tp_size, init_times=init_times)
    return init_times[0]




# start a model for inference
def print_a_model_size(model_path: str, tp_size: int, model_sizes: dict):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_per_layer_and_extra_param_and_buffer_byte, model_path, tp_size)
        per_layer, extra = future.result()
        model_sizes[(model_path, tp_size)] = (per_layer, extra)



# print parameters
def print_a_model_parameter(model_path: str, tp_size: int):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(get_vLLM_LLM, model_path, tp_size)


# print flops coeff
def print_a_model_flops_coeff(model_path: str, tp_size: int, model_coeffs: dict):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_non_attention_flops_coefficient, model_path, tp_size)
        coeff = future.result()
        model_coeffs[(model_path, tp_size)] = coeff


# print model size and flops coeff
def print_a_model_size_and_flops_coeff(model_path: str, tp_size: int, model_sizes: dict, model_coeffs: dict):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            get_per_layer_and_extra_param_and_buffer_byte_AND_non_attention_flops_coefficient, 
            model_path, tp_size)
        per_layer, extra, coeff = future.result()
        model_sizes[(model_path, tp_size)] = (per_layer, extra)
        model_coeffs[(model_path, tp_size)] = coeff



# get the model initialization time
def get_model_initialization_time(model_path: str, tp_size: int, model_init_costs: dict):
    # use a child process to run benchmark_throughput.main so that the cuda memory can be released completely when finishing inference
    init_costs = list()
    for _ in range(3):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_model_init_cost, model_path, tp_size)
            init_cost = future.result()
            print(f"model: {model_path}, init_cost: {init_cost}")
            init_costs.append(init_cost)
    
    model_init_costs[(model_path, tp_size)] = min(init_costs)




if __name__ == "__main__":

    model_paths = [
                #     'NousResearch/Llama-2-7b-hf', 
                # #    'NousResearch/Llama-2-7b-chat-hf',
                #    'NousResearch/Llama-2-13b-hf',
                #    'NousResearch/Llama-2-70b-hf',
                #    'THUDM/chatglm3-6b',
                #    'EleutherAI/gpt-j-6b', 
                #    'EleutherAI/gpt-neox-20b',
                #    'baichuan-inc/Baichuan2-13B-Chat',
                #    'baichuan-inc/Baichuan-7B',
                #    'mistralai/Mixtral-8x7B-v0.1',
                # 
                # 'lmsys/vicuna-13b-v1.5',
                # 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
                # 'chavinlo/alpaca-13b',
                # 'project-baize/baize-v2-13b',
                # 'TheBloke/koala-13B-HF',
                # 'databricks/dolly-v2-12b',
                # 'mosaicml/mpt-7b-chat',
                # 'THUDM/chatglm3-6b',
                'stabilityai/stablelm-tuned-alpha-7b',
                # 
                # 'meta-llama/Llama-2-70b-chat-hf', 
                # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                # 'WizardLMTeam/WizardLM-13B-V1.2',
                # 'meta-llama/CodeLlama-34b-Instruct-hf',
                # 'mistralai/Mistral-7B-Instruct-v0.2',
                ]

    # model_sizes = dict()
    # for model_path in model_paths:
    #     for tp_size in [2**i for i in range( int(math.log(4, 2))+1 )]:
    #         print_a_model_size(model_path, tp_size, model_sizes)

    # with open('model_size_database.py', 'a') as file:
    #     # file.write(f"model_sizes = {model_sizes}\n")
    #     file.write(f"model_sizes.update({model_sizes})\n")


    # # for model_path in model_paths:
    # #     print_a_model_parameter(model_path, 1)


    # model_coeffs = dict()
    # for model_path in model_paths:
    #     for tp_size in [2**i for i in range(0, int(math.log(4, 2))+1 )]:
    #         print_a_model_flops_coeff(model_path, tp_size, model_coeffs)

    # with open('model_coeff_database.py', 'a') as file:
    #     # file.write(f"model_coeffs = {model_coeffs}\n")
    #     file.write(f"model_coeffs.update({model_coeffs})\n")




    model_sizes = dict()
    model_coeffs = dict()
    for model_path in model_paths:
        for tp_size in [2**i for i in range(0, int(math.log(4, 2))+1 )]:
        # for tp_size in [1]:
            try:
                print_a_model_size_and_flops_coeff(model_path, tp_size, model_sizes, model_coeffs)
            except Exception as e:
                print(e)

    with open('model_size_database.py', 'a') as file:
        # file.write(f"model_sizes = {model_sizes}\n")
        file.write(f"model_sizes.update({model_sizes})\n")

    with open('model_coeff_database.py', 'a') as file:
        # file.write(f"model_coeffs = {model_coeffs}\n")
        file.write(f"model_coeffs.update({model_coeffs})\n")
    




    # get the model init cost
    model_init_costs = dict()
    for model_path in model_paths:
        for tp_size in [2**i for i in range(0, int(math.log(4, 2))+1 )]:
        # for tp_size in [1]:
            try:
                get_model_initialization_time(model_path, tp_size, model_init_costs)
            except Exception as e:
                print(e)

    with open('model_initcost_database.py', 'a') as file:
        # file.write(f"model_coeffs = {model_coeffs}\n")
        file.write(f"model_init_costs.update({model_init_costs})\n")

