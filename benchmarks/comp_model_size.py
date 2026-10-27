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
    weight_load_degree = '2'
    gpu_use_ratio = 0.9


    if tokenizer is None:
        tokenizer = model


    import os

    os.environ['USE_VLLM']='True'
    os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'False'

    os.environ['RUN_MULTI_MODEL'] = 'False'
    os.environ['SOFT_RESCHEDULE'] = 'False'
    os.environ['NO_PREEMPT'] = 'True'

    os.environ['SORT_REQS'] = 'True'


    os.environ['WEIGHT_LOAD_DEGREE'] = weight_load_degree
    if backend == "ours":
        os.environ['USE_VLLM'] = 'False'
        os.environ['DYNAMIC_INCREASE_ONCARD_WEIGHTS'] = 'True'


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


        gpu_memory_utilization=gpu_use_ratio,
        max_num_seqs=512,
        max_paddings=512,
    )

    end_time = time.perf_counter()
    init_times.append(end_time - start_time)


    return llm


def get_per_layer_and_extra_param_and_buffer_byte(
        model_path: str, tp_size: int):
    '''
        Input:
            comp_worker_num: the number of tensor parallel workers.

        NOTE: compute according to model.parameters() and model.buffers().
    '''

    llm = get_vLLM_LLM(model_path=model_path, tensor_parallel_size=tp_size)
    named_parameters = llm.llm_engine.driver_worker.model_runner.model.named_parameters()
    named_buffers = llm.llm_engine.driver_worker.model_runner.model.named_buffers()
    per_layer = 0
    total = 0
    for k, v in named_parameters:
        if '.0.' in k:

            per_layer += (v.numel()*v.element_size())
        total += (v.numel()*v.element_size())
        print(k, v.shape)

    for k, v in named_buffers:
        if '.0.' in k:

            per_layer += (v.numel()*v.element_size())
        total += (v.numel()*v.element_size())
        print(k, v.shape)

    layer_num = llm.llm_engine.model_config.get_num_layers(llm.llm_engine.parallel_config)
    extra = total - layer_num * per_layer

    return per_layer, extra


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

            coeff += (math.prod(v.shape))
    return coeff


def get_per_layer_and_extra_param_and_buffer_byte_AND_non_attention_flops_coefficient(
        model_path: str, tp_size: int):
    '''
        Input:
            comp_worker_num: the number of tensor parallel workers.

        NOTE: compute according to model.parameters() and model.buffers().

        Also get the coefficient for the non-attention flops.
        Total flops = coefficient * #token + attention_flops.
    '''

    llm = get_vLLM_LLM(model_path=model_path, tensor_parallel_size=tp_size)
    named_parameters = llm.llm_engine.driver_worker.model_runner.model.named_parameters()
    named_buffers = llm.llm_engine.driver_worker.model_runner.model.named_buffers()
    per_layer = 0
    total = 0
    coeff = 0
    for k, v in named_parameters:
        if '.0.' in k:

            per_layer += (v.numel()*v.element_size())
            print("per layer params")
        total += (v.numel()*v.element_size())
        print(k, v.shape)


        if ('.0.' in k) and (len(v.shape)>1):

            coeff += (math.prod(v.shape))
            print("flops coeffs")

    for k, v in named_buffers:


        total += (v.numel()*v.element_size())
        print(k, v.shape)

    layer_num = llm.llm_engine.model_config.get_num_layers(llm.llm_engine.parallel_config)
    extra = total - layer_num * per_layer

    return per_layer, extra, coeff


def get_model_init_cost(model_path: str, tp_size: int):
    init_times=list()
    get_vLLM_LLM(model_path, tp_size, init_times=init_times)
    return init_times[0]


def print_a_model_size(model_path: str, tp_size: int, model_sizes: dict):

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_per_layer_and_extra_param_and_buffer_byte, model_path, tp_size)
        per_layer, extra = future.result()
        model_sizes[(model_path, tp_size)] = (per_layer, extra)


def print_a_model_parameter(model_path: str, tp_size: int):

    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(get_vLLM_LLM, model_path, tp_size)


def print_a_model_flops_coeff(model_path: str, tp_size: int, model_coeffs: dict):

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_non_attention_flops_coefficient, model_path, tp_size)
        coeff = future.result()
        model_coeffs[(model_path, tp_size)] = coeff


def print_a_model_size_and_flops_coeff(model_path: str, tp_size: int, model_sizes: dict, model_coeffs: dict):

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            get_per_layer_and_extra_param_and_buffer_byte_AND_non_attention_flops_coefficient,
            model_path, tp_size)
        per_layer, extra, coeff = future.result()
        model_sizes[(model_path, tp_size)] = (per_layer, extra)
        model_coeffs[(model_path, tp_size)] = coeff


def get_model_initialization_time(model_path: str, tp_size: int, model_init_costs: dict):

    init_costs = list()
    for _ in range(3):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_model_init_cost, model_path, tp_size)
            init_cost = future.result()
            print(f"model: {model_path}, init_cost: {init_cost}")
            init_costs.append(init_cost)

    model_init_costs[(model_path, tp_size)] = min(init_costs)


def extract_peak_mem_from_log(filename: str):
    peak_mem_dict = dict()
    with open(filename, 'r', errors='ignore') as f:
        key = None
        for line in f.readlines():
            if 'model_path: ' in line:
                modelpath, tpsize = line.split(',')
                modelpath = modelpath.split(':')[-1].strip()
                tpsize = int(tpsize.split(':')[-1])
                key = (modelpath, tpsize)
            elif 'peak_memory:' in line:
                peak = int(line.split(',')[-2].split(':')[-1])
                peak_mem_dict[key] = peak
                key = None
    return peak_mem_dict


if __name__ == "__main__":

    model_paths = [


                'meta-llama/Llama-2-70b-chat-hf',


                ]


    model_sizes = dict()
    model_coeffs = dict()
    for model_path in model_paths:

        for tp_size in [2**i for i in range(0, int(math.log(2, 2))+1 )]:

            try:
                print(f"model_path: {model_path}, tp_size: {tp_size}")
                print_a_model_size_and_flops_coeff(model_path, tp_size, model_sizes, model_coeffs)
            except Exception as e:
                print(e)
