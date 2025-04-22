"""
This file draws the inp len distribution curves of different experiments.
"""


import time
import argparse
import schedule_multi_model
from schedule_multi_model import *
import matplotlib.pyplot as plt



def _get_inp_lens(test_cases, versions, max_token_nums, specify_outlens, num_prompts, model_paths, inp_seq_ids_dict):
    funcs = [schedule_multi_model._get_req_len_funcs(
        test_case=test_case, version=version, max_token_num=max_token_num, specify_outlen=specify_outlen) \
        for test_case, version, max_token_num, specify_outlen in zip(test_cases, versions, max_token_nums, specify_outlens)]
    inp_generators = [_[0] for _ in funcs]
    inp_lens_dict = {i:inp_generators[i](num_prompts, i, model_path, inp_seq_ids_dict[i]) for i, model_path in enumerate(model_paths)}
    if test_cases[0] == 'router':
        return np.concatenate(list(inp_lens_dict.values()))
    elif test_cases[0] == 'general':
        return inp_lens_dict[0]
    elif test_cases[0] == 'chain-summary':
        seq_lens = dict()
        for i in range(len(model_paths)-1):
            for l, seq_id in zip(inp_lens_dict[i], inp_seq_ids_dict[i]):
                if seq_id not in seq_lens:
                    seq_lens[seq_id] = 0
                seq_lens[seq_id] += l
        return list(seq_lens.values())
    assert False



class Args:
    """Arguments for multi-model scheduling."""
    def __init__(self,
        reqnum,
        specify_outlen,
        ratio_seed=0, 
        ratio_set=1,
        gen_execplans_baseline='ours', 
        search_method_baseline='ours',
        test_case='router',
        router_question_version='multiple_choice_question',
        max_token_num=900,
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num=8,
        max_group_seq_num=1,
        top_k=20,
        similar_threshold=0.2,
        fully_connected_gpu_unit=2,
        machine_name='machine2',
        evaluator_num=5,
        summarize_model='lmsys/vicuna-13b-v1.5',
        evaluator_model='meta-llama/Llama-2-70b-chat-hf',
        test_id=0
    ):
        self.gen_execplans_baseline = gen_execplans_baseline
        self.search_method_baseline = search_method_baseline
        self.test_case = test_case
        self.ratio_seed = ratio_seed
        self.ratio_set = ratio_set
        self.reqnum = reqnum
        self.router_question_version = router_question_version
        self.max_token_num = max_token_num
        self.specify_outlen = specify_outlen
        self.gpu_name = gpu_name
        self.byte_per_gpu = byte_per_gpu
        self.tot_gpu_num = tot_gpu_num
        self.max_group_seq_num = max_group_seq_num
        self.top_k = top_k
        self.similar_threshold = similar_threshold
        self.fully_connected_gpu_unit = fully_connected_gpu_unit
        self.machine_name = machine_name
        self.evaluator_num = evaluator_num
        self.summarize_model = summarize_model
        self.evaluator_model = evaluator_model
        self.test_id = test_id





def get_inp_lens(
        reqnum,
        specify_outlen,
        ratio_seed=0, 
        ratio_set=1,
        gen_execplans_baseline='ours', 
        search_method_baseline='ours',
        test_case='router',
        router_question_version='multiple_choice_question',
        max_token_num=900,
        gpu_name='A100-80G',
        byte_per_gpu=80*(1024**3),
        tot_gpu_num=8,
        max_group_seq_num=1,
        top_k=20,
        similar_threshold=0.2,
        fully_connected_gpu_unit=2,
        machine_name='machine2',
        evaluator_num=5,
        summarize_model='lmsys/vicuna-13b-v1.5',
        evaluator_model='meta-llama/Llama-2-70b-chat-hf',
        test_id=0
):
    args=Args(
        reqnum=reqnum,
        specify_outlen=specify_outlen,
        ratio_seed=ratio_seed, 
        ratio_set=ratio_set,
        gen_execplans_baseline=gen_execplans_baseline, 
        search_method_baseline=search_method_baseline,
        test_case=test_case,
        router_question_version=router_question_version,
        max_token_num=max_token_num,
        gpu_name=gpu_name,
        byte_per_gpu=byte_per_gpu,
        tot_gpu_num=tot_gpu_num,
        max_group_seq_num=max_group_seq_num,
        top_k=top_k,
        similar_threshold=similar_threshold,
        fully_connected_gpu_unit=fully_connected_gpu_unit,
        machine_name=machine_name,
        evaluator_num=evaluator_num,
        summarize_model=summarize_model,
        evaluator_model=evaluator_model,
        test_id=test_id)
    
    model_paths, check_gap, sort_input, in_edge_dict_with_dummy_inp_nodes, \
        num_prompts, inp_seq_ids_dict, inp_generator, inp_merger, outlen_generator, \
            prompt_templates_lens, node_dataset_chunk_mapping, \
                inp_req_ids, inp_req_from_which_model_which_out_reqs, \
                    out_req_id_mapping, new_out_req_part_num, independent_srcs, prompt_template_args, \
                        sampling_args_dict, seq_outlen_dict, \
                            test_case, version, max_token_num, specify_outlen = \
        get_schedule_setting(args, test_case=test_case, version=router_question_version, max_token_num=max_token_num, specify_outlen=args.specify_outlen,
                             use_real_dataset=True, ratio_seed=ratio_seed, ratio_set=ratio_set, reqnum=reqnum,
                             evaluator_num=args.evaluator_num, summarize_model=args.summarize_model, evaluator_model=args.evaluator_model)
    test_cases, versions, max_token_nums, specify_outlens = [], [], [], []
    if isinstance(test_case, List):
        test_cases, versions, max_token_nums, specify_outlens = test_case, version, max_token_num, specify_outlen
    else:
        test_cases, versions, max_token_nums, specify_outlens = [test_case], [version], [max_token_num], [specify_outlen]

    inp_lens=_get_inp_lens(test_cases, versions, max_token_nums, specify_outlens, num_prompts, model_paths, inp_seq_ids_dict)
    return inp_lens







if __name__ == "__main__":

  
    
    
    
    specify_outlen=False
    inp_lens_dict = dict()
    
    for summarize_model in ['lmsys/vicuna-13b-v1.5']: 
        
        for reqnum in [100, 200, 300, 400, 500]:
            inp_lens = get_inp_lens(reqnum, specify_outlen, test_case='chain-summary', summarize_model=summarize_model)
            inp_lens_dict[reqnum] = inp_lens


    print(f"inp_lens_dict: {inp_lens_dict}")

    
    plt.rcParams['font.size'] = 16

    
    
    methods = [100, 200, 300, 400, 500]


    
    
    
    
    data = {i:(np.arange(len(k)), k) for i, k in inp_lens_dict.items()}

    
    bar_width = 0.05


    
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']


    
    
    
    method_names=methods

    
    


    
    plt.figure(figsize=(12, 6))
    

    
    
    

    for i, method in enumerate(methods):
        len_ids = np.asarray(data[method][0])
        lens = np.asarray(data[method][1])
        p1=plt.bar(len_ids, lens, label=f'{method_names[i]}', color=colors[i], alpha=1-0.1*i)


    plt.xlabel('Id')
    plt.ylabel('Input length')
    
    
    plt.legend()

    
    plt.tight_layout()
    plt.savefig('./figures/chain_summary_inplens.png', format='png')

        


