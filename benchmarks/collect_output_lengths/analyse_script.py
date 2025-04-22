



import json





prefix = 'collect_output_lengths/no_robot/NEWROUND_'


model_names = [
                
                'Llama-2-70b-chat-hf',
                'Mixtral-8x7B-Instruct-v0.1',
                'WizardLM-13B-V1.2',
                'CodeLlama-34b-Instruct-hf',
                'Mistral-7B-Instruct-v0.2',     
                
                'vicuna-13b-v1.5',
                'oasst-sft-4-pythia-12b-epoch-3.5',
                'alpaca-13b',
                'baize-v2-13b',
                'koala-13B-HF',
                'dolly-v2-12b',
                'mpt-7b-chat',
                'chatglm3-6b',
                'stablelm-tuned-alpha-7b'
            ]


model_paths = [
    'meta-llama/Llama-2-70b-chat-hf',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'WizardLMTeam/WizardLM-13B-V1.2',
    'meta-llama/CodeLlama-34b-Instruct-hf',
    'mistralai/Mistral-7B-Instruct-v0.2',     
    
    'lmsys/vicuna-13b-v1.5',
    'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
    'chavinlo/alpaca-13b',
    'project-baize/baize-v2-13b',
    'TheBloke/koala-13B-HF',
    'databricks/dolly-v2-12b',
    'mosaicml/mpt-7b-chat',
    'THUDM/chatglm3-6b',
    'stabilityai/stablelm-tuned-alpha-7b'
]




suffix = lambda temp: '_tp2_10kreq_1.log' if temp == 1.0 else f'_tp2_temp{temp}_10kreq_1.log'
temp = 0.8



dataset = {key:list() for key in model_names}

max_model_len_dict = {key:None for key in model_names}


def update_dataset(dataset, key, filename):
    try:
        with open(prefix+filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'output_lens = ' in line:
                    data = line[len('output_lens = '):]
                    data = json.loads(data)
                    dataset[key].append(data)
    except:
        print(f"Failed: file {prefix+filename}")



def get_max_model_len(max_model_len_dict, key, filename):
    
    try:
        with open(prefix+filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'max_model_len:' in line:
                    data = line[len('max_model_len:'):]
                    data = json.loads(data)
                    max_model_len_dict[key] = data
    except:
        print(f"Failed: file {prefix+filename}")    




def get_max_model_len_from_engine_args(max_model_len_dict, model_paths):
    import search_exec_plans 
    for model_path in model_paths:
        (model_config, cache_config, parallel_config, scheduler_config,
        device_config, lora_config) = search_exec_plans.get_engin_args(model_path, 1)
        max_model_len = model_config.max_model_len
        pos = model_path.find('/')
        model_name = model_path[pos+1:]
        max_model_len_dict[model_name] = max_model_len


for model in dataset.keys():
    
    filename = f'{model}{suffix(temp=temp)}'
    update_dataset(dataset, model, filename)
    


get_max_model_len_from_engine_args(max_model_len_dict, model_paths)



for k, vs in dataset.items():
    print(f"{k:<20}TOT_len: {str([sum([sum(i[:2]) for i in v]) for v in vs]):<20}AVG_out: {str([sum([i[1] for i in v]) / len(v) for v in vs]):<20}")









import matplotlib.pyplot as plt
import numpy as np
fig_path_prefix = 'Cost_Model_per_iter_machine2/figures/cdf'
fig_path_suffix = 'norobot_1.pdf'
fig_path_suffix = f'norobot_temp{temp}_1.pdf'

pdf_dict = dict()
for k, vs in dataset.items():   
    if len(vs) == 0:
        continue
    lens = vs[0]
    
    max_model_len = max_model_len_dict[k]
    event_nums = np.zeros(max_model_len, dtype=np.int32)
    sample_sizes = np.zeros(max_model_len, dtype=np.int32)
    
    inps = np.asarray([i[0] for i in lens])
    outs = np.asarray([i[1] for i in lens])
    uniq_inps = set([i[0] for i in lens])
    assert min(outs) >= 1
    
    bins = {inp: np.unique(outs[inps==inp], return_counts=True) for inp in uniq_inps}
    for inp in bins:
        sample_size = sum(bins[inp][1])
        
        for out, count in zip(*(bins[inp])):
            if inp + out < max_model_len:
                
                event_nums[out-1] += count
        sample_sizes[:max_model_len-inp-1] = sample_sizes[:max_model_len-inp-1] \
            + sample_size
    sample_sizes[event_nums==0] = 1 
    pdf = event_nums / sample_sizes
    cdf = np.cumsum(pdf)
    
    
    assert (pdf[-1] == 0) and (cdf[-1] == cdf[-2])
    pdf[-1] = 1 - cdf[-2]
    assert sum(pdf) == 1
    
    pdf_dict[(k, temp)] = pdf.tolist()
    
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, max_model_len+1), cdf, marker='.', markersize=10, label = f'cdf: max {cdf[-1]:.2f}', color='tan')
    
    
    tots = inps + outs
    inps = inps[tots<=max_model_len]
    outs = outs[tots<=max_model_len]
    assert max(tots) <= max_model_len, f"{k} max tot > max_model_len {max_model_len}"
    interval = 100
    for i in range(max(inps)//100):
        start, end = (i*interval, (i+1)*interval)
        indices = (inps>start) * (inps<=end)
        tmp_outs = outs[indices]
        if len(tmp_outs) == 0:
            continue
        
        elements, counts = np.unique(tmp_outs, return_counts=True)
        cum_counts = np.cumsum(counts)
        cum_counts = cum_counts/cum_counts[-1]
        
        ax.plot(elements, cum_counts, marker='1', markersize=8, label = f'{start, end} 
    ax.set(xlabel=f'out len (max_model_len:{max_model_len})', ylabel='cum prob')
    
    ax.grid()
    plt.legend()
    fig.savefig(f"{fig_path_prefix}_{k}_{fig_path_suffix}")
    plt.show()    




with open('./collect_output_lengths/no_robot/out_len_sampler_2.py', 'a') as file:
    
    file.write(f"\npdf_dict.update({pdf_dict})\n")





