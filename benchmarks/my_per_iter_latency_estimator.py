'''
This file estimates the per iter exec latency based on the collected measurement records
'''
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict
from vllm.engine.metrics import MyThroughputLogger, cal_flops
from vllm.transformers_utils.config import get_config
import numpy as np



def get_tp_size_from_logfile(filename:str):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'tensor_parallel_size=' in line:
                pos0 = line.find('tensor_parallel_size=')+len('tensor_parallel_size=')
                pos1 = line[pos0:].find(',')+pos0
                tp_size = int(line[pos0:pos1])
                return tp_size




class CostTable:
    """ Stores the collected measurement records, and can estimate per-iter exec costs based on them. """
    def __init__(self, 
                 logfiles: List[Tuple[str, str]], 
                 prepInp_decode_logfiles: List[Tuple[str, str]], 
                 model_infos: List[Tuple[str, bool, Optional[str]]],
                 machine_name: str,
                 metadata=None
                 ) -> None:
        
        if metadata != None:
            self.prefill_table, self.decode_table,\
                self.sample_prefill_table, self.sample_decode_table, \
                self.prepInp_prefill_table, self.prepInp_decode_table, \
                self.model_configs, self.prepare_cost_table = metadata
            return

        # {(model_name, exec_plan), table for that model}; 
        # for specific model table: key of table: (seqnum), value of table: (flops, per-iter exec latency)
        self.prefill_table: Dict[Tuple[str, Any], Dict[int, List[Tuple[float, float]]]] = dict()
        self.decode_table: Dict[Tuple[str, Any], Dict[int, List[Tuple[float, float]]]] = dict()
        self.sample_prefill_table: Dict[Tuple[str, Any], Dict[int, List[Tuple[float, float]]]] = dict()
        self.sample_decode_table: Dict[Tuple[str, Any], Dict[int, List[Tuple[float, float]]]] = dict()
        self.prepInp_prefill_table: Dict[Tuple[str, Any], Dict[int, List[Tuple[float, float]]]] = dict()
        self.prepInp_decode_table: Dict[Tuple[str, Any], Dict[int, List[Tuple[float, float]]]] = dict()
        # store the arguments we need to compute flops for a iteration
        self.model_configs: Dict[Tuple[str, bool, Optional[str]], Tuple[int, int, int, int, int, int]] = dict()
        self.prepare_cost_table: Dict[Tuple[str, Tuple[int, int, int, int, int, int]], float] = dict()
        # 
        for model_info in model_infos:
            self.init_model_config(*model_info)
        for file_info in logfiles:
            self.init_table(*file_info)
        for file_info in prepInp_decode_logfiles:
            self.init_prepInp_decode_table(*file_info)
        # 
        self.init_prepare_cost(machine_name)





    def store_meta_data(self, filename: str):
        
        model_paths = [k[0] for k in self.prefill_table.keys()]
        (prefill_table, decode_table,\
         sample_prefill_table, sample_decode_table, \
            prepInp_prefill_table, prepInp_decode_table, \
                model_configs, prepare_cost_table) = self.serialize(model_paths=model_paths)
        
        with open(filename, 'w') as f:
            f.write('import my_per_iter_latency_estimator\n')
            # f.write('from collections import defaultdict\n')
            f.write(f"prefill_table={prefill_table}\n")
            f.write(f"decode_table={decode_table}\n")
            f.write(f"sample_prefill_table={sample_prefill_table}\n")
            f.write(f"sample_decode_table={sample_decode_table}\n")
            f.write(f"prepInp_prefill_table={prepInp_prefill_table}\n")
            f.write(f"prepInp_decode_table={prepInp_decode_table}\n")
            f.write(f"model_configs={model_configs}\n")
            f.write(f"prepare_cost_table={prepare_cost_table}\n")
            f.write(f"cost_table = my_per_iter_latency_estimator.get_cost_table_from_serialized_data(prefill_table, decode_table,"\
                "sample_prefill_table, sample_decode_table, "\
                "prepInp_prefill_table, prepInp_decode_table, "\
                "model_configs, prepare_cost_table)")
        return 

        def _to_dict(to_convert):
            return {k:dict(v) for k,v in to_convert.items()}
        
        prefill_table=_to_dict(self.prefill_table)
        decode_table=_to_dict(self.decode_table)
        sample_prefill_table=_to_dict(self.sample_prefill_table)
        sample_decode_table=_to_dict(self.sample_decode_table)
        prepInp_prefill_table=_to_dict(self.prepInp_prefill_table)
        prepInp_decode_table=_to_dict(self.prepInp_decode_table)

        with open(filename, 'w') as f:
            f.write('import my_per_iter_latency_estimator\n')
            # f.write('from collections import defaultdict\n')
            f.write(f"prefill_table={prefill_table}\n")
            f.write(f"decode_table={decode_table}\n")
            f.write(f"sample_prefill_table={sample_prefill_table}\n")
            f.write(f"sample_decode_table={sample_decode_table}\n")
            f.write(f"prepInp_prefill_table={prepInp_prefill_table}\n")
            f.write(f"prepInp_decode_table={prepInp_decode_table}\n")
            f.write(f"model_configs={self.model_configs}\n")
            f.write(f"prepare_cost_table={self.prepare_cost_table}\n")
            metadata = (prefill_table, decode_table,\
                sample_prefill_table, sample_decode_table, \
                prepInp_prefill_table, prepInp_decode_table, \
                self.model_configs, self.prepare_cost_table)
            f.write(f"metadata = {metadata}\n")
            f.write(f"cost_table = my_per_iter_latency_estimator.CostTable([],[],[],metadata=metadata)\n")




    def update_meta_data(self, filename: str):
        
        model_paths = [k[0] for k in self.prefill_table.keys()]
        (prefill_table, decode_table,\
         sample_prefill_table, sample_decode_table, \
            prepInp_prefill_table, prepInp_decode_table, \
                model_configs, prepare_cost_table) = self.serialize(model_paths=model_paths)
        
        with open(filename, 'a') as f:
            # f.write('import my_per_iter_latency_estimator\n')
            f.write(f"\n")
            f.write(f"prefill_table[0].extend({prefill_table[0]})\n")
            f.write(f"prefill_table[1].extend({prefill_table[1]})\n")
            f.write(f"decode_table[0].extend({decode_table[0]})\n")
            f.write(f"decode_table[1].extend({decode_table[1]})\n")
            f.write(f"sample_prefill_table[0].extend({sample_prefill_table[0]})\n")
            f.write(f"sample_prefill_table[1].extend({sample_prefill_table[1]})\n")
            f.write(f"sample_decode_table[0].extend({sample_decode_table[0]})\n")
            f.write(f"sample_decode_table[1].extend({sample_decode_table[1]})\n")
            f.write(f"prepInp_prefill_table[0].extend({prepInp_prefill_table[0]})\n")
            f.write(f"prepInp_prefill_table[1].extend({prepInp_prefill_table[1]})\n")
            f.write(f"prepInp_decode_table[0].extend({prepInp_decode_table[0]})\n")
            f.write(f"prepInp_decode_table[1].extend({prepInp_decode_table[1]})\n")
            f.write(f"model_configs.update({model_configs})\n")
            f.write(f"prepare_cost_table.update({prepare_cost_table})\n")
            f.write(f"cost_table = my_per_iter_latency_estimator.get_cost_table_from_serialized_data(prefill_table, decode_table,"\
                "sample_prefill_table, sample_decode_table, "\
                "prepInp_prefill_table, prepInp_decode_table, "\
                "model_configs, prepare_cost_table)")
        return 





    def serialize(self, model_paths: List[str]):
        def _to_list(to_convert, model_paths):
            keys = [k for k in to_convert.keys() if k[0] in model_paths]
            values = list()
            for k in keys:
                subdict = to_convert[k]
                subvalues = list()
                for a, b in subdict.items():
                    subvalues.append((a, *(b[0]), *(b[1])))
                    assert len(subvalues[-1]) == 5
                values.append(subvalues)
            return keys, values
        
        prefill_table=_to_list(self.prefill_table, model_paths)
        decode_table=_to_list(self.decode_table, model_paths)
        sample_prefill_table=_to_list(self.sample_prefill_table, model_paths)
        sample_decode_table=_to_list(self.sample_decode_table, model_paths)
        prepInp_prefill_table=_to_list(self.prepInp_prefill_table, model_paths)
        prepInp_decode_table=_to_list(self.prepInp_decode_table, model_paths)
        
        return (prefill_table, decode_table,\
                sample_prefill_table, sample_decode_table, \
                prepInp_prefill_table, prepInp_decode_table, \
                self.model_configs, self.prepare_cost_table)
    

        



    def get_weight_loading_time_for_each_seqnum(self, decode_table, profile_num):
        seqnum=1
        ys = np.asarray([i[1] for i in decode_table[seqnum]]).reshape((-1, profile_num))
        return np.min(ys)






    def _select_cost_model_samples_from_records(
            self, table: Dict[int, List[Tuple[float, float]]], piece_num: int, profile_num: int, threshold: float,
            weight_loading_time: float):
        '''
            This function select the measurement samples for the linear/piecewise linear functions from the records.
            NOTE: 
            1. for exec latency cost model, we use piecewise linear function in the shape of ``relu''.
            2. for other cost models, we use linear function.
            INPUT:
                piece_num:  
                    if 1, linear function of form: y=ax+b;
                    if 2, ``relu'' like function: y=c if x<=x1; y=ax+b if x > x1.
                threshold: the threshold to check two latencys are close enough to be ``the same'', 
                    should be the same in the cost model data collection process.
        '''
        for seqnum in table:
            # select one record for each row
            xs = np.asarray([i[0] for i in table[seqnum]]).reshape((-1, profile_num))
            ys = np.asarray([i[1] for i in table[seqnum]]).reshape((-1, profile_num))
            indices = np.argmin(ys, axis=1)
            xs = xs[range(len(xs)), indices]
            ys = ys[range(len(ys)), indices]

            if piece_num == 1:
                # select the records with the smallest xs and the largest xs
                ind1 = np.argmin(xs)
                ind2 = np.argmax(xs)
                table[seqnum] = [(xs[ind], ys[ind]) for ind in [ind1, ind2]]
            else:
                # select three records: 
                # the smallest xs, the point on the increasing part of the ``relu'' shape, and the largest xs
                order = np.argsort(xs)
                xs = xs[order]
                ys = ys[order]
                for x, y in zip(xs, ys):
                    
                    # TODO here we need to deal with weight_loading_time (some samples are skipped during cost data collection)
                    if (y > (ys[0] + threshold)) and (x < xs[-1]):
                    # if (y > (weight_loading_time + threshold)) and (x < xs[-1]):
                        # if y is large but x == xs[-1], then we still think all ys are close enough to each other
                        # compute the intersection point
                        intersection_x = (ys[0] - y) / ((ys[-1] - y)/(xs[-1] - x)) + x
                        table[seqnum] = [(intersection_x, ys[0]), (xs[-1], ys[-1])]
                        break
                else:
                    # all ys are close enough to ys[0]
                    table[seqnum] = [(xs[0], ys[0]), (xs[-1], ys[-1])]
                
                
                # do some checks
                # (x1, y1), (x2, y2) = table[seqnum]
                # for x, y in zip(xs, ys):
                #     if x <= x1:
                #         # assert abs(y-y1) <= threshold, f"{y, y1}"
                #         if abs(y-y1) > 10*threshold:
                #             print(f"{y, y1}")
                #     else:
                #         # assert abs(y-((x-x1)*(y2-y1)/(x2-x1)+y1)) <= threshold, f"{y, ((x-x1)*(y2-y1)/(x2-x1)+y1)}; {seqnum, x1, y1, x2, y2, x}"
                #         if abs(y-((x-x1)*(y2-y1)/(x2-x1)+y1)) > 10*threshold:
                #             print(f"{y, ((x-x1)*(y2-y1)/(x2-x1)+y1)}; {seqnum, x1, y1, x2, y2, x}")



    def _need_model_weight_loading(self, exec_plan):
        '''
            exec_plan: (tp, gpu_ratio, wldeg, cache_gpu_num)
        '''
        return exec_plan[2] > 2



    def _get_tp_size_from_exec_plan(self, exec_plan):
        '''
            exec_plan: (tp, gpu_ratio, wldeg, cache_gpu_num)
        '''
        return exec_plan[0]




    def init_table(self, model_name, exec_plan, sample_config, filename,
                   profile_num, threshold,
                   trust_remote_code:bool, revision:Optional[str] = None):
        '''
            INPUT:
                profile_num, threshold: settings of the cost model data collection process.
        '''
        print(model_name, exec_plan)

        tp_size = get_tp_size_from_logfile(filename)

        for table in [self.prefill_table, self.decode_table, 
                      self.prepInp_prefill_table, self.prepInp_decode_table]:
            table[(model_name, exec_plan)] = defaultdict(list)
        for table in [self.sample_prefill_table, self.sample_decode_table]:
            table[(model_name, exec_plan, sample_config)] = defaultdict(list)
        
        prefill_table = self.prefill_table[(model_name, exec_plan)]
        decode_table = self.decode_table[(model_name, exec_plan)]
        prepInp_prefill_table = self.prepInp_prefill_table[(model_name, exec_plan)]
        prepInp_decode_table = self.prepInp_decode_table[(model_name, exec_plan)]
        sample_prefill_table = self.sample_prefill_table[(model_name, exec_plan, sample_config)]
        sample_decode_table = self.sample_decode_table[(model_name, exec_plan, sample_config)]
        # 
        count = 0
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if '[[' == line[:2]:
                    seqnum, is_prompt, flops, exec_latency = \
                        MyThroughputLogger.get_seqnum_isprompt_flops_execLatency(line=line)
                    _, context_tot_len, _, sample_latency = \
                        MyThroughputLogger.get_seqnum_contextTotLen_isprompt_sampleLatency(line=line)
                    _, _, max_seqlen, _, prepInp_latency = \
                        MyThroughputLogger.get_seqnum_contextTotLen_maxSeqlen_isprompt_prepInpLatency(line=line)
                    count += 1
                    if count <= 2:
                        continue


                    # this record is useful and we need to store it
                    recompute_flops = self.comp_flops(
                        tp_size,
                        np.asarray([seqnum]), np.asarray([max_seqlen]), np.asarray([context_tot_len]), is_prompt,
                        model_name, trust_remote_code, revision)[0]/1e12
                    
                    # temporary accuracy check
                    if model_name in ['NousResearch/Llama-2-7b-hf', 'NousResearch/Llama-2-13b-hf']:
                        assert flops == recompute_flops*tp_size, f"wrong flops recomputation: {flops} VS recomp-> {recompute_flops}"
                    
                    flops = recompute_flops


                    cost_table, sample_cost_table, prepInp_cost_table = None, None, None
                    if is_prompt:
                        cost_table = prefill_table
                        sample_cost_table = sample_prefill_table
                        prepInp_cost_table = prepInp_prefill_table
                    else:
                        cost_table = decode_table
                        sample_cost_table = sample_decode_table
                        prepInp_cost_table = prepInp_decode_table
                    cost_table[seqnum].append((flops, exec_latency))
                    sample_cost_table[seqnum].append((context_tot_len, sample_latency))
                    prepInp_cost_table[seqnum].append((seqnum*max_seqlen, prepInp_latency))

                    # # NOTE: we assume there is 2 or 3 records for each seqnum
                    # if len(cost_table[seqnum])>2:
                    #     # we need to delete the record with the highest flops, i.e., only keep two records
                    #     cost_table[seqnum] = sorted(cost_table[seqnum], key=lambda i: i[0])
                    #     cost_table[seqnum] = cost_table[seqnum][:2]
        
        # select the necessary records for each seqnum in each table--------------
        weight_loading_time = self.get_weight_loading_time_for_each_seqnum(decode_table, profile_num)
        for table in [prefill_table, decode_table]:
            piece_num = 2 if self._need_model_weight_loading(exec_plan) else 1
            self._select_cost_model_samples_from_records(
                table, piece_num=piece_num, profile_num=profile_num, threshold=threshold, weight_loading_time=weight_loading_time)
        
        for table in [sample_prefill_table, sample_decode_table,
                      prepInp_prefill_table, prepInp_decode_table]:
            self._select_cost_model_samples_from_records(
                table, piece_num=1, profile_num=profile_num, threshold=threshold, weight_loading_time=weight_loading_time)            
        
        # # we only keep two endpoint records for each key in the cost table
        # for table in [prefill_table, decode_table, 
        #               sample_prefill_table, sample_decode_table,
        #               prepInp_prefill_table, prepInp_decode_table]:
        #     for seqnum in table:
        #         vs = sorted(table[seqnum], key=lambda i: i[0])
        #         assert len(vs)%2 == 0
        #         ys = np.asarray([i[1] for i in vs]).reshape(2, -1)
        #         indices = np.argmin(ys, axis=1)
        #         table[seqnum] = [vs[indices[0]], vs[indices[1]+ys.shape[1]]]
        



    def init_prepInp_decode_table(self, model_name, exec_plan, filename):
        self.prepInp_decode_table[(model_name, exec_plan)] = defaultdict(list)
        
        table = self.prepInp_decode_table[(model_name, exec_plan)]
        # 
        count = 0
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if '[[' in line:
                    seqnum, _, max_seqlen, is_prompt, prepInp_latency = \
                        MyThroughputLogger.get_seqnum_contextTotLen_maxSeqlen_isprompt_prepInpLatency(line=line)
                    count += 1
                    if count <= 2:
                        continue
                    # this record is useful and we need to store it
                    assert not is_prompt
                    table[seqnum].append((seqnum*max_seqlen, prepInp_latency))

        # we only keep two endpoint records for each key in the cost table
        for seqnum in table:
            vs = sorted(table[seqnum], key=lambda i: i[0])
            assert len(vs)%2 == 0
            ys = np.asarray([i[1] for i in vs]).reshape(2, -1)
            indices = np.argmin(ys, axis=1)
            table[seqnum] = [vs[indices[0]], vs[indices[1]+ys.shape[1]]]
        




    def init_model_config(self, model_name:str, trust_remote_code:bool, revision:Optional[str] = None):
        key = (model_name, trust_remote_code, revision)
        if key not in self.model_configs:
            hf_config = get_config(*key)
            # V: int = hf_config.vocab_size
            h: int = hf_config.hidden_size

            # I: int = hf_config.intermediate_size
            # I: int = getattr(hf_config, "intermediate_size", None) or \
            #                 getattr(hf_config, "n_inner", None) or \
            #                 getattr(hf_config, "inner_hidden_size", None) or \
            #                 (4*h)
            
            L: int = hf_config.num_hidden_layers
            
            # kv_head_num: int = hf_config.num_key_value_heads
            # self.kv_head_num: int = model_config.get_total_num_kv_heads() #hf_config.num_key_value_heads

            # head_dim: int = h // hf_config.num_attention_heads

            # self.model_configs[key] = (V,h,I,L, kv_head_num, head_dim)
            self.model_configs[key] = (h,L)



    def _linear_function(self, cost_table, seq_nums, Xs, piece_num):
        '''
            Compute the corresponding Ys for the given Xs.
            The cost model is a linear function
            Input: Xs should be np arrays.
        '''
        if piece_num == 1:
            records = [cost_table[seqnum] for seqnum in seq_nums]
            X1 = np.asarray([i[0][0] for i in records])
            Y1 = np.asarray([i[0][1] for i in records])
            X2 = np.asarray([i[1][0] for i in records])
            Y2 = np.asarray([i[1][1] for i in records])
            Ys = (Y2 - Y1) / (X2 - X1) * (Xs - X1) + Y1
            return Ys
        else:
            records = [cost_table[seqnum] for seqnum in seq_nums]
            X1 = np.asarray([i[0][0] for i in records])
            Y1 = np.asarray([i[0][1] for i in records])
            X2 = np.asarray([i[1][0] for i in records])
            Y2 = np.asarray([i[1][1] for i in records])
            Ys1 = (Xs <= X1) * Y1
            Ys2 = (Xs > X1) * ( (Y2 - Y1) / (X2 - X1) * (Xs - X1) + Y1 )
            Ys = Ys1 + Ys2
            return Ys




    def comp_flops(
            self,
            tp_size,
            B_array, s_array, context_tot_len_array, is_prompt: bool,
            model_path:str, trust_remote_code:bool, revision:Optional[str] = None):
        '''
            Call cal_flops to compute the flops for the given model.
            NOTE: B_array, s_array, context_tot_len_array should be Numpy Arrays.
        '''
        key = (model_path, trust_remote_code, revision)

        if key not in self.model_configs:
            # if no available model config data, return a small flops
            print(f"key {key} not in self.model_configs: {self.model_configs.keys()}")

            return np.asarray([1e-9]*len(B_array))

        # _, h,I,L, kv_head_num, head_dim = self.model_configs[key]
        h,L = self.model_configs[key]
        flops = cal_flops(model_path, h,L, tp_size, B_array, s_array, context_tot_len_array, is_prompt)
        return flops



    def estimate_cost(
            self, 
            B: List[int], s: List[int], context_tot_len: List[int], is_prompt: bool, 
            model_name:str, exec_plan, sample_config, trust_remote_code:bool, revision:Optional[str] = None):
        """
            Input:
                B, s, context_tot_len can be arrays of values, i.e., we can estimate costs in batch;
                But these iterations should all belong to prefill stages or all belong to decode stages.
                exec_plan: the execution plan of the model: (tp, gpu_ratio, wldeg, cache_gpu_num)
            Output:
                the estimated costs.
        """
        B_array = np.asarray(B)
        s_array = np.asarray(s)
        context_tot_len_array = np.asarray(context_tot_len)

        # compute model exec flops
        # key = (model_name, trust_remote_code, revision)
        # h,I,L = self.model_configs[key]
        # flops = cal_flops(h,I,L, B_array, s_array, context_tot_len_array, is_prompt)
        flops = self.comp_flops(
            self._get_tp_size_from_exec_plan(exec_plan),
            B_array, s_array, context_tot_len_array, is_prompt,
            model_name, trust_remote_code, revision)


        # fetch corresponding cost table records
        if (model_name, exec_plan) not in self.prefill_table:
            # if no available cost model, return a large latency <= we should prune such exec_plan in advance
            assert False
            return np.asarray([1e9] * len(B))
        
        cost_table = self.prefill_table[(model_name, exec_plan)]
        sample_cost_table = self.sample_prefill_table[(model_name, exec_plan, sample_config)]
        prepInp_cost_table = self.prepInp_prefill_table[(model_name, exec_plan)]
        if not is_prompt:
            cost_table = self.decode_table[(model_name, exec_plan)]
            sample_cost_table = self.sample_decode_table[(model_name, exec_plan, sample_config)]
            prepInp_cost_table = self.prepInp_decode_table[(model_name, exec_plan)]
        
        # estimate the total latency
        piece_num = 2 if self._need_model_weight_loading(exec_plan) else 1
        latencys = self._linear_function(cost_table, B, flops/1e12, piece_num=piece_num)
        latencys = latencys + self._linear_function(sample_cost_table, B, context_tot_len_array, piece_num=1)
        latencys = latencys + self._linear_function(prepInp_cost_table, B, B_array*s_array, piece_num=1)
        
        # cost_table = cost_table[(model_name, exec_plan)]
        # records = [cost_table[seqnum] for seqnum in B]
        # X1 = np.asarray([i[0][0] for i in records])
        # Y1 = np.asarray([i[0][1] for i in records])
        # X2 = np.asarray([i[1][0] for i in records])
        # Y2 = np.asarray([i[1][1] for i in records])
        # exec_latencys = (Y2 - Y1) / (X2 - X1) * (flops/1e12 - X1) + Y1

        return latencys


    def can_estimate_cost(self, model_path:str, exec_plan):
        return ((model_path, exec_plan) in self.prefill_table)

    def get_model_name_from_model_path(self, model_path:str):
        pos = model_path.find('/')
        model_name = model_path[pos+1:]
        return model_name


    def load_init_cost_from_database_file(self, machine_name:str):
        model_init_costs = None
        if machine_name == 'machine1':
            import model_initcost_database
            model_init_costs = model_initcost_database.model_init_costs
        elif machine_name == 'machine2':
            import model_initcost_database
            model_init_costs = model_initcost_database.model_init_costs
        
        for (model_path, tp_size), init_cost in model_init_costs.items():
            model_name = self.get_model_name_from_model_path(model_path)
            self.prepare_cost_table[(model_name, (tp_size, 0.9, 2, 0))] = init_cost


    def init_prepare_cost(self, machine_name:str):
        # self.prepare_cost_table
        # we first prepare some hard data here, and we will update the table later.
        self.prepare_cost_table[('Llama-2-7b-hf', (1, 0.9, 2, 0))] = 3.7
        self.prepare_cost_table[('Llama-2-7b-hf', (1, 0.9, 8, 3))] = 2.6
        self.prepare_cost_table[('Llama-2-7b-hf', (1, 0.9, 16, 3))] = 2.3
        self.prepare_cost_table[('Llama-2-7b-hf', (2, 0.9, 2, 0))] = 17.6
        self.prepare_cost_table[('Llama-2-7b-hf', (2, 0.9, 8, 2))] = 18.4
        self.prepare_cost_table[('Llama-2-7b-hf', (2, 0.9, 16, 2))] = 18.5
        self.prepare_cost_table[('Llama-2-7b-hf', (4, 0.9, 2, 0))] = 37.7
        # 
        self.prepare_cost_table[('Llama-2-13b-hf', (1, 0.9, 2, 0))] = 3.2
        self.prepare_cost_table[('Llama-2-13b-hf', (1, 0.9, 10, 3))] = 2.83
        self.prepare_cost_table[('Llama-2-13b-hf', (1, 0.9, 20, 3))] = 2.64
        self.prepare_cost_table[('Llama-2-13b-hf', (2, 0.9, 2, 0))] = 19.1
        self.prepare_cost_table[('Llama-2-13b-hf', (2, 0.9, 10, 2))] = 19.3
        self.prepare_cost_table[('Llama-2-13b-hf', (2, 0.9, 20, 2))] = 22.4
        self.prepare_cost_table[('Llama-2-13b-hf', (4, 0.9, 2, 0))] = 40.8
        # 
        self.prepare_cost_table[('Llama-2-70b-hf', (2, 0.9, 2, 0))] = 15.0
        self.prepare_cost_table[('Llama-2-70b-hf', (2, 0.9, 16, 2))] = 36.3
        self.prepare_cost_table[('Llama-2-70b-hf', (2, 0.9, 20, 2))] = 36.7
        self.prepare_cost_table[('Llama-2-70b-hf', (2, 0.9, 40, 2))] = 45.8
        self.prepare_cost_table[('Llama-2-70b-hf', (4, 0.9, 2, 0))] = 47.4
        # 
        self.prepare_cost_table[('chatglm3-6b', (1, 0.9, 2, 0))] = 3.6
        self.prepare_cost_table[('chatglm3-6b', (2, 0.9, 2, 0))] = 19.6
        self.prepare_cost_table[('chatglm3-6b', (4, 0.9, 2, 0))] = 41.4
        # 
        self.prepare_cost_table[('gpt-j-6b', (1, 0.9, 2, 0))] = 3.2
        self.prepare_cost_table[('gpt-j-6b', (2, 0.9, 2, 0))] = 31.8
        self.prepare_cost_table[('gpt-j-6b', (4, 0.9, 2, 0))] = 54.2
        # 
        self.prepare_cost_table[('gpt-neox-20b', (1, 0.9, 2, 0))] = 3.0
        self.prepare_cost_table[('gpt-neox-20b', (2, 0.9, 2, 0))] = 25.0
        self.prepare_cost_table[('gpt-neox-20b', (4, 0.9, 2, 0))] = 44.5
        # 
        self.prepare_cost_table[('Llama-2-7b-chat-hf', (1, 0.9, 2, 0))] = 2.6
        self.prepare_cost_table[('Llama-2-7b-chat-hf', (2, 0.9, 2, 0))] = 18.0
        self.prepare_cost_table[('Llama-2-7b-chat-hf', (4, 0.9, 2, 0))] = 41.8
        # 
        self.prepare_cost_table[('Baichuan2-13B-Chat', (1, 0.9, 2, 0))] = 3.0
        self.prepare_cost_table[('Baichuan2-13B-Chat', (2, 0.9, 2, 0))] = 25.6
        self.prepare_cost_table[('Baichuan2-13B-Chat', (4, 0.9, 2, 0))] = 50.0
        # 
        self.prepare_cost_table[('Baichuan-7B', (1, 0.9, 2, 0))] = 2.8
        self.prepare_cost_table[('Baichuan-7B', (2, 0.9, 2, 0))] = 22.8
        self.prepare_cost_table[('Baichuan-7B', (4, 0.9, 2, 0))] = 44.7
        # 
        self.prepare_cost_table[('Mixtral-8x7B-v0.1', (2, 0.9, 2, 0))] = 36.8
        self.prepare_cost_table[('Mixtral-8x7B-v0.1', (4, 0.9, 2, 0))] = 46.5        

        # get init cost from the database file
        self.load_init_cost_from_database_file(machine_name)



    def get_prepare_cost(self, model_name:str, exec_plan):
        '''
            This function returns the prepare cost of the model with the exec_plan.
            NOTE: we only consider hard preemption now, 
            i.e., (1) all model weights are reloaded, (2) the nccl group is reinitialized.
        '''
        tmp_exec_plan = (exec_plan[0], 0.9, exec_plan[2], exec_plan[3])
        return self.prepare_cost_table[(model_name, tmp_exec_plan)]




def get_cost_table():
    '''
        Return the log files and other parameters used to build our cost model.
    
        Example input for CostTable:
            model_path = 'NousResearch/Llama-2-7b-hf'
            exec_plan = (1, 0.9, 2, 0) # (tp, gpu_ratio, wldeg, cache_gpu_num)
            sample_config = (1, 1, -1, 0) #(temp, top_p, top_k, min_p)
    
    '''
    def get_model_name_from_path(model_path: str):
        pos = model_path.find('/')
        model_name = model_path[pos+1:]
        return model_name

    model_exec_settings = []
    model_exec_settings.extend(
        [('NousResearch/Llama-2-7b-hf', (tp, 0.9, wldeg, cache_gpu_num))
         for (tp, wldeg, cache_gpu_num) in 
         [(1, 2, 0), (2, 2, 0), (4, 2, 0), 
          (1, 8, 3), (1, 16, 3), 
          (2, 8, 2), (2, 16, 2)]]
    )
    model_exec_settings.extend(
        [('NousResearch/Llama-2-13b-hf', (tp, 0.9, wldeg, cache_gpu_num))
         for (tp, wldeg, cache_gpu_num) in 
         [(1, 2, 0), (2, 2, 0), (4, 2, 0), 
          (1, 10, 3), (1, 20, 3), 
          (2, 10, 2), (2, 20, 2)]]
    )
    model_exec_settings.extend(
        [('NousResearch/Llama-2-70b-hf', (tp, 0.9, wldeg, cache_gpu_num))
         for (tp, wldeg, cache_gpu_num) in 
         [(2, 2, 0), (4, 2, 0), 
          (2, 16, 2), (2, 20, 2), (2, 40, 2)]]
    )

    sample_config = (1, 1, -1, 0)
    logfiles = []
    profile_num = 2
    threshold = 1e-3

    # get log file info
    for model_exec_setting in model_exec_settings:
        model_path, exec_plan = model_exec_setting
        model_name = get_model_name_from_path(model_path)

        file_path = None
        if ((model_name == 'Llama-2-7b-hf') \
            and (exec_plan == (1, 0.9, 2, 0)) \
                and (sample_config == (1, 1, -1, 0))):
            file_path = './Cost_Model_per_iter/Formal_Llama-2-7b-hf_1_tp1_temp1.0_wldeg2_samp_prepInp.log'
            profile_num = 5
        elif ((model_name == 'Llama-2-70b-hf') \
            and (exec_plan in [(2, 0.9, 2, 0), (4, 0.9, 2, 0), (2, 0.9, 16, 2)]) \
                and (sample_config == (1, 1, -1, 0))):
            tp, _, wldeg, _ = exec_plan
            file_path = f'./Cost_Model_per_iter/Formal_Llama-2-70b-hf_tp{tp}_temp1.0_wldeg{wldeg}_2_2.log'
            profile_num = 2
        else:
            tp, _, wldeg, _ = exec_plan
            file_path = f'./Cost_Model_per_iter/Formal_{model_name}_tp{tp}_temp1.0_wldeg{wldeg}_2_4.log'
            profile_num = 2

        logfiles.append((model_path, exec_plan, sample_config, file_path, profile_num, threshold, 
                         True, None))




    model_exec_settings = []
    model_exec_settings.extend(
        [(model_path, (tp, 0.9, 2, 0))
         for model_path in [
            # 'NousResearch/Llama-2-7b-hf', 
            'NousResearch/Llama-2-7b-chat-hf',
            # 'NousResearch/Llama-2-13b-hf',
            # 'NousResearch/Llama-2-70b-hf',
            'THUDM/chatglm3-6b',
            'EleutherAI/gpt-j-6b', 
            'EleutherAI/gpt-neox-20b',
            'baichuan-inc/Baichuan2-13B-Chat',
            'baichuan-inc/Baichuan-7B',
            'mistralai/Mixtral-8x7B-v0.1',
         ] for tp in [1,2,4] if not ( (model_path == 'mistralai/Mixtral-8x7B-v0.1') and (tp == 1) )]
    )
    for model_path, exec_plan in model_exec_settings:
        model_name = get_model_name_from_path(model_path)
        tp, _, wldeg, _ = exec_plan
        profile_num = 2
        file_path = f'./Cost_Model_per_iter/Formal_{model_name}_tp{tp}_temp1.0_wldeg{wldeg}_3_4.log'
        logfiles.append((model_path, exec_plan, sample_config, file_path, profile_num, threshold, 
                    True, None))



    # add NEWROUND models from LLM-blender---------------------
    model_exec_settings = []
    model_exec_settings.extend(
        [(model_path, (tp, 0.9, 2, 0))
         for model_path in [
            'lmsys/vicuna-13b-v1.5',
            'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
            'chavinlo/alpaca-13b',
            'project-baize/baize-v2-13b',
            'TheBloke/koala-13B-HF',
            'databricks/dolly-v2-12b',
            'mosaicml/mpt-7b-chat',
         ] for tp in [1,2,4] if not ( (model_path == 'mistralai/Mixtral-8x7B-v0.1') and (tp == 1) )]
    )
    for model_path, exec_plan in model_exec_settings:
        model_name = get_model_name_from_path(model_path)
        tp, _, wldeg, _ = exec_plan
        profile_num = 2
        file_path = f'./Cost_Model_per_iter/NEWROUND_{model_name}_tp{tp}_4_temp1.0_wldeg{wldeg}_1.log'
        logfiles.append((model_path, exec_plan, sample_config, file_path, profile_num, threshold, 
                    True, None))





    # get decode prepInp log file info
    # each tuple in prepInp_decode_logfiles: (model_name, exec_plan, filename)
    prepInp_decode_logfiles = [
        ('NousResearch/Llama-2-7b-hf', (1, 0.9, 2, 0), 
         './Cost_Model_per_iter/Formal_Llama-2-7b-hf_1_tp1_temp1.0_wldeg2_DecodePrepInp.log')]

    # each item is (model_name, trust_remote_code, revision)
    model_infos = [(model_path, True, None) for model_path \
                   in [
                        'NousResearch/Llama-2-7b-hf', 
                        'NousResearch/Llama-2-7b-chat-hf',
                        'NousResearch/Llama-2-13b-hf',
                        'NousResearch/Llama-2-70b-hf',
                        'THUDM/chatglm3-6b',
                        'EleutherAI/gpt-j-6b', 
                        'EleutherAI/gpt-neox-20b',
                        'baichuan-inc/Baichuan2-13B-Chat',
                        'baichuan-inc/Baichuan-7B',
                        'mistralai/Mixtral-8x7B-v0.1',
                        # 
                        # add NEWROUND models from LLM-blender---------------------
                        'lmsys/vicuna-13b-v1.5',
                        'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
                        'chavinlo/alpaca-13b',
                        'project-baize/baize-v2-13b',
                        'TheBloke/koala-13B-HF',
                        'databricks/dolly-v2-12b',
                        'mosaicml/mpt-7b-chat',
                        ]
                #    in ['NousResearch/Llama-2-7b-hf', 
                #        'NousResearch/Llama-2-13b-hf',
                #        'NousResearch/Llama-2-70b-hf']
                ]
    

    for file_info in logfiles:
        print('!/benchmarks'+file_info[3][1:])
    for file_info in prepInp_decode_logfiles:
        print('!/benchmarks'+file_info[2][1:])

    cost_table = CostTable(
        logfiles=logfiles,
        prepInp_decode_logfiles=prepInp_decode_logfiles,
        model_infos=model_infos,
        machine_name='machine1')

    cost_table.store_meta_data(filename='get_my_cost_table_directly.py')

    return cost_table




def get_cost_table_2():
    '''
        Return the log files and other parameters used to build our cost model.
    
        Example input for CostTable:
            model_path = 'NousResearch/Llama-2-7b-hf'
            exec_plan = (1, 0.9, 2, 0) # (tp, gpu_ratio, wldeg, cache_gpu_num)
            sample_config = (1, 1, -1, 0) #(temp, top_p, top_k, min_p)
        NOTE: this version build cost table for machine2.
    '''
    def get_model_name_from_path(model_path: str):
        pos = model_path.find('/')
        model_name = model_path[pos+1:]
        return model_name
    
    
    sample_config = (1, 1, -1, 0)
    logfiles = []
    profile_num = 2
    threshold = 1e-3

    # add NEWROUND models from LLM-blender and routerbench---------------------
    model_exec_settings = []
    model_exec_settings.extend(
        [(model_path, (tp, 0.9, 2, 0))
         for model_path in [
            # 'lmsys/vicuna-13b-v1.5',
            # 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
            # 'chavinlo/alpaca-13b',
            # 'project-baize/baize-v2-13b',
            # 'TheBloke/koala-13B-HF',
            # 'databricks/dolly-v2-12b',
            # 'mosaicml/mpt-7b-chat',
            # 'meta-llama/Llama-2-70b-chat-hf',
            # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            # 'WizardLMTeam/WizardLM-13B-V1.2',
            # 'meta-llama/CodeLlama-34b-Instruct-hf',
            # 'mistralai/Mistral-7B-Instruct-v0.2',
            # 
            'THUDM/chatglm3-6b',
            'stabilityai/stablelm-tuned-alpha-7b',
         ] for tp in [1,2] if not \
            ( (model_path in [\
                'mistralai/Mixtral-8x7B-v0.1', \
                'meta-llama/Llama-2-70b-chat-hf', \
                'mistralai/Mixtral-8x7B-Instruct-v0.1',]) and (tp == 1) )]
    )
    for model_path, exec_plan in model_exec_settings:
        model_name = get_model_name_from_path(model_path)
        tp, _, wldeg, _ = exec_plan
        profile_num = 2
        file_path = f'./Cost_Model_per_iter_machine2/NEWROUND_{model_name}_tp{tp}_5_temp1.0_wldeg{wldeg}_1.log'
        logfiles.append((model_path, exec_plan, sample_config, file_path, profile_num, threshold, 
                    True, None))





    # get decode prepInp log file info
    # each tuple in prepInp_decode_logfiles: (model_name, exec_plan, filename)
    prepInp_decode_logfiles = []

    # each item is (model_name, trust_remote_code, revision)
    model_infos = [(model_path, True, None) for model_path \
                   in [
                        # 'NousResearch/Llama-2-7b-hf', 
                        # 'NousResearch/Llama-2-7b-chat-hf',
                        # 'NousResearch/Llama-2-13b-hf',
                        # 'NousResearch/Llama-2-70b-hf',
                        # 'THUDM/chatglm3-6b',
                        # 'EleutherAI/gpt-j-6b', 
                        # 'EleutherAI/gpt-neox-20b',
                        # 'baichuan-inc/Baichuan2-13B-Chat',
                        # 'baichuan-inc/Baichuan-7B',
                        # 'mistralai/Mixtral-8x7B-v0.1',
                        # 
                        # add NEWROUND models from LLM-blender---------------------
                        # 'lmsys/vicuna-13b-v1.5',
                        # 'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5',
                        # 'chavinlo/alpaca-13b',
                        # 'project-baize/baize-v2-13b',
                        # 'TheBloke/koala-13B-HF',
                        # 'databricks/dolly-v2-12b',
                        # 'mosaicml/mpt-7b-chat',
                        # # 
                        # 'meta-llama/Llama-2-70b-chat-hf',
                        # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                        # 'WizardLMTeam/WizardLM-13B-V1.2',
                        # 'meta-llama/CodeLlama-34b-Instruct-hf',
                        # 'mistralai/Mistral-7B-Instruct-v0.2',     
                        # 
                        'THUDM/chatglm3-6b',
                        'stabilityai/stablelm-tuned-alpha-7b',                   
                        ]
                #    in ['NousResearch/Llama-2-7b-hf', 
                #        'NousResearch/Llama-2-13b-hf',
                #        'NousResearch/Llama-2-70b-hf']
                ]
    

    for file_info in logfiles:
        print('!/benchmarks'+file_info[3][1:])
    for file_info in prepInp_decode_logfiles:
        print('!/benchmarks'+file_info[2][1:])

    cost_table = CostTable(
        logfiles=logfiles,
        prepInp_decode_logfiles=prepInp_decode_logfiles,
        model_infos=model_infos,
        machine_name='machine2')

    # cost_table.store_meta_data(filename='get_my_cost_table_directly2.py')
    cost_table.update_meta_data(filename='get_my_cost_table_directly2.py')

    return cost_table




def get_cost_table_from_serialized_data(
        prefill_table, decode_table,
        sample_prefill_table, sample_decode_table, 
        prepInp_prefill_table, prepInp_decode_table,
        model_configs, prepare_cost_table):
    def _to_dict(to_convert):
        keys, values = to_convert
        ret = dict()
        for k, subvalues in zip(keys, values):
            subdict = {a[0]: ((a[1], a[2]), (a[3], a[4])) for a in subvalues}
            ret[k] = subdict
        return ret
    
    prefill_table=_to_dict(prefill_table)
    decode_table=_to_dict(decode_table)
    sample_prefill_table=_to_dict(sample_prefill_table)
    sample_decode_table=_to_dict(sample_decode_table)
    prepInp_prefill_table=_to_dict(prepInp_prefill_table)
    prepInp_decode_table=_to_dict(prepInp_decode_table)

    metadata = (prefill_table, decode_table,\
        sample_prefill_table, sample_decode_table, \
        prepInp_prefill_table, prepInp_decode_table, \
        model_configs, prepare_cost_table)
    cost_table = CostTable([],[],[],None,metadata=metadata)
    
    return cost_table