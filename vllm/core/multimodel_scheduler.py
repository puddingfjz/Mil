""" Code for determine multimodel scheduling. """


from multiprocessing import Array, Event, Lock
import os
from typing import List, Optional, Tuple, Dict, Union
from collections import deque, defaultdict
import time

from ray.util.placement_group import (
            placement_group_table,
            remove_placement_group,
            get_placement_group,
        )


from vllm.core.block_manager import KVBlkPerLayerWeight


from vllm.engine.ray_utils import ray


from multiprocessing.managers import BaseManager
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
import traceback
from transformers import AutoTokenizer


class DummyLock:
    """
        A dummy lock which does nothing. Used by LLM_COMMUNICATOR.
    """
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self,*args):
        pass


class LLM_COMMUNICATOR:
    """
        A class used to communicate between different LLMs for input/output passing.
        NOTE:
            1. When a model only has one input model, the fused_inp_queue is the same as the queue in output_pool.
            2. for the models without input models, we also prepare their inputs in the output pool as the output of a dummy model.
            3. self._read_output_nums[model_1, model_2] records the number of seqs in self.output_pool of model_1 that has been read by model_2
            4. self.fetched_fused_inp_start[to_model_id][dp_i] records the number of complete inps that has been read by to_model's dp_i worker.
                this variable will be RESET every time the model is restarted.
            5. in ``in_edge_dict_with_dummy_inp_nodes``, we use NEGATIVE model_ids to represent DUMMY inp nodes.
    """
    def __init__(
            self, model_num: int,
            in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]],

            base_model_ids_dict: Dict[int, List[int]],
            inp_req_ids: Dict[int, Dict[int, List[int]]],
            inp_req_from_which_model_which_out_reqs: Dict[int, Dict[int, Dict[int, int]]],
            out_req_id_mapping:Dict[int, Dict[int, Tuple[int, int]]],
            new_out_req_part_num: Dict[int, int],
            independent_srcs: Dict[int, bool],
            prompt_template_args: Dict[int, Tuple],
            ):
        """
            NOTE:
                1. The input parameters: 
                    in_edge_dict_with_dummy_inp_nodes, inp_req_ids, out_req_id_mapping, new_out_req_part_num, independent_srcs are all
                    about original base models.
                2. model_num: the number of models in the system which may contain fused models
        """


        self.output_pool: Dict[int, Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]] = defaultdict(list)

        self._read_output_nums: Dict[Tuple[int, int], int] = defaultdict(int)

        self._available_srcs: Dict[int, Dict[int, Tuple[int, Union[List[str],List[List[int]]]]]] = defaultdict(dict)

        self.fused_inp_queues: Dict[int, Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]] = defaultdict(list)

        self.fetched_fused_inp_start: Dict[int, List[int]] = defaultdict(list)
        self.in_edge_dict_with_dummy_inp_nodes: Dict[int, List[int]] = in_edge_dict_with_dummy_inp_nodes

        self._unavailable_req_nums: Dict[int, int] = dict()
        self._ungened_out_req_nums: Dict[int, int] = dict()


        self.base_model_ids_dict: Dict[int, List[int]] = base_model_ids_dict


        self.inp_req_ids: Dict[int, Dict[int, List[int]]] = inp_req_ids
        self.inp_req_from_which_model_which_out_reqs: Dict[int, Dict[int, Dict[int, int]]] = inp_req_from_which_model_which_out_reqs
        self.inp_req_from_which_model_which_out_reqs_reversed: Dict[int, Dict[int, Dict[int, List[int]]]] = None
        self._init_inp_req_from_which_model_which_out_reqs_reversed()


        self.out_req_id_mapping: Dict[int, Dict[int, Tuple[int, int]]] = out_req_id_mapping

        self.new_out_req_part_num: Dict[int, Dict[int, int]] = new_out_req_part_num

        self._meta_output_pool: Dict[int, Union[Dict[int, Tuple[int, List[str]]], Dict[int, Tuple[int, List[List[int]]]]]] = dict()
        self._init_meta_output_pool()


        self._independent_srcs: Dict[int, bool] = independent_srcs


        self.out_req_model_id_mapping: Dict[int, Dict[int, int]] = defaultdict(dict)


        self._add_req_locks: Dict[int, Lock] = {_: Lock() for _ in in_edge_dict_with_dummy_inp_nodes}
        self._fetch_req_locks: Dict[int, Lock] = {_: Lock() for _ in in_edge_dict_with_dummy_inp_nodes}


        self.prompt_template_args: Dict[int, Tuple] = prompt_template_args
        self.base_model_tokenizers: Dict[int, AutoTokenizer] = dict()
        self._init_model_tokenizers()


        self.outlen_dict: Dict[int, Dict[int, int]] = defaultdict(dict)


        self.non_compute_ranges = defaultdict(list)
        self.non_compute_ranges_lock = Lock()

        print(f"In INIT Communicator:")
        print(f"self._add_req_locks: {self._add_req_locks}")
        print(f"self._fetch_req_locks: {self._fetch_req_locks}")
        print(f"self._unavailable_req_nums: {self._unavailable_req_nums}")
        print(f"self._ungened_out_req_nums: {self._ungened_out_req_nums}")
        print(f"self.base_model_ids_dict: {self.base_model_ids_dict}")
        print(f"self.base_model_tokenizers: {self.base_model_tokenizers:}")


    def update_non_compute_ranges(
        self,
        non_compute_ranges: Tuple[List[float], List[float]],
        shared_id: int, gpu_ids: List[int]):
        """
            INPUT: non_compute_ranges: [list of range starts, list of range ends].
        """
        with self.non_compute_ranges_lock:
            for gpu_id in gpu_ids:
                self.non_compute_ranges[gpu_id].append([shared_id, non_compute_ranges])


    def get_non_compute_ranges(self):
        return self.non_compute_ranges


    def get_all_model_outputs(self):
        """
            This function returns the output length for each req of each model.
        """
        try:
            ret = {base_model_id:v for base_model_id, v in self.outlen_dict.items() if base_model_id >= 0}
            print(f"outlen_dict: {ret}")
            return ret
            outlen_dict: Dict[int, Dict[int, int]] = dict()
            for base_model_id, out_contents in self.output_pool.items():
                if base_model_id < 0:
                    continue
                if isinstance(out_contents[0][1], list):

                    outlen_dict[base_model_id] = {req_id:len(out) for req_id, out in out_contents}
                else:
                    assert isinstance(out_contents[0][1], str), "the existing output is either token id list nor output string"
                    tokenizer = self.base_model_tokenizers[base_model_id]
                    out_strs = [out for req_id, out in out_contents]
                    token_ids_list = tokenizer(out_strs, add_special_tokens=False, padding=False, truncation=False).input_ids
                    outlen_dict[base_model_id] = {req_id:len(token_ids) for (req_id, _), token_ids in zip(out_contents, token_ids_list)}
            print(f"outlen_dict: {outlen_dict}")
            return outlen_dict
        except Exception as e:
            print(f"ERROR: {e}!")
            print(traceback.format_exc())


    def fuse_inp_srcs(self, src_seqs):
        ret = src_seqs[0]
        for src in src_seqs[1:]:
            ret += src
        return ret


    def fuse_inp_srcs_based_on_template(self, src_seqs_list, base_model_id, one_src: bool):
        if len(self.prompt_template_args[base_model_id]) <= 2:

            if one_src:
                return src_seqs_list
            else:
                rets = [(req_id, ''.join(src_seqs)) for req_id, src_seqs in src_seqs_list]
                return rets
        else:
            template = self.prompt_template_args[base_model_id][2]
            if one_src:
                return [(req_id, template.format(src_seqs)) for req_id, src_seqs in src_seqs_list]
            else:
                srcs_in_order = self.prompt_template_args[base_model_id][3]
                inps = self.in_edge_dict_with_dummy_inp_nodes[base_model_id]
                mapping = {inp:i for i, inp in enumerate(inps)}
                return [(req_id, template.format(*[src_seqs[mapping[src]] for src in srcs_in_order])) \
                        for req_id, src_seqs in src_seqs_list]


    def _init_inp_req_from_which_model_which_out_reqs_reversed(self):
        self.inp_req_from_which_model_which_out_reqs_reversed = dict()
        for to_model_id, from_info in self.inp_req_from_which_model_which_out_reqs.items():
            self.inp_req_from_which_model_which_out_reqs_reversed[to_model_id] = dict()
            to_info = self.inp_req_from_which_model_which_out_reqs_reversed[to_model_id]
            for from_model_id, from_req_info in from_info.items():
                to_info[from_model_id] = {out_req_id:list() for out_req_id in from_req_info.values()}
                to_req_info = to_info[from_model_id]
                for inp_req_id, out_req_id in from_req_info.items():
                    to_req_info[out_req_id].append(inp_req_id)


    def _init_model_tokenizers(self) -> None:
        """
            Prepare tokenizers for the base models.
        """
        print(f"in _init_model_tokenizers")
        _tmp_tokenizer_dict = dict()

        for base_model_id, args in self.prompt_template_args.items():
            tokenizer_name, trust_remote_code = args[:2]


            tokenizer = None
            if (tokenizer_name, trust_remote_code) in _tmp_tokenizer_dict:
                tokenizer = _tmp_tokenizer_dict[(tokenizer_name, trust_remote_code)]
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=trust_remote_code)
                _tmp_tokenizer_dict[(tokenizer_name, trust_remote_code)] = tokenizer


            self.base_model_tokenizers[base_model_id]=tokenizer


    def _init_meta_output_pool(self):
        """
            Init self._meta_output_pool according to self.out_req_id_mapping, self.new_out_req_part_num.
        """
        print(f"self.new_out_req_part_num: {self.new_out_req_part_num}")
        print(f"self.out_req_id_mapping: {self.out_req_id_mapping}")
        for model_id in self.new_out_req_part_num:
            self._meta_output_pool[model_id] = dict()
            for new_out_req_id, tot_part_num in self.new_out_req_part_num[model_id].items():
                self._meta_output_pool[model_id][new_out_req_id] = [0, [None for _ in range(tot_part_num)]]


    def _get_correct_lock(
            self, model_id: int, lock_dict: Dict[int, Lock], read_only: bool
        ) -> Union[DummyLock, Lock]:
        """
            Get the correct lock from the lock list (if any).
            NOTE:
                1. if to get read_only _add_req_locks and the inp models have sent out all out reqs,
                    return a dummy lock.
        """


        if model_id < 0:
            return DummyLock()
        if (lock_dict == self._add_req_locks) and read_only \
            and (self._ungened_out_req_nums[model_id] == 0):
            return DummyLock()
        return lock_dict[model_id]


    def init_unavailable_req_nums_and_ungened_out_req_nums(
            self, unavailable_req_nums: Dict[int, int], ungened_out_req_nums: Dict[int, int]):
        """
            NOTE: do not need lock here, because it is called before launching LLM infer processes.
        """
        self._unavailable_req_nums.update(unavailable_req_nums)
        self._ungened_out_req_nums.update(ungened_out_req_nums)
        print(f"self._unavailable_req_nums: {self._unavailable_req_nums}")
        print(f"self._ungened_out_req_nums: {self._ungened_out_req_nums}")


    def _reset_state_for_model(self, model_id: int, dp_size: int):
        """
            When a model is restarted, we need to update some variables.
            Update: self.fused_inp_queues, self.fetched_fused_inp_start.
            NOTE: 
                1. do not need lock here, will add lock where this method is called.
                2. the input model_id can be of fused models or base models, but 
                ``fused_inp_queues`` and ``fetched_fused_inp_start`` are for base models only.
        """
        print(f"In _reset_state_for_model, parameters: (base_model_id, dp_size): {model_id, dp_size}-----------------\n")

        old_dp_size = len(self.fetched_fused_inp_start[model_id])
        remaining = list()
        for dp_worker_fetched_start in self.fetched_fused_inp_start[model_id]:
            remaining.extend(self.fused_inp_queues[model_id][dp_worker_fetched_start::old_dp_size])

        remaining = sorted(remaining, key=lambda inp: inp[0])
        self.fused_inp_queues[model_id] = remaining
        self.fetched_fused_inp_start[model_id] = [i for i in range(dp_size)]


    def reset_state_for_model(self, shared_id: int, dp_size: int):
        """
            When a model is restarted, we need to update some variables.
            Update: self.fused_inp_queues, self.fetched_fused_inp_start.
            NOTE: 
                1. do not need lock here, will add lock where this method is called.
                2. the input model_id can be of fused models or base models, but 
                ``fused_inp_queues`` and ``fetched_fused_inp_start`` are for base models only.
        """
        print(f"In reset_state_for_model, parameters: shared_id: {shared_id} dp_size: {dp_size}-----------------\n")


        base_model_ids = self.base_model_ids_dict[shared_id]
        for base_model_id in base_model_ids:
            self._reset_state_for_model(base_model_id, dp_size)


    def _process_outputs(self, model_id: int, seqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]):
        """
            Process the outputs, e.g., concat different ori seqs into new out seqs.
            May be called when horizontally fusing models, e.g., Map-reduce Summary.
        """
        print(f"in _process_outputs")

        print(f"in _process_outputs: self._meta_output_pool[model_id]: model_id: {model_id}, {self._meta_output_pool[model_id]}")

        ret = list()
        for req_id, req_content in seqs:
            new_out_id, ind = self.out_req_id_mapping[model_id][req_id]

            print(f"_process_outputs: req_id: {req_id}, model_id: {model_id}, new_out_id, ind: {new_out_id, ind}")

            self._meta_output_pool[model_id][new_out_id][1][ind] = req_content
            self._meta_output_pool[model_id][new_out_id][0] = self._meta_output_pool[model_id][new_out_id][0] + 1
            if self._meta_output_pool[model_id][new_out_id][0] == len(self._meta_output_pool[model_id][new_out_id][1]):

                new_out_content = self.fuse_inp_srcs(self._meta_output_pool[model_id][new_out_id][1])
                ret.append((new_out_id, new_out_content))


        for v in self._meta_output_pool[model_id].values():
            print(f"v[0]: {v[0]}")
        print(f"ret: {[_[0] for _ in ret]}")
        print(f"in _process_outputs: self._meta_output_pool[model_id]: model_id: {model_id}, {self._meta_output_pool[model_id]}")

        return ret


    def add_seqs_base_model(self, model_id: int, seqs: Union[List[Tuple[int, str, int]],List[Tuple[int, List[int], int]]]):
        """
            Send newly generated outputs to the model communicator.
            Request IDs associate outputs with downstream inputs. A downstream input
            is complete only after all of its required upstream outputs are available.

            INPUT:
                seqs: list of (req_id, req_content); req_content may be generated text or generated token ids.
            Modify:
                self.output_pool, self._ungened_out_req_nums
            NOTE: 
                1. ensure the input ``seqs`` only contains seqs which have not been added before.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
            NOTE:
                1. this function is only for base models.
        """


        seqlens = {req_id: seqlen for req_id, seq, seqlen in seqs}
        seqs = [(req_id, seq) for req_id, seq, seqlen in seqs]


        to_add = seqs


        with self._get_correct_lock(model_id, self._add_req_locks, read_only=False):

            if model_id in self.new_out_req_part_num:
                to_add = self._process_outputs(model_id=model_id, seqs=seqs)


            self.output_pool[model_id].extend(to_add)

            self._ungened_out_req_nums[model_id] = self._ungened_out_req_nums[model_id] - len(to_add)


            self.outlen_dict[model_id].update(seqlens)

            print(f"ADDING SEQS of model {model_id} TO THE POOL:")


    def add_seqs(self, shared_id: int, seqs: Union[List[Tuple[int, str, int]],List[Tuple[int, List[int], int]]]):
        """
            Send newly generated outputs to the model communicator.
            Request IDs associate outputs with downstream inputs. A downstream input
            is complete only after all of its required upstream outputs are available.

            INPUT:
                seqs: list of (req_id, req_content, req_content_len); req_content may be generated text or generated token ids.
            Modify:
                self.output_pool, self._ungened_out_req_nums
            NOTE: 
                1. ensure the input ``seqs`` only contains seqs which have not been added before.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
            NOTE:
                1. shared_id can be negative values for dummy input nodes, or shared ids for nomal models. 
        """
        try:
            if (shared_id not in self.base_model_ids_dict):
                assert shared_id < 0
                self.add_seqs_base_model(model_id=shared_id, seqs=seqs)
            elif len(self.base_model_ids_dict[shared_id]) == 1:
                self.add_seqs_base_model(model_id=self.base_model_ids_dict[shared_id][0], seqs=seqs)
            else:

                base_model_seqs_dict = defaultdict(list)
                for seq in seqs:
                    req_id = seq[0]
                    base_model_id = self.out_req_model_id_mapping[shared_id][req_id]
                    base_model_seqs_dict[base_model_id].append(seq)


                for base_model_id, base_model_seqs in base_model_seqs_dict.items():
                    self.add_seqs_base_model(model_id=base_model_id, seqs=base_model_seqs)

        except Exception as e:
            print(f"Exception in add_seqs: {e}")
            print(traceback.format_exc())
            assert False


    def _get_selected_outputs(
            self, from_model_id: int, to_model_id: int,
    ):
        """
            Try to fetch available outputs from the model whose id is ``model_id``, 
            where the ids of the outputs are in req_ids_requested.
            NOTE: 1. assume the corresponding lock has been obtained.
        """
        end = len(self.output_pool[from_model_id])
        start = self._read_output_nums[(from_model_id, to_model_id)]
        self._read_output_nums[(from_model_id, to_model_id)] = end


        if (to_model_id in self.inp_req_from_which_model_which_out_reqs_reversed) \
            and (from_model_id in self.inp_req_from_which_model_which_out_reqs_reversed[to_model_id]):
            out_req_ids = self.inp_req_from_which_model_which_out_reqs_reversed[to_model_id][from_model_id]
            _ret = [req for req in self.output_pool[from_model_id][start:end] if req[0] in out_req_ids]
            ret = [(inp_req_id, req[1]) for req in _ret for inp_req_id in out_req_ids[req[0]]]
            return ret
        else:

            return self.output_pool[from_model_id][start:end]


    def _get_fused_inp_queues_limited_version(self, to_model_id: int):
        """
            Update self.fused_inp_queues based on the new outputs from inp models of the to_model.
            Modify:
                self._read_output_nums, self._available_srcs, self.fused_inp_queues, self._unavailable_req_nums
            NOTE:
                1. need lock here, as there may be multiple dp workers: one lock for one model.
            NOTE:
                1. this version does not support model horizontal fusion.
        """

        if self._unavailable_req_nums[to_model_id] == 0:

            return


        inps = self.in_edge_dict_with_dummy_inp_nodes[to_model_id]
        if len(inps) == 1:

            with self._get_correct_lock(inps[0], self._add_req_locks, read_only=True):
                end = len(self.output_pool[inps[0]])
                start = self._read_output_nums[(inps[0], to_model_id)]
                reqs = self.output_pool[inps[0]][start:end]

                reqs = sorted(reqs, key=lambda x: x[0])
            self.fused_inp_queues[to_model_id].extend(reqs)
            self._read_output_nums[(inps[0], to_model_id)] = end
            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)
        else:

            new_complete_req_ids = list()
            inp_src_num = len(inps)
            for inp_i, from_model_id in enumerate(inps):


                with self._get_correct_lock(from_model_id, self._add_req_locks, read_only=True):

                    end = len(self.output_pool[from_model_id])
                    start = self._read_output_nums[(from_model_id, to_model_id)]
                    for req in self.output_pool[from_model_id][start:end]:
                        if req[0] not in self._available_srcs[to_model_id]:
                            self._available_srcs[to_model_id][req[0]] = [0, [None for _ in range(inp_src_num)]]
                        self._available_srcs[to_model_id][req[0]][0] += 1
                        self._available_srcs[to_model_id][req[0]][1][inp_i] = req[1]
                        if self._available_srcs[to_model_id][req[0]][0] == inp_src_num:
                            new_complete_req_ids.append(req[0])
                self._read_output_nums[(from_model_id, to_model_id)] = end
            new_complete_req_ids = sorted(new_complete_req_ids)
            reqs = [(req_id, self.fuse_inp_srcs(self._available_srcs[to_model_id][req_id][1])) for req_id in new_complete_req_ids]
            self.fused_inp_queues[to_model_id].extend(reqs)
            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)


    def _get_fused_inp_queues(self, to_model_id: int):
        """
            Update self.fused_inp_queues based on the new outputs from inp models of the to_model.
            Modify:
                self._read_output_nums, self._available_srcs, self.fused_inp_queues, self._unavailable_req_nums
            NOTE:
                1. need lock here, as there may be multiple dp workers: one lock for one model.
                2. this version support model horizontal fusion, i.e., the reqs from different models may not be fused together.
        """


        if self._unavailable_req_nums[to_model_id] == 0:

            return


        inps = self.in_edge_dict_with_dummy_inp_nodes[to_model_id]


        if (len(inps) == 1) or (self._independent_srcs[to_model_id]):


            reqs = list()
            for from_model_id in inps:
                with self._get_correct_lock(from_model_id, self._add_req_locks, read_only=True):
                    reqs.extend(
                        self._get_selected_outputs(from_model_id=from_model_id, to_model_id=to_model_id) )


            reqs = sorted(reqs, key=lambda x: x[0])


            reqs = self.fuse_inp_srcs_based_on_template(reqs, to_model_id, one_src=True)

            self.fused_inp_queues[to_model_id].extend(reqs)


            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)
        else:

            new_complete_req_ids = list()
            inp_src_num = len(inps)
            for inp_i, from_model_id in enumerate(inps):


                with self._get_correct_lock(from_model_id, self._add_req_locks, read_only=True):


                    reqs = self._get_selected_outputs(from_model_id=from_model_id, to_model_id=to_model_id)


                    for req in reqs:
                        if req[0] not in self._available_srcs[to_model_id]:
                            self._available_srcs[to_model_id][req[0]] = [0, [None for _ in range(inp_src_num)]]
                        self._available_srcs[to_model_id][req[0]][0] += 1
                        self._available_srcs[to_model_id][req[0]][1][inp_i] = req[1]
                        if self._available_srcs[to_model_id][req[0]][0] == inp_src_num:
                            new_complete_req_ids.append(req[0])


            new_complete_req_ids = sorted(new_complete_req_ids)


            reqs = [(req_id, self._available_srcs[to_model_id][req_id][1]) for req_id in new_complete_req_ids]
            reqs = self.fuse_inp_srcs_based_on_template(reqs, to_model_id, one_src=False)

            self.fused_inp_queues[to_model_id].extend(reqs)


            self._unavailable_req_nums[to_model_id] = self._unavailable_req_nums[to_model_id] - len(reqs)


    def _get_seqs(self, to_model_id: int, dp_id: int, dp_size: int
        ) -> Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]:
        """
            Get the available inp reqs for a specific dp worker.
            Update:
                self.fetched_fused_inp_start
        """


        start = self.fetched_fused_inp_start[to_model_id][dp_id]
        reqs = self.fused_inp_queues[to_model_id]
        ret = reqs[start::dp_size]
        end = start + len(ret)*dp_size
        self.fetched_fused_inp_start[to_model_id][dp_id] = end

        return ret


    def _update_out_req_model_id_mapping(
            self, shared_id: int, base_model_id: int,
            reqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]],
    ) -> None:


        if len(self.base_model_ids_dict[shared_id]) == 1:

            return

        new_mapping = {req[0]:base_model_id for req in reqs}
        self.out_req_model_id_mapping[shared_id].update(new_mapping)


    def get_seqs_base_model(self, to_shared_id: int, to_model_id: int, dp_id: int, dp_size: int
        ) -> Tuple[Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]], bool]:
        """
            Get the available inp reqs for a specific dp worker.
            Update:
                self.fetched_fused_inp_start, self.fused_inp_queues
            Output:
                1. ret: the reqs assigned to the dp worker
                2. possible_to_get_future_reqs: whether it is possible for the dp worker to get more new available reqs in the future
                    True if there are/will be remaining available inp reqs for this model.
            NOTE:
                1. When the dp worker cannot get available reqs, reorganize the fused_inp_queue and assign reqs to the worker again.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
                3. ``to_model_id`` is the id of a base model.
            NOTE:
                1. this function is for a base model.
                2. to_model_id: the base model id.
        """


        with self._get_correct_lock(to_model_id, self._fetch_req_locks, read_only=False):


            self._get_fused_inp_queues(to_model_id)


            ret = self._get_seqs(to_model_id, dp_id, dp_size)


            possible_to_get_future_reqs: bool = True

            if len(ret) == 0:


                self._reset_state_for_model(to_model_id, dp_size)


                start = self.fetched_fused_inp_start[to_model_id][dp_id]


                if start >= len(self.fused_inp_queues[to_model_id]):


                    if len(self.fused_inp_queues[to_model_id])>0:
                        ret = [self.fused_inp_queues[to_model_id][0]]
                        self.fused_inp_queues[to_model_id] = self.fused_inp_queues[to_model_id][1:]

                        self._update_out_req_model_id_mapping(to_shared_id, base_model_id=to_model_id, reqs=ret)


                        return ret, possible_to_get_future_reqs
                    else:
                        if self._unavailable_req_nums[to_model_id] == 0:


                            possible_to_get_future_reqs = False

                        assert self._unavailable_req_nums[to_model_id] >= 0, f"assert self._unavailable_req_nums[to_model_id] >= 0 wrong: {self._unavailable_req_nums[to_model_id]}"


                        return [], possible_to_get_future_reqs
                else:

                    ret = self._get_seqs(to_model_id, dp_id, dp_size)


                    self._update_out_req_model_id_mapping(to_shared_id, base_model_id=to_model_id, reqs=ret)
                    return ret, possible_to_get_future_reqs

            else:


                self._update_out_req_model_id_mapping(to_shared_id, base_model_id=to_model_id, reqs=ret)


                return ret, possible_to_get_future_reqs


    def _apply_prompt_template(
            self,
            base_model_id: int,
            seqs: Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]
        ) -> Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]]:
        """
            This function applies prompt template to the fetched inp seqs.
            E.g., apply chat template
        """
        ret = seqs

        tokenizer = self.base_model_tokenizers[base_model_id]


        if tokenizer.chat_template != None:


            ret = [(req_i, tokenizer.apply_chat_template([{"role": "user", "content": ori_prompt}], add_generation_prompt=True, tokenize=False)) \
                      for req_i, ori_prompt in seqs]

        return ret


    def get_seqs(self, to_shared_id: int, dp_id: int, dp_size: int
        ) -> Tuple[Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]], bool, List[int]]:
        """
            Get the available inp reqs for a specific dp worker.
            Update:
                self.fetched_fused_inp_start, self.fused_inp_queues
            Output:
                1. ret: the reqs assigned to the dp worker
                2. possible_to_get_future_reqs: whether it is possible for the dp worker to get more new available reqs in the future
                    True if there are/will be remaining available inp reqs for this model.
                3. req_base_model_ids: the corresponding base model each req belongs to.
            NOTE:
                1. When the dp worker cannot get available reqs, reorganize the fused_inp_queue and assign reqs to the worker again.
                2. need lock here, as there may be multiple dp workers: one lock for one model.
                3. ``to_model_id`` is the id of a base model.
            NOTE:
                1. this function is for any model: fused or base models.
        """
        try:

            print(f"self.base_model_ids_dict: {self.base_model_ids_dict}")


            base_model_ids = self.base_model_ids_dict[to_shared_id]


            possible_to_get_future_reqs = False
            ret = list()
            req_base_model_ids = list()
            for base_model_id in base_model_ids:
                tmp_ret, tmp_possible_to_get_future_reqs = \
                    self.get_seqs_base_model(to_shared_id=to_shared_id, to_model_id=base_model_id, dp_id=dp_id, dp_size=dp_size)
                tmp_ret = self._apply_prompt_template(base_model_id, tmp_ret)
                ret.extend(tmp_ret)

                if tmp_possible_to_get_future_reqs:
                    possible_to_get_future_reqs = True


                if len(base_model_ids) > 1:

                    req_base_model_ids.extend([base_model_id]*len(tmp_ret))

            if len(base_model_ids) == 1:
                req_base_model_ids = [base_model_ids[0]]

            print(f"in communicator, get reqs: {[_[0] for _ in ret]}, req_base_model_ids: {req_base_model_ids}")

            return ret, possible_to_get_future_reqs, req_base_model_ids

        except Exception as e:
            print(f"Exception in llm.get_seqs: {e}")
            print(traceback.format_exc())
            assert False


    def get_info(self):
        return self.output_pool, self._read_output_nums, self._available_srcs, \
            self.fused_inp_queues, self.fetched_fused_inp_start, self.in_edge_dict_with_dummy_inp_nodes


class MyManager(BaseManager):
    pass

MyManager.register('Communicator', LLM_COMMUNICATOR)


class SHARED_CONTECT():
    shared_setting: Array = None
    shared_id: int = 0

    dp_id: int = 0


    events: List[Event] = None
    started_status: List[Event] = None
    gened_outputs = deque()
    remaining_requests = deque()
    shared_finish_status = [False]


    execution_plan_size: int = None

    execution_plan_non_gpu_para_num: int = 4


    communicator: LLM_COMMUNICATOR = None

    return_str: bool = False

    check_in_gap: int = int(1e9)

    check_out_gaps: Array = None
    tot_req_num_remained: int = None


    sampling_args_dict: Dict[int, SamplingParams] = None

    seq_outlen_dict: Dict[int, Dict[int, int]] = None


    wait_stage_synchs: List[bool] = None
    stage1_started: Event = None


    @classmethod
    def set_execution_plan_size(cls, tot_gpu_num):

        cls.execution_plan_size = tot_gpu_num + cls.execution_plan_non_gpu_para_num


    @classmethod
    def get_tp_size(cls):

        offset = cls.shared_id * cls.execution_plan_size
        tensor_parallel_size: int = cls.shared_setting[offset]
        return tensor_parallel_size

    @classmethod
    def get_gpu_mem_utilization(cls):

        offset = cls.shared_id * cls.execution_plan_size
        gpu_memory_utilization: float = cls.shared_setting[offset+1]/10
        return gpu_memory_utilization

    @classmethod
    def get_wldegree(cls):

        offset = cls.shared_id * cls.execution_plan_size
        weight_load_degree: int = cls.shared_setting[offset+2]
        return weight_load_degree

    @classmethod
    def get_dp_size(cls):

        offset = cls.shared_id * cls.execution_plan_size
        dp_size: int = cls.shared_setting[offset+3]
        return dp_size


    @classmethod
    def has_dp_parallel(cls):

        if cls.dp_id > 0:
            return True
        offset = cls.shared_id * cls.execution_plan_size
        dp_size: int = cls.shared_setting[offset+3]
        return dp_size > 1


    @classmethod
    def get_gpus(cls):

        offset = cls.shared_id * cls.execution_plan_size
        gpus = cls.shared_setting[offset+cls.execution_plan_non_gpu_para_num:offset+cls.execution_plan_size]
        return gpus


    @classmethod
    def get_gpus_used(cls):
        gpus = os.environ['TOT_ORDERED_GPUS'].split(',')
        gpus = gpus[:cls.get_tp_size()]
        return gpus


    @classmethod
    def get_sampling_args(cls, base_model_id: int, req_id: int)->SamplingParams:
        if len(cls.sampling_args_dict[base_model_id]) == 1:

            return list(cls.sampling_args_dict[base_model_id].values())[0]
        else:
            return cls.sampling_args_dict[base_model_id][req_id]


    @classmethod
    def prepare_for_reschedule(cls, outputs, remaining_requests, llm_engine, mark_finish:bool=True) -> None:


        start_wrapup = time.perf_counter()


        cls.gened_outputs.extend(outputs)


        cls.remaining_requests = remaining_requests

        print(f"event list status in prepare--model {cls.shared_id} 2", flush=True)


        exit_actor = (os.environ['SOFT_RESCHEDULE'] == 'False')
        llm_engine.delete_workers(exit_actor)

        print(f"event list status in prepare--model {cls.shared_id} 3", flush=True)


        if (os.environ['SOFT_RESCHEDULE'] == 'False') and (llm_engine.parallel_config.world_size > 1):

            print(f"dp worker id {cls.dp_id}, placement_group_table(): {placement_group_table()}")


            pg_info_name = 'my_pg'+os.environ['DP_WORKER_I']
            if os.environ['USE_VLLM']=='True':
                pg_info_name = 'vllm_pg'
            pg = get_placement_group(pg_info_name)
            remove_placement_group(pg)


        print(f"event list status in prepare--model {cls.shared_id} 4", flush=True)


        KVBlkPerLayerWeight.reset()

        print(f"event list status in prepare--model {cls.shared_id} 5", flush=True)


        if mark_finish:
            cls.events[cls.shared_id + 2].set()

        if cls.events != None:

            print(f"event list status in prepare--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")


        end_wrapup = time.perf_counter()
        print(f"total time to wrap up for reschedule: {end_wrapup-start_wrapup}s")


    @classmethod
    def update_execution_plan(cls, tensor_parallel_size, gpu_memory_utilization):

        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return tensor_parallel_size, gpu_memory_utilization


        tensor_parallel_size: int = cls.get_tp_size()
        gpu_memory_utilization: float = cls.get_gpu_mem_utilization()
        weight_load_degree: int = cls.get_wldegree()


        gpus: str = ','.join([str(i) for i in cls.get_gpus()])

        os.environ['WEIGHT_LOAD_DEGREE'] = str(weight_load_degree)


        os.environ['TOT_ORDERED_GPUS'] = gpus

        print(f"in update_execution_plan, shared_id: {cls.shared_id}, TOT_ORDERED_GPUS: {gpus}", flush=True)

        return tensor_parallel_size, gpu_memory_utilization


    @classmethod
    def set_execution_plan(cls, setting, shared_ids=list()) -> None:
        '''
            If model_ids is given, only change the setting for specific models
        '''
        iter_range = None
        if len(shared_ids) == 0:
            iter_range = range(len(cls.shared_setting))
        else:
            iter_range = list()
            for shared_id in shared_ids:
                iter_range.extend(range(cls.execution_plan_size * shared_id, cls.execution_plan_size * (shared_id+1)))


        for i, v in zip(iter_range, setting):
            cls.shared_setting[i] = v


    @classmethod
    def is_finished(cls) -> None:
        return cls.shared_finish_status[cls.shared_id]


    @classmethod
    def set_finished(cls, gened_output_num: int, set_event_state_anyway: bool = False) -> None:
        """
            Update the remaining request count and mark completion when it reaches zero.
            Rescheduling readiness is signaled separately; this method does not set events.
            set_event_state_anyway is retained for call compatibility.
        """
        print(f"in set_finished, model id: {cls.shared_id}, cls.tot_req_num_remained: {cls.tot_req_num_remained}, gened_output_num: {gened_output_num}, set_event_state_anyway: {set_event_state_anyway}")


        cls.tot_req_num_remained = cls.tot_req_num_remained - gened_output_num
        finished = (cls.tot_req_num_remained == 0)
        if finished:
            cls.shared_finish_status[cls.shared_id] = finished


        if os.environ['RUN_MULTI_MODEL'] == 'True':
            print(f"event list status in set finish--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")


    @classmethod
    def set_finish_preparation_before_init_LLM(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] == 'True':
            cls.events[cls.shared_id + 2].set()
            print(f"event list status in set finish before init LLM--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}")


    @classmethod
    def set_finish_preparation_for_reschedule(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] == 'True':
            cls.events[cls.shared_id + 2].set()
            print(f"event list status in set finish prepare for reschedule--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}, --abs: {time.perf_counter()}")


    @classmethod
    def wait_all_models_to_finish_prepare_for_reschedule(cls, shared_ids: List[int]) -> None:
        assert os.environ['RUN_MULTI_MODEL'] == 'True'


        print(f"event list status in waiting all to finish 1: {[event.is_set() for event in cls.events[2:]]}, --abs: {time.perf_counter()}")


        for shared_id in shared_ids:
            cls.events[2+shared_id].wait()

        print(f"event list status in waiting all to finish 2: {[event.is_set() for event in cls.events[2:]]}")


    @classmethod
    def wait_all_models_to_finish_preparation_before_init_LLM(cls, shared_ids: List[int]) -> None:
        '''
            Call this function in the multi-model scheduler process.
        '''
        assert os.environ['RUN_MULTI_MODEL'] == 'True'


        print(f"event list status in waiting all to finish 1: {[event.is_set() for event in cls.events[2:]]}")


        for shared_id in shared_ids:
            cls.events[2+shared_id].wait()

        print(f"event list status in waiting all to finish 2: {[event.is_set() for event in cls.events[2:]]}")


    @classmethod
    def start_specific_models(cls, shared_ids) -> None:
        '''
            (1) set the started_status;
            (2) clear the finishing_preparation_for_reschedule event;
        '''

        assert os.environ['RUN_MULTI_MODEL'] == 'True'

        for shared_id in shared_ids:
            cls.started_status[shared_id].set()
            cls.events[2+shared_id].clear()

        print(f"start event list status in start_specific_models--model {cls.shared_id}: {[event.is_set() for event in cls.started_status]}")
        print(f"event list status in start_specific_models--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}, --abs: {time.perf_counter()}")


    @classmethod
    def is_started(cls) -> bool:
        return cls.started_status[cls.shared_id].is_set()


    @classmethod
    def wait_to_be_started(cls) -> None:


        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return
        cls.started_status[cls.shared_id].wait()

        print(f"event list status in wait_to_be_started--model {cls.shared_id}: {[event.is_set() for event in cls.started_status]}, --abs: {time.perf_counter()}")


    @classmethod
    def query_finish_status(cls, shared_id) -> None:
        assert os.environ['RUN_MULTI_MODEL'] == 'True'

        return cls.shared_finish_status[shared_id]


    @classmethod
    def should_reschedule(cls) -> bool:
        '''
        check whether this model should stop inference and rescheduled
        Call this function after every inference iteration.
        '''
        if os.environ['RUN_MULTI_MODEL'] != 'True':
            return False


        return not cls.started_status[cls.shared_id].is_set()


    @classmethod
    def stop_specific_models(cls, shared_ids: List[int]) -> bool:
        '''
        Call this function in the model scheduler process.
        '''
        for shared_id in shared_ids:
            cls.started_status[shared_id].clear()

        print(f"start event list status in stop_specific_models--model {cls.shared_id}: {[event.is_set() for event in cls.started_status]}")
        print(f"event list status in stop_specific_models--model {cls.shared_id}: {[event.is_set() for event in cls.events[2:]]}, --abs: {time.perf_counter()}")


    @classmethod
    def wait_for_stage1_to_start(cls):
        if (cls.wait_stage_synchs!=None) and (cls.wait_stage_synchs[cls.shared_id]):
            cls.stage1_started.wait()


    @classmethod
    def start_stage1_if_possible(cls):
        if cls.wait_stage_synchs != None:
            for shared_id in range(len(cls.wait_stage_synchs)):
                if (not cls.wait_stage_synchs[shared_id]) and (not cls.query_finish_status(shared_id)):

                    return
            cls.stage1_started.set()


    @classmethod
    def init_shared_vars(cls, model_setting: Array) -> None:
        cls.shared_setting = model_setting


    @classmethod
    def increment(cls) -> None:
        with cls.shared_setting.get_lock():
            cls.shared_setting[cls.shared_id] += 1

    @classmethod
    def test_and_print(cls) -> None:
        if os.environ['RUN_MULTI_MODEL'] == 'True':
            print(cls.shared_setting[cls.shared_id])
            cls.increment()


    @classmethod
    def test_task(cls, identifier=-1):


        if identifier == -1:
            identifier = cls.shared_id
        print(f'Task {identifier} waiting...', flush=True)
        cls.events[cls.shared_id].wait()

        value = 1


        print(f'Task {identifier} completed with {value}', flush=True)
