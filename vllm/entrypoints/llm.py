from typing import List, Optional, Union, Tuple

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter


from vllm.core.multimodel_scheduler import SHARED_CONTECT


PRINT_LOGS=True


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq" and "squeezellm". If None, we first check
            the `quantization_config` attribute in the model config file. If
            that is None, we assume the model weights are not quantized and use
            `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()


    @classmethod
    def get_engine_configs_only(cls,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ):
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        (model_config, cache_config, parallel_config, scheduler_config,
                device_config, lora_config) = engine_args.create_engine_configs()
        return (model_config, cache_config, parallel_config, scheduler_config,
                device_config, lora_config)


    def update_llm_engine(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine.update_from_engine_args(engine_args)


    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        prefix_pos: Optional[Union[int, List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            prefix_pos: If not None, we use the given position as the prefix
                position for each prompt. We will cache the prefix's KV
                cache and reuse it for the next request with the same prefix.
                This is an experimental feature, and may be replaced with
                automatic prefix caching in the future.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):

            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:

            sampling_params = SamplingParams()


        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            prefix_pos_i = prefix_pos[i] if prefix_pos is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt,
                              sampling_params,
                              token_ids,
                              lora_request=lora_request,
                              prefix_pos=prefix_pos_i)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        lora_request: Optional[LoRARequest] = None,
        prefix_pos: Optional[int] = None,

        request_id: Optional[int] = None
    ) -> None:

        if request_id == None:
            request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    prefix_pos=prefix_pos)


    def _add_new_available_reqs(
        self,
        new_inps:Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]],

        sort_inps,
        req_base_model_ids: List[int]) -> None:
        """
            Check whether there are newly available input reqs.
            Update: the waiting list of reqs.
            INPUT: 
                new_inps: list of (req_id, req_content)
                sampling_parameters is a dict containing values for [n, temperature, top_p, use_beam_search, ignore_eos, max_tokens]
                sort_inps: if True, we will sort the inp seqs by their lengths every time.
                req_base_model_ids: the corresponding base model each req belongs to.
                    if this model is a base model, req_base_model_ids=[base_model_id]; else, len(req_base_model_ids)=len(new_inps)

        """
        if len(new_inps) == 0:

            return


        if sort_inps:
            if len(req_base_model_ids) == len(new_inps):

                order = sorted(range(len(new_inps)), key=lambda i: len(new_inps[i][1]), reverse=True)
                new_inps = [new_inps[i] for i in order]
                req_base_model_ids = [req_base_model_ids[i] for i in order]
            else:
                new_inps = sorted(new_inps, key=lambda inp: len(inp[1]), reverse=True)
                assert len(req_base_model_ids) == 1
                req_base_model_ids = req_base_model_ids * len(new_inps)


        if isinstance(new_inps[0][1], str):
            for (req_id, prompt), base_model_id in zip(new_inps, req_base_model_ids):
                sampling_params = SHARED_CONTECT.get_sampling_args(base_model_id=base_model_id, req_id=req_id)
                self._add_request(
                    prompt=prompt,
                    prompt_token_ids=None,
                    sampling_params=sampling_params,
                    request_id=req_id,
                )
        else:
            for (req_id, prompt_token_ids), base_model_id in zip(new_inps, req_base_model_ids):
                sampling_params = SHARED_CONTECT.get_sampling_args(base_model_id=base_model_id, req_id=req_id)
                self._add_request(
                    prompt=None,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                    request_id=req_id,
                )


    def _get_outputs_to_system_communicator(
            self,
            outputs: List[RequestOutput], output_num_sent_out:int, return_str:bool
        ) -> Tuple[int, Union[List[Tuple[int, str, int]],List[Tuple[int, List[int], int]]]]:
        """
            Update ``output_num_sent_out`` and return the outputs sent to the system communicator.
        """

        new_outputs = outputs[output_num_sent_out:]
        output_num_sent_out = len(outputs)
        if return_str:
            return output_num_sent_out, [(output.request_id, output.outputs[0].text, len(output.outputs[0].token_ids)) for output in new_outputs]
        else:
            return output_num_sent_out, [(output.request_id, output.outputs[0].token_ids, len(output.outputs[0].token_ids)) for output in new_outputs]


    def _correct_seq_ids(self):
        """
            This function is called in the beginning. 
            Useful when the last round of inference has data parallelism and there are remaining requests left to this round.
        """
        for req in self.llm_engine.scheduler.waiting:
            new_seqs_dict = dict()
            for seq in req.seqs_dict.values():
                new_seq_id = next(self.llm_engine.seq_counter)
                seq.seq_id = new_seq_id
                new_seqs_dict[new_seq_id] = seq
            req.seqs_dict = new_seqs_dict


    def _run_engine(
            self, use_tqdm: bool,


            sampling_parameters=None) -> List[RequestOutput]:
        """
            We add some input parameters to support constructing new requests in multi-level LLM systems.
        """
        import os


        if os.environ['RUN_MULTI_MODEL'] == 'True':
            self._correct_seq_ids()


        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")

        outputs: List[RequestOutput] = []


        import time


        non_comp_ranges = [list(), list()]


        import torch
        step_i = 0


        sort_inps = os.getenv("MY_SORT_INPS", "False") == 'True'

        output_num_sent_out = 0


        possible_to_get_future_reqs: bool = True

        while True:


            if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
                possible_to_get_future_reqs and (len(self.llm_engine.scheduler.waiting)+len(self.llm_engine.scheduler.swapped) == 0):


                enter_while_loop: bool = False


                has_unfinished_requests = self.llm_engine.has_unfinished_requests()
                while not (SHARED_CONTECT.should_reschedule() and (os.environ['NO_PREEMPT'] == 'False')):


                    if has_unfinished_requests and (step_i % SHARED_CONTECT.check_in_gap != 0):

                        break

                    if not enter_while_loop:

                        non_comp_ranges[0].append(time.perf_counter())
                        enter_while_loop = True

                    new_inps, possible_to_get_future_reqs, req_base_model_ids = \
                        SHARED_CONTECT.communicator.get_seqs(SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id, SHARED_CONTECT.get_dp_size())


                    print(f"GET SEQS for model shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}  FROM THE POOL: len(new_inps): {len(new_inps)} seq ids: {[_[0] for _ in new_inps]}")


                    self._add_new_available_reqs(new_inps, sort_inps, req_base_model_ids)


                    if not possible_to_get_future_reqs:

                        break

                    if (len(new_inps) == 0) and (not has_unfinished_requests):


                        time.sleep(10)
                    else:
                        break


                if enter_while_loop:
                    non_comp_ranges[1].append(time.perf_counter())


            if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
                (not possible_to_get_future_reqs) and (not self.llm_engine.has_unfinished_requests()):


                break
            elif (os.environ['RUN_MULTI_MODEL'] == 'False') and (not self.llm_engine.has_unfinished_requests()):
                break


            step_i+=1
            global PRINT_LOGS
            if PRINT_LOGS:
                print(f"model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}")
                print(f"step i: {step_i}", flush=True)


            if (step_i == 300):

                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStart()
            elif (step_i == 310):


                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStop()


            if SHARED_CONTECT.should_reschedule() and (os.environ['NO_PREEMPT'] == 'False'):


                print(f"THIS MODEL IS STOPPED!  model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}")


                break


            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)

                    if PRINT_LOGS:
                        print(f"finish one req: model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}, output.request_id: {output.request_id}")


                    if use_tqdm:
                        pbar.update(1)


            if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
                ((step_i % SHARED_CONTECT.check_out_gaps[SHARED_CONTECT.shared_id] == 0) or (not self.llm_engine.has_unfinished_requests())):


                output_num_sent_out, new_outputs = \
                    self._get_outputs_to_system_communicator(outputs, output_num_sent_out, SHARED_CONTECT.return_str)


                print(f"writing results back!  output_num_sent_out: {output_num_sent_out}, len(new_outputs): {len(new_outputs)}------------------")

                SHARED_CONTECT.communicator.add_seqs(SHARED_CONTECT.shared_id, new_outputs)


        if use_tqdm:
            pbar.close()

        print(f"total steps: {step_i}")

        from vllm._C import cache_ops
        if os.environ['USE_VLLM']!='True':


            self.llm_engine.disable_P2P_access()


        if (os.environ['RUN_MULTI_MODEL'] == 'True'):
            output_num_sent_out, new_outputs = \
                self._get_outputs_to_system_communicator(outputs, output_num_sent_out, SHARED_CONTECT.return_str)
            SHARED_CONTECT.communicator.add_seqs(SHARED_CONTECT.shared_id, new_outputs)

            print(f"writing results back!  output_num_sent_out: {output_num_sent_out}, len(new_outputs): {len(new_outputs)}------------------")


        if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
            (not SHARED_CONTECT.has_dp_parallel()):

            remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            SHARED_CONTECT.set_finished(len(outputs))
            SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine, mark_finish=False)


        elif (os.environ['RUN_MULTI_MODEL'] == 'True'):


            remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine, mark_finish=False)


        print(f"model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id} SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)


        if (os.environ['RUN_MULTI_MODEL'] == 'True'):
            SHARED_CONTECT.communicator.update_non_compute_ranges(
                non_comp_ranges, SHARED_CONTECT.shared_id, SHARED_CONTECT.get_gpus_used())


        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs


    def _run_engine_for_debug(self, use_tqdm: bool) -> List[RequestOutput]:

        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")

        outputs: List[RequestOutput] = []


        import os


        from vllm.core.multimodel_scheduler import SHARED_CONTECT


        import torch
        step_i = 0


        while self.llm_engine.has_unfinished_requests():


            step_i+=1
            print(f"step i: {step_i}", flush=True)


            if (step_i == 1) and (self.llm_engine.parallel_config.world_size==4):

                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStart()
            elif (step_i == 3) and (self.llm_engine.parallel_config.world_size==4):


                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStop()


            if (step_i>3):


                break


            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()

        print(f"total steps: {step_i}")

        from vllm._C import cache_ops
        if os.environ['USE_VLLM']!='True':


            self.llm_engine.disable_P2P_access()


        SHARED_CONTECT.set_finished(not self.llm_engine.scheduler.has_unfinished_seqs())
        if not SHARED_CONTECT.is_finished():
            remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine)


        print(f"SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)


        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs


    def _is_redundant_iter_workload(
            self, record_dict,
            set_max_num_seqs, is_prompt, corr_context_tot_len,
            last_seqnum, last_is_prompt, last_corr_context_tot_len,
            weight_loading_time):


        threshold = 1e-3


        if (last_seqnum, last_is_prompt) != (set_max_num_seqs, is_prompt):
            return False


        if corr_context_tot_len == last_corr_context_tot_len:
            return False

        related_records = record_dict[(set_max_num_seqs, is_prompt)]


        if (weight_loading_time[0] == None) and (len(related_records) == 2):
            weight_loading_time[0] = min(related_records.values())


        if len(related_records) == 2:
            if max(related_records.values()) <= (min(related_records.values()) + threshold):
                return True


        if len(related_records) == 3:
            if min(related_records.values()) > (weight_loading_time[0]+threshold):
                return True


        if len(related_records) <= 3:
            return False


        xs = sorted(related_records.keys())
        if False not in [related_records[x] > (related_records[xs[0]]+threshold) for x in xs[-3:-1]]:
            return True

        return False


    def _update_record_dict(
            self, my_throughput_logger, record_dict,
            set_max_num_seqs, is_prompt, corr_context_tot_len):
        last_exec_latency = my_throughput_logger.get_last_record_execLatency()
        related_records = record_dict[(set_max_num_seqs, is_prompt)]
        if corr_context_tot_len not in related_records:
            related_records[corr_context_tot_len] = last_exec_latency
        else:
            if last_exec_latency < related_records[corr_context_tot_len]:
                related_records[corr_context_tot_len] = last_exec_latency


    def _profile_per_iter_latency(
            self, sampling_params_dict, iter_workloads,
            set_seqlens_list: List[Tuple[bool, List[int]]] = list(),
            piecewise_cost_model_build_mode = False) -> None:
        '''
            Input:
                iter_workloads: dict of lists 
                    i.e., {is_prompt: list(), set_max_num_batched_tokens: list(), set_max_num_seqs: list()}
                piecewise_cost_model_build_mode: 
                    if True, some of the given iter_workloads are redundant and will not be profiled based 
                    on the actual per-iter cost distribution.
        '''

        import os
        from collections import defaultdict
        my_throughput_logger = self.llm_engine.driver_worker.model_runner.my_throughput_logger
        record_dict = defaultdict(dict)
        weight_loading_time = [None]


        self.llm_engine._profile_per_iter_latency(
            is_prompt=True, sampling_params_dict=sampling_params_dict,
            set_max_num_batched_tokens=2048, set_max_num_seqs=2)

        test_iter_num = len(iter_workloads['is_prompt'])
        last_seqnum, last_is_prompt, last_corr_context_tot_len = None, None, None
        for i in range(test_iter_num):
            is_prompt = iter_workloads['is_prompt'][i]
            set_max_num_batched_tokens = iter_workloads['set_max_num_batched_tokens'][i]
            set_max_num_seqs = iter_workloads['set_max_num_seqs'][i]
            corr_context_tot_len = iter_workloads['corr_context_tot_len'][i]


            if piecewise_cost_model_build_mode:
                if self._is_redundant_iter_workload(
                    record_dict, set_max_num_seqs, is_prompt, corr_context_tot_len,
                    last_seqnum, last_is_prompt, last_corr_context_tot_len,
                    weight_loading_time):
                    continue

            self.llm_engine._profile_per_iter_latency(
                is_prompt=is_prompt,
                sampling_params_dict=sampling_params_dict,
                set_max_num_batched_tokens=set_max_num_batched_tokens,
                set_max_num_seqs=set_max_num_seqs)


            self._update_record_dict(
                my_throughput_logger, record_dict, set_max_num_seqs, is_prompt, corr_context_tot_len)
            last_seqnum, last_is_prompt, last_corr_context_tot_len = set_max_num_seqs, is_prompt, corr_context_tot_len


        for is_prompt, set_seqlens in set_seqlens_list:
            self.llm_engine._profile_per_iter_latency(
                is_prompt=is_prompt,
                sampling_params_dict=sampling_params_dict,
                set_seqlens=set_seqlens)


        from vllm._C import cache_ops
        if os.environ['USE_VLLM']!='True':


            self.llm_engine.disable_P2P_access()


        return
