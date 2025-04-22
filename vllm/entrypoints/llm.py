from typing import List, Optional, Union, Tuple

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.lora.request import LoRARequest
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter


# <> support multimodel scheduling
from vllm.core.multimodel_scheduler import SHARED_CONTECT


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




    # <>
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
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
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
        # <> support model-level pipeline
        request_id: Optional[int] = None 
    ) -> None:
        # request_id = str(next(self.request_counter))
        if request_id == None:
            request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    lora_request=lora_request,
                                    prefix_pos=prefix_pos)



    
    # <> support model-level pipeline
    def _add_new_available_reqs(
        self, 
        new_inps:Union[List[Tuple[int, str]],List[Tuple[int, List[int]]]], 
        # sampling_parameters, 
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
            # there is no new input
            return

        # if sampling_parameters['ignore_eos']:
        #     assert sampling_parameters['max_tokens'] != None
        # else:
        #     # we can set the max token as the max_model_len because it is also a stop condition
        #     sampling_parameters['max_tokens'] = self.llm_engine.model_config.max_model_len

        # sampling_params = SamplingParams(**sampling_parameters)

        if sort_inps:
            if len(req_base_model_ids) == len(new_inps):
                # this model is a fused model, or len(new_inps) == 1
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




    # <> support data parallel + model-level pipeline
    def _run_engine(
            self, use_tqdm: bool, 
            # <> add a new parameter to support constructing 
            # new requests in multi-level LLM systems.
            sampling_parameters=None) -> List[RequestOutput]:
        """
            We add some input parameters to support constructing new requests in multi-level LLM systems.
        """
        import os

        # print(f"1 INIT REQ iDS: shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id} #: {self.llm_engine.get_num_unfinished_requests()}, req ids: {[(_.request_id, list(_.seqs_dict.keys())) for list_ in [self.llm_engine.scheduler.waiting, self.llm_engine.scheduler.swapped, self.llm_engine.scheduler.running] for _ in list_]}")
        # print(f"1 INIT REQ iDS: shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}, GPU block #: {self.llm_engine.scheduler.block_manager.num_total_gpu_blocks}, block_manager: {list(self.llm_engine.scheduler.block_manager.block_tables.keys())}")

        # correct the seqs ids in case we get remaining seqs from last round inference.
        if os.environ['RUN_MULTI_MODEL'] == 'True':
            self._correct_seq_ids()


        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []

        # <> init KV_blk_per_layer_weights


        # <> support multimodel scheduling
        # from vllm.core.multimodel_scheduler import SHARED_CONTECT
        import time


        # store the computation time ranges [stores the absolute timestamps]
        non_comp_ranges = [list(), list()]


        # <> For Profiling--------------------
        import torch
        step_i = 0
        # -------------------------------------------


        # <> support model-level pipeline
        # check_gap = int(os.environ['MY_CHECK_GAP'])
        # sort_inps = os.environ['MY_SORT_INPS'] == 'True'
        sort_inps = os.getenv("MY_SORT_INPS", "False") == 'True'
        # the number of outputs we have send to the system communicator
        output_num_sent_out = 0

        

        # while self.llm_engine.has_unfinished_requests():
        # <>
        possible_to_get_future_reqs: bool = True
        # while SHARED_CONTECT.tot_req_num_remained > len(outputs):
        while True:

            # <> support model-level pipeline
            # # we need to check whether there are new available seqs every check_in_gap steps
            # if step_i % SHARED_CONTECT.check_in_gap == 0:
            #     new_inps = SHARED_CONTECT.communicator.get_seqs(SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id, SHARED_CONTECT.get_dp_size())
            #     self._add_new_available_reqs(new_inps, sampling_parameters, sort_inps)


            # <> change the condition to check new seqs: 
            # (1) is the first iter or (2) it is possible to get future reqs and has no pending (waiting+swapped) reqs
            # if (step_i == 0) or \
            #     (SHARED_CONTECT.check_in and (len(self.llm_engine.scheduler.waiting)+len(self.llm_engine.scheduler.swapped) == 0)):
            # <> change the condition to check new seqs: 
            # (1) it is possible to get future reqs and (2) has no pending (waiting+swapped) reqs

            # print(f"{SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id}: len(self.llm_engine.scheduler.running): {len(self.llm_engine.scheduler.running)}")

            if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
                possible_to_get_future_reqs and (len(self.llm_engine.scheduler.waiting)+len(self.llm_engine.scheduler.swapped) == 0):

                enter_while_loop: bool = False

                # print(f"try to get new reqs: {SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id}")

                # NOTE: when all the requests are finished, we must (1) wait for new requests to continue the inference 
                # or (2) break the loop when there is a stop signal
                has_unfinished_requests = self.llm_engine.has_unfinished_requests()
                while not (SHARED_CONTECT.should_reschedule() and (os.environ['NO_PREEMPT'] == 'False')):

                    # print(f"try to get new reqs 111: {SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id}")

                    if has_unfinished_requests and (step_i % SHARED_CONTECT.check_in_gap != 0):
                        # when there are running reqs, we check every check_in_gap steps
                        break

                    if not enter_while_loop:
                        # start of non-compute time
                        non_comp_ranges[0].append(time.perf_counter())
                        enter_while_loop = True

                    new_inps, possible_to_get_future_reqs, req_base_model_ids = \
                        SHARED_CONTECT.communicator.get_seqs(SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id, SHARED_CONTECT.get_dp_size())

                    # print(f"try to get new reqs 222: {SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id}, new_inps: {new_inps}, possible_to_get_future_reqs: {possible_to_get_future_reqs}")
                    print(f"GET SEQS for model shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}  FROM THE POOL: len(new_inps): {len(new_inps)} seq ids: {[_[0] for _ in new_inps]}")
                    # for _ in new_inps:
                    #     print(_[0], _[1][:10])
                    # with open(f"./test_end2end_schedule/model_IO.log", 'a') as file:
                    #     for _ in new_inps:
                    #         file.write(f"Get: shared id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}: {str(_)}\n")

                    self._add_new_available_reqs(new_inps, sort_inps, req_base_model_ids)

                    # print(f"2 INIT REQ iDS: shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id} #: {self.llm_engine.get_num_unfinished_requests()}, req ids: {[(_.request_id, list(_.seqs_dict.keys())) for list_ in [self.llm_engine.scheduler.waiting, self.llm_engine.scheduler.swapped, self.llm_engine.scheduler.running] for _ in list_]}")
                    # print(f"2 INIT REQ iDS: shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}, GPU block #: {self.llm_engine.scheduler.block_manager.num_total_gpu_blocks}, block_manager: {list(self.llm_engine.scheduler.block_manager.block_tables.keys())}")


                    # print(f"try to get new reqs 333: {SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id}, self.llm_engine.scheduler.waiting: {self.llm_engine.scheduler.waiting}")
                    
                    if not possible_to_get_future_reqs:
                        # print(f"Break after get seqs: {SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id} possible_to_get_future_reqs: {possible_to_get_future_reqs}")
                        break
                    
                    if (len(new_inps) == 0) and (not has_unfinished_requests):
                        # we query the communicator every 1 second
                        # print(f"Sleep after get seqs: {SHARED_CONTECT.shared_id, SHARED_CONTECT.dp_id} len(new_inps): {len(new_inps)}, has_unfinished_requests: {has_unfinished_requests}")
                        # time.sleep(1)
                        # TODO: <> FOR DEBUG
                        time.sleep(10)
                    else:
                        break
            
                # end of non-compute time
                if enter_while_loop:
                    non_comp_ranges[1].append(time.perf_counter())




            # check whether we need to stop the inference process
            if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
                (not possible_to_get_future_reqs) and (not self.llm_engine.has_unfinished_requests()):
                # (1) not possible to get reqs in the future and (2) all assigned reqs have been finished
                # print(f"All requests have been finished!")
                # print(f"possible_to_get_future_reqs: {possible_to_get_future_reqs}")
                # print(f"self.llm_engine.has_unfinished_requests(): {self.llm_engine.has_unfinished_requests()}")
                break
            elif (os.environ['RUN_MULTI_MODEL'] == 'False') and (not self.llm_engine.has_unfinished_requests()):
                break 


            # <> For Profiling-----------------
            step_i+=1
            print(f"model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}")
            print(f"step i: {step_i}", flush=True)
            # print(f"INIT REQ iDS: shared id {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id} #: {self.llm_engine.get_num_unfinished_requests()}, req ids: {[(_.request_id, list(_.seqs_dict.keys()), len(self.llm_engine.scheduler.block_manager.block_tables[list(_.seqs_dict.keys())[0]]), (list(_.seqs_dict.values())[0].get_len()-1)//self.llm_engine.cache_config.block_size) for list_ in [self.llm_engine.scheduler.running] for _ in list_]}")
            # print(f"model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}  possible_to_get_future_reqs: {possible_to_get_future_reqs}  self.llm_engine.has_unfinished_requests(): {self.llm_engine.has_unfinished_requests()}")


            # <> For DEBUG
            # stop_step = 155
            # if os.environ['USE_VLLM']=='False':
            #     stop_step = 128
            # if step_i > stop_step: # 155 vllm  128 ours
            #     break


            if (step_i == 300): #140):
                # print(f"step_i: {step_i}, step_start: {step_start}, step_end:{step_end}")
                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStart()
            elif (step_i == 310): #200):
                # elif (step_i == step_end) and (run_profile):
                # print(f"step_i: {step_i}, step_start: {step_start}, step_end:{step_end}")
                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStop()
            # ---------------------------------------- 



            # <> to support multi-model scheduling
            if SHARED_CONTECT.should_reschedule() and (os.environ['NO_PREEMPT'] == 'False'):
            # if step_i>1 and (os.environ['NO_PREEMPT'] == 'False'):
                # remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
                # SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests)
                # # now we are ready to reload the model according to the new execution plan and restart the inference

                print(f"THIS MODEL IS STOPPED!  model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}")

                # NOTE: after changing to multiprocessing dp worker processes, we do not need to 
                #       notify the dp ray actors as all workers can share the latest SHARED_CONTECT content
                # # notify other data parallel ray actors if any
                # if SHARED_CONTECT.has_dp_parallel():
                #     SHARED_CONTECT.notify_dp_actors_should_reschedule()

                break



            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)

                    print(f"finish one req: model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id}, output.request_id: {output.request_id}")


                    if use_tqdm:
                        pbar.update(1)
            
            # send the outputs to the model communicator every check_out_gap steps or when all current reqs are finished
            # <> multi-level LLM system
            # print(f"SHARED_CONTECT.check_out_gaps[SHARED_CONTECT.shared_id]: {SHARED_CONTECT.check_out_gaps[SHARED_CONTECT.shared_id]}")
            if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
                ((step_i % SHARED_CONTECT.check_out_gaps[SHARED_CONTECT.shared_id] == 0) or (not self.llm_engine.has_unfinished_requests())):
                
                # print(f"writing results back!  SHARED_CONTECT.shared_id: {SHARED_CONTECT.shared_id}------------------")
                output_num_sent_out, new_outputs = \
                    self._get_outputs_to_system_communicator(outputs, output_num_sent_out, SHARED_CONTECT.return_str)
                
                # print(f"writing results back!  output_num_sent_out: {output_num_sent_out}, new_outputs: {new_outputs}------------------")
                print(f"writing results back!  output_num_sent_out: {output_num_sent_out}, len(new_outputs): {len(new_outputs)}------------------")

                SHARED_CONTECT.communicator.add_seqs(SHARED_CONTECT.shared_id, new_outputs)


        if use_tqdm:
            pbar.close()

        print(f"total steps: {step_i}")
        # <> For layer-by-layer params loading------------------------------------------------
        from vllm._C import cache_ops
        if os.environ['USE_VLLM']!='True':
            # for cache_device_i in self.llm_engine.workers[0].model_runner.model.model.cache_device_ids:
            #     cache_ops.disable_P2P_access(cache_device_i, 0, 0)
            # in distributed inference, the current device for this worker may not be cuda:0
            self.llm_engine.disable_P2P_access()
        # -------------------------------------------------------------------------------------------



        
        # <> support multi-level model system
        if (os.environ['RUN_MULTI_MODEL'] == 'True'):
            output_num_sent_out, new_outputs = \
                self._get_outputs_to_system_communicator(outputs, output_num_sent_out, SHARED_CONTECT.return_str)
            SHARED_CONTECT.communicator.add_seqs(SHARED_CONTECT.shared_id, new_outputs)

            print(f"writing results back!  output_num_sent_out: {output_num_sent_out}, len(new_outputs): {len(new_outputs)}------------------")
        
        
        
        # <> support data parallelism
        if (os.environ['RUN_MULTI_MODEL'] == 'True') and \
            (not SHARED_CONTECT.has_dp_parallel()):
            # <> support multi-model inference
            remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            SHARED_CONTECT.set_finished(len(outputs))
            SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine, mark_finish=False)
            # if not SHARED_CONTECT.is_finished():
            #     SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine)
            # 
                # now we are ready to reload the model according to the new execution plan and restart the inference
            # ----
            # # <> support multi-model inference
            # SHARED_CONTECT.set_finished(not self.llm_engine.scheduler.has_unfinished_seqs())
            # if not SHARED_CONTECT.is_finished():
            #     remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            #     SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine)
            #     # now we are ready to reload the model according to the new execution plan and restart the inference            
        elif (os.environ['RUN_MULTI_MODEL'] == 'True'):
            # data parallelism is used
            # for dp ray actors, we will kill them no matter they finish or not, 
            # for dp main actor, we will kill its tp workers no matter it finishes or not
            # for all dp actors, we will store their finished and remaining requests info
            # the ``is_finished'' status of the dp main actor will be set after the main actor received all request info from other actors
            # the preparation for reschedule will also be marked as finished after that
            remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine, mark_finish=False)



        print(f"model id: {SHARED_CONTECT.shared_id} dp id: {SHARED_CONTECT.dp_id} SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)



        # store the non_compute_ranges back
        SHARED_CONTECT.communicator.update_non_compute_ranges(
            non_comp_ranges, SHARED_CONTECT.shared_id, SHARED_CONTECT.get_gpus_used())


        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
    











    # TODO () this function is used for DEBUG
    def _run_engine_for_debug(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []

        # <> init KV_blk_per_layer_weights
        import os

        # <> support multimodel scheduling
        from vllm.core.multimodel_scheduler import SHARED_CONTECT


        # <> For Profiling--------------------
        import torch
        step_i = 0
        # -------------------------------------------


        while self.llm_engine.has_unfinished_requests():

            # <> For Profiling-----------------
            step_i+=1
            print(f"step i: {step_i}", flush=True)


            # <> For DEBUG
            # stop_step = 155
            # if os.environ['USE_VLLM']=='False':
            #     stop_step = 128
            # if step_i > stop_step: # 155 vllm  128 ours
            #     break


            if (step_i == 1) and (self.llm_engine.parallel_config.world_size==4): #140):
                # print(f"step_i: {step_i}, step_start: {step_start}, step_end:{step_end}")
                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStart()
            elif (step_i == 3) and (self.llm_engine.parallel_config.world_size==4): #200):
                # elif (step_i == step_end) and (run_profile):
                # print(f"step_i: {step_i}, step_start: {step_start}, step_end:{step_end}")
                print(f"step_i: {step_i}")
                torch.cuda.cudart().cudaProfilerStop()
            # ---------------------------------------- 



            # <> to support multi-model scheduling
            # if SHARED_CONTECT.should_reschedule():
            if (step_i>3):
                # remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
                # SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests)
                # # now we are ready to reload the model according to the new execution plan and restart the inference
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
        # <> For layer-by-layer params loading------------------------------------------------
        from vllm._C import cache_ops
        if os.environ['USE_VLLM']!='True':
            # for cache_device_i in self.llm_engine.workers[0].model_runner.model.model.cache_device_ids:
            #     cache_ops.disable_P2P_access(cache_device_i, 0, 0)
            # in distributed inference, the current device for this worker may not be cuda:0
            self.llm_engine.disable_P2P_access()
        # -------------------------------------------------------------------------------------------



        # <> support multi-model inference
        SHARED_CONTECT.set_finished(not self.llm_engine.scheduler.has_unfinished_seqs())
        if not SHARED_CONTECT.is_finished():
            remaining_requests = self.llm_engine.scheduler.get_unfinished_seqs()
            SHARED_CONTECT.prepare_for_reschedule(outputs, remaining_requests, self.llm_engine)
            # now we are ready to reload the model according to the new execution plan and restart the inference


        print(f"SHARED_CONTECT.is_finished: {SHARED_CONTECT.is_finished()}", flush=True)



        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs



    # <> check whether a iter workload is worthy to be profiled
    def _is_redundant_iter_workload(
            self, record_dict,
            set_max_num_seqs, is_prompt, corr_context_tot_len,
            last_seqnum, last_is_prompt, last_corr_context_tot_len,
            weight_loading_time):
        
        # if the latency difference is within this threshold, we think the two latencys are the same
        threshold = 1e-3

        # 1. about another seqnum and another ``is_prompt'', then not redundant
        if (last_seqnum, last_is_prompt) != (set_max_num_seqs, is_prompt):
            return False
        
        # 1.5 corresponds to the same sampled context_tot_len interesting point as the last record, then not redundant
        if corr_context_tot_len == last_corr_context_tot_len:
            return False

        related_records = record_dict[(set_max_num_seqs, is_prompt)]
        
        # try to set weight_loading_time
        if (weight_loading_time[0] == None) and (len(related_records) == 2):
            weight_loading_time[0] = min(related_records.values())


        # 2. if 2 valid points and min == max, redundant
        if len(related_records) == 2:
            if max(related_records.values()) <= (min(related_records.values()) + threshold):
                return True
        

        # 3. if 3 valid points, y1, y2 and y3 all > weight_loading_time[0]+threshold, redundant
        if len(related_records) == 3:
            if min(related_records.values()) > (weight_loading_time[0]+threshold):
                return True
        

        # 4. if <= 3 valid points, not redundant
        if len(related_records) <= 3:
            return False
        
        # 5. compare the last two exec latency with the smallest latency
        xs = sorted(related_records.keys())
        if False not in [related_records[x] > (related_records[xs[0]]+threshold) for x in xs[-3:-1]]:
            return True
        
        return False



    # <>
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




    # <> run per iter latency profile
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
        # <> init KV_blk_per_layer_weights
        import os
        from collections import defaultdict
        my_throughput_logger = self.llm_engine.driver_worker.model_runner.my_throughput_logger
        record_dict = defaultdict(dict)
        weight_loading_time = [None]

        # run a measurement first as the first measurement result is higher than expected [should not be like this as we have profiled cache]
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

            # check whether we still need to profile this iter workloads
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
            
            # store results in record_dict
            self._update_record_dict(
                my_throughput_logger, record_dict, set_max_num_seqs, is_prompt, corr_context_tot_len)
            last_seqnum, last_is_prompt, last_corr_context_tot_len = set_max_num_seqs, is_prompt, corr_context_tot_len
        
        
        for is_prompt, set_seqlens in set_seqlens_list:
            self.llm_engine._profile_per_iter_latency(
                is_prompt=is_prompt, 
                sampling_params_dict=sampling_params_dict,
                set_seqlens=set_seqlens)            
        # ------------------------------------------------------------------------------
        # self.llm_engine._profile_per_iter_latency(
        #     is_prompt=True, sampling_params_dict=sampling_params_dict,
        #     set_max_num_batched_tokens=2048, set_max_num_seqs=2)
        # self.llm_engine._profile_per_iter_latency(
        #     is_prompt=True, sampling_params_dict=sampling_params_dict,
        #     set_max_num_batched_tokens=2048, set_max_num_seqs=2)
        # self.llm_engine._profile_per_iter_latency(
        #     is_prompt=False, sampling_params_dict=sampling_params_dict,
        #     set_max_num_batched_tokens=112394, set_max_num_seqs=512)
        # self.llm_engine._profile_per_iter_latency(
        #     is_prompt=False, sampling_params_dict=sampling_params_dict,
        #     set_max_num_batched_tokens=112394, set_max_num_seqs=256)
        # self.llm_engine._profile_per_iter_latency(
        #     is_prompt=False, sampling_params_dict=sampling_params_dict)
        # self.llm_engine._profile_per_iter_latency(
        #     is_prompt=False, sampling_params_dict=sampling_params_dict,
        #     set_max_num_seqs=120)

        # <> For layer-by-layer params loading------------------------------------------------
        from vllm._C import cache_ops
        if os.environ['USE_VLLM']!='True':
            # for cache_device_i in self.llm_engine.workers[0].model_runner.model.model.cache_device_ids:
            #     cache_ops.disable_P2P_access(cache_device_i, 0, 0)
            # in distributed inference, the current device for this worker may not be cuda:0
            self.llm_engine.disable_P2P_access()
        # -------------------------------------------------------------------------------------------

        return 
  





