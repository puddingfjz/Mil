from vllm.logger import init_logger
from aioprometheus import Counter, Gauge, Histogram

import time
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


# <>
from vllm.config import ModelConfig, ParallelConfig
import model_coeff_database

logger = init_logger(__name__)

labels = {}


def add_global_metrics_labels(**kwargs):
    labels.update(kwargs)


# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.

# begin-metrics-definitions
gauge_avg_prompt_throughput = Gauge("vllm:avg_prompt_throughput_toks_per_s",
                                    "Average prefill throughput in tokens/s.")
gauge_avg_generation_throughput = Gauge(
    "vllm:avg_generation_throughput_toks_per_s",
    "Average generation throughput in tokens/s.")
counter_prompt_tokens = Counter("vllm:prompt_tokens_total",
                                "Number of prefill tokens processed.")
counter_generation_tokens = Counter("vllm:generation_tokens_total",
                                    "Number of generation tokens processed.")

gauge_scheduler_running = Gauge(
    "vllm:num_requests_running",
    "Number of requests currently running on GPU.")
gauge_scheduler_swapped = Gauge("vllm:num_requests_swapped",
                                "Number of requests swapped to CPU.")
gauge_scheduler_waiting = Gauge("vllm:num_requests_waiting",
                                "Number of requests waiting to be processed.")

gauge_gpu_cache_usage = Gauge(
    "vllm:gpu_cache_usage_perc",
    "GPU KV-cache usage. 1 means 100 percent usage.")
gauge_cpu_cache_usage = Gauge(
    "vllm:cpu_cache_usage_perc",
    "CPU KV-cache usage. 1 means 100 percent usage.")

histogram_time_to_first_token = Histogram(
    "vllm:time_to_first_token_seconds",
    "Histogram of time to first token in seconds.",
    buckets=[
        0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0
    ])
histogram_time_per_output_tokens = Histogram(
    "vllm:time_per_output_token_seconds",
    "Histogram of time per output token in seconds.",
    buckets=[
        0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5
    ])
histogram_e2e_request_latency = Histogram(
    "vllm:e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds.",
    buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
# end-metrics-definitions


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float

    # System stats.
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float

    # Raw stats from last model iteration.
    num_prompt_tokens: int
    num_generation_tokens: int
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    time_e2e_requests: List[float]


class StatLogger:
    """StatLogger is used LLMEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float) -> None:
        # Metadata for logging locally.
        self.last_local_log = time.monotonic()
        self.local_interval = local_interval

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def _log_prometheus(self, stats: Stats) -> None:
        # Set system stat gauges.
        gauge_scheduler_running.set(labels, stats.num_running)
        gauge_scheduler_swapped.set(labels, stats.num_swapped)
        gauge_scheduler_waiting.set(labels, stats.num_waiting)
        gauge_gpu_cache_usage.set(labels, stats.gpu_cache_usage)
        gauge_cpu_cache_usage.set(labels, stats.cpu_cache_usage)

        # Add to token counters.
        counter_prompt_tokens.add(labels, stats.num_prompt_tokens)
        counter_generation_tokens.add(labels, stats.num_generation_tokens)

        # Observe request level latencies in histograms.
        for ttft in stats.time_to_first_tokens:
            histogram_time_to_first_token.observe(labels, ttft)
        for tpot in stats.time_per_output_tokens:
            histogram_time_per_output_tokens.observe(labels, tpot)
        for e2e in stats.time_e2e_requests:
            histogram_e2e_request_latency.observe(labels, e2e)

    def _log_prometheus_interval(self, prompt_throughput: float,
                                 generation_throughput: float) -> None:
        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on the vLLM side.
        # Moving forward, we should use counters like counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the grafana/prometheus side.
        # See https://github.com/vllm-project/vllm/pull/2316#discussion_r1464204666
        gauge_avg_prompt_throughput.set(labels, prompt_throughput)
        gauge_avg_generation_throughput.set(labels, generation_throughput)

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to prometheus and tracked stats every iteration. 
           Logs to Stdout every self.local_interval seconds."""

        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(stats.num_prompt_tokens)
        self.num_generation_tokens.append(stats.num_generation_tokens)

        # Log locally every local_interval seconds.
        if self._local_interval_elapsed(stats.now):

            # Compute summary metrics for tracked stats (and log them to promethus if applicable).
            prompt_throughput = self._get_throughput(self.num_prompt_tokens,
                                                     now=stats.now)
            generation_throughput = self._get_throughput(
                self.num_generation_tokens, now=stats.now)
            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput)

            # Log to stdout.
            logger.info(
                f"Avg prompt throughput: {prompt_throughput:.1f} tokens/s, "
                f"Avg generation throughput: {generation_throughput:.1f} tokens/s, "
                f"Running: {stats.num_running} reqs, "
                f"Swapped: {stats.num_swapped} reqs, "
                f"Pending: {stats.num_waiting} reqs, "
                f"GPU KV cache usage: {stats.gpu_cache_usage * 100:.1f}%, "
                f"CPU KV cache usage: {stats.cpu_cache_usage * 100:.1f}%")

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now





# <>
def cal_flops_wrong_version(h,I,L, B, s, context_tot_len, is_prompt):
    '''
    s = max_seq_len
    For prefill (padding to max_seq_len):
        L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)
    For decoding:
        L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)
    '''
    # h = 4096
    # I = 11008
    # L = 32
    # h,I,L = self.h, self.I, self.L
    if is_prompt:
        return L*( 4*B*s*h*h + 2*B*s*s*h + 3*B*s*h*I)
    else:
        # we must set s=1 here, because for decode stages, we store the real max_seqlen in the flop_metadata as s
        s = 1
        return L*( 4*B*s*h*h + 2*h*context_tot_len + 3*I*B*s*h)


# <>
def cal_flops_manual_computation(
        h,I,L, 
        kv_head_num, tp_size, head_dim, 
        B, s, context_tot_len, is_prompt):
    '''
    s = max_seq_len
    For prefill (padding to max_seq_len):
        L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)
    For decoding:
        L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)
        
    '''
    # h = 4096
    # I = 11008
    # L = 32
    # h,I,L = self.h, self.I, self.L


    if is_prompt:
        return L*( B*s*h*(2*max(kv_head_num/tp_size,1))*head_dim + \
                   2*B*s*h/tp_size*h +  \
                   2*B*s*h*s/tp_size + \
                   3*B*s*I/tp_size*h)
    else:
        # we must set s=1 here, because for decode stages, we store the real max_seqlen in the flop_metadata as s
        s = 1
        # return L*( B*s*h*[(2*max(kv_head_num/tp_size,1))*head_dim] + 
        return L*( B*s*h*(2*max(kv_head_num/tp_size,1))*head_dim + 
                   2*B*s*h/tp_size*h +  
                   2*h*context_tot_len/tp_size + 
                   3*B*s*I/tp_size*h)





# <>
def cal_flops(
        model_path,
        h,L, tp_size, 
        B, s, context_tot_len, is_prompt):
    '''
    s = max_seq_len
    For prefill (padding to max_seq_len):
        L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)
    For decoding:
        L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)
        
    '''
    # h = 4096
    # I = 11008
    # L = 32
    # h,I,L = self.h, self.I, self.L

    coeff = model_coeff_database.model_coeffs[(model_path,tp_size)]
    if is_prompt:
        return L*( B*s*coeff + 2*B*s*h*s/tp_size)
    else:
        # we must set s=1 here, because for decode stages, we store the real max_seqlen in the flop_metadata as s
        s = 1
        # return L*( B*s*h*[(2*max(kv_head_num/tp_size,1))*head_dim] + 
        return L*( B*s*coeff + 2*h*context_tot_len/tp_size)









# <>
class MyThroughputLogger:
    """MyThroughputLogger is used in model_runner to record the flops, latency, and throughput."""

    def __init__(self, model_config: ModelConfig, parallel_config: ParallelConfig) -> None:
        self.flop_metadata: List[Tuple[int, int]] = list()
        self.flops_list: List[int] = list()
        self.latencys: List[float] = list()
        self.throughputs: List[float] = list()
        self.schedule_time: List[float] = list()
        self.exec_time: List[float] = list()
        self.sample_time: List[float] = list()
        self.prepInp_time: List[float] = list()
        self.print_flopMetadata_time: List[float] = list()

        # self.V: int = model_config.hf_config["vocab_size"]
        # self.h: int = model_config.hf_config["hidden_size"]
        # self.I: int = model_config.hf_config["intermediate_size"]
        # self.L: int = model_config.hf_config["num_hidden_layers"]
        self.V: int = model_config.hf_config.vocab_size
        self.h: int = model_config.hf_config.hidden_size
        self.I: int = getattr(model_config.hf_config, "intermediate_size", None) or \
                        getattr(model_config.hf_config, "n_inner", None) or \
                        getattr(model_config.hf_config, "inner_hidden_size", None) or \
                        (4*self.h)
        self.L: int = model_config.hf_config.num_hidden_layers
        self.kv_head_num: int = model_config.get_total_num_kv_heads() #hf_config.num_key_value_heads
        self.tp_size: int = parallel_config.tensor_parallel_size
        self.head_dim: int = self.h // model_config.hf_config.num_attention_heads
        self.model_path: str = model_config.model



    def append_flop_metadata(self, seq_num, context_tot_len, attention_sum, max_seqlen, is_prompt):
        # seq_num: the real seq_num
        # equivalent_seq_num: the seq_num when regarded as a decoding stage
        self.flop_metadata.append((seq_num, context_tot_len, attention_sum, max_seqlen, is_prompt))

    def append_latency(self, latency):
        self.latencys.append(latency)

    def append_schedule_time(self, latency):
        self.schedule_time.append(latency)


    def append_exec_time(self, latency):
        self.exec_time.append(latency)


    def append_sample_time(self, latency):
        self.sample_time.append(latency)

    
    def append_prepInp_time(self, latency):
        self.prepInp_time.append(latency)

    
    def append_print_flopMetadata_time(self, latency):
        self.print_flopMetadata_time.append(latency)


    def cal_flops_WRONG_VERSION(self, T,V,h,I,L,context_tot_len):
        # context_tot_len = sum(contexts): the total context length for the decoding stage
        # for the prefill stage, it can be regarded as a decoding stage, and we can compute flops in a similar way
        return 2*T*V*h+L*(4*T*h*h+2*context_tot_len*h+3*I*h*T)
    
    
    # def move_flop_metadata_to_cpu(self):
    #     self.flop_metadata = [(
    #         int(i.item()) if not isinstance(i, int) else i, 
    #         int(j.item()) if not isinstance(j, int) else j,
    #         int(k.item()) if not isinstance(k, int) else k) \
    #         for i, j, k in self.flop_metadata]
    #     self.flop_metadata = [[int(j.item()) if not isinstance(j, int) else j for j in i] \
    #         for i in self.flop_metadata]



    def cal_throughput(self):        
        # must change the flop metadata to python int instead of torch int32 to avoid result overflow
        # self.move_flop_metadata_to_cpu()
        h,I,L = self.h, self.I, self.L
        kv_head_num, tp_size, head_dim = self.kv_head_num, self.tp_size, self.head_dim
        # self.flops_list = [cal_flops(h,I,L, 
        #                              kv_head_num, tp_size, head_dim,
        #                              B, s, context_tot_len, is_prompt) \
        #                                 for B, context_tot_len, _, s, is_prompt in self.flop_metadata]
        self.flops_list = [cal_flops(self.model_path, 
                                     h,L, tp_size,
                                     B, s, context_tot_len, is_prompt) \
                                        for B, context_tot_len, _, s, is_prompt in self.flop_metadata]
        
        record_num = len(self.latencys)
        self.throughputs = [flops / latency for flops, latency in zip(self.flops_list[:record_num], self.latencys)]


    def print_by_record(self):
        import json
        flop_metadata = self.flop_metadata
        flops_list = self.flops_list
        latencys = self.latencys
        throughputs = self.throughputs
        schedule_time = self.schedule_time
        exec_time = self.exec_time
        sample_time = self.sample_time
        prepInp_time = self.prepInp_time
        print_flopMetadata_time = self.print_flopMetadata_time
        for i in range(len(latencys)):
            res = [
                flop_metadata[i],
                flops_list[i]/1e12,
                (latencys[i], exec_time[i], sample_time[i], prepInp_time[i], print_flopMetadata_time[i]),
                (throughputs[i]/1e12, flops_list[i]/exec_time[i]/1e12),
                self.tp_size]
            print(json.dumps(res))
            # 
            # print(flop_metadata[-i], 
            #       f'{flops_list[-i]/1e12} TFLOPs', 
            #       "Time:",
            #       f'{latencys[-i]} s', 
            #     #   f'{schedule_time[-i]} s', 
            #       f'{exec_time[-i]} s', 
            #       f'{sample_time[-i]} s', 
            #       "Throughput:",
            #       f'{throughputs[-i]/1e12} TFLOPs/s',
            #       f'{flops_list[-i]/exec_time[-i]/1e12} TFLOPs/s',)


    def get_last_record_execLatency(self):
        '''
            Return the execLatency of the last record.
        '''
        return self.exec_time[-1]



    @classmethod
    def load_record(cls, line):
        import json
        tp_size = None
        if len(json.loads(line)) > 4:
            tp_size = json.loads(line)[4]
        flop_metadata, flops, times, throughputs =  json.loads(line)[:4]
        return flop_metadata, flops, times, throughputs, tp_size
    

    @classmethod
    def get_seqnum_isprompt_flops_execLatency(cls, line):
        import json
        flop_metadata, flops, times, throughputs =  json.loads(line)[:4]
        return flop_metadata[0], flop_metadata[-1], flops, times[1]


    @classmethod
    def get_seqnum_contextTotLen_isprompt_sampleLatency(cls, line):
        import json
        flop_metadata, flops, times, throughputs =  json.loads(line)[:4]
        return flop_metadata[0], flop_metadata[1], flop_metadata[-1], times[2]


    @classmethod
    def get_seqnum_contextTotLen_maxSeqlen_isprompt_prepInpLatency(cls, line):
        import json
        flop_metadata, flops, times, throughputs =  json.loads(line)[:4]
        return flop_metadata[0], flop_metadata[1], flop_metadata[3], flop_metadata[-1], times[3]


    @classmethod
    def get_all_type_latencys(cls, line):
        import json
        flop_metadata, flops, times, throughputs =  json.loads(line)[:4]
        return times


    def __repr__(self) -> str:
        return (
            "MyThroughputLogger(\n"
            f"flop_metadata={self.flop_metadata}, \n"
            f"flops_list={self.flops_list}, \n"
            f"latencys={self.latencys}, \n"
            f"throughputs={self.throughputs})\n"
            f"schedule_time={self.schedule_time})\n"
            f"exec_time={self.exec_time})\n"
            f"sample_time={self.sample_time})\n"
            f"prepInp_time={self.prepInp_time})\n"
            f"print_flopMetadata_time={self.print_flopMetadata_time})\n"
            f"tp_size={self.tp_size}"
            )


