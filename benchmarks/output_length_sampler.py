"""
Sample the output length for each req following the pdf we obtained from experiments on the no-robot dataset.
"""

from typing import List
import numpy as np
import collect_output_lengths.no_robot.out_len_sampler_2 as out_len_sampler2

def sample_out_len_for_given_model(model: str, inp_lens: List[int]):
    # get the pdf dict
    pdf_dict = out_len_sampler2.pdf_dict
    rng = np.random.default_rng(seed=0)
    max_model_len = len(pdf_dict[model])
    req_num = len(inp_lens)
    out_lens = rng.choice(np.arange(1, max_model_len+1), req_num, replace=True, p=pdf_dict[model])
    # the out_lens is limited by max_model_len
    out_lens = np.minimum(max_model_len - np.asarray(inp_lens), out_lens)
    return out_lens



# NOTE: we cannot directly compute the expectation output lengths and do fake scheduling based on them, 
# because the total latency is not a linear function of the output lengths.
# 
# def get_expectation_out_len_for_given_model(model: str, inp_lens: List[int]):
#     '''
#         Directly compute the expectation output length for the given model and given input lengths.
#     '''
#     # get the pdf dict
#     pdf_dict = out_len_sampler.pdf_dict
#     pdf = pdf_dict[model]
#     max_model_len = len(pdf)
#     max_outlens = max_model_len - np.asarray(inp_lens)

#     # the out_lens is limited by max_model_len
#     expected_outlens = np.cumsum(pdf*np.arange(1, max_model_len+1))[max_outlens - 1]
#     expected_outlens = expected_outlens + (1 - np.cumsum(pdf)[max_outlens-1])*max_outlens
#     return expected_outlens



