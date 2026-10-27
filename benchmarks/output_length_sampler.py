"""
Sample the output length for each req following the pdf we obtained from experiments on the no-robot dataset.
"""

from typing import List
import numpy as np
import collect_output_lengths.no_robot.out_len_sampler_2 as out_len_sampler2
from experiment_reproducibility import experiment_seed


def sample_uniform_output_lengths(model: str, inp_lens: List[int], max_token_num=None):
    """Table 3 Rand: uniformly sample each request's legal output interval."""
    highs = len(out_len_sampler2.pdf_dict[model]) - np.asarray(inp_lens)
    if max_token_num is not None:
        highs = np.minimum(highs, max_token_num)
    if np.any(highs < 1):
        raise ValueError("input leaves no legal output token")
    rng = np.random.default_rng(seed=experiment_seed())
    return rng.integers(1, highs, len(inp_lens), endpoint=True)

def sample_out_len_for_given_model_ori(model: str, inp_lens: List[int]):

    pdf_dict = out_len_sampler2.pdf_dict
    rng = np.random.default_rng(seed=experiment_seed())
    max_model_len = len(pdf_dict[model])
    req_num = len(inp_lens)
    out_lens = rng.choice(np.arange(1, max_model_len+1), req_num, replace=True, p=pdf_dict[model])

    out_lens = np.minimum(max_model_len - np.asarray(inp_lens), out_lens)
    return out_lens


def sample_out_len_for_given_model(model: str, inp_lens: List[int], noise_level: float):

    pdf_dict = out_len_sampler2.pdf_dict
    rng = np.random.default_rng(seed=experiment_seed())
    max_model_len = len(pdf_dict[model])

    pdf = np.asarray(pdf_dict[model])

    if noise_level == -2:

        out_lens = sum(np.arange(1, max_model_len+1) * pdf)
        out_lens = int(out_lens)
        out_lens = np.full(len(inp_lens), out_lens)
        return out_lens


    if noise_level != 0:

        pdf = rng.uniform(np.maximum(0, pdf-noise_level), pdf+noise_level, max_model_len)

        pdf = pdf / np.sum(pdf)


    req_num = len(inp_lens)

    out_lens = rng.choice(np.arange(1, max_model_len+1), req_num, replace=True, p=pdf)

    out_lens = np.minimum(max_model_len - np.asarray(inp_lens), out_lens)

    print(f"mean (with noise): {np.mean(out_lens)}")

    return out_lens


def sample_out_len_for_given_model_with_noise(model: str, inp_lens: List[int], noise_level:float, max_token_num: int):
    """
        直接对outlen分布添加噪声 然后再进行sample
    """

    pdf_dict = out_len_sampler2.pdf_dict
    rng = np.random.default_rng(seed=experiment_seed())
    max_model_len = len(pdf_dict[model])
    req_num = len(inp_lens)
    pdf = pdf_dict[model]

    noise = rng.uniform(max(1e-9, 1-noise_level), 1+noise_level, max_model_len)
    pdf = noise*pdf
    pdf = pdf / np.sum(pdf)

    out_lens = rng.choice(np.arange(1, max_model_len+1), req_num, replace=True, p=pdf)

    out_lens = np.minimum(max_model_len - np.asarray(inp_lens), out_lens)

    print(f"out_lens with noises: {out_lens}")

    return out_lens


def add_noise_to_out_lens(outlen_generator, model_id: int, model_name: str, inp_lens: List[int], noise_level:float, max_token_num: int):
    """
        Add noise to the predicted (or real) output lengths and 
    """
    out_lens = outlen_generator(model_id, model_name, inp_lens)
    out_lens = np.asarray(out_lens)
    ori_out_lens = out_lens
    print(f"ori out_lens: {out_lens}, {type(out_lens)}")


    if noise_level == -2:
        return np.full(len(inp_lens), np.mean(out_lens))


    pdf_dict = out_len_sampler2.pdf_dict
    max_model_len = len(pdf_dict[model_name])
    highs = max_model_len - np.asarray(inp_lens)
    if max_token_num != None:
        highs = np.minimum(max_token_num, highs)


    rng = np.random.default_rng(seed=experiment_seed())
    if noise_level > 0:

        out_lens = rng.integers(np.maximum(1, (out_lens*(1-noise_level)).astype(int)), (out_lens*(1+noise_level)).astype(int), len(inp_lens), endpoint=True)

        out_lens = np.minimum(highs, out_lens)
    else:
        out_lens = rng.integers(1, highs, len(inp_lens), endpoint=True)

    print(f"out_lens with noises: {sorted(out_lens)}")
    print(f"max_token_num: {max_token_num}")

    errors = np.mean(np.abs(out_lens - ori_out_lens) / ori_out_lens)
    print(f"avg errors: {errors}, mean (with noise): {np.mean(out_lens)}, ori mean: {np.mean(ori_out_lens)}")

    return out_lens
