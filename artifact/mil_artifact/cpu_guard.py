"""Fail closed on GPU operators while importing the legacy CPU simulator."""
import os
import sys
from types import ModuleType


def forbidden_gpu_operation(*args, **kwargs):
    raise RuntimeError('GPU operators are forbidden in CPU scheduling')


class ForbiddenOps(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return forbidden_gpu_operation


def install():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    extension = ModuleType('vllm._C')
    extension.__file__ = '<CPU-only forbidden GPU operators>'
    extension.ops = ForbiddenOps('vllm._C.ops')
    extension.cache_ops = ForbiddenOps('vllm._C.cache_ops')
    sys.modules['vllm._C'] = extension
    # The cost simulator uses Python configuration/cache-size calculations.
    # Every compiled inference/cache operator fails if accidentally reached.
    import torch
    torch.cuda._lazy_init = forbidden_gpu_operation
