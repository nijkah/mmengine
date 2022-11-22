# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import List, Union

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch
import torch.nn as nn

from mmengine.config import Config, ConfigDict
from mmengine.device import is_npu_available
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS
from .optimizer_wrapper import OptimWrapper


def register_torch_optimizers() -> List[str]:
    """Register optimizers in ``torch.optim`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


def register_deepspeed_optimizers() -> List[str]:
    """Register optimizers in ``deepspeed.ops.adam`` and ``deepspeed.ops.lamb``
    and ``deepspeed.runtime.fp16.onebit`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """

    assert deepspeed is not None, 'DeepSpeed should be installed!'
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from deepspeed.ops.lamb import FusedLamb
    from deepspeed.runtime.fp16.onebit.adam import OnebitAdam
    from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb
    from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

    deepspeed_optimizers = []
    optims = [
        FusedAdam, DeepSpeedCPUAdam, FusedLamb, OnebitAdam, OnebitLamb,
        ZeroOneAdam
    ]
    for optim in optims:
        if inspect.isclass(optim) and issubclass(optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module(module=optim)
            deepspeed_optimizers.append(optim.__name__)
    return deepspeed_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()
if deepspeed is not None:
    DEEPSPEED_OPTIMIZERS = register_deepspeed_optimizers()


def build_optim_wrapper(model: nn.Module,
                        cfg: Union[dict, Config, ConfigDict]) -> OptimWrapper:
    """Build function of OptimWrapper.

    If ``constructor`` is set in the ``cfg``, this method will build an
    optimizer wrapper constructor, and use optimizer wrapper constructor to
    build the optimizer wrapper. If ``constructor`` is not set, the
    ``DefaultOptimWrapperConstructor`` will be used by default.

    Args:
        model (nn.Module): Model to be optimized.
        cfg (dict): Config of optimizer wrapper, optimizer constructor and
            optimizer.

    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    optim_wrapper_cfg = copy.deepcopy(cfg)
    constructor_type = optim_wrapper_cfg.pop('constructor',
                                             'DefaultOptimWrapperConstructor')
    paramwise_cfg = optim_wrapper_cfg.pop('paramwise_cfg', None)

    # Since the current generation of NPU(Ascend 910) only supports
    # mixed precision training, here we turn on mixed precision by default
    # on the NPU to make the training normal
    if is_npu_available():
        optim_wrapper_cfg['type'] = 'AmpOptimWrapper'

    optim_wrapper_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
        dict(
            type=constructor_type,
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg))
    optim_wrapper = optim_wrapper_constructor(model)
    return optim_wrapper
