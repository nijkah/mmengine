# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch
import torch.nn as nn

from mmengine.registry import OPTIM_WRAPPERS
from .optimizer_wrapper import OptimWrapper


@OPTIM_WRAPPERS.register_module()
class DeepSpeedOptimWrapper(OptimWrapper):
    """DeepSpeedOptimWrapper provides a common interface for updating
    parameters when using ``DeepSpeedEngine``.

    The main difference compared to the original one is that
    ``loss.backward()`` and ``optimizer.step()`` and `optimizer.zero_grad()`
    are done in ``DeepSpeedEngine``.
    """

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation with ``DeepSpeedEngine``.

        Compared to the original method, the ``backward`` is done in
        ``DeepSpeedEngine`` with the given loss.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`.

        Examples:
        >>> # original ``backward``
        >>> loss.backward(**kwargs)
        >>> self._inner_count += 1
        >>>
        >>> # Edited ``backward``
        >>> model.backward(loss, **kwargs)
        >>> self._inner_count += 1
        """
        self.model.backward(loss, **kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Compared to the original method, the ``zero_grad`` is done in
        ``DeepSpeedEngine.step()``. So just passed.
        """
        pass

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Compared to the original method, the ``step`` is done with
        ``DeepSpeedEngine``.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.

        Examples:
        >>> # original ``step``
        >>> self.optimizer.step(**kwargs)
        >>>
        >>> # Edited ``step``
        >>> self.model.step(**kwargs)
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.model.step(**kwargs)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        Compared to the original method, this saves model information as
        a member variable in order to use in the training step.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        self.model = model
        yield super().optim_context(model)

    def load_state_dict(self, state_dict: dict, **kwargs) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Provide unified ``load_state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be loaded when training with ``torch.cuda.amp``.

        Args:
            state_dict (dict): The state dictionary of :attr:`optimizer`.
        """
        self.optimizer.load_state_dict(state_dict, **kwargs)
