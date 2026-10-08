#!/usr/bin/env python3

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from ..constraints import Interval, Positive
from .kernel import Kernel


class _RobustKernel(Kernel):
    """
    Base class for the robust kernels (Tukey, Huber, Cauchy and Andrews).

    A robust kernel is a function of the Euclidean distance r = ||x1 - x2|| and a positive
    scale parameter `c`. Subclasses only need to implement :meth:`_robust_function`.

    :param batch_shape: Set this if you want a separate `c` for each batch of input
        data. It should be B_1 x ... x B_k if `x1` is a B_1 x ... x B_k x N x D tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param c_constraint: Set this if you want to apply a constraint to the
        scale parameter `c`. (Default: `Positive`.)

    :ivar torch.Tensor c: The scale parameter. Size/shape of parameter depends on the batch_shape argument.
    """

    def __init__(self, c_constraint: Interval | None = None, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name="raw_c", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if c_constraint is None:
            c_constraint = Positive()

        self.register_constraint("raw_c", c_constraint)
        self.initialize(c=1.0)

    @abstractmethod
    def _robust_function(self, dist: Tensor, c: Tensor) -> Tensor:
        """
        Evaluates the kernel as a function of the (non-squared) distance `dist` and the scale `c`.
        """
        raise NotImplementedError()

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params):
        c = self.c
        if not diag:
            c = c.unsqueeze(-1)

        if last_dim_is_batch:
            c = c.unsqueeze(-1)

        dist = self.covar_dist(x1, x2, square_dist=False, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        return self._robust_function(dist, c)

    @property
    def c(self) -> Tensor:
        return self.raw_c_constraint.transform(self.raw_c)

    @c.setter
    def c(self, value: Tensor | float) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_c)
        self.initialize(raw_c=self.raw_c_constraint.inverse_transform(value))
