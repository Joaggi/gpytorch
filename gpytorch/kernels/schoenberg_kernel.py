#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator.operators import to_dense
from torch import Tensor

from ..constraints import Interval, Positive
from .kernel import Kernel


class SchoenbergKernel(Kernel):
    """
    Decorates a robust kernel (e.g. :class:`gpytorch.kernels.CauchyKernel`) with the Schoenberg
    construction::

        K(x1, x2) = exp(-gamma * Phi(x1, x2)),   gamma > 0

    where Phi(x1, x2) = -k_base(x1, x2) and `k_base` is the base kernel. The Huber, Cauchy and
    Andrews kernels are <= 0 with a maximum of 0 at x1 = x2, so negating them gives a distance-like
    Phi, the same role ||x1 - x2||^2 plays in the Gaussian kernel.

    .. note::

        The result is only guaranteed to be positive semi-definite when Phi is conditionally
        negative definite. This holds for :class:`gpytorch.kernels.CauchyKernel`, but it is not
        guaranteed for the other robust kernels.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    :param base_kernel: The robust kernel to be wrapped. Its scale parameter `c` is learned
        together with `gamma`.
    :param batch_shape: Set this if you want a separate `gamma` for each batch of input
        data. It should be B_1 x ... x B_k if `x1` is a B_1 x ... x B_k x N x D tensor.
    :param gamma_constraint: Set this if you want to apply a constraint to the
        parameter `gamma`. (Default: `Positive`.)

    :ivar gpytorch.kernels.Kernel base_kernel: The wrapped kernel.
    :ivar torch.Tensor gamma: The Schoenberg parameter, initialized to 1.0. Size/shape of parameter
        depends on the batch_shape argument.

    Example:
        >>> x = torch.randn(10, 5)
        >>> base_covar_module = gpytorch.kernels.CauchyKernel()
        >>> covar_module = gpytorch.kernels.SchoenbergKernel(base_covar_module)
        >>> covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel: Kernel, gamma_constraint: Interval | None = None, **kwargs):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super().__init__(**kwargs)
        if gamma_constraint is None:
            gamma_constraint = Positive()

        self.base_kernel = base_kernel
        gamma = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_gamma", parameter=torch.nn.Parameter(gamma))
        self.register_constraint("raw_gamma", gamma_constraint)
        self.initialize(gamma=1.0)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Tensor:
        base = to_dense(self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params))
        gamma = self.gamma
        if last_dim_is_batch:
            gamma = gamma.unsqueeze(-1)
        if diag:
            gamma = gamma.unsqueeze(-1)
        else:
            gamma = gamma.view(*gamma.shape, 1, 1)
        # exp(-gamma * Phi) with Phi = -base
        return torch.exp(gamma * base)

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        return self.base_kernel.num_outputs_per_input(x1, x2)

    @property
    def gamma(self) -> Tensor:
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value: Tensor | float) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma)
        self.initialize(raw_gamma=self.raw_gamma_constraint.inverse_transform(value))
