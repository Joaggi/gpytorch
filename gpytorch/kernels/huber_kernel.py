#!/usr/bin/env python3

from __future__ import annotations

import torch
from torch import Tensor

from .robust_kernel import _RobustKernel


class HuberKernel(_RobustKernel):
    """
    Computes a covariance matrix based on the Huber robust kernel
    between inputs `x1` and `x2`::

        k(x1, x2) = -1/4 * r^2               if r/c <= 1
        k(x1, x2) = -c/2 * r + c^2/4         if r/c > 1

    where r = ||x1 - x2|| is the Euclidean distance between the inputs and `c` is a positive scale parameter.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    .. note::

        This kernel is not guaranteed to be positive semi-definite, so it may not be a valid
        covariance function for every set of inputs.

    :param batch_shape: Set this if you want a separate `c` for each batch of input
        data. It should be B_1 x ... x B_k if `x1` is a B_1 x ... x B_k x N x D tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param c_constraint: Set this if you want to apply a constraint to the
        scale parameter `c`. (Default: `Positive`.)

    :ivar torch.Tensor c: The scale parameter. Size/shape of parameter depends on the batch_shape argument.

    Example:
        >>> x = torch.randn(10, 5)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.HuberKernel())
        >>> covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)
    """

    def _robust_function(self, dist: Tensor, c: Tensor) -> Tensor:
        return torch.where(dist / c <= 1, -0.25 * dist.square(), -0.5 * c * dist + 0.25 * c.square())
