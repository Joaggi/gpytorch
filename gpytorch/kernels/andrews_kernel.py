from __future__ import annotations

import math

import torch
from torch import Tensor

from .robust_kernel import _RobustKernel


class AndrewsKernel(_RobustKernel):
    """
    Computes a covariance matrix based on the Andrews robust kernel
    between inputs `x1` and `x2`::

        k(x1, x2) = -c/2 * (1 - cos(r/c))    if r/c <= pi
        k(x1, x2) = -c                       if r/c > pi

    where r = ||x1 - x2|| is the Euclidean distance between the inputs and `c` is a positive scale parameter.

    This implementation is derived from J. A. Gallego, F. A. González, and O. Nasraoui,
    "Robust kernels for robust location estimation," Neurocomputing, vol. 429, no. 1, pp. 174-186, 2021.
    DOI: 10.1016/j.neucom.2020.10.090

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

    """

    def _robust_function(self, dist: Tensor, c: Tensor) -> Tensor:
        ratio = dist / c
        return torch.where(ratio <= math.pi, -0.5 * c * (1 - torch.cos(ratio)), -c.expand_as(ratio))
