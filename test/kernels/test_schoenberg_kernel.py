#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import AndrewsKernel, CauchyKernel, HuberKernel, SchoenbergKernel, TukeyKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

BASE_KERNELS = [TukeyKernel, HuberKernel, CauchyKernel, AndrewsKernel]


class TestSchoenbergKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return SchoenbergKernel(CauchyKernel(), **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return SchoenbergKernel(CauchyKernel(ard_num_dims=num_dims), **kwargs)

    def test_matches_definition(self):
        x1 = torch.randn(6, 3, dtype=torch.float64)
        x2 = torch.randn(4, 3, dtype=torch.float64)
        for base_cls in BASE_KERNELS:
            base = base_cls().double()
            base.initialize(c=2.0)
            kernel = SchoenbergKernel(base).double()
            kernel.initialize(gamma=0.7)
            kernel.eval()
            actual = torch.exp(-0.7 * -base(x1, x2).to_dense())
            self.assertAllClose(kernel(x1, x2).to_dense(), actual)

    def test_known_value_cauchy(self):
        # Phi = -cauchy(c=2) = 0.81093..., so K = exp(-0.81093...)
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        base = CauchyKernel()
        base.initialize(c=2.0)
        kernel = SchoenbergKernel(base)
        kernel.eval()
        self.assertAlmostEqual(kernel(a, b).to_dense().item(), 0.44444, places=4)

    def test_gamma_default_is_one(self):
        kernel = SchoenbergKernel(CauchyKernel())
        self.assertAlmostEqual(kernel.gamma.item(), 1.0, places=5)

    def test_initialize_gamma(self):
        kernel = SchoenbergKernel(CauchyKernel())
        kernel.initialize(gamma=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.gamma)
        self.assertLess(torch.norm(kernel.gamma - actual_value), 1e-5)

    def test_unit_diagonal(self):
        # Huber, Cauchy and Andrews are 0 at x == y, so exp(-gamma * 0) == 1
        x = torch.randn(5, 3)
        for base_cls in (HuberKernel, CauchyKernel, AndrewsKernel):
            kernel = SchoenbergKernel(base_cls())
            kernel.eval()
            self.assertAllClose(kernel(x).to_dense().diagonal(), torch.ones(5))
            self.assertAllClose(kernel(x, diag=True), torch.ones(5))

    def test_batch_gamma(self):
        x = torch.randn(2, 5, 3)
        gamma = torch.tensor([0.5, 2.0])
        kernel = SchoenbergKernel(CauchyKernel(), batch_shape=torch.Size([2]))
        kernel.initialize(gamma=gamma)
        kernel.eval()
        res = kernel(x).to_dense()
        for i in range(2):
            single = SchoenbergKernel(CauchyKernel())
            single.initialize(gamma=gamma[i])
            single.eval()
            self.assertAllClose(res[i], single(x[i]).to_dense())

    def test_gradients(self):
        x = torch.randn(5, 2)
        kernel = SchoenbergKernel(CauchyKernel())
        kernel.eval()
        kernel(x).to_dense().sum().backward()
        self.assertTrue(torch.isfinite(kernel.raw_gamma.grad).all())
        self.assertTrue(torch.isfinite(kernel.base_kernel.raw_c.grad).all())

    def test_cauchy_is_positive_semi_definite(self):
        # Phi is conditionally negative definite for Cauchy, so Schoenberg guarantees PSD.
        # (Not asserted for the other robust kernels.)
        torch.manual_seed(0)
        x = torch.randn(30, 2, dtype=torch.float64) * 2
        base = CauchyKernel().double()
        base.initialize(c=1.0)
        kernel = SchoenbergKernel(base).double()
        kernel.eval()
        self.assertGreater(torch.linalg.eigvalsh(kernel(x).to_dense()).min().item(), -1e-8)


if __name__ == "__main__":
    unittest.main()
