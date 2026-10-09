#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import HuberKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


def _expected(dist, c):
    ratio = dist / c
    return torch.where(ratio <= 1, -0.25 * dist**2, -c / 2 * dist + c**2 / 4)


class TestHuberKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return HuberKernel(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return HuberKernel(ard_num_dims=num_dims, **kwargs)

    def test_matches_closed_form(self):
        # Compare against the closed-form definition for several scales.
        a = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
        b = torch.tensor([[0.0, 1.0], [3.0, 4.0]], dtype=torch.float64)

        for c in (0.5, 2.0, 10.0):
            kernel = HuberKernel().double()
            kernel.initialize(c=c)
            kernel.eval()
            res = kernel(a, b).to_dense()
            actual = _expected(torch.cdist(a, b), torch.tensor(c, dtype=torch.float64))
            self.assertAllClose(res, actual)

    def test_diag(self):
        x = torch.randn(6, 3)
        kernel = HuberKernel()
        kernel.initialize(c=2.0)
        kernel.eval()
        full = kernel(x).to_dense()
        diag = kernel(x, diag=True)
        self.assertAllClose(diag, full.diagonal(dim1=-1, dim2=-2))

    def test_initialize_c(self):
        kernel = HuberKernel()
        kernel.initialize(c=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.c)
        self.assertLess(torch.norm(kernel.c - actual_value), 1e-5)

    def test_initialize_c_batch(self):
        kernel = HuberKernel(batch_shape=torch.Size([2]))
        c_init = torch.tensor([3.14, 4.13])
        kernel.initialize(c=c_init)
        actual_value = c_init.view_as(kernel.c)
        self.assertLess(torch.norm(kernel.c - actual_value), 1e-5)

    def test_batch_c(self):
        x = torch.randn(2, 5, 3)
        c = torch.tensor([1.0, 4.0])
        kernel = HuberKernel(batch_shape=torch.Size([2]))
        kernel.initialize(c=c)
        kernel.eval()
        res = kernel(x).to_dense()
        for i in range(2):
            single = HuberKernel()
            single.initialize(c=c[i])
            single.eval()
            self.assertAllClose(res[i], single(x[i]).to_dense())

    def test_c_gradient(self):
        x = torch.randn(5, 2)
        kernel = HuberKernel()
        kernel.initialize(c=2.0)
        kernel.eval()
        kernel(x).to_dense().sum().backward()
        self.assertIsNotNone(kernel.raw_c.grad)
        self.assertTrue(torch.isfinite(kernel.raw_c.grad).all())

    def test_linear_beyond_c(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[3.0, 4.0]])  # distance 5
        kernel = HuberKernel()
        kernel.initialize(c=4.0)
        kernel.eval()
        self.assertAlmostEqual(kernel(a, b).to_dense().item(), -2.0 * 5 + 4.0, places=5)
        kernel.initialize(c=10.0)
        self.assertAlmostEqual(kernel(a, b).to_dense().item(), -6.25, places=5)


if __name__ == "__main__":
    unittest.main()
