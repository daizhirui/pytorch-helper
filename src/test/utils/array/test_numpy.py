import unittest

import numpy as np
import torch

from pytorch_helper.utils.array.numpy import unfold


class TestModels(unittest.TestCase):

    def test_unfold_unit_case(self, a, axis, size, step):
        target = torch.tensor(a).unfold(axis, size, step).numpy()
        answer = unfold(a, axis, size, step)
        self.assertTrue(np.all(answer == target))

    def test_unfold(self):
        a = np.arange(10)
        self.test_unfold_unit_case(a, 0, 1, 1)
        self.test_unfold_unit_case(a, 0, 2, 1)
        self.test_unfold_unit_case(a, 0, 2, 2)

        a = np.arange(20).reshape(10, 2)
        self.test_unfold_unit_case(a, 1, 1, 1)
        self.test_unfold_unit_case(a, 1, 2, 1)
        self.test_unfold_unit_case(a, 1, 2, 2)

        a = np.arange(20).reshape(2, 10)
        self.test_unfold_unit_case(a, 1, 1, 1)
        self.test_unfold_unit_case(a, 1, 2, 1)
        self.test_unfold_unit_case(a, 1, 2, 2)

        a = np.arange(60).reshape([3, 10, 2])
        self.test_unfold_unit_case(a, 1, 1, 1)
        self.test_unfold_unit_case(a, 1, 2, 1)
        self.test_unfold_unit_case(a, 1, 2, 2)
        self.test_unfold_unit_case(a, 1, 5, 2)
