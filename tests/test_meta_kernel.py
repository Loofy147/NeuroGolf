import unittest
import numpy as np
from core.meta_kernel import MetaKernel
from kernels.gravity import GravityKernel
from registry.manager import registry

class TestMetaKernel(unittest.TestCase):
    def setUp(self):
        registry.register(GravityKernel)
        self.meta = MetaKernel()

    def test_learn_and_solve(self):
        grid_in = [[1, 0], [0, 0]]
        grid_out = [[0, 0], [1, 0]]
        chain = [("gravity_drop", {})]

        self.meta.learn(grid_in, grid_out, chain)

        # Test on a similar grid
        test_in = [[2, 0], [0, 0]]
        # We need an example to solve zero-shot in this implementation
        result = self.meta.solve(test_in, grid_in, grid_out)

        expected = [[0, 0], [2, 0]]
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
