import unittest
import numpy as np
from kernels.gravity import GravityKernel
from kernels.flood import FloodKernel

class TestKernels(unittest.TestCase):
    def test_gravity(self):
        kernel = GravityKernel()
        grid = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        expected = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0]
        ]
        result = kernel.execute(grid)
        np.testing.assert_array_equal(result, expected)

    def test_flood(self):
        kernel = FloodKernel()
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        expected = [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1]
        ]
        result = kernel.execute(grid, color=2)
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
