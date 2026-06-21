import unittest
from kernels.base import BaseKernel
from registry.manager import KernelRegistry

class MockKernel(BaseKernel):
    @property
    def name(self): return "mock"
    @property
    def parameters(self): return 10
    def execute(self, grid, **kwargs): return grid

class TestRegistry(unittest.TestCase):
    def test_registration(self):
        reg = KernelRegistry()
        reg.register(MockKernel)
        self.assertIn("mock", reg.list_kernels())
        kernel = reg.get_kernel("mock")
        self.assertEqual(kernel.parameters, 10)

if __name__ == '__main__':
    unittest.main()
