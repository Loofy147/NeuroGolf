from abc import ABC, abstractmethod
import numpy as np

class BaseKernel(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def parameters(self):
        """Returns the number of parameters."""
        pass

    @abstractmethod
    def execute(self, grid, **kwargs):
        """Executes the transformation on the grid."""
        pass

    @abstractmethod
    def to_onnx_nodes(self, input_name, output_name, **kwargs):
        """
        Generates ONNX nodes and initializers for this kernel.
        Returns (nodes, initializers).
        """
        pass

    def compute_cost(self, flops=0):
        """
        Compute cost based on parameters and FLOPs.
        As per ARC stress test: C_FLOPs + lambda * params
        """
        return self.parameters + flops * 0.001
