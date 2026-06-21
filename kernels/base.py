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

    def compute_cost(self, flops=0):
        """
        Compute cost based on parameters and FLOPs.
        As per ARC stress test: C_FLOPs + lambda * params
        """
        # For now, a simple placeholder
        return self.parameters + flops * 0.001
