import numpy as np
from kernels.base import BaseKernel

class GravityKernel(BaseKernel):
    @property
    def name(self):
        return "gravity_drop"

    @property
    def parameters(self):
        # 4 parameters as mentioned: direction, speed, collision_mode, padding
        return 4

    def execute(self, grid, direction='down', **kwargs):
        grid = np.array(grid)
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)

        if direction == 'down':
            for c in range(cols):
                # Get non-zero elements in column
                col_data = grid[:, c]
                non_zero = col_data[col_data != 0]
                # Place them at the bottom
                new_grid[rows - len(non_zero):, c] = non_zero

        return new_grid
