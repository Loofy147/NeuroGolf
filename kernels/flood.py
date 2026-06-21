import numpy as np
from kernels.base import BaseKernel
from scipy.ndimage import binary_fill_holes

class FloodKernel(BaseKernel):
    @property
    def name(self):
        return "color_flood"

    @property
    def parameters(self):
        return 2 # color, connectivity

    def execute(self, grid, color=1, **kwargs):
        grid = np.array(grid)
        # Binary mask of all objects
        mask = (grid != 0)
        # Fill holes in the mask
        filled_mask = binary_fill_holes(mask)

        # New pixels are filled with the specified color
        new_grid = np.copy(grid)
        new_grid[(filled_mask == 1) & (mask == 0)] = color

        return new_grid
