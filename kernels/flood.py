import numpy as np
from kernels.base import BaseKernel
from scipy.ndimage import binary_fill_holes
from onnx import helper, TensorProto, numpy_helper

class FloodKernel(BaseKernel):
    @property
    def name(self):
        return "color_flood"

    @property
    def parameters(self):
        return 9 # 3x3 kernel per color? Let's say 100 for now.

    def execute(self, grid, color=1, **kwargs):
        grid = np.array(grid)
        mask = (grid != 0)
        filled_mask = binary_fill_holes(mask)
        new_grid = np.copy(grid)
        new_grid[(filled_mask == 1) & (mask == 0)] = color
        return new_grid

    def to_onnx_nodes(self, input_name, output_name, color=1, **kwargs):
        # Implementation for morph dilation to fill areas
        nodes = []
        initializers = []

        # Shared 3x3 weights for dilation
        weights = np.ones((10, 10, 3, 3), dtype=np.float32)
        initializer = numpy_helper.from_array(weights, name="flood_dilation_weights")
        initializers.append(initializer)

        # Simple placeholder logic:
        # For demo, let's just use Identity for now as Flood is more complex
        nodes.append(helper.make_node('Identity', [input_name], [output_name]))

        return nodes, initializers
