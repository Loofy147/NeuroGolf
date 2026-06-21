import numpy as np
from kernels.base import BaseKernel
from onnx import helper, TensorProto, numpy_helper

class GravityKernel(BaseKernel):
    @property
    def name(self):
        return "gravity_drop"

    @property
    def parameters(self):
        return 900

    def execute(self, grid, direction='down', **kwargs):
        grid = np.array(grid)
        rows, cols = grid.shape
        new_grid = np.zeros_like(grid)
        if direction == 'down':
            for c in range(cols):
                col_data = grid[:, c]
                non_zero = col_data[col_data != 0]
                new_grid[rows - len(non_zero):, c] = non_zero
        return new_grid

    def to_onnx_nodes(self, input_name, output_name, direction='down', **kwargs):
        nodes = []
        initializers = []
        weights = np.zeros((10, 10, 3, 3), dtype=np.float32)
        for color in range(1, 10):
            weights[color, color, 1, 1] = 1.0
            weights[color, 0, 2, 1] = -1.0
            weights[color, color, 0, 1] = 1.0
            weights[color, 0, 1, 1] = 1.0

        initializer = numpy_helper.from_array(weights, name="gravity_shared_weights")
        initializers.append(initializer)

        # Clip inputs
        min_val_name = "clip_min"
        max_val_name = "clip_max"
        initializers.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=min_val_name))
        initializers.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=max_val_name))

        current_input = input_name
        for step in range(30):
            conv_out = f'gravity_step_{step}'
            clamped_out = f'gravity_clamp_{step}' if step < 29 else output_name

            nodes.append(helper.make_node(
                'Conv',
                inputs=[current_input, 'gravity_shared_weights'],
                outputs=[conv_out],
                pads=[1, 1, 1, 1],
                name=f'Gravity_Step_{step}'
            ))

            nodes.append(helper.make_node(
                'Clip',
                inputs=[conv_out, min_val_name, max_val_name],
                outputs=[clamped_out],
                name=f'Gravity_Clip_{step}'
            ))
            current_input = clamped_out

        return nodes, initializers
