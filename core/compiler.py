import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

class ONNXCompiler:
    def __init__(self, task_id):
        self.task_id = task_id
        self.nodes = []
        self.initializers = []
        self.inputs = [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])]
        self.outputs = [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])]

    def inject_border_mask(self, input_name, mask_output_name):
        # Modern ONNX expects axes as an input, not an attribute for ReduceSum
        axes_name = f'axes_{self.task_id}'
        axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
        self.initializers.append(axes_init)

        self.nodes.append(
            helper.make_node(
                'ReduceSum',
                inputs=[input_name, axes_name],
                outputs=[mask_output_name],
                keepdims=1,
                name=f'Extract_Grid_Boundary_{self.task_id}'
            )
        )

    def add_primitive_chain(self, start_input, end_output, kernel_chain):
        current_input = start_input
        for idx, (kernel, params) in enumerate(kernel_chain):
            layer_output = f'layer_{idx}_out' if idx < len(kernel_chain) - 1 else end_output
            k_nodes, k_initializers = kernel.to_onnx_nodes(current_input, layer_output, **params)
            self.nodes.extend(k_nodes)
            for init in k_initializers:
                if not any(i.name == init.name for i in self.initializers):
                    self.initializers.append(init)
            current_input = layer_output

    def compile(self, kernel_chain, output_path):
        self.inject_border_mask('input', 'border_mask')
        self.add_primitive_chain('input', 'raw_result', kernel_chain)
        self.nodes.append(
            helper.make_node(
                'Mul',
                inputs=['raw_result', 'border_mask'],
                outputs=['output'],
                name='Final_Boundary_Enforcement'
            )
        )
        graph = helper.make_graph(self.nodes, f'task_{self.task_id}', self.inputs, self.outputs, self.initializers)
        model = helper.make_model(graph, producer_name='Moaziz-NeuroGolf')
        onnx.save(model, output_path)
        return model
