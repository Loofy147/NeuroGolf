from registry.manager import registry
from core.hashing import RelationalHasher
from core.compiler import ONNXCompiler

class CompositionalAssembly:
    def __init__(self, kernel_chain):
        self.kernel_chain = kernel_chain # List of (kernel_name, params)

    def execute(self, grid):
        current_grid = grid
        for kernel_name, params in self.kernel_chain:
            kernel = registry.get_kernel(kernel_name)
            if kernel:
                current_grid = kernel.execute(current_grid, **params)
        return current_grid

class MetaKernel:
    def __init__(self):
        self.hasher = RelationalHasher()
        self.knowledge_base = {} # signature -> kernel_chain

    def learn(self, input_grid, output_grid, kernel_chain):
        signature = self.hasher.compute_signature(input_grid, output_grid)
        self.knowledge_base[signature] = kernel_chain

    def solve(self, input_grid, example_input=None, example_output=None):
        """
        Solves a task by matching the signature of a provided example.
        """
        if example_input is not None and example_output is not None:
            signature = self.hasher.compute_signature(example_input, example_output)
            kernel_chain = self.knowledge_base.get(signature)
            if kernel_chain:
                assembly = CompositionalAssembly(kernel_chain)
                return assembly.execute(input_grid)
        return None

    def synthesize_onnx(self, task_id, example_input, example_output, output_path):
        """
        Translates a discovered kernel chain into a compliant ONNX file.
        """
        signature = self.hasher.compute_signature(example_input, example_output)
        kernel_chain_names = self.knowledge_base.get(signature)

        if not kernel_chain_names:
            return None

        # Convert names/params to actual kernel objects
        kernel_chain = []
        for name, params in kernel_chain_names:
            kernel = registry.get_kernel(name)
            if kernel:
                kernel_chain.append((kernel, params))

        compiler = ONNXCompiler(task_id)
        model = compiler.compile(kernel_chain, output_path)
        return model
