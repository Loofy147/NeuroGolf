import numpy as np
from registry.manager import registry

class SimpleSolver:
    def __init__(self, meta_kernel):
        self.meta = meta_kernel

    def discover_and_learn(self, task_data):
        """
        Tries all registered kernels to find a match for the task.
        In a real scenario, this would be a combinatorial search.
        """
        train_pairs = task_data.get('train', [])
        if not train_pairs:
            return False

        # Try single kernels first
        for kernel_name in registry.list_kernels():
            kernel = registry.get_kernel(kernel_name)

            # For simplicity, we try a few common parameter sets
            param_grid = [{}]
            if kernel_name == "color_flood":
                param_grid = [{"color": c} for c in range(1, 10)]

            for params in param_grid:
                match = True
                for pair in train_pairs:
                    inp = np.array(pair['input'])
                    out = np.array(pair['output'])

                    try:
                        # Rescale grid if necessary for internal execution
                        # but ARC grids are already 30x30 max
                        result = kernel.execute(inp, **params)
                        if not np.array_equal(result, out):
                            match = False
                            break
                    except:
                        match = False
                        break

                if match:
                    # Found a solution!
                    self.meta.learn(train_pairs[0]['input'], train_pairs[0]['output'], [(kernel_name, params)])
                    return True

        return False
