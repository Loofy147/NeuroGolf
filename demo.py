import numpy as np
from core.meta_kernel import MetaKernel
from kernels.gravity import GravityKernel
from kernels.flood import FloodKernel
from registry.manager import registry

def print_grid(grid, title):
    print(f"\n{title}:")
    for row in grid:
        print(" ".join(map(str, row)))

def main():
    # 1. Setup Registry
    registry.register(GravityKernel)
    registry.register(FloodKernel)

    meta = MetaKernel()

    # 2. Define a "Gravity" task
    train_in = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    train_out = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0]
    ]

    # "Learn" the transformation
    meta.learn(train_in, train_out, [("gravity_drop", {})])

    # 3. Solve a novel instance
    test_in = [
        [0, 2, 0],
        [2, 0, 0],
        [0, 0, 0]
    ]

    print("Solving 'Gravity Drop' task...")
    result = meta.solve(test_in, train_in, train_out)

    print_grid(test_in, "Input Grid")
    print_grid(result, "Output Grid")

    # 4. Define a complex "Flood + Gravity" task
    # Step 1: Flood the hole (1)
    # Step 2: Drop it
    complex_in = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    # Intermediate flooded:
    # [1, 1, 1]
    # [1, 2, 1]
    # [1, 1, 1]

    complex_out = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1] # This is a bit simplified for the demo
    ]

    # For demo, let's just show one more: Color Flood
    meta.learn(complex_in, [[1,1,1],[1,2,1],[1,1,1]], [("color_flood", {"color": 2})])

    print("\nSolving 'Color Flood' task...")
    result_flood = meta.solve(complex_in, complex_in, [[1,1,1],[1,2,1],[1,1,1]])
    print_grid(complex_in, "Input Grid")
    print_grid(result_flood, "Output Grid (Flooded)")

if __name__ == "__main__":
    main()
