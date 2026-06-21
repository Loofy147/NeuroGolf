import os
import json
import zipfile
import numpy as np
from core.meta_kernel import MetaKernel
from core.solver import SimpleSolver
from kernels.gravity import GravityKernel
from kernels.flood import FloodKernel
from registry.manager import registry

def main():
    # 1. Setup
    registry.register(GravityKernel)
    registry.register(FloodKernel)
    meta = MetaKernel()
    solver = SimpleSolver(meta)

    submission_dir = "submission"
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    # 2. Iterate through tasks (demonstration)
    task_files = [f for f in os.listdir('.') if f.startswith('task') and f.endswith('.json')]

    print(f"Processing {len(task_files)} tasks...")

    solved_count = 0
    for task_file in sorted(task_files):
        task_id = task_file.replace('.json', '')
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        if solver.discover_and_learn(task_data):
            output_path = os.path.join(submission_dir, f"{task_id}.onnx")
            # We use Float16 for final submission to maximize score
            meta.synthesize_onnx(task_id, task_data['train'][0]['input'], task_data['train'][0]['output'], output_path, precision='float16')
            print(f"  [+] Solved {task_id}")
            solved_count += 1

    # 3. Create ZIP
    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    print(f"\nFinalized submission.zip with {solved_count} models.")

if __name__ == "__main__":
    main()
