import numpy as np
from core.ingestion import SpectralIngestor

class RelationalHasher:
    def __init__(self):
        self.ingestor = SpectralIngestor()

    def compute_signature(self, input_grid, output_grid):
        """
        Computes a scale-invariant spectral signature for the transformation.
        S_task = f( lambda_max(A_in -> A_out) + Delta Trace(X) )
        """
        objs_in, adj_in = self.ingestor.grid_to_graph(input_grid)
        objs_out, adj_out = self.ingestor.grid_to_graph(output_grid)

        # Spectral signature components
        # 1. Largest eigenvalue of adjacency matrix delta
        if adj_in.shape == adj_out.shape and adj_in.size > 0:
            diff = adj_out - adj_in
            eigvals = np.linalg.eigvals(diff)
            lambda_max = np.max(np.abs(eigvals))
        else:
            # If topology changed (number of objects), use a different metric
            lambda_max = abs(len(objs_out) - len(objs_in))

        # 2. Trace delta (feature matrix change - simplified to color distribution)
        colors_in = sorted([o['color'] for o in objs_in])
        colors_out = sorted([o['color'] for o in objs_out])

        # 3. Geometric delta
        centroid_in = np.mean([o['coords'].mean(axis=0) for o in objs_in], axis=0) if objs_in else [0, 0]
        centroid_out = np.mean([o['coords'].mean(axis=0) for o in objs_out], axis=0) if objs_out else [0, 0]
        geo_delta = np.linalg.norm(np.array(centroid_out) - np.array(centroid_in))

        # Combine into a robust signature hash
        signature_tuple = (
            round(float(lambda_max), 4),
            tuple(colors_in),
            tuple(colors_out),
            round(float(geo_delta), 4)
        )
        return str(hash(signature_tuple))
