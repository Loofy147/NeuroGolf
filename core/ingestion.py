import numpy as np
from scipy.ndimage import label

class SpectralIngestor:
    def __init__(self, background_color=0):
        self.background_color = background_color

    def extract_objects(self, grid):
        """
        Identifies cohesive objects in the grid.
        Returns a list of objects, where each object is a dictionary containing:
        - color: integer color of the object
        - mask: binary mask of the object
        - coords: list of (r, c) coordinates
        """
        grid = np.array(grid)
        objects = []

        # We can extract objects by color
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color == self.background_color:
                continue

            color_mask = (grid == color)
            # Use 8-connectivity for ARC objects
            structure = np.ones((3, 3), dtype=int)
            labeled_array, num_features = label(color_mask, structure=structure)

            for i in range(1, num_features + 1):
                mask = (labeled_array == i)
                coords = np.argwhere(mask)
                objects.append({
                    'color': int(color),
                    'mask': mask,
                    'coords': coords,
                    'bbox': (coords.min(axis=0), coords.max(axis=0))
                })

        return objects

    def grid_to_graph(self, grid):
        """
        Maps the grid to a relational graph.
        """
        objects = self.extract_objects(grid)
        num_objects = len(objects)
        adj_matrix = np.zeros((num_objects, num_objects))

        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                # Simple spatial relations: distance between centroids or bounding boxes
                dist = self._compute_distance(objects[i], objects[j])
                adj_matrix[i, j] = dist
                adj_matrix[j, i] = dist

        return objects, adj_matrix

    def _compute_distance(self, obj1, obj2):
        # Euclidean distance between centroids
        c1 = obj1['coords'].mean(axis=0)
        c2 = obj2['coords'].mean(axis=0)
        return np.linalg.norm(c1 - c2)
