import unittest
import numpy as np
from core.ingestion import SpectralIngestor

class TestIngestion(unittest.TestCase):
    def test_extract_objects(self):
        ingestor = SpectralIngestor()
        # Create a simple grid with two objects
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]
        ]
        objects = ingestor.extract_objects(grid)
        self.assertEqual(len(objects), 2)

        colors = [obj['color'] for obj in objects]
        self.assertIn(1, colors)
        self.assertIn(2, colors)

    def test_grid_to_graph(self):
        ingestor = SpectralIngestor()
        grid = [
            [1, 0, 2],
            [0, 0, 0],
            [0, 0, 0]
        ]
        objects, adj = ingestor.grid_to_graph(grid)
        self.assertEqual(len(objects), 2)
        self.assertEqual(adj.shape, (2, 2))
        self.assertGreater(adj[0, 1], 0)

if __name__ == '__main__':
    unittest.main()
