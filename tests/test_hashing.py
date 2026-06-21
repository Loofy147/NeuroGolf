import unittest
from core.hashing import RelationalHasher

class TestHashing(unittest.TestCase):
    def test_signature_consistency(self):
        hasher = RelationalHasher()
        grid1 = [[1, 0], [0, 0]]
        grid2 = [[0, 0], [1, 0]]

        sig1 = hasher.compute_signature(grid1, grid2)
        sig2 = hasher.compute_signature(grid1, grid2)

        self.assertEqual(sig1, sig2)

        # Different transformation should have different signature
        grid3 = [[1, 1], [1, 1]]
        sig3 = hasher.compute_signature(grid1, grid3)
        self.assertNotEqual(sig1, sig3)

if __name__ == '__main__':
    unittest.main()
