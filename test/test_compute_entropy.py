from unittest import TestCase
import pandas as pd
import compute_entropy as target


class Test(TestCase):
    def test_compute_entropy(self):
        df = pd.DataFrame({
            'cat1': [1,   1, 100,   0,   1,  50, 100, 0, 1000],
            'cat2': [1,   1, 100, 100, 100, 100,   0, 0,  100],
            'cat3': [1, 100, 100, 100, 100, 100,   0, 0,  100],
            'cat4': [1,   1, 100,   0,   1,  50,   0, 0,  100],
            'cat5': [1,   1, 100, 100, 100, 100,   0, 0,  100],
            'cat6': [1, 100, 100, 100, 100, 100,   0, 0,  100],
        })

        entr = target.compute_entropy(df)
        self.assertAlmostEqual(entr[0], 2.585, delta=0.001)  # scipy.stats.entropy([1, 1, 1, 1, 1, 1], base=2)
        self.assertAlmostEqual(entr[1], 1.159, delta=0.001)
        self.assertAlmostEqual(entr[2], 2.585, delta=0.001)  # scipy.stats.entropy([100, 100, 100, 100, 100, 100], base=2)
        self.assertAlmostEqual(entr[3],   2.0, delta=0.001)
        self.assertAlmostEqual(entr[4], 2.040, delta=0.001)
        self.assertAlmostEqual(entr[5], 2.521, delta=0.001)
        self.assertAlmostEqual(entr[6],   0.0, delta=0.001)
        self.assertAlmostEqual(entr[7],   0.0, delta=0.001)
        self.assertAlmostEqual(entr[8], 1.692, delta=0.001)

    def test_compute_entropy_all_zeros(self):
        df = pd.DataFrame({
            'cat1': [0],
            'cat2': [0],
            'cat3': [0],
            'cat4': [0],
            'cat5': [0],
            'cat6': [0],
        })

        entr = target.compute_entropy(df)
        self.assertAlmostEqual(entr[0], 0, delta=0.001)