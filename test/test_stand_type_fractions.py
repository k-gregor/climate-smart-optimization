from unittest import TestCase
import numpy as np

import stand_type_fractions as stf


class Test(TestCase):
    def test_get_stand_type_fractions_1800_2200_all(self):
        enhanced_stf = stf.get_stand_type_fractions_1800_2200_all(file='stand_type_fracs_3_cells.txt')
        np.testing.assert_equal(len(enhanced_stf), 1203)
        np.testing.assert_equal(enhanced_stf.loc[(25.75, 50.75, 1800)].values, [0, 0, 0, 0])
        np.testing.assert_equal(enhanced_stf.loc[(25.75, 50.75, 2200)].values, [0.0754890, 0.0735410, 0.0000000, 0.1486350])

    def test_get_stand_type_fractions_1800_2200(self):
        enhanced_stf = stf.get_stand_type_fractions_1800_2200(25.75, 50.75, file='stand_type_fracs_3_cells.txt')
        np.testing.assert_equal(len(enhanced_stf), 401)
        np.testing.assert_equal(enhanced_stf.loc[(25.75, 50.75, 1800)].values, [0, 0, 0, 0])
        np.testing.assert_almost_equal(enhanced_stf.loc[(25.75, 50.75, 2200)].values, [0.0754890, 0.0735410, 0.0000000, 0.1486350], 10)

    def test_get_forest_type_fractions_1800_2200_all(self):
        enhanced_stf = stf.get_stand_type_fractions_1800_2200_all(file='nat_frac.txt')
        np.testing.assert_equal(len(enhanced_stf), 802)
        np.testing.assert_equal(enhanced_stf.loc[(-9.75, 51.75, 1800)].values, [0, 0, 0])
        np.testing.assert_almost_equal(enhanced_stf.loc[(-9.75, 51.75, 2200)]['NATURAL'], 0.2177500, 10)

