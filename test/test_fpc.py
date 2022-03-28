from unittest import TestCase
import surface_roughness as target
import numpy as np


class FpcTest(TestCase):
    def test_get_fpc_complete(self):
        fpc = target.get_real_fpc_forest('./', only_managed=True)
        self.assertEqual(4, len(fpc))
        np.testing.assert_almost_equal([0.1*0.4/0.6, 0.1*0.4/0.6, 0.1*0.4/0.8, 0.1*0.5/0.8], fpc['Pin_hal'])
        np.testing.assert_almost_equal([0, 0, 0.3*0.4/0.8, 0.3*0.5/0.8], fpc['Pin_syl'])
        np.testing.assert_almost_equal([0.3*0.2/0.6, 0.3*0.2/0.6, 0, 0], fpc['Que_rob'])

        # used do do stupid things with multiplying and taking the root, leading to errors. Checking here that negative Lons or Lats remain negative.
        self.assertEqual(-0.25, fpc.index.get_level_values('Lon')[0])

    def test_get_fpc_for_gridcell_only(self):
        fpc = target.get_real_fpc_forest('./', lons_lats_of_interest=[(-1.11, 38.75)], only_managed=True)
        self.assertEqual(2, len(fpc))
        np.testing.assert_almost_equal([0.1*0.4/0.8, 0.1*0.5/0.8], fpc['Pin_hal'])
        fpc = target.get_real_fpc_forest('./', lons_lats_of_interest=[(-0.25, 38.75)], only_managed=True)
        np.testing.assert_almost_equal([0.1*0.4/0.6, 0.1*0.4/0.6], fpc['Pin_hal'])
        self.assertEqual(2, len(fpc))

    def test_get_fpc_for_year_only(self):
        fpc = target.get_real_fpc_forest('./', years_of_interest=[1800], only_managed=True)
        self.assertEqual(2, len(fpc))
        np.testing.assert_almost_equal([0.1*0.4/0.6, 0.1*0.4/0.8], fpc['Pin_hal'])
        fpc = target.get_real_fpc_forest('./', years_of_interest=[1801], only_managed=True)
        self.assertEqual(2, len(fpc))
