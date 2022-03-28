from unittest import TestCase

import analyze_lpj_output as target
import pandas as pd
import numpy as np


class TestStuff(TestCase):

    def test_10_year_average(self):
        avg = target.get_monthly_data_accumulated_to_year_averaged_over_10_years("dummy_average_file.csv", 2010)
        np.testing.assert_allclose(avg['YearlyTotal'].values, [135.9266, 127.7048, 113.4905, 119.3979], rtol=1e-10)

        avg = target.get_monthly_data_accumulated_to_year_averaged_over_10_years("dummy_average_file.csv", 2020)
        np.testing.assert_allclose(avg['YearlyTotal'].values, [126.9655, 131.1944, 117.6291, 147.7358], rtol=1e-10)

    def test_length_of_longitude(self):
        self.assertAlmostEqual(target.compute_length_of_longitude(0), 111320, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(15), 107550, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(30), 96486, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(45), 78847, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(60), 55800, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(75), 28902, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(90), 0, delta=1)

        self.assertAlmostEqual(target.compute_length_of_longitude(-15), 107551, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(-30), 96486, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(-45), 78847, delta=2)
        self.assertAlmostEqual(target.compute_length_of_longitude(-60), 55800, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(-75), 28902, delta=1)
        self.assertAlmostEqual(target.compute_length_of_longitude(-90), 0, delta=1)


    def test_length_of_latitude(self):
        self.assertAlmostEqual(target.compute_length_of_latitude(0), 110574, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(15), 110649, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(30), 110852, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(45), 111132, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(60), 111412, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(75), 111618, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(90), 111694, delta=1)

        self.assertAlmostEqual(target.compute_length_of_latitude(-15), 110649, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(-30), 110852, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(-45), 111132, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(-60), 111412, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(-75), 111618, delta=1)
        self.assertAlmostEqual(target.compute_length_of_latitude(-90), 111694, delta=1)

    def test_get_area_for_lon_and_lat(self):
        self.assertAlmostEqual(target.get_area_for_lon_and_lat_with_formulas_specific_gc_size(45), 111133/2*78847/2, delta=100000)

    def test_compute_total_c_pool_in_giga_tons_for_single_cell(self):
        df = pd.DataFrame({'Year': [1234],
                           'Lat': [45],
                           'Lon':  [24.25],
                           'Total': [12]}).set_index(['Lon', 'Lat'])

        # 111133/2 * 78847/2 *  12 / 1000**4
        self.assertAlmostEqual(target.compute_total_c_pool_in_giga_tons_new(df, 'Total'), 0.026288, delta=0.01)

    def test_compute_total_c_pool_in_giga_tons_for_multiple_times_same_cell(self):
        df = pd.DataFrame({'Year': [1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234, 1234],
                           'Lat': [51.25, 51.25, 51.25, 51.25, 51.25, 51.25, 51.25, 51.25, 51.25, 51.25],
                           'Lon':  [24.25, 24.25, 24.25, 24.25, 24.25, 24.25, 24.25, 24.25, 24.25, 24.25],
                           'Total': [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]}
                          ).set_index(['Lon', 'Lat'])
        self.assertRaises(AssertionError, target.compute_total_c_pool_in_giga_tons_new, df, 'Total')

    def test_compute_total_c_pool_in_giga_tons_for_multiple_cells_same_year(self):
        # check https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
        df = pd.DataFrame({'Year': [1234, 1234, 1234, 1234],
                           'Lat': [30, 45, 60, 75],
                           'Lon':  [24.25, 33.25, 48.75, 60.25],  # pretty much irrelevant
                           'Total': [12, 9, 14, 14]}
                          ).set_index(['Lon', 'Lat'])

        # 30: 0.032087
        # 45: 0.019715633
        # 60: 0.021758764
        # 75: 0.01129
        self.assertAlmostEqual(target.compute_total_c_pool_in_giga_tons_new(df, 'Total'), 0.08485, delta=0.001)

    def test_time_series_cpool(self):
        df = pd.DataFrame({'Year': [1, 2, 3, 4],
                           'Lat': [75, 75, 75, 75],
                           'Lon': [24.25, 24.25, 24.25, 24.25],
                           'Total': [12, 12, 12, 16]}
                          ).set_index(['Lon', 'Lat'])

        cpool = target.get_time_series_of_total_cpool(df, 'Total')
        np.testing.assert_almost_equal(list(cpool.values), [0.00967795, 0.00967795, 0.00967795, 0.012903934], decimal=3)
