from unittest import TestCase

import optimization_europe_aggregation as target
import pandas as pd

class TestBiodivIndicatorTest(TestCase):

    def test_biodiv_indicator_for_single_gc(self):

        portfolios = {
            'optimized' : pd.DataFrame({
                'Lon': [5.5],
                'Lat': [7.5],
                'base': [0.1],
                'toCoppice': [0.4],
                'unmanaged': [0.5],
                'forest_frac': [0.2],
                'has_forest': [True],
            }).set_index(['Lon', 'Lat'])
        }
        rcp26_simulations = dict(
            base='./resources/base_rcp26/',
            toCoppice='./resources/toCoppice_rcp26/',
            unmanaged='./resources/unmanaged_rcp26/'
        )

        simulations = {'rcp26': rcp26_simulations}

        # old trees: coppice 0 | base (1.3-0.8)/(1.8-0.8) = 0.5 | unmanaged 1
        # cwd: coppice 0 | base (1-0.5)/(3-0.5) = 0.2 | unmanaged 1
        # diversity: base 0 | coppice 1 | unmanaged 1
        # combined: unmanaged 1, coppice 1/3, base 0.233333
        # optimized portfolio: 0.5 * 1 + 0.1 * 0.233333 + 0.4 * 0.33333 = 0.65666
        indicator_value = target.get_combined_biodiv_indicator(portfolios, [(5.5, 7.5)], 'rcp26', simulations, ['base', 'unmanaged', 'toCoppice'], min_year=2100, max_year=2100)
        indicator_value_at_gc = indicator_value.loc[(5.5, 7.5)]

        self.assertAlmostEqual(indicator_value_at_gc['base'], 0.233333, delta=0.0001)
        self.assertAlmostEqual(indicator_value_at_gc['toCoppice'], 1/3, delta=0.0001)
        self.assertAlmostEqual(indicator_value_at_gc['unmanaged'], 1, delta=0.0001)
        self.assertAlmostEqual(indicator_value_at_gc['optimized'], 0.65666, delta=0.0001)

    def test_biodiv_indicator_for_single_gc_nan_in_values(self):

        portfolios = {
            'optimized' : pd.DataFrame({
                'Lon': [5.5],
                'Lat': [7.5],
                'base': [0.1],
                'toCoppice': [0.4],
                'unmanaged': [0.5],
                'forest_frac': [0.2],
                'has_forest': [True],
            }).set_index(['Lon', 'Lat'])
        }
        rcp26_simulations = dict(
            base='./resources/toCoppice_rcp26/',
            toCoppice='./resources/toCoppice_rcp26/',
            unmanaged='./resources/unmanaged_rcp26/'
        )

        simulations = {'rcp26': rcp26_simulations}

        # old trees: coppice 0 | base 0 | unmanaged 1
        # cwd: coppice 0 | base 0 | unmanaged 1
        # diversity: coppice 0 | base 0 | unmanaged 0
        # combined: unmanaged 2/3, coppice 0, base 0
        # optimized portfolio: 0.5 * 1 + 0.1 * 0 + 0.4 * 0 = 0.5
        indicator_value = target.get_combined_biodiv_indicator(portfolios, [(5.5, 7.5)], 'rcp26', simulations, ['base', 'unmanaged', 'toCoppice'], min_year=2100, max_year=2100)
        indicator_value_at_gc = indicator_value.loc[(5.5, 7.5)]

        self.assertAlmostEqual(indicator_value_at_gc['base'], 0, delta=0.0001)
        self.assertAlmostEqual(indicator_value_at_gc['toCoppice'], 0, delta=0.0001)
        self.assertAlmostEqual(indicator_value_at_gc['unmanaged'], 2/3, delta=0.0001)
        self.assertAlmostEqual(indicator_value_at_gc['optimized'], 1/3, delta=0.0001)