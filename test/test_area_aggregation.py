from unittest import TestCase
import analyze_lpj_output as target
import pandas as pd


class AggregationTest(TestCase):
    def test_compute_avg_val_over_gridcell_areas(self):
        albedos = pd.read_csv('example_albedo.csv', delim_whitespace=True).set_index(['Lon', 'Lat'])
        avg_albedo = target.compute_avg_val_over_gridcell_areas(albedos, 'Albedo')
        self.assertAlmostEqual(avg_albedo, 0.2544, 4)

    def test_compute_avg_val_over_gridcell_areas_duplicated_cells(self):
        albedos = pd.read_csv('example_albedo_duplicated_cells.csv', delim_whitespace=True).set_index(['Lon', 'Lat'])
        self.assertRaises(AssertionError, target.compute_avg_val_over_gridcell_areas, albedos, 'Albedo')

    def test_compute_avg_val_over_gridcell_areas_multiple_es(self):
        albedos = pd.read_csv('example_albedo_jan_and_jul.csv', delim_whitespace=True).set_index(['Lon', 'Lat'])
        avg_albedo = target.compute_avg_val_over_gridcell_areas(albedos, ['AlbedoJan', 'AlbedoJul'])
        self.assertAlmostEqual(avg_albedo['AlbedoJan'], 0.2544, 4)
        self.assertAlmostEqual(avg_albedo['AlbedoJul'], 0.1544, 4)
