import pandas as pd

import pandas_helper


def get_stand_type_fractions_1800_2200(lon, lat, file='/home/konni/Documents/konni/projekte/phd/data/simulation_inputs/other_inputs/global_forest_ST_1871-2011_CORRECTED_1y_mask_gridlist_europe_forexclim.txt'):
    """
    We need the stand type fractions because the cmass_wood_*_sts.out files are per stand type area and we need to adapt them.
    The fractions file only goes from 1870 to 2010, so we simply need to extend them to [1800, 2200] by copying the first and last entry repectively
    """
    iter_csv = pd.read_csv(file, delim_whitespace=True, iterator=True)
    stand_type_fractions = pd.concat([chunk[(chunk['Lat'] == lat) & (chunk['Lon'] == lon)] for chunk in iter_csv])
    stf = stand_type_fractions.rename(columns = {'year':'Year'}).set_index(['Lon', 'Lat', 'Year'])
    stf = stf.loc[~stf.index.duplicated(keep='first')]
    idx_future = pd.MultiIndex.from_tuples([(lon, lat, yr) for yr in range(2011, 2201)], names=('Lon', 'Lat', 'Year'))
    stf_future = pd.DataFrame([stf.loc[(lon, lat, 2010)]], index=idx_future, columns=stf.columns)
    idx_past = pd.MultiIndex.from_tuples([(lon, lat, yr) for yr in range(1800, 1870)], names=('Lon', 'Lat', 'Year'))
    stf_past = pd.DataFrame(0, index=idx_past, columns=stf.columns)
    return pd.concat((stf_past, stf, stf_future))


def get_stand_type_fractions_1800_2200_new(lons_lats_of_interest, min_year, max_year, file='/home/konni/Documents/konni/projekte/phd/data/simulation_inputs/other_inputs/global_forest_ST_1871-2011_CORRECTED_1y_mask_gridlist_europe_forexclim.txt_capitalized'):

    stf = pandas_helper.read_for_years(file, min_year, max_year, lons_lats_of_interest=lons_lats_of_interest)
    stf = stf.set_index(['Lon', 'Lat', 'Year'])
    stf = stf.loc[~stf.index.duplicated(keep='first')]
    res = pd.DataFrame()
    for d in stf.groupby(['Lon', 'Lat'], as_index=False):
        res = pd.concat([res, enhance_past_future(d[1], max_year)])

    return res



def enhance_past_future(stf, max_year=2200):
    lon = stf.index.get_level_values('Lon').values[0]
    lat = stf.index.get_level_values('Lat').values[0]

    idx_future = pd.MultiIndex.from_tuples([(lon, lat, yr) for yr in range(2011, max_year + 1)], names=('Lon', 'Lat', 'Year'))
    stf_future = pd.DataFrame([stf.loc[(lon, lat, 2010)]], index=idx_future, columns=stf.columns)
    idx_past = pd.MultiIndex.from_tuples([(lon, lat, yr) for yr in range(1800, 1870)], names=('Lon', 'Lat', 'Year'))
    stf_past = pd.DataFrame(0, index=idx_past, columns=stf.columns)

    stf_combined = pd.concat((stf_past, stf, stf_future))
    return stf_combined


def get_stand_type_fractions_1800_2200_all(file='/home/konni/Documents/konni/projekte/phd/data/simulation_inputs/other_inputs/global_forest_ST_1871-2011_CORRECTED_1y_mask_gridlist_europe_forexclim.txt'):
    """
    We need the stand type fractions because the cmass_wood_*_sts.out files are per stand type area and we need to adapt them.
    The fractions file only goes from 1870 to 2010, so we simply need to extend them to [1800, 2200] by copying the first and last entry repectively
    """
    iter_csv = pd.read_csv(file, delim_whitespace=True, iterator=True)
    stand_type_fractions = pd.concat([chunk for chunk in iter_csv])
    stf = stand_type_fractions.rename(columns={'year': 'Year'})
    stf = stf.set_index(['Lon', 'Lat', 'Year'])
    stf = stf.loc[~stf.index.duplicated(keep='first')]

    res = pd.DataFrame()
    for d in stf.groupby(['Lon', 'Lat'], as_index=False):
        res = pd.concat([res, enhance_past_future(d[1])])

    return res


def get_stand_type_fractions_all(min_year=2000, max_year=2010, file='/home/konni/Documents/konni/projekte/phd/data/simulation_inputs/other_inputs/global_forest_ST_1871-2011_CORRECTED_1y_mask_gridlist_europe_forexclim.txt'):
    """
    We need the stand type fractions because the cmass_wood_*_sts.out files are per stand type area and we need to adapt them.
    The fractions file only goes from 1870 to 2010, so we simply need to extend them to [1800, 2200] by copying the first and last entry repectively
    """
    iter_csv = pd.read_csv(file, delim_whitespace=True, iterator=True)
    stand_type_fractions = pd.concat([chunk[(chunk['year'] >= min_year) & (chunk['year'] <= max_year)] for chunk in iter_csv])
    stf = stand_type_fractions.rename(columns={'year': 'Year'})
    stf = stf.set_index(['Lon', 'Lat', 'Year'])
    stf = stf.loc[~stf.index.duplicated(keep='first')]

    return stf
