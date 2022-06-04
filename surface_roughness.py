import pandas as pd
from helper.chunk_filter import ChunkFilter
import numpy as np

def get_real_fpc_forest(basepath, lon_of_interest=None, lat_of_interest=None, years_of_interest=None, only_managed=False, fpc_type='fpc'):
    return get_real_fpc_forest(basepath, [(lon_of_interest, lat_of_interest)], years_of_interest, only_managed, fpc_type)


def get_real_fpc_forest(basepath, lons_lats_of_interest=None, years_of_interest=None, only_managed=False, fpc_type='fpc'):
    """
    FPC is tricky: LPJ puts out the FPC of all PFTs but only related to the are where they are 'allowed' to be active.
    Since management changes this 'allowed' area, we need to also output this area and we call it 'active_fraction_forest'
    So here we scale the fpc_forest output file such that we really have the FPCs related to the whole forest area.
    """

    chunk_filter = ChunkFilter(years_of_interest=years_of_interest, lons_lats_of_interest=lons_lats_of_interest)

    if only_managed:
        filename_active_fraction = 'active_fraction_forest.out'
        filename_fpc = fpc_type + '_forest.out'
    else:
        filename_active_fraction = 'active_fraction.out'
        filename_fpc = fpc_type + '.out'

    iter_csv = pd.read_csv(basepath + filename_active_fraction, delim_whitespace=True, iterator=True)
    active_fraction_forest = pd.concat([chunk_filter.filter_chunk(chunk) for chunk in iter_csv])
    iter_csv = pd.read_csv(basepath + filename_fpc, delim_whitespace=True, iterator=True)
    fpc_forest2 = pd.concat([chunk_filter.filter_chunk(chunk) for chunk in iter_csv])

    active_fraction_forest['Total'] = 0.0  # a0dd dummy dimension to make multiplication work
    active_fraction_forest = active_fraction_forest.set_index(['Lon', 'Lat', 'Year'])
    active_fraction_forest = active_fraction_forest.div(active_fraction_forest.C3_gr, axis=0)  # put active fraction in relation to total grass active fraction (=total forest area)
    fpc_forest2 = fpc_forest2.set_index(['Lon', 'Lat', 'Year'])

    fpc_forest2 = fpc_forest2.loc[~fpc_forest2.index.duplicated(keep='first')]
    active_fraction_forest = active_fraction_forest.loc[~active_fraction_forest.index.duplicated(keep='first')]

    if fpc_type == 'lai':
        fpc_forest2 = 1-np.exp(-0.5*fpc_forest2)

    if fpc_type == 'dens':
        fpc_forest2 = fpc_forest2.mul(crownarea)


    mul = fpc_forest2.mul(active_fraction_forest)
    if not only_managed:
        mul.drop(['Barren_sum', 'Forest_sum', 'Natural_sum'], inplace=True, axis=1)
    return mul


def get_real_cc_forest(basepath, lon_of_interest, lat_of_interest):
    return get_real_fpc_forest(basepath, lon_of_interest=lon_of_interest, lat_of_interest=lat_of_interest, fpc_type='crownarea')
