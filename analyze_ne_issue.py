import pandas as pd

iter_csv = pd.read_csv('/home/konni/Documents/konni/projekte/phd/data/simulation_inputs/global_forest_ST_1871-2011_CORRECTED_1y_mask_gridlist_europe_forexclim.txt', delim_whitespace=True, iterator=True)
forest_share = pd.concat([chunk[chunk['year'] == 2010] for chunk in iter_csv])

forest_share_be = forest_share[forest_share['ForestBE'] > 0]

forest_share_be = forest_share_be.set_index(['Lon', 'Lat'])

iter_csv = pd.read_csv('/home/konni/Documents/konni/projekte/phd/data/mats_data/stormbringer.nateko.lu.se/public/mats/FOREXCLIM/ToAnja200121/190909_forest_europe_45/dens_forest.out', delim_whitespace=True, iterator=True)
dens = pd.concat([chunk[(chunk['Year'] == 2010)] for chunk in iter_csv])

dens = dens.set_index(['Lon', 'Lat'])

relevant_dens = dens[forest_share_be.index]

# no, the combination of Lat and Lon needs to be in the forest_share thing.
# relevant_dens = dens[dens['Lon'].isin(forest_share_be['Lon'].values)]
# relevant_dens = relevant_dens[relevant_dens['Lat'].isin(forest_share_be['Lat'].values)]

print(relevant_dens)

relevant_que_ile_dens = relevant_dens['Que_ile']

print(relevant_que_ile_dens)