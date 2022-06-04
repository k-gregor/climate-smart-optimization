import numpy as np
from enum import Enum
import surface_roughness as sr
import pandas as pd

import analyze_lpj_output as analysis

import pandas_helper as ph
from compute_entropy import compute_entropy
from constants import months
from optimization_plots import NoManagementFeasibleError

SLOW_POOL_DECAY_RATE = 24 / 25

MIN_FPC_DEF_FOREST = 0.1


class OptimizationType(Enum):
    SINGLE_RCP = 1
    MINMAX = 2
    ZEROMAX = 3
    ZSCORE = 4


def normalize_for_uncertainty_scenarios(rcps, scores, lower_is_better=False):
    """
    Currently have wrong order of axis like below f        fpc = sr.get_real_fpc_forest('./', only_managed=True)
        snow = pd.read_csv('snow_matching_fpc_forest.out', delim_whitespace=True).set_index(['Lon', 'Lat', 'Year'])

        snow = snow.loc[~snow.index.duplicated(keep='first')]

        albedo = target.get_forest_area_albedo(fpc, snow)or legacy reasons
    scores: dict[simulation] --> [values per rcp]
    If we stick with this optimization with the rcps as uncertainty scenarios, we should switch the order to avoid bugs
    """

    # first, convert the scores map of the current variable (simulation --> [value_rcp1, value_rcp2, ...] )
    # to rcp --> [value_simulation1, value_simulation_2]
    # because we want to normalize the values inside an rcp. note that values across rcps should not be mixed, they are the different points in our uncertainty space.
    normalized_scores_per_rcp = {}
    for rcp in rcps:
        normalized_scores_per_rcp[rcp] = []

    for simulation in scores.keys():
        for idx, rcp in enumerate(rcps):
            normalized_scores_per_rcp[rcp].append(scores[simulation][idx])

    # normalize for each rcp separately because they are disjoint futures and should not be mixed together.
    for rcp in rcps:

        # Unlike python3, python2 will still do integer division, e.g. 1/2=0 so just to avoid anyone using this with python2 and getting strange results, cast one variable to float.
        max_score = float(np.max(normalized_scores_per_rcp[rcp]))
        min_score = np.min(normalized_scores_per_rcp[rcp])

        if lower_is_better:
            if min_score == max_score:
                normalized_scores_per_rcp[rcp] = np.array(normalized_scores_per_rcp[rcp]) * 0
            else:
                normalized_scores_per_rcp[rcp] = (max_score - np.array(normalized_scores_per_rcp[rcp])) / (max_score - min_score)
        else:
            if min_score == max_score:
                normalized_scores_per_rcp[rcp] = np.array(normalized_scores_per_rcp[rcp]) * 0
            else:
                normalized_scores_per_rcp[rcp] = (np.array(normalized_scores_per_rcp[rcp]) - min_score) / (max_score - min_score)

    # at this point, we have normalizes_scores_per_rcp, so 1 array of length n_similations of normalized values per rcp.
    return normalized_scores_per_rcp


def get_mean_over_rcps(dataframe: pd.DataFrame):
    return dataframe.mean()


def get_1sem_worst_case(dataframe: pd.DataFrame):
    return dataframe.mean() - dataframe.sem()


def aggregation_mean_lambda(min_year, max_year):
    return lambda dataframe, field: aggregation_mean(dataframe, field, min_year, max_year)


def aggregation_mean(dataframe: pd.DataFrame, field, min_year, max_year):
    return dataframe[field].loc[(dataframe['Year'] >= min_year) & (dataframe['Year'] <= max_year)].mean()


def get_es_vals_new(rcps,
                    simulation_paths,
                    simulation_names,
                    lon, lat,
                    used_variables,
                    future_year1,
                    future_year2,
                    optimizationType=OptimizationType.MINMAX,
                    boundary_simulations=[],
                    rounding=False,
                    variables_for_additional_constraints=[],
                    discounting=True):
    """
    :param simulation_names: you can enter a subset of management strategies. If left empty, all are used.
    :param boundary_simulations: additional simulations that will not be part of the final solution but may be used in the normalization process
    """

    assert set(simulation_names).isdisjoint(set(
        boundary_simulations)), "used_simulations and boundary_simulations share elements!"  # should not matter for the optimization, but this check could prevent from accidentaly passing wrong lists.

    all_scores_dict_u = {}

    all_scores_dict_raw = {}

    et = {}
    swp = {}
    harvest = {}
    hlp = {}
    albedo_jul = {}
    albedo_jan = {}
    csequestration = {}
    vegc = {}
    biodiversity_cwd = {}
    biodiversity_big_trees = {}
    biodiversity_size_diversity = {}
    surface_roughness = {}
    forest_fpc = {}
    mitigation = {}

    infeasible_forest_types = set()

    compute_biodiv = True in (ele.startswith('biodiv') for ele in used_variables)

    aggregation = aggregation_mean_lambda(future_year1, future_year2)

    for idx, simulation in enumerate(simulation_names + boundary_simulations):
        et[simulation] = []
        swp[simulation] = []
        harvest[simulation] = []
        hlp[simulation] = []
        csequestration[simulation] = []
        albedo_jul[simulation] = []
        albedo_jan[simulation] = []
        vegc[simulation] = []
        biodiversity_cwd[simulation] = []
        biodiversity_big_trees[simulation] = []
        biodiversity_size_diversity[simulation] = []
        surface_roughness[simulation] = []
        forest_fpc[simulation] = []
        mitigation[simulation] = []
        for rcp in rcps:
            basepath = simulation_paths[rcp][simulation]

            # need to read from 1990 to get present value too for feasibility checks
            fpc = get_fpc(basepath, 1990, future_year2, [(lon, lat)])

            forest_fpc[simulation].append(aggregation(fpc, 'forest_fpc'))

            # Here we check if _converted_ stands yield an FPC of >=10%.
            # If not, we do not allow a conversion to this type of forest.
            if simulation.startswith('to'):
                future_fpc = fpc[(fpc['Year'] >= future_year1) & (fpc['Year'] <= future_year2)].mean()
                tree_type = simulation[2:].lower() if simulation != 'toCoppice' else 'bd'
                if future_fpc[tree_type] < MIN_FPC_DEF_FOREST:
                    print(rcp, ': Converting to', simulation, 'not sensible, because converted stands will have less than 10% FPC. Removing this management option from the optimization.')
                    infeasible_forest_types.add(idx)

            et[simulation].append(aggregation(get_forest_et(basepath, future_year1, future_year2, [(lon, lat)]), 'YearlyTotal'))
            swp[simulation].append(aggregation(get_forest_swp(basepath, future_year1, future_year2, [(lon, lat)]), 'min'))

            # need to read in more as we want to get present day values as well.
            cpool = get_cpool(basepath, 1990, future_year2, [(lon, lat)])
            csequestration[simulation].append(aggregation(cpool, 'Total'))
            vegc[simulation].append(aggregation(cpool, 'VegC'))
            # need to read in more as we want to get present day values as well.
            fluxes = get_fluxes_with_new_harvests(basepath, 1990, future_year2, [(lon, lat)])
            hlp[simulation].append(aggregation(fluxes, 'slow_harv'))

            albedo_jul[simulation].append(aggregation(get_albedo(basepath, future_year1, future_year2, [(lon, lat)]), 'albedo_jul'))
            albedo_jan[simulation].append(aggregation(get_albedo(basepath, future_year1, future_year2, [(lon, lat)]), 'albedo_jan'))

            stem_harvests_m3_per_ha = get_harvests_via_species_file(basepath, 1800, 2200, lons_lats_of_interest=[(lon, lat)], residuals=False)
            harvest[simulation].append(aggregation(stem_harvests_m3_per_ha, 'total_harv_m3_wood_per_ha'))

            should_discount = rcp if discounting else None
            mitigationvals = get_new_total_mitigation(fluxes, cpool, discounting=should_discount)
            mitigation[simulation].append(aggregation(mitigationvals, 'total_mitigation'))

            if compute_biodiv:
                biodiv_size = get_biodiversity_tree_sizes(basepath, future_year1, future_year2, [(lon, lat)])
                biodiv_cwd = get_biodiversity_cwd(basepath, future_year1, future_year2, [(lon, lat)])
                biodiversity_cwd[simulation].append(aggregation(biodiv_cwd, 'ForestCWD'))
                biodiversity_big_trees[simulation].append(aggregation(biodiv_size, 'thick_trees'))
                biodiversity_size_diversity[simulation].append(aggregation(biodiv_size, 'size_diversity'))

            surface_roughness[simulation].append(aggregation(get_surface_roughness(basepath, future_year1, future_year2, [(lon, lat)]), 'z0'))

    add_constraint_base_values = dict(
        csequestration=cpool[cpool['Year'] == 2010]['Total'].values[0],
        hlp=fluxes[(fluxes['Year'] >= 1990) & (fluxes['Year'] <= 2010)]['slow_harv'].mean(),
        vegc=0.5 * cpool[(cpool['Year'] >= 1990) & (cpool['Year'] <= 2010)]['VegC'].mean(),
        harvest=stem_harvests_m3_per_ha[(stem_harvests_m3_per_ha['Year'] >= 1990) & (stem_harvests_m3_per_ha['Year'] <= 2010)]['total_harv_m3_wood_per_ha'].mean(),
        forest_fpc=fpc[(fpc['Year'] >= 1990) & (fpc['Year'] <= 2010)]['forest_fpc'].mean()
    )

    all_scores_dict_raw['et'] = et
    all_scores_dict_raw['harvest'] = harvest
    all_scores_dict_raw['hlp'] = hlp
    all_scores_dict_raw['csequestration'] = csequestration
    all_scores_dict_raw['vegc'] = vegc
    all_scores_dict_raw['albedo_jan'] = albedo_jan
    all_scores_dict_raw['albedo_jul'] = albedo_jul
    all_scores_dict_raw['biodiversity_cwd'] = biodiversity_cwd
    all_scores_dict_raw['biodiversity_big_trees'] = biodiversity_big_trees
    all_scores_dict_raw['biodiversity_size_diversity'] = biodiversity_size_diversity
    all_scores_dict_raw['surface_roughness'] = surface_roughness
    all_scores_dict_raw['swp'] = swp
    all_scores_dict_raw['forest_fpc'] = forest_fpc
    all_scores_dict_raw['mitigation'] = mitigation

    feasible_managements = []
    feasible_managements_ids = []
    for idx, sim in enumerate(simulation_names):
        is_feasible = idx not in infeasible_forest_types
        for idxr, rcp in enumerate(rcps):
            # print(sim, rcp, all_scores_dict_raw['forest_fpc'][sim][idxr], 'compare to', add_constraint_base_values['forest_fpc'])
            if all_scores_dict_raw['forest_fpc'][sim][idxr] < MIN_FPC_DEF_FOREST:
                print(sim + ' infeasible for ' + rcp + ' since fpc=' + "{:.3f}".format(all_scores_dict_raw['forest_fpc'][sim][idxr]) + ' compared to ' + "{:.3f}".format(min(add_constraint_base_values['forest_fpc'], MIN_FPC_DEF_FOREST)))
                infeasible_forest_types.add(idx)
                is_feasible = False
        if is_feasible:
            feasible_managements.append(sim)
            feasible_managements_ids.append(idx)

    print('infeasible managements', infeasible_forest_types)
    print('feasible managements', feasible_managements)
    print('feasible managements ids', feasible_managements_ids)

    for inf_man in infeasible_forest_types:
        for varrr in all_scores_dict_raw.keys():
            all_scores_dict_raw[varrr].pop(simulation_names[inf_man], None)

    if not feasible_managements:
        raise NoManagementFeasibleError('No management feasible!')


    for variable in used_variables + [x for x in variables_for_additional_constraints if x not in used_variables]:
        lower_is_better = True if variable in ['water_avail'] else False
        if variable == 'biodiversity_combined':
            biodiversity_cwd_norm = normalize_for_uncertainty_scenarios(rcps, all_scores_dict_raw['biodiversity_cwd'], lower_is_better=lower_is_better)
            biodiversity_big_trees_norm = normalize_for_uncertainty_scenarios(rcps, all_scores_dict_raw['biodiversity_big_trees'], lower_is_better=lower_is_better)
            biodiversity_size_diversity_norm = normalize_for_uncertainty_scenarios(rcps, all_scores_dict_raw['biodiversity_size_diversity'], lower_is_better=lower_is_better)

            biodiversity_combined_norm = {}
            for rcp in rcps:
                biodiversity_combined_norm[rcp] = np.mean(pd.DataFrame([biodiversity_big_trees_norm,biodiversity_size_diversity_norm,biodiversity_cwd_norm])[rcp].values)

            all_scores_dict_u[variable] = biodiversity_combined_norm
        else:
            all_scores_dict_u[variable] = normalize_for_uncertainty_scenarios(rcps, all_scores_dict_raw[variable], lower_is_better=lower_is_better)

    es_vals_u = np.zeros((len(used_variables) * len(rcps), len(feasible_managements)))

    print('additional_constraints', variables_for_additional_constraints, 'rcps', len(rcps))

    additional_constraints = dict(
        lhs=np.zeros((len(variables_for_additional_constraints) * len(rcps), len(feasible_managements))),
        rhs=np.zeros(0)
    )
    for es in variables_for_additional_constraints:
        # wo do not need to normalize the additional constraints.
        # they will look like: 12.4w_base + 13.5w_toNe + 18.2w_baseRefrain >= 15.2
        # 15.2 is the value of 2010 for example, and this enforces (in this example) that we have to take some baseRefrain, otherwise we will never reach this value.
        # it does not go into the maximization, hence it need not be normalized
        add_const = np.tile(add_constraint_base_values[es], len(rcps))
        additional_constraints['rhs'] = np.concatenate((additional_constraints['rhs'], add_const))

    idx = 0
    idx2 = 0
    for es in used_variables:
        vall_u = all_scores_dict_u[es]  # vall is a dict of rcps now

        for rcp in rcps:
            es_vals_u[idx, :] = vall_u[rcp][:len(feasible_managements)]
            idx += 1

        idx2 += 1

    idx_add = 0
    for es in variables_for_additional_constraints:
        for idx2, rcp in enumerate(rcps):
            raw_vals_of_rcp = []
            for idx, simulation in enumerate(feasible_managements):
                raw_vals_of_rcp.append(all_scores_dict_raw[es][simulation][idx2])

            additional_constraints['lhs'][idx_add, :] = raw_vals_of_rcp
            idx_add += 1

    delete_rows = []
    for idx, constraint in enumerate(additional_constraints['lhs']):
        if sum(constraint) == 0:
            delete_rows.append(idx)
    if delete_rows:
        print('warning, additional contraint empty, deleting it.')
        additional_constraints['lhs'] = np.delete(additional_constraints['lhs'], delete_rows, axis=0)
        additional_constraints['rhs'] = np.delete(additional_constraints['rhs'], delete_rows)

    if optimizationType == OptimizationType.MINMAX:
        return es_vals_u, all_scores_dict_u, all_scores_dict_raw, additional_constraints, list(infeasible_forest_types), feasible_managements
    else:
        raise ValueError('Specified an optimization method that was not implemented')


def get_fpc(basepath, min_year, max_year, lons_lats_of_interest, only_managed=True, fpc_type='crownarea'):
    # need to take only managed FPC here since only this tells us whether the management leads to sustained forest at the end of the century
    fpc_forest_real = sr.get_real_fpc_forest(basepath, years_of_interest=[min_year, max_year], lons_lats_of_interest=lons_lats_of_interest, only_managed=only_managed, fpc_type=fpc_type)
    fpc_forest_real = analysis.normalize_fpc(fpc_forest_real)
    fpc_forest_real['forest_fpc'] = fpc_forest_real['deciduous'] + fpc_forest_real['evergreen']
    return fpc_forest_real.reset_index()


to_t_dry_biomass_per_ha = 10 / 0.47  # convert kgC/m2 to t dry mass / ha @McGroddy2004


def compute_mitigation(cflux, discounting_factors, base_year = 2010):

    cflux = cflux.set_index('Year')

    #note: cflux['slow_h'] is accounted for in C stocks!!

    #Harvest = H_R_atm + H_S_atm
    #Lu_ch = LU_R_atm + Lu_S_atm
    # harvest residues and parts of stem harvests are used as fuelwood
    harvested_fuel_wood = cflux['H_R_atm'] + cflux['LU_R_atm'] + (cflux['H_S_atm'] + cflux['LU_S_atm']) * 0.305
    harvested_short_pool = (cflux['H_S_atm'] + cflux['LU_S_atm']) * (1-0.305)
    harvests_for_medium_and_slow_pool = (cflux['H_slow'] + cflux['Slow_LU'])

    # the Knauf value does not contain end of life burning of material in the substitution factor!
    # 1.5: wood usage in Germany Knauf2015, cf. factors on average 2.1 for Sathre2010
    # now also account for landfilling, 1.1 from Sathre2010 for landfilled wood products, 23% of waste is landfilled in Europe
    material_substitution_factor = 0.77 * 1.5 + 0.23 * 1.1
    cflux['material_substitution'] = (harvests_for_medium_and_slow_pool) * material_substitution_factor

    # now, all material usage has been accounted for including end-of-life, except the 77% of products that are not landfilled, for those we assume energy recovery
    decayed_medium_and_long_pool = cflux['Slow_h']
    energy_substitution_factor = 0.67  # Knauf2015
    cflux['fuel_substitution']  = (harvested_fuel_wood + harvested_short_pool * 0.77 + decayed_medium_and_long_pool * 0.77) * energy_substitution_factor

    if discounting_factors is not None:
        cflux['fuel_substitution'] *= discounting_factors['factor']
        cflux['material_substitution'] *= discounting_factors['factor']

    cflux['acc_fuel_substitution'] = cflux['fuel_substitution'].cumsum()
    cflux['fuel_mitigation'] = cflux['acc_fuel_substitution'] - cflux.loc[base_year, 'acc_fuel_substitution']

    cflux['acc_material_substitution'] = cflux['material_substitution'].cumsum()
    cflux['material_mitigation'] = cflux['acc_material_substitution'] - cflux.loc[base_year, 'acc_material_substitution']

    cflux['cstorage_mitigation'] = cflux['c_storage'] - cflux.loc[base_year, 'c_storage']

    cflux['total_mitigation'] = cflux['cstorage_mitigation'] + cflux['material_mitigation'] + cflux['fuel_mitigation']

    # for some reason Lon and Lat are in the index _and_ in the columns after this, so we delete the columns here...
    del cflux['Lon']
    del cflux['Lat']

    return cflux


def get_new_total_mitigation(cflux, cpool, discounting=None, base_year = 2010):
    cflux = cflux.set_index(['Lon', 'Lat', 'Year'])
    cpool = cpool.set_index(['Lon', 'Lat', 'Year'])

    discounting_factors = None
    if discounting:
        discounting_factors = get_discounting_factors_rcp(discounting)

    cflux['c_storage'] = cpool['Total']
    return cflux.reset_index().groupby(['Lon', 'Lat'], as_index=True).apply(lambda group: compute_mitigation(group, discounting_factors, base_year)).reset_index()


def get_discounting_factors_rcp(rcp):
    co2_emissions = pd.read_csv('total_co2_emissions_oecd_rcp_db-1.csv').drop(columns=['Region', 'Variable', 'Unit']).rename(columns={'Scenario':'Year'}).set_index('Year').transpose()
    co2_emissions.index = co2_emissions.index.astype(int)
    for year in range(2000, 2201):
        if year not in co2_emissions.index:
            co2_emissions.loc[year] = np.nan
    co2_emissions = co2_emissions.sort_index()
    co2_emissions = co2_emissions.interpolate()
    discount_rates = co2_emissions / co2_emissions.loc[2010, 'rcp26']
    discount_rates['rcp85'].loc[2010:2200] = 1
    discount_rates['rcp26'][discount_rates['rcp26'] < 0] = 0
    discount_rates_df = pd.DataFrame(discount_rates[rcp]).rename(columns={rcp: 'factor'})
    discount_rates_df.index.name = 'Year'
    return discount_rates_df


def get_fluxes_with_new_harvests(basepath, min_year, max_year, lons_lats_of_interest=None):
    cflux = ph.read_for_years(basepath + 'cflux.out', min_year, max_year, lons_lats_of_interest)
    cflux['slow_harv'] = cflux[['H_slow', 'Slow_LU']].sum(axis=1)
    # Harvest (total emissions to atmosphere from harvests) plus H_slow (Harvests that went into one of the longer lived pools) is total harvested wood
    cflux['total_harv'] = cflux[['Harvest', 'H_slow', 'Slow_LU', 'LU_S_atm', 'LU_R_atm']].sum(axis=1)

    cflux['total_harv_t'] = cflux['total_harv'] * to_t_dry_biomass_per_ha
    cflux['slow_harv_t'] = cflux['slow_harv'] * to_t_dry_biomass_per_ha

    return cflux


def get_harvests_via_species_file(basepath, min_year, max_year, lons_lats_of_interest=None, residuals=False, slow_only=False):

    # no need to multiply with active fraction! It is already contained in the output!
    harvest_file_type = 'slow' if slow_only else 'stem'
    harvests = ph.read_for_years(basepath + 'cmass_harv_per_species_' + harvest_file_type + '.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    luc = ph.read_for_years(basepath + 'cmass_luc_per_species_' + harvest_file_type + '.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    harvests += luc

    if residuals:
        assert slow_only is False
        harvests_residuals = ph.read_for_years(basepath + 'cmass_harv_per_species_residue_to_atm.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        luc_residuals = ph.read_for_years(basepath + 'cmass_luc_per_species_residue_to_atm.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        harvests += harvests_residuals
        harvests += luc_residuals

    # wooddens -- mostly from Savill 2019 -- is measured in kg/m3 at 15% moisture content
    # 15% is considered "air dry"
    # lpj output is kgC
    # the 0.47 carbon content refers to biomass
    # I am not calculating out the 15% since the values here are for dry mass and 15% is considered air dry
    wooddens_t_per_m3 = pd.read_csv('wood_density.csv')
    wooddens2 = wooddens_t_per_m3.transpose()
    wooddens2.columns = wooddens2.iloc[0]
    wooddens2 = wooddens2.drop(wooddens2.index[0])
    wooddens3 = wooddens2.loc['Value']
    wooddens3['BES'] = 1
    wooddens3['C3_gr'] = 1

    woodens_kg_per_m3 = wooddens3 * 1000.0

    m3_per_kg_wood = 1 / woodens_kg_per_m3
    m3_per_kg_wood['BES'] = 0
    m3_per_kg_wood['C3_gr'] = 0

    harvests_in_wood_kg = harvests / 0.47

    harvests_in_wood_kg = harvests_in_wood_kg.mul(m3_per_kg_wood)

    harvests_in_wood_kg['total_harv_m3_wood_per_m2'] = harvests_in_wood_kg.sum(axis=1)
    harvests_in_wood_kg['total_harv_m3_wood_per_ha'] = harvests_in_wood_kg.sum(axis=1) * 10000
    harvests_in_wood_kg['total_harv_kg_C_per_m2'] = harvests.sum(axis=1)

    return harvests_in_wood_kg.reset_index()


def get_cpool(basepath, year1, year2, lons_lats_of_interest):
    cpool = ph.read_for_years(basepath + 'cpool.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
    return cpool


def get_albedo(basepath, year1, year2, lons_lats_of_interest):
    fpc_forest_real = sr.get_real_fpc_forest(basepath, years_of_interest=[year1, year2], lons_lats_of_interest=lons_lats_of_interest, fpc_type='lai', only_managed=False)

    snow = ph.read_for_years(basepath + 'mysnow.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    snow = snow.loc[~snow.index.duplicated(keep='first')]

    fpc_with_albedo = analysis.get_forest_area_albedo(fpc_forest_real, snow)
    fpc_with_albedo = fpc_with_albedo.reset_index()
    return fpc_with_albedo


def get_forest_et(basepath, year1, year2, lons_lats_of_interest, aet_only=False):

    aet = ph.read_for_years(basepath + 'maet_forest.out', year1, year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    aet['YearlyTotal'] = aet.loc[:, months].sum(axis=1)

    if not aet_only:
        soil_et = ph.read_for_years(basepath + 'mevap_forest.out', year1, year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        soil_et['YearlyTotal'] = soil_et.loc[:, months].sum(axis=1)

        interception = ph.read_for_years(basepath + 'mintercep_forest.out', year1, year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        interception['YearlyTotal'] = interception.loc[:, months].sum(axis=1)

        aet['YearlyTotal'] += soil_et['YearlyTotal'] + interception['YearlyTotal']

    return aet.reset_index()


def get_forest_swp(basepath, year1, year2, lons_lats_of_interest, only_forest=True):
    swp_upper = ph.read_for_years(basepath + 'mpsi_s_upper_forest.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
    swp_lower = ph.read_for_years(basepath + 'mpsi_s_lower_forest.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
    if not only_forest:
        swp_upper = ph.read_for_years(basepath + 'mpsi_s_upper.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
        swp_lower = ph.read_for_years(basepath + 'mpsi_s_lower.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)

    # upper layer is 50cm, lower layer is 100cm deep
    swp_upper.iloc[:, 3:15] = (0.5 * swp_upper.iloc[:, 3:15] + swp_lower.iloc[:, 3:15]) / 1.5

    swp_upper['min'] = swp_upper.loc[:, months].min(axis=1)
    swp_upper['mean'] = swp_upper.loc[:, months].mean(axis=1)
    return swp_upper


def get_biodiversity_tree_sizes(basepath, year1, year2, lons_lats_of_interest):
    diams = ph.read_for_years(basepath + 'diamstruct_forest.out', year1, year2, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])

    tmp = compute_entropy(diams)  # store in temporary so that it does not affect thick_trees
    # index 0: 1-10, index 5: 51-60
    diams['thick_trees'] = diams.iloc[:, 5:].sum(axis=1)
    diams['size_diversity'] = tmp
    return diams.reset_index()


def get_biodiversity_cwd(basepath, year1, year2, lons_lats_of_interest):
    return ph.read_for_years(basepath + 'diversity.out', year1, year2, lons_lats_of_interest)


def get_surface_roughness(basepath, year1, year2, lons_lats_of_interest):
    roughness = ph.read_for_years(basepath + 'canopy_height.out', year1, year2, lons_lats_of_interest)
    roughness['z0'] = 0.5 * roughness['z0'] + 0.5 * roughness['z0_Win']
    return roughness
