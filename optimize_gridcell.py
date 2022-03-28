import optimization as opt
import optimization_preparation as oprep
import optimization_plots as oplot
import numpy as np

luyssaert_variables = ['base_broadleaved_frac', 'base_coniferous_frac', 'total_unmanaged', 'total_broadleaved', 'total_coniferous', 'total_grass', 'total_coppice', 'total_high', 'forest_frac']


def optimize_gridcell(rcps, simulations_paths, used_simulations, es, constraints, lambda_opt, weights, gcs2, row_nr, row, min_year, max_year, file_basepath=None, plot=False, gc_location=None, discounting=True, plot_for_paper_name=None):
    if row_nr > 0:
        print(str(row_nr) + ' ' + str(row['Lon']) + ' ' + str(row['Lat']) + ', progress ' + "{:.2f}".format(row_nr / len(gcs2)))

    row.loc[used_simulations] = 0
    row.loc['no_management_feasible'] = False
    row.loc['has_forest'] = True
    row.loc['all_managements_possible'] = True

    feasible_managements = []
    portfolio_fractions = np.zeros_like(used_simulations)

    all_scores_raw = None  # need to assign here to avoid UnboundLocalError in infeasible grid cell case
    all_rcp_solution_new = None  # need to assign here to avoid UnboundLocalError in infeasible grid cell case
    es_vals_all_rcps_new = None  # need to assign here to avoid UnboundLocalError in infeasible grid cell case
    all_scores_all_rcps_new = None  # need to assign here to avoid UnboundLocalError in infeasible grid cell case

    for idx, management in enumerate(used_simulations):
        row.loc['feasible_' + management] = False

    try:
        es_vals_all_rcps_new, all_scores_all_rcps_new, all_scores_raw, additional_constraints, infeasible_managements, feasible_managements = oprep.get_es_vals_new(rcps,
                                                                                                                                          simulations_paths,
                                                                                                                                          future_year1=min_year,
                                                                                                                                          future_year2=max_year,
                                                                                                                                          simulation_names=used_simulations,
                                                                                                                                          optimizationType=oprep.OptimizationType.MINMAX,
                                                                                                                                          used_variables=es,
                                                                                                                                          boundary_simulations=[], lon=row[0],
                                                                                                                                          lat=row[1],
                                                                                                                                          variables_for_additional_constraints=constraints)

        all_rcp_solution_new = opt.solve_optimization_for_gridcell_general_min_max_distance(es_vals_all_rcps_new, rcps,
                                                                                            lambda_opt=lambda_opt,
                                                                                            es_weights=weights,
                                                                                            additional_constraints=additional_constraints,
                                                                                            infeasible_management_idxs=[])


        portfolio_fractions = all_rcp_solution_new.x[1:]

        row.loc[feasible_managements] = portfolio_fractions

        row.loc['feasible'] = all_rcp_solution_new.success

        if infeasible_managements:
            row.loc['all_managements_possible'] = False

        for idx, management in enumerate(feasible_managements):
            row.loc['feasible_' + management] = True

    except oplot.NoManagementFeasibleError:
        row.loc['feasible'] = False
        row.loc['no_management_feasible'] = True
        print('No management is feasible for this gridcell')

    try:
        luyssaert_vals = oplot.luyssaert_values(feasible_managements, portfolio_fractions, (row['Lon'], row['Lat']))
        for luyssaert_variable in luyssaert_variables:
            row.loc[luyssaert_variable] = luyssaert_vals[luyssaert_variable]
    except oplot.NoManagedForestError:
        row.loc['has_forest'] = False
        print('This gridcell does not have any managed forest.')

    # we can get values like -1E-16 sometimes in the optimization.optimize
    for man in used_simulations:
        if row.loc[man] < 0:
            row.loc[man] = 0

    output = dict(
        row=row,
        es=es,
        feasible_managements=feasible_managements,
        es_vals_all_rcps_new=es_vals_all_rcps_new,
        all_rcp_solution_new=all_rcp_solution_new,
        scores=all_scores_all_rcps_new,
        rcps=rcps,
        all_scores_raw=all_scores_raw
    )

    if plot:
        if file_basepath:
            oplot.plot_optimization_results2(output, luyssaert=False, save_to=file_basepath + '_' + str(row[0]) + '_' + str(row[1]) + '.png')
        elif plot_for_paper_name:
            oplot.plot_optimization_results2(output, luyssaert=False, row_nr=row_nr, save_to='/home/konni/Documents/konni/projekte/phd/my_papers/optimization_paper-Review/' + plot_for_paper_name + '.png')
        else:
            oplot.plot_optimization_results2(output, luyssaert=False, row_nr=row_nr)

    return output
