import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
from constants import *
import numpy as np
import pandas as pd
from variable_units import *


def my_plot_radar_chart(the_fig, es_names, simulation_names, es_vals, row, col, simulations_of_interest=None):
    if len(es_vals.shape) == 1:
        es_vals = np.expand_dims(es_vals, axis=1)  # only a hack to get the indexing below correct for a 1-dim array

    if not simulations_of_interest:
        simulations_of_interest = simulation_names

    for idx, simulation in enumerate(simulation_names):
        if simulation not in simulations_of_interest:
            continue
        es_scores = es_vals[:, idx]
        the_fig.add_trace(go.Scatterpolar(
            r=np.concatenate((es_scores, [es_scores[0]])),
            theta=[table_names_no_latex[es] for es in (es_names + [es_names[0]])],
            name=simulation,
            line_color=color_discrete_map[simulation],
            mode='lines'
        ), row, col)


# from https://community.plotly.com/t/scatter-plot-fill-with-color-how-to-set-opacity-of-fill/29591
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def my_plot_radar_chart_min_max(the_fig, es_names, simulation_names, mins, maxs, row, col, simulations_of_interest=None):
    if len(mins.shape) == 1:
        mins = np.expand_dims(mins, axis=1)  # only a hack to get the indexing below correct for a 1-dim array
    if len(maxs.shape) == 1:
        maxs = np.expand_dims(maxs, axis=1)  # only a hack to get the indexing below correct for a 1-dim array

    if not simulations_of_interest:
        simulations_of_interest = simulation_names

    for idx, simulation in enumerate(simulation_names):
        if simulation not in simulations_of_interest:
            continue
        es_scores = mins[:, idx]
        the_fig.add_trace(go.Scatterpolar(
            r=np.concatenate((es_scores, [es_scores[0]])),
            theta=es_names + [es_names[0]],
            name=simulation,
            line_color=color_discrete_map[simulation],
            mode='lines',
        ), row, col)
        es_scores = maxs[:, idx]
        the_fig.add_trace(go.Scatterpolar(
            r=np.concatenate((es_scores, [es_scores[0]])),
            theta=es_names + [es_names[0]],
            name=simulation,
            line_color=color_discrete_map[simulation],
            fill='tonext',
            fillcolor=f"rgba{(*hex_to_rgb(color_discrete_map[simulation]), 0.1)}",
            mode='lines'
        ), row, col)


def my_plot_line_chart_min_max(the_fig, es_names, simulation_names, mins, maxs, row, col, simulations_of_interest=None):
    if len(mins.shape) == 1:
        mins = np.expand_dims(mins, axis=1)  # only a hack to get the indexing below correct for a 1-dim array

    if len(maxs.shape) == 1:
        maxs = np.expand_dims(maxs, axis=1)  # only a hack to get the indexing below correct for a 1-dim array

    if not simulations_of_interest:
        simulations_of_interest = simulation_names

    for idx, simulation in enumerate(simulation_names):
        if simulation not in simulations_of_interest:
            continue
        es_scores = mins[:, idx]
        trace1 = go.Box(y=es_scores, x=es_names, name=simulation, line_color=color_discrete_map[simulation])
        trace1['showlegend'] = False
        the_fig.add_trace(trace1, row, col)
        es_scores = maxs[:, idx]
        # trace2 = go.Scatter(y=es_scores, x=es_names, name=simulation, line_color=color_discrete_map[simulation],
        #                      fill='tonexty', fillcolor=f"rgba{(*hex_to_rgb(color_discrete_map[simulation]), 0.1)}",
        #                      mode='markers')
        # trace2['showlegend'] = False
        # the_fig.add_trace(trace2, row, col)


def my_plot_line_chart_all(the_fig, es_names, simulation_names, all_vals, row, col, simulations_of_interest=None):

    if not simulations_of_interest:
        simulations_of_interest = simulation_names

    for idx, simulation in enumerate(simulation_names):
        if simulation not in simulations_of_interest:
            continue

        for rcp, vals in all_vals.items():
            es_scores = vals[:, idx]
            the_fig.add_trace(go.Scatter(
                y=es_scores,
                x=es_names,
                name=simulation,
                line_color=color_discrete_map[simulation],
                mode='lines'
            ), row, col)


def plot_optimization_results2(output, luyssaert=False, save_to=None, plot_only_rcp=None, radar=True, min_only=True, simulations_of_interest=None, row_nr=0):
    es_list = output['es']
    simulation_list = output['feasible_managements']
    all_es_vals = output['es_vals_all_rcps_new']
    optimized_solution = output['all_rcp_solution_new']
    rcps = output['rcps']
    gridcell = output['row']

    if not simulations_of_interest:
        simulations_of_interest = simulation_list

    col_widths = None if radar else [0.45, 0.3, 0.25]

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'polar' if radar else 'xy'}, {'type': 'polar' if radar else 'xy'}, {'type': 'domain'}]], column_widths=col_widths,
                        subplot_titles=(" ", " ", ' '),
                        shared_yaxes=True)


    # Don't let subplot titles go over the content
    for annotation in fig['layout']['annotations']:
        annotation['yanchor'] = 'bottom'
        annotation['y'] = 1.05
        annotation['yref'] = 'paper'

    # all_es_vals = es1_rcp1, es1_rcp2, ..., es2_rcp1, es2_rcp2, ...

    worst_cases = np.ones((len(es_list), len(simulation_list)))
    best_cases = np.zeros((len(es_list), len(simulation_list)))
    means = np.zeros((len(es_list), len(simulation_list)))
    all_vals = {'rcp26': np.zeros((len(es_list), len(simulation_list))), 'rcp45': np.zeros((len(es_list), len(simulation_list))), 'rcp60': np.zeros((len(es_list), len(simulation_list))), 'rcp85': np.zeros((len(es_list), len(simulation_list)))}
    for idx in range(len(es_list)):
        if plot_only_rcp:
            rcp_idx = rcps.index(plot_only_rcp)
            vals = all_es_vals[(idx * len(rcps) + rcp_idx):(idx * len(rcps) + rcp_idx + 1), :]
            worst_cases[idx, :] = vals
        else:
            vals = all_es_vals[idx * len(rcps):(idx + 1) * len(rcps), :]  # all values for the current es, for all the rcps
            worst_cases[idx, :] = np.min(vals, axis=0)  # min by columns, so worst case for each simulation/management
            best_cases[idx, :] = np.max(vals, axis=0)  # min by columns, so worst case for each simulation/management
            means[idx, :] = np.mean(vals, axis=0)  # mean by columns, so mean for each simulation/management
            for rcp in rcps:
                rcp_idx = rcps.index(rcp)
                vals = all_es_vals[(idx * len(rcps) + rcp_idx):(idx * len(rcps) + rcp_idx + 1), :]
                all_vals[rcp][idx, :] = vals

    if radar:
        if min_only:
            my_plot_radar_chart(fig, list(es_list), list(simulation_list), worst_cases, 1, 1, simulations_of_interest)
        else:
            my_plot_radar_chart_min_max(fig, list(es_list), list(simulation_list), worst_cases, best_cases, 1, 1, simulations_of_interest)
    else:
        for es, values in output['scores'].items():
            # quick hack to put unmanaged as the lowest, since it will span the whole y axis for mitigation and will block the others.
            for idx in range(len(simulation_list)):
                man_idx = len(simulation_list) - 1 - idx
                man = simulation_list[man_idx]
                # for man_idx, man in enumerate(simulation_list):
                if man in simulations_of_interest:
                    vvv = [values[rcp][man_idx] for rcp in rcps]
                    trace1 = go.Box(y=vvv, name=table_names_no_latex[es], line_color=color_discrete_map[man], offsetgroup=man, showlegend=False)
                    fig.add_trace(trace1, 1, 1)

    portfolio_fractions = optimized_solution.x[1:]
    portfolio_worst = np.ones(len(es_list))
    portfolio_best = np.zeros(len(es_list))
    for idx, rcp in enumerate(rcps):
        vals_for_rcp = all_es_vals[idx::len(rcps)]
        optimized_scores = np.dot(portfolio_fractions, np.transpose(vals_for_rcp))
        if not plot_only_rcp:
            portfolio_worst = np.minimum(portfolio_worst, optimized_scores)
            portfolio_best = np.maximum(portfolio_worst, optimized_scores)
            # portfolio_scores += portfolio_scores/len(rcps)
        elif plot_only_rcp == rcp:
            portfolio_worst = optimized_scores

    if radar:
        if min_only:
            my_plot_radar_chart(fig, list(es_list), ['optimal portfolio'], portfolio_worst, 1, 2)
        else:
            my_plot_radar_chart_min_max(fig, list(es_list), ['optimal portfolio'], portfolio_worst, portfolio_best, 1, 2)
    else:
        for es, values in output['scores'].items():
            vvv = np.zeros(len(rcps))
            for man_idx, man in enumerate(simulation_list):
                vvv += portfolio_fractions[man_idx] * np.array([values[rcp][man_idx] for rcp in rcps])
            trace1 = go.Box(y=vvv, name=table_names_no_latex[es], line_color=color_discrete_map['optimal portfolio'], width=0.7)
            trace1['showlegend'] = False
            fig.add_trace(trace1, 1, 2)

    fig.update_polars(radialaxis=dict(range=[0, 1], showticklabels=True, showline=False), gridshape='circular')
    fig.update_layout(yaxis_range=[-0.05, 1.05], yaxis_title='Normalized ESI Performance')

    fig.update_xaxes(tickangle=45)
    fig.update_layout(boxmode='group')

    # fig.update_yaxes(showline=True, showgrid=True, gridwidth=1, gridcolor='LightGrey')

    the_colors = [color_discrete_map[simulation] for simulation in simulation_list]

    fig.add_trace(go.Pie(labels=np.array(simulation_list), sort=False, values=np.array(portfolio_fractions), hole=.3, marker=dict(colors=np.array(the_colors)), textposition='inside'), 1, 3)

    title = '<b>'
    if gridcell[2]:
        title += str(gridcell[2]) + ' '
    title += '(' + str(gridcell[0]) + ', ' + str(gridcell[1]) + ')' + '</b>'
    if plot_only_rcp:
        title += ', RCP: ' + str(rcps)
    if not optimized_solution.success:
        title += ', INFEASIBLE!'

        fig.update_layout(
            yaxis=dict(
                tickmode='linear',
                tick0=0.5,
                dtick=0.75
            )
        )

    fig.update_layout(height=380, width=1200, margin=dict(l=80, r=80, t=100, b=20))#, plot_bgcolor='rgba(0,0,0.5,0.0)')
    #
    fig.add_annotation(x=-0.05, y=1.3,
                       xref="paper", yref="paper",
                       text=title,
                       font=dict(size=16),
                       showarrow=False)

    start_lettering = row_nr * 3

    # add a), b), c)
    position_change_for_a = 0.2 if radar else 0.25
    fig.add_annotation(x=fig['layout']['annotations'][0]['x'] - position_change_for_a, y=1.2,
                       xref="paper", yref="paper",
                       text="<b>" + chr(ord('a') + start_lettering) + ")</b> ESI Performance 2100-2130 (indiv. management)",
                       font=dict(size=15),
                       showarrow=False)
    fig.add_annotation(x=fig['layout']['annotations'][1]['x'] + 0.02, y=1.2,
                       xref="paper", yref="paper",
                       text="<b>" + chr(ord('b') + start_lettering) + ")</b> ESI Performance 2100-2130 (optimized portfolio)",
                       font=dict(size=15),
                       showarrow=False)
    fig.add_annotation(x=fig['layout']['annotations'][2]['x'] + 0.12, y=1.2,
                       xref="paper", yref="paper",
                       text="<b>" + chr(ord('c') + start_lettering) + ")</b> Optimized Portfolio Shares",
                       font=dict(size=15),
                       showarrow=False)


    if save_to:
        fig.write_image(save_to)
    else:
        fig.show()

    if luyssaert:
        fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
                            subplot_titles=("Species in 2010", "Species for portfolio", 'Management 2100 Luyssaert color scheme'))

        lv = luyssaert_values(simulation_list, portfolio_fractions, gridcell)

        fig.add_trace(go.Pie(labels=['coniferous', 'broadleaved'], sort=False, values=[lv['base_coniferous_frac'], lv['base_broadleaved_frac']], hole=.3,
                             marker=dict(colors=['green', 'lightgreen'])), 1, 1)
        fig.add_trace(go.Pie(labels=['coniferous', 'broadleaved'], sort=False, values=[lv['total_coniferous'], lv['total_broadleaved'], lv['total_grass']], hole=.3,
                             marker=dict(colors=['green', 'lightgreen'])), 1, 2)
        fig.add_trace(go.Pie(labels=['high stand', 'unmanaged', 'coppice'], sort=False, values=[lv['total_high'], lv['total_unmanaged'], lv['total_coppice']], hole=.3,
                             marker=dict(colors=['blue', 'red', 'orange'])), 1, 3)

        fig.update_layout(title_text="Comparison of Species Distribution (Luyssaert2018 color scheme)")

        fig.show()


class NoManagedForestError(Exception):
    pass


class NoManagementFeasibleError(Exception):
    pass


def luyssaert_values(simulation_list, portfolio_fractions, gridcell):
    forest_fractions = pd.read_csv(forest_fraction_file, delim_whitespace=True)

    forest_fractions = forest_fractions.query('Lon==' + str(gridcell[0]) + ' and Lat==' + str(gridcell[1])).set_index(['year'])

    forest_fractions['Total'] = forest_fractions['ForestBD'] + forest_fractions['ForestBE'] + forest_fractions['ForestND'] + forest_fractions['ForestNE']

    total_forest_fraction = forest_fractions['Total'].values[-1]

    if total_forest_fraction == 0:
        raise NoManagedForestError('Gridcell without managed forest! ' + str(gridcell[0]) + ', ' + str(gridcell[1]))

    base_broadleaved_frac = (forest_fractions.iloc[-1, :]['ForestBD'] + forest_fractions.iloc[-1, :]['ForestBE']) / total_forest_fraction
    base_coniferous_frac = (forest_fractions.iloc[-1, :]['ForestND'] + forest_fractions.iloc[-1, :]['ForestNE']) / total_forest_fraction

    total_broadleaved = 0.0
    total_coniferous = 0.0
    total_grass = 0.0
    total_high = 0.0
    total_coppice = 0.0
    total_unmanaged = 0.0

    for idx, simulation in enumerate(simulation_list):
        if simulation == 'toBd' or simulation == 'toBe':
            total_broadleaved += portfolio_fractions[idx]
            total_high += portfolio_fractions[idx]
        elif simulation == 'toCoppice':
            total_broadleaved += portfolio_fractions[idx]
            total_coppice += portfolio_fractions[idx]
        elif simulation == 'toNe':
            total_coniferous += portfolio_fractions[idx]
            total_high += portfolio_fractions[idx]
        elif simulation == 'base':
            total_broadleaved += portfolio_fractions[idx] * base_broadleaved_frac
            total_coniferous += portfolio_fractions[idx] * base_coniferous_frac
            total_high += portfolio_fractions[idx]
        elif simulation == 'baseRefrain' or simulation == 'unmanaged':
            total_broadleaved += portfolio_fractions[idx] * base_broadleaved_frac
            total_coniferous += portfolio_fractions[idx] * base_coniferous_frac
            total_unmanaged += portfolio_fractions[idx]
        elif simulation == 'toGrass':
            total_grass += portfolio_fractions[idx]
        else:
            raise ValueError('A simulation was used that was not accounted for in the species distribution computation. It was: ' + str(simulation))

    return dict(base_broadleaved_frac=base_broadleaved_frac, base_coniferous_frac=base_coniferous_frac, total_unmanaged=total_unmanaged, total_broadleaved=total_broadleaved,
                total_coniferous=total_coniferous, total_grass=total_grass, total_high=total_high, total_coppice=total_coppice, forest_frac=total_forest_fraction)


def plot_absolute_values(all_scores_raw, portfolio, used_simulations, boundary_simulations):
    for i in range(4):  # rcps
        used_variables = all_scores_raw.keys()
        absolute_vals_first_rcp = pd.DataFrame(all_scores_raw).applymap(lambda x: x[i])
        fig, axs = plt.subplots(1, len(used_variables), figsize=(25, 5))
        fig.tight_layout()

        for idx, es in enumerate(used_variables):
            absolute_vals_first_rcp[es].plot(ax=axs[idx], kind='bar', color=[color_discrete_map[x] for x in used_simulations + boundary_simulations], alpha=0.5)

            optimum = np.dot(absolute_vals_first_rcp[es][used_simulations], portfolio)

            axs[idx].plot(used_simulations + boundary_simulations, optimum * np.ones(len(used_simulations + boundary_simulations)), linewidth=3, color='k')

            rel_perf = optimum / np.max(absolute_vals_first_rcp[es][:])

            if es == 'flood_risk':
                rel_perf = optimum / np.min(absolute_vals_first_rcp[es][:])
                axs[idx].set_title(es + ' (lower=better) ' + "{:.2f}".format(rel_perf))
            else:
                axs[idx].set_title(es + ' ' + "{:.2f}".format(rel_perf))

        #     fig.update_layout(title_text="Absolute values of single management and optimum (RCP 4.5)")
        fig.suptitle('Absolute values of single management and optimum (RCP 4.5)')
        plt.show()
