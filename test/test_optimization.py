import unittest
import numpy as np
import optimization as target

# es x strategies
vals = np.array([[1., 0., 0.96047829, 0.6653241, 0.85235997],
                 [1., 0.35606453, 0.93303329, 0.48346691, 0.76368886],
                 [0.46187047, 1., 0.25879147, 0.38772973, 0.44590858],
                 [0.55439808, 0.29434444, 0.70920116, 0., 0.39333201],
                 [0.45993177, 0.79656691, 0., 0.32361473, 0.4351323],
                 [0., 0.42329683, 0.00730904, 0.40301537, 0.15722702],
                 [0.85653956, 0., 1., 0.41511496, 0.71333659],
                 [0.86312464, 1., 0.67491613, 0.86146339, 0.82313601],
                 [0.34504458, 1., 0., 0.76893615, 0.47803942]
                 ])


class OptimizationTest(unittest.TestCase):

    @staticmethod
    def test_optimization_minimize_max_distance_to_optimum_no_weights_and_equal_weights():
        solution_no_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'])

        # not testing the z value as it can change depending on the weights.
        # e.g. using weights [1, 1, 1, 1] and [2, 2, 2, 2] will lead to the same optimal values except for the z.
        np.testing.assert_almost_equal(solution_no_weights.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        equal_weights = np.ones(vals.shape[0])
        solution_equal_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], es_weights=equal_weights)
        np.testing.assert_almost_equal(solution_equal_weights.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

    @staticmethod
    def test_optimization_minimize_max_distance_to_optimum_zero_row():
        vals_with_zero = np.vstack((vals, np.array([0., 0., 0., 0., 0.])))

        solution_no_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals_with_zero, rcps=['rcp1'])

        # not testing the z value as it can change depending on the weights.
        # e.g. using weights [1, 1, 1, 1] and [2, 2, 2, 2] will lead to the same optimal values except for the z.
        np.testing.assert_almost_equal(solution_no_weights.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        equal_weights = np.ones(vals.shape[0])
        solution_equal_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], es_weights=equal_weights)
        np.testing.assert_almost_equal(solution_equal_weights.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

    @staticmethod
    def test_optimization_minimize_max_distance_to_optimum_weights():
        # increasing the first ES
        weights_one_spike = list(np.ones(9))
        weights_one_spike[0] = 2

        solution_spiked_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], es_weights=np.array(weights_one_spike))

        # doubling the weight for the first es must decrease the preference of the 2nd management, as it provides the worst case for that!
        np.testing.assert_almost_equal(solution_spiked_weights.x[1:], [0., 0.18005742, 0.32008861, 0.49985398, 0.], decimal=7)

    @staticmethod
    def test_optimization_minimize_max_distance_to_optimum_bounds():
        solution_standard_bounds = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], m_upper_bounds=np.ones(5), m_lower_bounds=np.zeros(5))
        # same result as for no bounds
        np.testing.assert_almost_equal(solution_standard_bounds.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        upper_bounds = np.ones(5)
        upper_bounds[1] = 0.5
        solution_upper_bounds = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], m_upper_bounds=upper_bounds, m_lower_bounds=np.zeros(5))
        np.testing.assert_almost_equal(solution_upper_bounds.x[1:], [0., 0.5, 0.2407295, 0.2592705, 0.], decimal=7)

        lower_bounds = np.zeros(5)
        lower_bounds[4] = 0.1
        solution_both_bounds = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], m_upper_bounds=upper_bounds, m_lower_bounds=lower_bounds)
        np.testing.assert_almost_equal(solution_both_bounds.x[1:], [0., 0.5, 0.1828858, 0.2171142, 0.1], decimal=7)

    @staticmethod
    def test_optimization_minimize_max_distance_to_optimum_bounds_and_weights():
        upper_bounds = np.ones(5)
        upper_bounds[1] = 0.5
        lower_bounds = np.zeros(5)
        lower_bounds[4] = 0.1

        # increasing the first ES
        weights_one_spike = list(np.ones(9))
        weights_one_spike[0] = 2

        solution_bounds_and_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], m_upper_bounds=upper_bounds, m_lower_bounds=lower_bounds,
                                                                                                      es_weights=weights_one_spike)
        # without the weights it would be [0., 0.5, 0.1828858, 0.2171142, 0.1], but we move away from m[1] since it performs poorly on es[0].
        np.testing.assert_almost_equal(solution_bounds_and_weights.x[1:], [0., 0.1831606, 0.2614751, 0.4553643, 0.1], decimal=7)

    @staticmethod
    def test_optimization_minimize_with_infeasible_management():
        solution_normal = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'])
        np.testing.assert_almost_equal(solution_normal.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        # management 3 cannot be used
        solution_infeasible_management = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], infeasible_management_idxs=[3])
        np.testing.assert_almost_equal(solution_infeasible_management.x[1:], [0., 0.5678027, 0., 0., 0.4321973], decimal=7)

        # management 3 and 4 cannot be used
        solution_infeasible_management = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], infeasible_management_idxs=[3, 4])
        np.testing.assert_almost_equal(solution_infeasible_management.x[1:], [0.0659366, 0.694719, 0.2393444, 0., 0.], decimal=7)

    @staticmethod
    def test_optimization_with_one_additional_constraint():
        solution_no_add_constraints = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'])
        np.testing.assert_almost_equal(solution_no_add_constraints.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        outcomes_for_es = np.dot(vals, solution_no_add_constraints.x[1:])
        np.testing.assert_almost_equal(outcomes_for_es[0], 0.3705, decimal=4)

        additional_constraints = {'lhs': vals[0, :], 'rhs': [0.4]}

        solution_add_constraints = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], additional_constraints=additional_constraints)
        # np.testing.assert_almost_equal(solution_equal_weights.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)
        outcomes_for_es_with_add_constraints = np.dot(vals, solution_add_constraints.x[1:])
        np.testing.assert_almost_equal(outcomes_for_es_with_add_constraints[0], 0.4, decimal=4)

    @staticmethod
    def test_optimization_with_multiple_additional_constraints():
        solution_no_add_constraints = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'])
        np.testing.assert_almost_equal(solution_no_add_constraints.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        outcomes_for_es = np.dot(vals, solution_no_add_constraints.x[1:])
        np.testing.assert_almost_equal(outcomes_for_es[0], 0.3705, decimal=4)
        np.testing.assert_almost_equal(outcomes_for_es[1], 0.5172, decimal=4)

        additional_constraints = {'lhs': vals[0:2, :], 'rhs': [0.4, 0.6]}

        solution_add_constraints = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], additional_constraints=additional_constraints)
        outcomes_for_es_with_add_constraints = np.dot(vals, solution_add_constraints.x[1:])

        # the values for one of those es will be exactly at the provided lower bound because they would be lower otherwise, so the optimization
        # will just give as much to them as absolutely necessary to fulfil the additional constraint.
        np.testing.assert_almost_equal(outcomes_for_es_with_add_constraints[0], 0.5272, decimal=4)
        np.testing.assert_almost_equal(outcomes_for_es_with_add_constraints[1], 0.6, decimal=4)

    @staticmethod
    def test_additional_constraints_already_achieved_will_not_alter_result():
        solution_no_add_constraints = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'])
        np.testing.assert_almost_equal(solution_no_add_constraints.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        outcomes_for_es = np.dot(vals, solution_no_add_constraints.x[1:])
        np.testing.assert_almost_equal(outcomes_for_es[0], 0.3705, decimal=4)
        np.testing.assert_almost_equal(outcomes_for_es[1], 0.5172, decimal=4)

        # the additional constraints are lower than what is already achieved
        additional_constraints = {'lhs': vals[0:2, :], 'rhs': [0.3, 0.5]}

        solution_add_constraints = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], additional_constraints=additional_constraints)
        outcomes_for_es_with_add_constraints = np.dot(vals, solution_add_constraints.x[1:])

        # same result as before
        np.testing.assert_almost_equal(outcomes_for_es_with_add_constraints[0], 0.3705, decimal=4)
        np.testing.assert_almost_equal(outcomes_for_es_with_add_constraints[1], 0.5172, decimal=4)

    @staticmethod
    def test_with_multiple_rcps_same_vals_for_each_rcp():
        # increasing the first ES
        weights_one_spike = list(np.ones(9))
        weights_one_spike[0] = 2

        vals_2_rcps = np.repeat(vals, 2, axis=0)

        solution_spiked_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals_2_rcps, rcps=['rcp1', 'rcp2'], es_weights=np.array(weights_one_spike))

        # same result as in the 1 rcp weighting test, since the es_vals are the exact same, just doubled.
        np.testing.assert_almost_equal(solution_spiked_weights.x[1:], [0., 0.18005742, 0.32008861, 0.49985398, 0.], decimal=7)

    @staticmethod
    def test_with_multiple_rcps_small_example_different_vals():
        vals_simple = np.array([[1.0, 0.0, 1.0, 1.0],
                                [0.0, 0.5, 0.5, 1.0],
                                [1.0, 1.0, 0.0, 0.5]])
        solution = target.solve_optimization_for_gridcell_general_min_max_distance(vals_simple, rcps=['rcp1'])
        np.testing.assert_almost_equal(solution.x[1:], [0.1428571, 0.2857143, 0., 0.5714286], decimal=7)

        # same result when adding equal weights
        weights_equal = list(np.ones(3))
        solution2 = target.solve_optimization_for_gridcell_general_min_max_distance(vals_simple, rcps=['rcp1'], es_weights=np.array(weights_equal))
        np.testing.assert_almost_equal(solution2.x[1:], [0.1428571, 0.2857143, 0., 0.5714286], decimal=7)

        # adding more weight to 3rd es decreases preference for 4th strategy (has only 0.5 performance)
        weights_spiked = list(np.ones(3))
        weights_spiked[2] = 3
        solution2 = target.solve_optimization_for_gridcell_general_min_max_distance(vals_simple, rcps=['rcp1'], es_weights=np.array(weights_spiked))
        np.testing.assert_almost_equal(solution2.x[1:], [0.2307692, 0.4615385, 0., 0.3076923], decimal=7)

        # another rcp added:
        vals_2_rcps = np.array([[1.0, 0.0, 1.0, 1.0],
                                [1.0, 0.0, 1.0, 1.0],
                                # es2
                                [0.0, 0.5, 0.5, 1.0],
                                [0.0, 0.5, 0.5, 1.0],
                                # es3
                                [1.0, 1.0, 0.0, 0.5],
                                [1.0, 1.0, 0.0, 0.4]])

        # with the spiked weight, the amount of 4th strategy is even lower than for 1 rcp since in the second rcp, the 4th strategy is even worse in es 3.
        solution3 = target.solve_optimization_for_gridcell_general_min_max_distance(vals_2_rcps, rcps=['rcp1', 'rcp2'], es_weights=np.array(weights_spiked))
        np.testing.assert_almost_equal(solution3.x[1:], [0.2432432, 0.4864865, 0., 0.2702703], decimal=7)

    def test_optimization_minimize_max_distance_to_optimum_lambda(self):
        # solution_no_weights = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'])
        #
        # # not testing the z value as it can change depending on the weights.
        # # e.g. using weights [1, 1, 1, 1] and [2, 2, 2, 2] will lead to the same optimal values except for the z.
        # np.testing.assert_almost_equal(solution_no_weights.x[1:], [0., 0.54488303, 0.22959669, 0.22552028, 0.], decimal=7)

        solution_lambda = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=0.5)
        np.testing.assert_almost_equal(solution_lambda.x[1:], [0., 0.5311108, 0., 0.0660581, 0.4028311], decimal=7)

        weights_equal = list(np.ones(vals.shape[0]))
        solution_lambda = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=0.5, es_weights=weights_equal)
        np.testing.assert_almost_equal(solution_lambda.x[1:], [0., 0.5311108, 0., 0.0660581, 0.4028311], decimal=7)

        weights_equal = list(np.ones(vals.shape[0])/vals.shape[0])
        solution_lambda = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=0.5, es_weights=weights_equal)
        np.testing.assert_almost_equal(solution_lambda.x[1:], [0., 0.5311108, 0., 0.0660581, 0.4028311], decimal=7)

        weights_spiked = list(np.ones(vals.shape[0]))
        weights_spiked[0] = 2  # strategy 1 is worst for this one, so the value of that strategy should decrease

        # strategy 1 is now not as important anymore!
        solution_lambda = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=0.5, es_weights=weights_spiked)
        np.testing.assert_almost_equal(solution_lambda.x[1:], [0., 0.1970041, 0., 0.2568956, 0.5461003], decimal=7)

        weights_spiked = list(np.ones(vals.shape[0])/vals.shape[0])
        weights_spiked[0] *= 2  # strategy 1 is worst for this one, so the value of that strategy should decrease

        solution_lambda = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=0.5, es_weights=weights_spiked)
        np.testing.assert_almost_equal(solution_lambda.x[1:], [0., 0.1970041, 0., 0.2568956, 0.5461003], decimal=7)

    @staticmethod
    def test_optimization_minimize_average_ecosystem_service():
        solution_lambda = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=1.0)
        np.testing.assert_almost_equal(solution_lambda.x[1:], [1, 0, 0, 0, 0], decimal=7)

        weights_equal = list(np.ones(vals.shape[0]))
        solution_lambda_equally_weighted = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=1.0, es_weights=weights_equal)
        np.testing.assert_almost_equal(solution_lambda_equally_weighted.x[1:], [1, 0, 0, 0, 0], decimal=7)

        weights_spiked = weights_equal
        weights_spiked[5] = 2  # strategy 0 is bad in this es, portfolio should move away from this one.
        # but it is not weighted high enough, portfolio remains the same.
        solution_lambda_weighted = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=1.0, es_weights=weights_spiked)
        np.testing.assert_almost_equal(solution_lambda_weighted.x[1:], [1, 0, 0, 0, 0], decimal=7)

        weights_spiked[5] = 3
        # now the portfolio moves away from strategy 0 and it puts 100% in strategy 1.
        # checked in excel that this is indeed the right solution.
        solution_lambda_weighted = target.solve_optimization_for_gridcell_general_min_max_distance(vals, rcps=['rcp1'], lambda_opt=1.0, es_weights=weights_spiked)
        np.testing.assert_almost_equal(solution_lambda_weighted.x[1:], [0, 1, 0, 0, 0], decimal=7)


if __name__ == '__main__':
    unittest.main()
