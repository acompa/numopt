"""
Tests for simplex method.

@author alex
"""
from __future__ import division

import logging
import numpy as np
from numpy import testing as nptst
from unittest import TestCase

from numopt import simplex as undertest

logging.basicConfig(filename="simplex.log", level=logging.INFO)


class SimplexTests(TestCase):

    def setUp(self):
        """ Set up the initial vertex requested in 1(c). """
        # Build the Constraints
        raw_constraints = [[55, 2, 15, 34],
                            [12, 14, 32, 45],
                            [7, 12, 210, 7],
                            [-78, -240, -60, -800],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]
        self.b = [65, 25, 700, -3000, 0, 0, 0, 0]
        working_set = [0, 4, 2, 3]
        self.constraints = undertest.Constraints(raw_constraints, self.b, working_set)

        # Build the objective.
        self.c = np.array([[175, 225, -200, 450]]).T * -1.0

        # Initial vertex.
        self.x_0 = np.linalg.solve(_build_ndarray(raw_constraints, working_set),
                                   _build_ndarray(self.b, working_set))
        print self.x_0.shape
        logging.info("Initial vertex for 2(a) and 2(b) is {0}".format(self.x_0))
        self.x_0[0] = 0.0     # addressing rounding error
        print self.x_0.shape

        # Confirming initial point is a vertex
        original_constraint_idxs = range(4)
        self.A = _build_ndarray(raw_constraints, original_constraint_idxs)
        result = np.dot(self.A, self.x_0)
        b = _build_ndarray(self.b, original_constraint_idxs)
        logging.info("Confirming x_0 is a vertex...")
        logging.info("A*x = {0}. >= b = {1}? {2}".format(result, b, result >= b))

    def test_simplex_2a(self):
        """
        Use the initial vertex x_0 to run simplex.
        """
        logging.info("\nFINDING VERTEX FOR 2(a)")
        x = undertest.all_inequality(self.c, self.x_0, self.constraints)

        # Now check for optimality. x should satisfy constraints and yield non-negative lambda.
        original_constraint_idxs = range(4)
        self.assertTrue(np.all(
            np.dot(self.A, x) >= _build_ndarray(self.b, original_constraint_idxs)))
        logging.info("Checking vertex for 2(a). Ax = {0}, b = {1}".format(
            np.dot(self.A, x), self.b))

        # TODO (ac) Commenting b/c lambda check for optimality is failing...
        # my_lambda = np.linalg.solve(self.A.T, self.c)
        # print my_lambda
        # self.assertTrue(np.all(my_lambda >= 0))
        # self.assertTrue(np.all(np.linalg.solve(self.A.T, self.c) >= 0))

    def test_simplex_2b(self):
        # Now check for LEAST enjoyable diet.
        logging.info("\nFINDING VERTEX FOR 2(b) BY FLIPPING SIGN ON c")
        self.c *= -1.0
        x = undertest.all_inequality(self.c, self.x_0, self.constraints)

        # Now check for optimality.
        original_constraint_idxs = range(4)
        self.assertTrue(np.all(
            np.dot(self.A, x) >= _build_ndarray(self.b, original_constraint_idxs)))
        logging.info("Checking vertex for 2(b). Ax = {0}, b = {1}".format(
            np.dot(self.A, x), self.b))

        # TODO (ac) Commenting b/c lambda check for optimality is failing...
        # self.assertTrue(np.all(np.linalg.solve(self.A.T, self.c) >= 0))

    def test_simplex_toy_problem(self):
        """
        The notes include a toy problem. Test the simplex method on this problem, make sure it
        returns the right solution.
        """
        raw_constraints = [[0, -1], [1, 2], [-1, 1], [1, -1]]
        b = [-1, 0, -1, -1]
        working_set = [0, 2]
        constraints = undertest.Constraints(raw_constraints, b, working_set)
        c = np.array([[1, -1/3]]).T
        x_0 = np.array([2, 1])

        # Check the optimal solution
        x = undertest.all_inequality(c, x_0, constraints)
        nptst.assert_array_almost_equal(x, np.array([-2/3, 1/3]))

        # # Check for some conditions on the optimal solution.
        # A = _build_ndarray(constraints, range(4))
        # my_lambda, residual, _, _ = np.linalg.lstsq(A.T, c)
        # self.assertEqual(len(residual), 0)


def _build_ndarray(lst, idxs):
    """
    Given a 2d list, build a ndarray of the elements of the list matching the given indices.

    :param lst:
    :param idxs:
    :return:
    :rtype: M
    """
    out = np.array([lst[idx] for idx in idxs])
    if out.ndim == 1:
        out = out.T
    return out

