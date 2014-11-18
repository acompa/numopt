"""
Tests for pure Newton method.

@author alex
"""

import logging
import numpy as np
from numpy import testing as tst
import unittest

from numopt import newton as undertest

logger = logging.getLogger('newton.testing')


class PureNewtonTest(unittest.TestCase):

    def setUp(self):
        self.my_func = lambda x: np.array([x[0] ** 2 + 4 * x[1],
                                           -0.5 * x[0] + 8 * x[1] ** 2])
        self.my_jacobian = lambda x: np.array([[2 * x[0], 4],
                                               [-0.5, 16 * x[1]]])
        self.merit_fn = np.linalg.norm      # Merit function: 2-norm of F
        self.zero = np.array([0.0, 0.0])
        self.other_zero = np.array([1, -0.25])
        self.ftol = 1.0e-8

        # Testing function
        tst.assert_almost_equal(self.my_func(self.zero), np.zeros(2))
        tst.assert_almost_equal(self.my_func(self.other_zero), np.zeros(2))

    def test_51b(self):
        logger.debug("RUNNING 5.1B")
        x0 = np.array([1.0, -1])
        maxit = 12
        run_line_search = True
        result = undertest.pure_newton(x0, self.my_func, self.my_jacobian, self.merit_fn, maxit,
                                       self.ftol, run_line_search)
        logger.debug("5.1B yields root == {0}".format(result))
        tst.assert_array_almost_equal(result, self.other_zero, decimal=4)

    def test_51c(self):
        logger.debug("RUNNING 5.1C")
        x0 = np.array([1.99, 0])
        maxit = 9
        run_line_search = False
        result = undertest.pure_newton(x0, self.my_func, self.my_jacobian, self.merit_fn, maxit,
                                       self.ftol, run_line_search)
        self.assertFalse(np.all(result == self.zero) and np.all(result == self.other_zero))

    def test_51d(self):
        logger.debug("RUNNING 5.1D")
        x0 = np.array([1.99, 0])
        maxit = 15
        run_line_search = True
        result = undertest.pure_newton(x0, self.my_func, self.my_jacobian, self.merit_fn, maxit,
                                       self.ftol, run_line_search)
        tst.assert_array_almost_equal(result, self.zero, decimal=4)

    def test_51e(self):
        logger.debug("RUNNING 5.1E")
        x0 = np.array([1.0, 0])
        maxit = 8
        run_line_search = False
        result = undertest.pure_newton(x0, self.my_func, self.my_jacobian, self.merit_fn, maxit,
                                       self.ftol, run_line_search)
        self.assertFalse(np.all(result == self.zero) and np.all(result == self.other_zero))

    def test_51f(self):
        logger.debug("RUNNING 5.1F")
        x0 = np.array([1.0, 0])
        maxit = 8
        run_line_search = True
        result = undertest.pure_newton(x0, self.my_func, self.my_jacobian, self.merit_fn, maxit,
                                       self.ftol, run_line_search)
        tst.assert_array_almost_equal(result, self.zero, decimal=4)


class QuasiNewtonTest(unittest.TestCase):

    def setUp(self):
        c = np.array([2, -1, 2, -1])
        self.hessian = np.diag([5, 1, 0.01, 0.0001])
        self.q_k = lambda x: np.dot(c, x) + 0.5 * np.dot(np.dot(x, self.hessian), x)
        self.gradient = lambda x: c + np.dot(self.hessian, x)

    def test_55a(self):
        logger.debug("RUNNING 5.5A")
        x0 = np.array([-1, 0, 1, 1])
        maxit = 10
        ftol = 1.0e-8
        root, hessian = undertest.quasi_newton(x0, self.q_k, self.gradient, maxit, ftol,
                                               check_armijo=False, keep_hessian=True)
        tst.assert_approx_equal(hessian, self.hessian)

    def test_55b(self):
        logger.debug("RUNNING 5.5B")
        x0 = np.array([-0.4, 0, 1, 1])
        maxit = 10
        ftol = 1.0e-8
        root, hessian = undertest.quasi_newton(x0, self.q_k, self.gradient, maxit, ftol,
                                               check_armijo=True, keep_hessian=True)
        tst.assert_approx_equal(hessian, self.hessian)

    def test_55c(self):
        logger.debug("RUNNING 5.5C")
        x0 = np.array([-1, 0, 1, 1])
        maxit = 30
        ftol = 1.0e-8
        root, hessian = undertest.quasi_newton(x0, self.q_k, self.gradient, maxit, ftol,
                                               check_armijo=False, keep_hessian=True)
        tst.assert_approx_equal(hessian, self.hessian)

    def test_55d(self):
        logger.debug("RUNNING 5.5D")
        x0 = np.array([-1, 0, 1, 1])
        maxit = 30
        ftol = 1.0e-9
        root, hessian = undertest.quasi_newton(x0, self.q_k, self.gradient, maxit, ftol,
                                               check_armijo=True, keep_hessian=True)
        tst.assert_approx_equal(hessian, self.hessian)

if __name__ == "__main__":
    unittest.main()
