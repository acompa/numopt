"""
Tests for root-finding methods.
"""

from __future__ import division
import logging
import unittest

from numopt import root as undertest

logging.basicConfig(filename="root.log", level=logging.INFO)


class TestOneDimRootFinding(unittest.TestCase):

    def setUp(self):
        self.func = lambda x: x**3 - 8
        self.derivative = lambda x: 3 * x**2
        self.maxit = 12
        self.desired_root = 2.0

    def test_root(self):
        pass

    def test_bisection_1b(self):
        """
        Unit test for exercise 1.1b.
        """
        logging.info("\nANSWERS TO EXERCISE 1.1B")
        left = 0.5
        right = 3.1

        # The final interval should contain the desired root.
        root, (left, right) = undertest.bisection(self.func, left, right, self.maxit)
        self.assertTrue(_root_in_interval(self.desired_root, left, right))

    def test_bisection_1de(self):
        """
        Unit test for exercise 1.1d and 1.1e.
        """
        logging.info("\nANSWERS TO EXERCISE 1.1D")
        func = lambda x: x**7 - 7 * x**6 + 21 * x**5 - 35 * x**4 + 35 * x**3 - 21 * x**2 + 7 * x - 1
        starting_left = 0.95
        starting_right = 1.01

        # Margaret chose this case as a funky example of bisection. The root should NOT be in
        # the interval for this function.
        root, (left, right) = undertest.bisection(func, starting_left, starting_right, self.maxit)
        desired_root = 1.0
        self.assertFalse(_root_in_interval(desired_root, left, right))

        # Now let's try the factored form of func. Here the interval SHOULD contain the true root.
        logging.info("\nRUNNING EXERCISE 1.1E")
        factored_func = lambda x: (x - 1)**7
        root, (left, right) = undertest.bisection(
            factored_func, starting_left, starting_right, self.maxit)
        self.assertTrue(_root_in_interval(desired_root, left, right))

    def test_3(self):
        """
        Unit test for exercise 1.3c-g.
        """
        # Run Newton's method. Last starting point is wildly high. How will Newton perform??
        starting_points = (0.1, 4.0, -0.2, -0.1)
        assert len(starting_points) == 4

        logging.info("\nRUNNING EXERCISE 1.3C")
        newton_roots = [undertest.newton(self.func, self.derivative, x0, 50)
                for x0 in starting_points]

        # Run secant-based methods. Last interval has a high right endpoint. How will the algos do?
        secant_intervals = [(0.9, 10.0), (-0.2, 3.0), (0.1, 6.0), (1.9, 20.0), (20.0, 1.9)]
        assert len(secant_intervals) == 5
        logging.info("\nRUNNING EXERCISE 1.3D")
        secant_results = [undertest.secant(self.func, prev, current, self.maxit)
                for (prev, current) in secant_intervals]
        logging.info("\nRUNNING EXERCISE 1.3E")
        regula_falsi_results = [undertest.regula_falsi(self.func, prev, current, 100)
                for (prev, current) in secant_intervals]
        logging.info("\nRUNNING EXERCISE 1.3F")
        wheeler_results = [undertest.wheeler(self.func, prev, current, 20)
                for (prev, current) in secant_intervals]

    def test_figure34(self):
        """
        Unit test confirming iterations in figure 3.4.
        """
        star = 0.1
        current = 1.37
        func = lambda x: x**6 + 3 * x - 4

        logging.info("\nCONFIRMING FIGURE 3.4")
        rf_results = undertest.regula_falsi(func, star, current, 100)


def _root_in_interval(root, left, right):
    return left <= root <= right
