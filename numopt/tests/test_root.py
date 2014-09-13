"""
Tests for root-finding methods.
"""

from __future__ import division
import logging
import unittest

from numopt import root as undertest

logging.basicConfig(filename="root.log", level=logging.INFO)


class TestOneDimRootFinding(unittest.TestCase):

    def test_root(self):
        pass

    def test_bisection_1b(self):
        """
        This is the unit test for exercise 1.1b.

        Test bisection method using parameters listed below.
        """
        logging.info("\nANSWERS TO EXERCISE 1.1B")
        func = lambda x: x**3 - 8
        left = 0.5
        right = 3.1
        maxit = 12

        # The final interval should contain the desired root.
        root, (left, right) = undertest.bisection(func, left, right, maxit)
        desired_root = 2.0
        self.assertTrue(left < desired_root and right > desired_root)

    def test_bisection_1d(self):
        """
        Unit test for exercise 1.1d.
        """
        logging.info("\nANSWERS TO EXERCISE 1.1D")
        func = lambda x: x**7 - 7 * x**6 + 21 * x**5 - 35 * x**4 + 35 * x**3 - 21 * x**2 + 7 * x - 1
        left = 0.95
        right = 1.01
        maxit = 12

        root, (left, right) = undertest.bisection(func, left, right, maxit)
