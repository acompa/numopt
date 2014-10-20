"""
Running FZERO on various problems, producing logging output for responses.

@author alex
"""

import logging
import math
from unittest import TestCase

from numopt import fzero as undertest
from numopt.root import wheeler


class FZeroUnitTest(TestCase):

    def test_case_one(self):
        f1 = lambda x: 2 * x ** 3 - 4 * x ** 2 + 3 * x + 1
        x0, x1 = (-2, 2)
        logging.info("ANALYZING f = 2x^3 - 4x^2 + 3x + 1")
        out = undertest.fzerotx(f1, x0, x1)
        self.assertIsNotNone(out)

    def test_case_two(self):
        f2 = lambda x: 1.1 * x ** 3 - 2.6 * x - 2.6049
        x0, x1 = (-1.95, 2.4)
        logging.info("\nANALYZING f = 1.1x^3 - 2.6x - 2.6049")
        out = undertest.fzerotx(f2, x0, x1)
        self.assertIsNotNone(out)

        wheeler(f2, x0, x1, 15)

    def test_case_three(self):
        f3 = lambda x: math.exp(x) - 10**-3
        x0, x1 = (-5.0, -20)
        logging.info("\nANALYZING f = e^x - 0.001")
        out = undertest.fzerotx(f3, x0, x1)
        self.assertIsNotNone(out)

    def test_case_four(self):
        f4 = lambda x: x**7 - 7 * x**6 + 21 * x**5 - 35 * x**4 + 35 * x**3 - 21 * x**2 + 7 * x - 1
        x0, x1 = (0.95, 1.01)
        logging.info("\nANALYZING CANCELLATION FUNCTION FROM FIRST HOMEWORK")
        out = undertest.fzerotx(f4, x0, x1)
        self.assertIsNotNone(out)
