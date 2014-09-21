"""
Python implementation of the textbook version of FZERO in Matlab. See p9 of

http://www.mathworks.com/moler/zeros.pdf

for details.

@author alex, 9/21/2014
"""

import math
import logging
from sys import stdout

# logging.basicConfig(filename="fzerotx.log", level=logging.INFO)
logging.basicConfig(level=logging.INFO, stream=stdout)

def fzerotx(f, x0, x1):
    """
    Find a root of f within the interval [x0, x1].

    :param function f: function to evaluate
    :param float x0: initial left endpoint
    :param float x1: initial right endpoint
    :return: zero or None
    :rtype: float
    """
    # eps is not specified but I suppose its very very small
    eps = 1e-8

    a = x0
    b = x1
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a)={0} and f(b)={1} have the same sign".format(fa, fb))

    c = a
    fc = fa
    d = b - c
    e = d

    # The main loop
    while fb != 0:
        if fa * fb > 0:
            a = c
            fa = fc
            d = b - c
            e = d
        if abs(fa) < abs(fb):
            c = b
            b = a
            a = c
            fc = fb
            fb = fa
            fa = fc

        # Check for convergence
        m = 0.5 * (a - b)
        tol = 2.0 * eps * max([abs(b), 1.0])
        if abs(m) <= tol or fb == 0.0:
            break

        # Bisection or an interpolation step?
        if abs(e) < tol or abs(fc) <= abs(fb):
            logging.info("Taking a bisection step.")
            d = m
            e = m
        else:
            s = fb / fc
            if a == c:
                logging.info("a={0} == c={1}. Taking a secant step.".format(a, c))
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                logging.info("a={0} != c={1}. Taking an IQI step.".format(a, c))
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q - r) - (b - c) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

        if p > 0:
            logging.info("Flipping sign on q = {0}".format(q))
            q = -q
        else:
            logging.info("Flipping sign on p = {0}".format(p))
            p = -p

        # Do we accept the interpolation (secant/IQI) step?
        if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * e * q):
            e = d
            d = p / q
            logging.info("Accepting interpolation step d={0}".format(d))
        else:
            d = m
            e = m
            logging.info(
                "Rejecting interpolation step d={0}. Using bisection={1}.".format(p/q, d))

        # Evaluate f at next iterate
        c = b
        fc = fb
        if abs(d) > tol:
            b += d
        else:
            b -= math.copysign(tol, b - a)
        fb = f(b)
        logging.info("Next iterate = {0}".format(b))

    logging.info("CONVERGED! b = {0}, f(b) = {1}".format(b, fb))
    return b


if __name__ == "__main__":
    f1 = lambda x: 2 * x ** 3 - 4 * x ** 2 + 3 * x + 1
    x0, x1 = (-2, 2)
    logging.info("ANALYZING f = 2x^3 - 4x^2 + 3x + 1")
    fzerotx(f1, x0, x1)

    f2 = lambda x: 1.1 * x ** 3 - 2.6 * x - 2.6049
    x0, x1 = (-1.95, 2.4)
    logging.info("\nANALYZING f = 1.1x^3 - 2.6x - 2.6049")
    fzerotx(f2, x0, x1)
