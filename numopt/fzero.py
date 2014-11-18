"""
Python implementation of the textbook version of FZERO in Matlab. See p9 of

http://www.mathworks.com/moler/zeros.pdf

for details.

@author alex, 9/21/2014
"""
from __future__ import division

import math
import logging
from sys import stdout

fh = logging.FileHandler("fzerotx.log")
fh.setLevel(logging.INFO)
logger = logging.getLogger('fzerotx')
logger.addHandler(fh)


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
    logger.info("Starting with a secant step...")
    fc = fa
    d = b - c
    e = d

    # The main loop
    while fb != 0:
        if fa * fb > 0:
            logger.info("f({0}) = {1} and f({2}) = {3} have the same sign. Will not use IQI.".format(
                a, fa, b, fb))
            a = c
            fa = fc
            d = b - c
            e = d
        if abs(fa) < abs(fb):
            logger.info("|f({0})| = {1} < |f({2})| = {3}. Will not use IQI.".format(
                a, abs(fa), b, abs(fb)))
            c = b
            b = a
            a = c
            fc = fb
            fb = fa
            fa = fc
        logger.info("** NEW ITERATION! Examining [a, b] = [{0}, {1}]".format(a, b))

        # Check for convergence
        m = 0.5 * (a - b)
        tol = 2.0 * eps * max([abs(b), 1.0])
        if abs(m) <= tol or fb == 0.0:
            break

        # Bisection or an interpolation step?
        if abs(e) < tol or abs(fc) <= abs(fb):
            logger.info("Taking a bisection step.")
            d = m
            e = m
        else:
            s = fb / fc
            logger.info("a={0}, c={1}. Taking a {2} step.".format(
                a, c, "secant" if a == c else "IQI"))
            # Secant
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            # IQI
            else:
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q - r) - (b - c) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                logger.info("Flipping sign on q = {0}".format(q))
                q = -q
            else:
                logger.info("Flipping sign on p = {0}".format(p))
                p = -p

            # Do we accept the interpolation (secant/IQI) step?
            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * e * q):
                e = d
                d = p / q
                logger.info("Accepting interpolation step d={0}".format(d))
            else:
                log_string = "Rejecting interpolation step d={0} (iterate={1}) since ".format(
                    p/q, b + (p/q))
                if 2.0 * p >= 3.0 * m * q:
                    log_string += "2.0 * {0} => 3.0 * {1} * {2} = {3}".format(p, m, q, 3.0*m*q)
                else:
                    log_string += "{0} => abs(0.5 * {1} * {2}) = {3}".format(p, e, q, abs(0.5*e*q))
                logger.info(log_string)
                d = m
                e = m
                logger.info("Using bisection step d={0}".format(d))

        # Evaluate f at next iterate
        c = b
        fc = fb
        if abs(d) > tol:
            b += d
        else:
            b -= math.copysign(tol, b - a)
        fb = f(b)
        logger.info("Next iterate = {0}. f(x) = {1}".format(b, fb))

    logger.info("CONVERGED! b = {0}, f(b) = {1}".format(b, fb))
    return b


