"""
Methods for one-dimensional root-finding.

@author alex, 9/.16/2014
"""
from __future__ import division
import logging

logging.basicConfig(filename="root.log", level=logging.INFO)


def bisection(func, left, right, maxit, epsilon=0.0):
    """
    Use the bisection method to find roots for the given function f. Stopping conditions:

    >> we find an exact zero of f
    >> the interval of uncertainty is <= epsilon
    >> we perform $maxit #iterations

    :param function func: function to evaluate; must be one-dimensional
    :param float left: left bound on search interval
    :param float right: right bound on search interval
    :param float epsilon: minimum interval size
    :param int maxit: max #iterations
    :return: tuple: root of f and final interval; None if no root is found
    :rtype: tuple(float, tuple(float, float))
    """
    if func(left) * func(right) > 0:
        raise ValueError("Left and right interval endpoints do not seem to contain a zero!")

    iter = 0
    while iter < maxit:
        f_left = func(left)
        f_right = func(right)
        logging.info("Iteration {0}:  [{1:.16f}, {2:.16f}], f(left) = {3:.16f}, "
                     "f(right) = {4:.16f}".format(iter, left, right, f_left, f_right))

        # Evaluate midpoint.
        mid = (left + right) / 2
        f_mid = func(mid)
        if f_mid == 0:
            return mid, (left, right)
        elif f_mid * f_left < 0:
            right = mid     # narrow interval from the right
        else:
            left = mid      # narrow interval from the left

        # Terminate if |interval| < epsilon
        if right - left < epsilon:
            logging.info("Interval [{0:.16f}, {1:.16f}] is smaller than epsilon={3:.16f}. "
                    "Terminating...".format(left, right, epsilon))
            return None
        iter += 1
    logging.info("Final interval: [{0:.16f}, {1:.16f}]".format(left, right))
    return None, (left, right)


def newton(func, initial):
    """
    Use Newton's method to find func's roots.

    :param function func: find this function's roots
    :param float initial: initial guess for root
    :return: root or None if none found
    :rtype: float
    """
    pass


def regula_falsi(func, x0, x1):
    """
    Use the regula falsi method to find a root for func.

    :param function func: find this function's roots
    :param x0: initial guess x0
    :param x1: initial guess x1
    :return: root or None if none found
    :rtype: float
    """
    pass


def secant(func, x0, x1):
    """
    Use the secant method to find a root for func.

    :param function func: find this function's roots
    :param x0: initial guess x0
    :param x1: initial guess x1
    :return: root or None if none found
    :rtype: float
    """
    pass


def wheeler(func, x0, x1):
    """
    Use Wheeler's method to find a root for func.

    :param function func: find this function's roots
    :param x0: initial guess x0
    :param x1: initial guess x1
    :return: root or None if none found
    :rtype: float
    """
    pass
