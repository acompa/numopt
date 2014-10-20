"""
Methods for one-dimensional root-finding.

@author alex, 9/16/2014
"""
from __future__ import division
import logging


def _check_initial_value(f_initial, initial):
    """
    Given an initial value, check whether it is a root. If so, log and return.

    :param f_initial:
    :param initial:
    :return:
    """
    if f_initial == 0:
        logging.info("Initial value = {0:.16f} is a root.".format(initial))
        return True
    return False


def _secant_update(current, prev, f_current, f_prev):
    """
    Return the secant update used in the secant method and regula falsi.

    :param float current: current iterate
    :param float prev: previous iterate
    :param float f_current: function value @ current iterate
    :param float f_prev: function value @ prev iterate
    :return: next iterate value
    """
    return current - f_current * (current - prev) / (f_current - f_prev)


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

    logging.info("\nStarting bisection method!")
    the_iter = 1
    while the_iter <= maxit:
        f_left = func(left)
        f_right = func(right)
        logging.info("Iteration {0}:  [{1:.16f}, {2:.16f}], f(left) = {3:.16f}, "
                     "f(right) = {4:.16f}".format(the_iter, left, right, f_left, f_right))

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
        the_iter += 1
    logging.info("Final interval: [{0:.16f}, {1:.16f}]".format(left, right))
    return None, (left, right)


def secant(func, prev, current, maxit):
    """
    Use the secant method to find a root for func.

    :param function func: find this function's roots
    :param float prev: initial guess x0
    :param float current: initial guess x1
    :param int maxit: maximum #iterations
    :return: root and interval; root is None if none found
    :rtype: tuple(float, tuple(float, float))
    """
    if current == prev:
        raise ValueError("x0 and x1 must differ.")

    # Initial checks
    f_current = func(current)
    f_prev = func(prev)
    if _check_initial_value(f_current, current): return current
    if _check_initial_value(f_prev, prev): return prev
    if f_current == f_prev:
        # TODO (ac) what do I do here?
        return None, (None, None)

    the_iter = 1
    logging.info("\nStarting secant iterations! Initial iterates: {0:.16f}, {1:.16f}".format(
        prev, current))
    while the_iter <= maxit:
        next_value = _secant_update(current, prev, f_current, f_prev)
        f_next = func(next_value)
        logging.info("Iteration {0}: x={1:.16f}, f(x)={2:.16f}".format(
            the_iter, next_value, f_next))
        if f_next == f_current or f_next == 0:
            logging.info("Root = {0:.16f} found! Last two iterates: {1:.16f}, {2:.16f}".format(
                next_value, prev, current))
            return next_value, (prev, current)

        # Update for next iteration
        f_prev = f_current
        prev = current
        f_current = f_next
        current = next_value

        the_iter += 1
    logging.info("No root found. Last two iterates: {0:.16f}, {1:.16f}".format(prev, current))
    return None, (prev, current)


def regula_falsi(func, star, current, maxit):
    """
    Use the regula falsi method to find a root for func.

    :param function func: find this function's roots
    :param float star: initial guess x0
    :param float current: initial guess x1
    :param int maxit: maximum #iterations
    :return: root and interval; root is None if none found
    :rtype: tuple(float, tuple(float, float))
    """
    if star == current:
        raise ValueError("x0 must differ.")
    f_star = func(star)
    f_current = func(current)
    if f_star * f_current >= 0:
        raise ValueError("f(x0) and f(x1) must have different signs.")

    the_iter = 1
    logging.info("\nStarting regula falsi iterations! Initial iterates: {0:.16f}, {1:.16f}".format(
        star, current))
    while the_iter <= maxit:
        next_value = _secant_update(current, star, f_current, f_star)
        f_next = func(next_value)
        logging.info("Iteration {0}: x={1:.16f}, f(x)={2:.16f}".format(
            the_iter, next_value, f_next))
        if f_next == 0:
            logging.info("Root = {0:.16f} found! x-star={1:.16f}, last iterate={2:.16f}".format(
                next_value, star, current))
            return next_value, (star, current)

        # Update k. If signs differ, interval still contains root. Save current value as star.
        if f_next * f_current < 0:
            logging.info("Updating x-star. {0:.16f} --> {1:.16f}".format(star, current))
            star = current
            f_star = f_current

        # Store the next iterate for the next iteration.
        current = next_value
        f_current = f_next
        the_iter += 1
    logging.info("No root found. x-star={0:.16f}, last iterate={1:.16f}".format(star, current))
    return None, (star, current)


def wheeler(func, star, current, maxit):
    """
    Use Wheeler's method to find a root for func.

    :param function func: find this function's roots
    :param float star: initial guess x0
    :param float current: initial guess x1
    :param int maxit: maximum #iterations
    :return: root or None if none found
    :rtype: float
    """
    f_star = func(star)
    f_current = func(current)
    if _check_initial_value(f_star, star): return star
    if _check_initial_value(f_current, current): return current
    if f_current * f_star > 0:
        # TODO (ac) what do i do here??
        return None, (None, None)

    mu = 1.0
    the_iter = 1
    logging.info("\nStarting Wheeler's iterations! Initial iterates: {0:.16f}, {1:.16f}".format(
        star, current))
    while the_iter <= maxit:
        next_value = _secant_update(current, star, f_current, mu * f_star)
        f_next = func(next_value)
        logging.info("Iteration {0}: x={1:.16f}, f(x)={2:.16f}".format(
            the_iter, next_value, f_next))

        # Reduce step size or reset it if signs change between values.
        if f_next * f_current < 0:
            mu = 1.0
            star = current
            f_star = f_current
        elif f_next == 0:
            logging.info("Root = {0:.16f} found! x-star={1:.16f}, last iterate={2:.16f}".format(
                next_value, star, current))
            return next_value, (star, current)
        else:
            mu /= 2.0

        current = next_value
        f_current = f_next
        the_iter += 1
    logging.info("No root found. x-star={0:.16f}, last iterate={1:.16f}".format(star, current))
    return None, (star, current)


def newton(func, derivative, value, maxit):
    """
    Use Newton's method to find func's roots.

    :param function func: find this function's roots
    :param function derivative: derivative of given function
    :param float value: initial guess for root
    :param int maxit: maximum #iterations
    :return: root or None if none found
    :rtype: float
    """
    f_value = func(value)
    the_iter = 1
    logging.info("\nStarting Newton's iterations! Initial value = {0:.16f}".format(value))
    while the_iter <= maxit:
        next_value = value - f_value / derivative(value)
        f_next = func(next_value)
        logging.info("Iteration {0}: x={1:.16f}, f(x)={2:.16f}".format(
            the_iter, next_value, f_next))

        # Set up next iteration.
        value = next_value
        f_value = f_next
        if f_value == 0:
            logging.info("Root = {0:.16f} found!".format(next_value))
            return value
        the_iter += 1
    logging.info("No root found. Last iterate={0:.16f}".format(value))
    return None

