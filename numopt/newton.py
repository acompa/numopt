"""
Pure Newton methods with optional backtracking line search (as requested by hw assignment).

@author alex
"""
from __future__ import division

import logging
import numpy as np

ALLOWED_UPDATES = ['bfgs']

fh = logging.FileHandler('newton.log')
fh.setLevel(logging.DEBUG)
logger = logging.getLogger('newton')
logger.addHandler(fh)


def pure_newton(x0, my_func, my_jacobian, merit_fn, maxit, ftol, run_line_search=True, eta=0.001,
                gamma=0.5):
    """
    Pure Newton method, with optional backtracking line search, for estimating zeros of functions
    in R^n.

    Pure Newton methods will operate on the Jacobian of the target function. If you do not have
    access to the Jacobian, you probably want a quasi-Newton method that estimates the Hessian on
    each iteration.

    :param np.ndarray x0:         initial estimate of root
    :param function my_func:      function for which we need root; takes x_k as arg
    :param function my_jacobian:  function returning m x n ndarray of Jacobian at x_k
    :param function merit_fn:     function used to evaluate step size
    :param int maxit:             max iterations
    :param float ftol:            tolerance for evaluating possible zero of my_func
    :param bool run_line_search:  if True, use backtracking line search to find step size
    :param float eta:             parameter for line search
    :param float gamma:           parameter for line search
    :return: zero if one is found, or last iterate if none found
    """
    if not 0 < eta < 1:
        raise ValueError("eta = {0} does not satisfy 0 < eta < 1.".format(eta))
    if not 0 < gamma < 1:
        raise ValueError("gamma = {0} does not satisfy 0 < gamma < 1.".format(gamma))
    logger.info("Pure Newton: x0={0}, eta={1}, gamma={2}".format(x0, eta, gamma))

    # Setup
    k = 0
    x_k = x0
    jacobian_k = my_jacobian(x_k)
    func_k = my_func(x_k)

    # Continue looping until convergence.
    while k < maxit:
        # Find direction
        # TODO (ac) replace inverse with other method
        p_k = -np.dot(np.linalg.inv(jacobian_k), func_k)

        # If line searching, reduce step size until merit function indicates sufficient decrease
        alpha_k = 1.0
        if run_line_search:
            j_k = 0
            alphas = []
            merits = []
            merit_k = merit_fn(my_func(x_k + alpha_k * p_k))
            while merit_k > (1 - alpha_k * eta) * merit_fn(my_func(x_k)):
                alpha_k *= gamma
                j_k += 1
                merit_k = merit_fn(my_func(x_k + alpha_k * p_k))

                # Append logger info
                alphas.append(alpha_k)
                merits.append(merit_k)
        else:
            merit_k = merit_fn(my_func(x_k))

        # Logging, as requested
        log_str = "Iter {0}: x_k = {1}, F_k = {2}, ||F_k|| = {3}.".format(k, x_k, func_k, merit_k)
        if run_line_search:
            log_str += " alphas: {0}. merit functions: {1}".format(alphas, merits)
        logger.info(log_str)

        # Stopping condition
        if merit_k < ftol:
            logger.info("MERIT FUNCTION ~= 0! x_k={0}.".format(x_k))
            break

        # Prepare for next iteration
        x_k += alpha_k * p_k
        k += 1
        jacobian_k = my_jacobian(x_k)
        func_k = my_func(x_k)

    logger.info("Final iterate x_k={0}".format(x_k))
    return x_k


def quasi_newton(x0, my_func, my_gradient, maxit, ftol, check_armijo=True, eta=0.001, gamma=0.5,
                 update='bfgs', keep_hessian=False):
    """
    Quasi-Newton method with Hessian approximation provided by user. Optionally checks updates
    against Armijo rule.

    If you have access to the function's Jacobian, consider a pure Newton method instead!

    :param np.ndarray x0:         initial estimate of root
    :param function my_func:      function for which we need root; takes x_k as arg
    :param my_gradient:           function returning value of gradient at a point
    :param int maxit:             max iterations
    :param float ftol:            tolerance for evaluating possible zero of my_func
    :param check_armijo:          if true, check step size against Armijo rule
    :param float eta:             parameter for line search
    :param float gamma:           parameter for line search
    :param update:                Hessian update type; 'bfgs' or 'dfp'
    :param keep_hessian:          if true, return last Hessian approximation
    :return: return last iterate (or zero if converged); return tuple is keep_hessian is true
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    if not 0 < eta < 1:
        raise ValueError("eta = {0} does not satisfy 0 < eta < 1.".format(eta))
    if not 0 < gamma < 1:
        raise ValueError("gamma = {0} does not satisfy 0 < gamma < 1.".format(gamma))

    # Hessian update types
    if update == 'dfp':
        # TODO (ac) implement DFP update
        raise NotImplementedError("DFP update not yet supported for this method!")

    if update not in ['bfgs', 'dfp']:
        raise ValueError("Provided quasi-Newton update {0}, must be {1}"
                         .format(update, ALLOWED_UPDATES))

    log_str = "Quasi-Newton: x0={0}.".format(x0)
    if check_armijo:
        log_str += " eta={1}, gamma={2}".format(x0, eta, gamma)
    logger.info(log_str)

    # Setup
    k = 0
    x_k = x0
    g_k = my_gradient(x_k)
    f_k = my_func(x_k)
    B_k = np.eye(x0.shape[0])
    norm_g_k = np.linalg.norm(g_k)

    logger.info("At starting iteration, g_k={0}, f_k={1}.".format(g_k, f_k))
    while k < maxit:
        if norm_g_k < ftol:
            logger.info("GRADIENT ~= 0! x_k = {0}.".format(x_k))
            break

        # Get direction
        # TODO (ac) replace inverse with other method
        p_k = -np.dot(np.linalg.inv(B_k), g_k)

        # Get step size using Armijo if requested.
        if check_armijo:
            alpha_k = 1.0
            while my_func(x_k + alpha_k * p_k) > (f_k + eta * alpha_k * np.dot(p_k, g_k)):
                alpha_k *= gamma
        else:
            alpha_k = -np.dot(g_k, p_k) / np.dot(np.dot(p_k, B_k), p_k)

        # Update step, x, gradient, y.
        s_k = alpha_k * p_k
        x_next = x_k + s_k
        g_next = my_gradient(x_next)
        y_k = g_next - g_k

        # Update Hessian approximation using BFGS
        if update == 'bfgs' and np.dot(y_k, s_k) > 0:
            logger.info("Updating B_k since {0} > 0".format(np.dot(y_k, s_k)))
            B_k = (B_k
                   - 1 / np.dot(np.dot(s_k, B_k), s_k)
                   * np.outer(np.dot(B_k, s_k), np.dot(s_k, B_k))
                   + 1 / np.dot(y_k, s_k) * np.outer(y_k, y_k))

        # Final updates.
        x_k = x_next
        f_k = my_func(x_k)
        g_k = g_next
        norm_g_k = np.linalg.norm(g_k)

        logger.info("Iteration {0}: x_k={1}, f_k={2}, ||g_k||={3}, alpha_k={4}, B_k={5}".format(
            k, x_k, f_k, norm_g_k, alpha_k, B_k))
        k += 1

    if keep_hessian:
        return x_k, B_k
    return x_k

