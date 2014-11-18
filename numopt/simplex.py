"""
Implementation of the simplex method.

@author alex, 10/4/2014
"""
from __future__ import division

import logging
import numpy as np


fh = logging.FileHandler('simplex.log')
fh.setLevel(logging.INFO)
logger = logging.getLogger('simplex')
logger.addHandler(fh)


class Constraints(object):
    """
    A linear program is characterized by an objective function and the set of
    constraints restricting possible solutions for the program.

    Constraints can be active or inactive. Active constraints are tracked
    in the working set. Inactive constraints are tracked in the deactivated
    set.
    """

    def __init__(self, constraints, values, working_set):
        """
        :param list[int] working_set: initial vertex working set
        :param list[list[float] constraints: list of all constraints
        :param np.ndarray values:  constraint values
        """
        if len(constraints) != len(values):
            raise ValueError("Each constraint does not have a corresponding inequality value!"
                             "Constraints: {0}. Values: {1}".format(constraints, values))

        self.constraints = constraints
        self.b = values
        self.working_set = working_set
        self._build_deactivated_set()

        # Track a pointer to the working set.
        self.W_k = None
        self.update_active_constraints()

        # Track a pointer to all decreasing constraints.
        self._lambda_k = None
        self.D_k = None

        # Indices to activate and deactivate.
        self.to_activate = None
        self.to_deactivate = None

    def _build_deactivated_set(self):
        self.deactivated_set = [idx
                                for idx in range(len(self.constraints))
                                if idx not in self.working_set]

    def __len__(self):
        """ #constraints. """
        return len(self.constraints)

    def update_active_constraints(self):
        """
        Return active constraints corresponding to indices tracked in the
        working set.
        """
        self.W_k = np.array([self.constraints[idx] for idx in self.working_set])

    def get_descent_direction(self, x_k, c):
        """
        Given a vertex x_k and an objective function coefficient vector c, find the descent
        direction for this iteration of the simplex method.

        :param np.ndarray x_k: vertex
        :param np.ndarray c: coefficients of obj fn
        :return: descent direction
        """
        # Find lambda and all negative indices. If multiple lambdas < 0, take the lowest.
        lambda_k = np.linalg.solve(self.W_k.T, c)
        e_i = lambda_k < 0
        if np.sum(e_i) == 0:
            logger.info("FOUND OPTIMAL VERTEX!! Vertex = {0}, lambdas = {1}".format(
                x_k, lambda_k))
            raise _FoundOptimalVertexException(x_k)     # Terminating case!
        elif np.sum(e_i) == 1:
            relaxed_working_set_row = np.where(e_i)[0]
        else:
            relaxed_working_set_row = np.argmin(lambda_k)
        self.to_deactivate = self.working_set[relaxed_working_set_row]

        # Pointers for logger and index accounting.
        self._lambda_k = lambda_k[relaxed_working_set_row]

        # Find descent direction and step size.
        # Note: normally we do not invert the working set, but this is a toy implementation.
        e_i = np.zeros(self.W_k.shape[0])
        e_i[relaxed_working_set_row] = 1.0

        return np.linalg.solve(self.W_k, e_i)

    def find_decreasing_constraints(self, p_k):
        """
        Given a descent direction, find all decreasing constraints. If we find
        no decreasing constraints, raise an exception indicating the objective
        function is unbounded.
        """
        self.D_k = {}

        # Tracking dot values for debugging purposes.
        dot_values = []
        print self.deactivated_set
        for idx in self.deactivated_set:
            possible_decrease = np.dot(p_k, self.constraints[idx])
            dot_values.append(possible_decrease)
            if possible_decrease < 0:
                self.D_k[idx] = possible_decrease

        if len(self.D_k) == 0:
            raise Exception("Objective function is unbounded since D_k*p > 0. Terminating.\n"
                            "D_k = {0}, p = {1}".format(dot_values, p_k))

    def compute_step_size(self, x_k):
        """
        Return the next step size, where each size candidate gamma is

        \gamma = (a_i^Tx_k - b_i) / (-a_i^Tp_k)

        Also track the corresponding constraint.

        :param np.ndarray x_k: current vertex
        :return: next step size
        """

        def _gamma(vertex, idx, denominator):
            return (np.dot(vertex.T, self.constraints[idx]) - self.b[idx]) / -denominator

        # Find the correct step size, activate the corresponding constraint.
        gammas = dict((idx, _gamma(x_k, idx, denominator))
                      for (idx, denominator) in self.D_k.iteritems())
        logger.info("Obtained possible step sizes (gammas) = {0}".format(gammas))
        print gammas
        to_activate, step_size = min(gammas.iteritems(), key=lambda x: x[1])
        self.to_activate = to_activate

        return step_size

    def log_new_vertex(self, x_k, step_size, p_k, c):
        """
        Log diagnostics for new vertex.
        """
        logger.info("New vertex = {0}. l_k = {1}, alpha_k = {2}, p_k = {3}, c^Tx = {4}".format(
            x_k, self._lambda_k, step_size, p_k, np.dot(c.T, x_k)))

    def complete_iteration(self):
        """
        After each simplex iteration, we will

         > update working set and deactivated set,
         > update working matrix W_k, and
         > reset index pointers
        """
        self.working_set.remove(self.to_deactivate)
        self.working_set.append(self.to_activate)
        self.update_active_constraints()
        self._build_deactivated_set()

        # Reset indices and working set.
        self.to_activate = None
        self.to_deactivate = None


def all_inequality(c, x_k, constraints):
    """
    Run the all-inequality simplex method to find <c,x> subject to Ax >= b.

    The simplex method takes the following steps at iteration k:
      >> Check whether x_k is optimal by computing \lambda_k using c = W_k.T * \lambda_k. If
         \lambda_k >= 0, x_k is optimal, and we return it.
      >> If \lambda_k{i} < 0, we can "deactivate" the i-th constraint (so that it is no longer a
         strict equality) and obtain a descent direction p_k by solving W_k * p_k = e_k{i}, where
         e_k{i} is the i-th coordinate vector, i.e. 1 at index i and 0 otherwise.
      >> Next we need a step size a_k to take along the descent direction.
        >> If D_k, the set of decreasing constraints, e.g. D_k = {i: a_i.T * p_k < 0}, is empty,
           the method signals an unbounded objective and terminates.
        >> Otherwise we generate \gamma_i = [a_i.T * x_k - b_i] / -a_i.T * p_k for each i \in D_k.
           The maximum feasible step size a_k = min(\gamma_i) for all i \in D_k.
      >> Finally, we specify the new working set W_{k+1} = W_k - {w_{i}} + {t_k}, where t_k
         is the constraint corresponding to the maximum feasible step size.

    (comments and implementation taken from notes provided by Margaret Wright)

    :param np.ndarray c:  vector used in objective function, i.e. c.T * x
    :param np.ndarray x_k: initial feasible vertex
    :param Constraints constraints: list of all constraints
    :return: an optimal vertex for the given optimization problem
    """
    while True:
        # Create working set
        W_k = constraints.W_k
        rank = np.linalg.matrix_rank(W_k)
        logger.info("Working set: {0}. Matrix: {1}".format(
            constraints.working_set, W_k))
        logger.info("Deactivated set: {0}".format(constraints.deactivated_set))

        if rank != W_k.shape[0]:
            raise ValueError("W_k is nonsingular! Shape: %s, rank: %i".format(
                str(W_k.shape), rank))

        # Find descent direction.
        try:
            p_k = constraints.get_descent_direction(x_k, c)
        except _FoundOptimalVertexException, fove:
            return fove.vertex

        # Find all decreasing constraints. We should have at least one!
        constraints.find_decreasing_constraints(p_k)

        # Create a map from deactivated constraint index to its gamma value
        step_size = constraints.compute_step_size(x_k)

        # Update x_k.
        x_k = x_k + step_size * p_k
        constraints.log_new_vertex(x_k, step_size, p_k, c)

        # Update deactivated constraint in working set with newly-activated constraint.
        # Replace newly-activated constraint in deactivated set with newly-deactivated constraint
        constraints.complete_iteration()


class _FoundOptimalVertexException(Exception):

    def __init__(self, vertex):
        self.vertex = vertex

    def __str__(self):
        return repr(self.vertex)
