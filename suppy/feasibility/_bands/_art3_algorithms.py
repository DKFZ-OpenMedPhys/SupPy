from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    no_gpu = False

except ImportError:
    no_gpu = True
    cp = None

from suppy.utils import LinearMapping
from suppy.utils import ensure_float_array
from suppy.feasibility._linear_algorithms import HyperslabFeasibility


class ART3plusAlgorithm(HyperslabFeasibility, ABC):
    """
    ART3plusAlgorithm class for implementing the ART3+ algorithm.

    Parameters
    ----------
    A : npt.ArrayLike
        The matrix A involved in the feasibility problem.
    lb : npt.ArrayLike
        The lower bounds for the feasibility problem.
    ub : npt.ArrayLike
        The upper bounds for the feasibility problem.
    algorithmic_relaxation : npt.ArrayLike or float, optional
        The relaxation parameter for the algorithm, by default 1.
    relaxation : float, optional
        The relaxation parameter for the feasibility problem, by default 1.
    proximity_flag : bool, optional
        Flag to indicate whether to use proximity in the algorithm, by default True.
    """

    def __init__(
        self,
        A: npt.ArrayLike,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
        algorithmic_relaxation: npt.ArrayLike | float = 1,
        relaxation: float = 1,
        proximity_flag=True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)


class SequentialART3plus(ART3plusAlgorithm):
    """
    SequentialART3plus is an implementation of the ART3plus algorithm for
    solving feasibility problems.

    Parameters
    ----------
    A : npt.ArrayLike
        The matrix representing the system of linear inequalities.
    lb : npt.ArrayLike
        The lower bounds for the variables.
    ub : npt.ArrayLike
        The upper bounds for the variables.
    cs : None or List[int], optional
        The control sequence for the algorithm. If None, it will be initialized to the range of the number of rows in A.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity in the algorithm. Default is True.

    Attributes
    ----------
    initial_cs : List[int]
        The initial control sequence.
    cs : List[int]
        The current control sequence.
    _feasible : bool
        A flag indicating whether the current solution is feasible.

    Methods
    -------
    _project(x)
        Projects the point x onto the feasible region defined by the constraints.
    solve(x, max_iter)
        Solves the feasibility problem using the ART3plus algorithm.
    """

    def __init__(
        self,
        A: npt.ArrayLike,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
        cs: None | List[int] = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, 1, 1, proximity_flag)
        if cs is None:
            self.initial_cs = np.arange(self.A.shape[0])
        else:
            self.initial_cs = cs

        self.cs = self.initial_cs.copy()
        self._feasible = True

    def _project(self, x: npt.ArrayLike) -> npt.ArrayLike:
        to_remove = []
        for i in self.cs:
            # TODO: add a boolean variable that skips this if the projection did not move the point?
            p_i = self.single_map(x, i)
            # should be precomputed
            if (
                3 / 2 * self.Bounds.l[i] - 1 / 2 * self.Bounds.u[i] <= p_i < self.Bounds.l[i]
            ):  # lowe bound reflection
                self.A.update_step(
                    x, 2 * self.inverse_row_norm[i] * (self.Bounds.l[i] - p_i), i
                )  # reflection
                self._feasible = False

            elif (
                self.Bounds.u[i] < p_i <= 3 / 2 * self.Bounds.u[i] - 1 / 2 * self.Bounds.l[i]
            ):  # upper bound reflection
                self.A.update_step(
                    x, 2 * self.inverse_row_norm[i] * (self.Bounds.u[i] - p_i), i
                )  # reflection
                self._feasible = False

            elif self.Bounds.u[i] - self.Bounds.l[i] < np.abs(
                p_i - (self.Bounds.l[i] + self.Bounds.u[i]) / 2
            ):
                self.A.update_step(
                    x,
                    self.inverse_row_norm[i] * (self.Bounds.l[i] + self.Bounds.u[i]) / 2 - p_i,
                    i,
                )  # projection onto center of hyperslab
                self._feasible = False

            else:  # constraint is already met
                to_remove.append(i)

        # after loop remove constraints that are already met
        self.cs = [i for i in self.cs if i not in to_remove]  # is this fast?
        return x

    def solve(self, x, max_iter):
        # option to reset control sequence?

        for i in range(max_iter):
            if len(self.cs) == 0 and self._feasible:  # ran over all constraints and still feasible
                return x
            elif len(self.cs) == 0 and not self._feasible:
                self.cs = self.initial_cs.copy()
                self._feasible = True

            x = self.step(x)
        return x


class SimultaneousART3plus(ART3plusAlgorithm):
    """
    SimultaneousART3plus is an implementation of the ART3plus algorithm for
    solving feasibility problems.

    Parameters
    ----------
    A : npt.ArrayLike
        The matrix representing the system of linear inequalities.
    lb : npt.ArrayLike
        The lower bounds for the variables.
    ub : npt.ArrayLike
        The upper bounds for the variables.
    weights : None | List[float] | npt.ArrayLike, optional
        The weights for the constraints. If None, default weights are used. Default is None.
    proximity_flag : bool, optional
        Flag to indicate whether to use proximity measure. Default is True.

    Attributes
    ----------
    weights : npt.ArrayLike
        The weights for the constraints.
    _feasible : bool
        Flag indicating whether the current solution is feasible.
    _not_met : npt.ArrayLike
        Indices of constraints that are not met.
    _not_met_init : npt.ArrayLike
        Initial indices of constraints that are not met.

    Methods
    -------
    _project(x)
    proximity(x)
        Calculate the proximity measure for a given solution vector x.
    solve(x, max_iter)
    """

    def __init__(
        self,
        A: npt.ArrayLike,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
        weights: None | List[float] | npt.ArrayLike = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, 1, 1, proximity_flag)

        if weights is None:
            self.weights = np.ones(self.A.shape[0]) / self.A.shape[0]
        elif np.abs((weights.sum() - 1)) > 1e-10:
            print("Weights do not add up to 1! Choosing default weight vector...")
            self.weights = np.ones(self.A.shape[0]) / self.A.shape[0]
        else:
            self.weights = weights

        self._feasible = True
        if self.A._gpu:
            self._not_met = cp.arange(self.A.shape[0])
        else:
            self._not_met = np.arange(self.A.shape[0])

        self._not_met_init = self._not_met.copy()
        self._feasible = True

    def _project(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Perform one step of the ART3plus algorithm.

        Args:
            x (npt.ArrayLike): The point to be projected.

        Returns:
            npt.ArrayLike: The projected point.
        """
        p = self.map(x)
        p = p[self._not_met]
        l_redux = self.Bounds.l[self._not_met]
        u_redux = self.Bounds.u[self._not_met]

        # following calculations are performed on subarrays
        # assign different subsets
        idx_1 = p < l_redux
        idx_2 = p > u_redux
        idx_3 = p < l_redux - (u_redux - l_redux) / 2
        idx_4 = p > u_redux + (u_redux - l_redux) / 2

        # sets on subarrays
        set_1 = idx_1 & (not idx_3)  # idxs for lower bound reflection
        set_2 = idx_2 & (not idx_4)  # idxs for upper bound reflection
        set_3 = idx_3 | idx_4  # idxs for projections
        # there should be no overlap between the different regions here!
        x += (
            self.weights[self._not_met][set_1]
            * self.inverse_row_norm[self._not_met][set_1]
            * (2 * (l_redux - p))[set_1]
            @ self.A[self._not_met][set_1, :]
        )
        x += (
            self.weights[self._not_met][set_2]
            * self.inverse_row_norm[self._not_met][set_2]
            * (2 * (u_redux - p))[set_2]
            @ self.A[self._not_met][set_2, :]
        )
        x += (
            self.weights[self._not_met][set_3]
            * self.inverse_row_norm[self._not_met][set_3]
            * ((l_redux + u_redux) / 2 - p)[set_3]
            @ self.A[self._not_met][set_3, :]
        )

        # remove constraints that were already met before
        self._not_met = self._not_met[(idx_1 | idx_2)]

        if idx_1.sum() > 0 or idx_2.sum() > 0:
            self._feasible = False

        return x

    def proximity(self, x: npt.ArrayLike) -> float:
        p = self.map(x)
        (res_u, res_l) = self.Bounds.residual(p)  # residuals are positive  if constraints are met
        d_idx = res_u < 0
        c_idx = res_l < 0
        return np.sum(self.weights[d_idx] * res_u[d_idx] ** 2) + np.sum(
            self.weights[c_idx] * res_l[c_idx] ** 2
        )

    @ensure_float_array
    def solve(self, x: npt.ArrayLike, max_iter: int):
        for i in range(max_iter):
            if (
                len(self._not_met) == 0 and self._feasible
            ):  # ran over all constraints and still feasible
                return x
            elif len(self._not_met) == 0 and not self._feasible:
                self._not_met = self._not_met_init.copy()
                self._feasible = True

            x = self.step(x)
            if i % 100 == 0:
                print("A")
        return x
