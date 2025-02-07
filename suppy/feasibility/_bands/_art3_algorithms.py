from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = np

from suppy.utils import ensure_float_array
from suppy.feasibility._linear_algorithms import HyperslabFeasibility


class ART3plusAlgorithm(HyperslabFeasibility, ABC):
    """
    ART3plusAlgorithm class for implementing the ART3+ algorithm.

    Parameters
    ----------
    A : npt.NDArray
        The matrix A involved in the feasibility problem.
    lb : npt.NDArray
        The lower bounds for the feasibility problem.
    ub : npt.NDArray
        The upper bounds for the feasibility problem.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.
    relaxation : float, optional
        The relaxation parameter for the feasibility problem, by default 1.
    proximity_flag : bool, optional
        Flag to indicate whether to use proximity in the algorithm, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1,
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
    A : npt.NDArray
        The matrix representing the system of linear inequalities.
    lb : npt.NDArray
        The lower bounds for the variables.
    ub : npt.NDArray
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
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        cs: None | List[int] = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, 1, 1, proximity_flag)
        xp = cp if self.A.gpu else np
        if cs is None:
            self.initial_cs = xp.arange(self.A.shape[0])
        else:
            self.initial_cs = cs

        self.cs = self.initial_cs.copy()

    def _project(self, x: npt.NDArray) -> npt.NDArray:
        to_remove = []
        for i in self.cs:
            # TODO: add a boolean variable that skips this if the projection did not move the point?
            p_i = self.single_map(x, i)
            # should be precomputed
            if (
                3 / 2 * self.bounds.l[i] - 1 / 2 * self.bounds.u[i] <= p_i < self.bounds.l[i]
            ):  # lowe bound reflection
                self.A.update_step(
                    x, 2 * self.inverse_row_norm[i] * (self.bounds.l[i] - p_i), i
                )  # reflection

            elif (
                self.bounds.u[i] < p_i <= 3 / 2 * self.bounds.u[i] - 1 / 2 * self.bounds.l[i]
            ):  # upper bound reflection
                self.A.update_step(
                    x, 2 * self.inverse_row_norm[i] * (self.bounds.u[i] - p_i), i
                )  # reflection

            elif self.bounds.u[i] - self.bounds.l[i] < abs(
                p_i - (self.bounds.l[i] + self.bounds.u[i]) / 2
            ):
                self.A.update_step(
                    x,
                    self.inverse_row_norm[i] * ((self.bounds.l[i] + self.bounds.u[i]) / 2 - p_i),
                    i,
                )  # projection onto center of hyperslab

            else:  # constraint is already met
                to_remove.append(i)

        # after loop remove constraints that are already met
        self.cs = [i for i in self.cs if i not in to_remove]  # is this fast?
        return x

    @ensure_float_array
    def solve(
        self,
        x: npt.NDArray,
        max_iter: int = 500,
        constr_tol: float = 1e-6,
        storage: bool = False,
        proximity_measures: List | None = None,
    ) -> npt.NDArray:
        """
        Solves the optimization problem using an iterative approach.

        Parameters
        ----------
        x : npt.NDArray
            Initial guess for the solution.
        max_iter : int, optional
            Maximum number of iterations to perform.
        storage : bool, optional
            Flag indicating whether to store the intermediate solutions, by default False.
        constr_tol : float, optional
            The tolerance for the constraints, by default 1e-6.
        proximity_measures : List, optional
            The proximity measures to calculate, by default None. Right now only the first in the list is used to check the feasibility.

        Returns
        -------
        npt.NDArray
            The solution after the iterative process.
        """
        self.cs = self.initial_cs.copy()
        xp = cp if isinstance(x, cp.ndarray) else np
        if proximity_measures is None:
            proximity_measures = [("p_norm", 2)]
        else:
            # TODO: Check if the proximity measures are valid
            _ = None

        self.proximities = []
        i = 0
        feasible = False

        if storage is True:
            self.all_x = []
            self.all_x.append(x.copy())

        while i < max_iter and not feasible:

            if len(self.cs) == 0:
                self.cs = self.initial_cs.copy()

            x = self.project(x)
            if storage is True:
                self.all_x.append(x.copy())
            self.proximities.append(self.proximity(x, proximity_measures))

            # TODO: If proximity changes x some potential issues!
            if self.proximities[-1][0] < constr_tol:

                feasible = True
            i += 1
        if self.all_x is not None:
            self.all_x = xp.array(self.all_x)
        return x


class SimultaneousART3plus(ART3plusAlgorithm):
    """
    SimultaneousART3plus is an implementation of the ART3plus algorithm for
    solving feasibility problems.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the system of linear inequalities.
    lb : npt.NDArray
        The lower bounds for the variables.
    ub : npt.NDArray
        The upper bounds for the variables.
    weights : None | List[float] | npt.NDArray, optional
        The weights for the constraints. If None, default weights are used. Default is None.
    proximity_flag : bool, optional
        Flag to indicate whether to use proximity measure. Default is True.

    Attributes
    ----------
    weights : npt.NDArray
        The weights for the constraints.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, 1, 1, proximity_flag)
        xp = cp if self.A.gpu else np
        if weights is None:
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        elif abs((weights.sum() - 1)) > 1e-10:
            print("Weights do not add up to 1! Choosing default weight vector...")
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        else:
            self.weights = weights

        self._not_met = xp.arange(self.A.shape[0])

        self._not_met_init = self._not_met.copy()

    def _project(self, x: npt.NDArray) -> npt.NDArray:
        """
        Perform one step of the ART3plus algorithm.

        Args:
            x (npt.NDArray): The point to be projected.

        Returns:
            npt.NDArray: The projected point.
        """
        p = self.map(x)
        p = p[self._not_met]
        l_redux = self.bounds.l[self._not_met]
        u_redux = self.bounds.u[self._not_met]

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

        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List[str]) -> float:
        p = self.map(x)
        # residuals are positive if constraints are met
        (res_l, res_u) = self.bounds.residual(p)
        res_u[res_u > 0] = 0
        res_l[res_l > 0] = 0
        res = -res_u - res_l
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(self.weights @ (res ** measure[1]))
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure)")
        return measures
