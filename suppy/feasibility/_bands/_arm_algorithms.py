from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt
from suppy.utils import LinearMapping
from suppy.feasibility._linear_algorithms import HyperslabFeasibility

try:
    import cupy as cp

    no_gpu = False

except ImportError:
    no_gpu = True
    cp = np


class ARMAlgorithm(HyperslabFeasibility, ABC):
    """
    ARMAlgorithm class for handling feasibility problems with additional
    algorithmic relaxation.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the linear mapping.
    lb : npt.NDArray
        The lower bounds for the feasibility problem.
    ub : npt.NDArray
        The upper bounds for the feasibility problem.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter specific to the algorithm, by default 1.
    relaxation : float, optional
        The general relaxation parameter, by default 1.
    proximity_flag : bool, optional
        Flag to indicate if proximity constraints should be considered, by default True.
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


class SequentialARM(ARMAlgorithm):
    """
    SequentialARM is a class that implements a sequential algorithm for
    Adaptive Relaxation Method (ARM).

    Parameters
    ----------
    A : npt.NDArray
        The matrix A used in the ARM algorithm.
    lb : npt.NDArray
        The lower bounds for the variables.
    ub : npt.NDArray
        The upper bounds for the variables.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.
    relaxation : float, optional
        The relaxation parameter, by default 1.
    cs : None or List[int], optional
        The list of indices for the constraints, by default None.
    proximity_flag : bool, optional
        Flag to indicate if proximity should be considered, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1,
        relaxation: float = 1,
        cs: None | List[int] = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        self._k = 0  # relaxation power
        if cs is None:
            self.cs = xp.arange(self.A.shape[0])
        else:
            self.cs = cs

    def _project(self, x: npt.NDArray) -> npt.NDArray:
        xp = cp if self._use_gpu else np

        for i in self.cs:
            p_i = self.single_map(x, i)
            d = p_i - self.Bounds.center[i]
            psi = (self.Bounds.u[i] - self.Bounds.l[i]) / 2
            if xp.abs(d) > psi:
                self.A.update_step(
                    x,
                    -1
                    * self.algorithmic_relaxation**self._k
                    / 2
                    * self.inverse_row_norm[i]
                    * ((d**2 - psi**2) / d),
                    i,
                )
        return x


class SimultaneousARM(ARMAlgorithm):
    """
    SimultaneousARM is a class that implements an ARM (Adaptive Relaxation
    Method) algorithm
    for solving feasibility problems.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the constraints.
    lb : npt.NDArray
        The lower bounds for the constraints.
    ub : npt.NDArray
        The upper bounds for the constraints.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm. Default is 1.
    relaxation : float, optional
        The relaxation parameter for the constraints. Default is 1.
    weights : None, List[float], or npt.NDArray, optional
        The weights for the constraints. If None, weights are set to be uniform. Default is None.
    proximity_flag : bool, optional
        Flag to indicate whether to use proximity in the algorithm. Default is True.

    Methods
    -------
    _project(x)
        Performs the simultaneous projection of the input vector x.
    _proximity(x)
        Computes the proximity measure of the input vector x.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1,
        relaxation: float = 1,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        self._k = 0
        xp = cp if self._use_gpu else np

        if weights is None:
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        elif xp.abs((weights.sum() - 1)) > 1e-10:
            print("Weights do not add up to 1! Renormalizing to 1...")
            self.weights = weights / weights.sum()
        else:
            self.weights = weights

    def _project(self, x):
        xp = cp if self._use_gpu else np
        # simultaneous projection
        p = self.map(x)
        d = p - self.Bounds.center
        psi = self.Bounds.half_distance
        d_idx = xp.abs(d) > psi
        x -= (
            self.algorithmic_relaxation**self._k
            / 2
            * (
                self.weights[d_idx]
                * self.inverse_row_norm[d_idx]
                * (d[d_idx] - (psi[d_idx] ** 2) / d[d_idx])
            )
            @ self.A[d_idx, :]
        )

        self._k += 1
        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List[str]) -> float:
        p = self.map(x)
        # residuals are positive if constraints are met
        (res_l, res_u) = self.Bounds.residual(p)
        res_u[res_u > 0] = 0
        res_l[res_l > 0] = 0
        res = -res_u - res_l
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(1 / len(res) * self.weights @ (res ** measure[1]))
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure)")
        return measures


class BIPARM(ARMAlgorithm):
    """
    BIPARM Algorithm for feasibility problems.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the constraints.
    lb : npt.NDArray
        The lower bounds for the constraints.
    ub : npt.NDArray
        The upper bounds for the constraints.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.
    relaxation : float, optional
        The relaxation parameter for the projections, by default 1.
    weights : None, List[float], or npt.NDArray, optional
        The weights for the constraints, by default None.
    proximity_flag : bool, optional
        Flag to indicate if proximity should be considered, by default True.

    Methods
    -------
    _project(x)
        Perform the simultaneous projection of x.
    _proximity(x)
        Calculate the proximity measure for x.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1,
        relaxation: float = 1,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        self._k = 0
        if weights is None:
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]

        # if check_weight_validity(weights):
        #    self.weights = weights
        elif xp.abs((weights.sum() - 1)) > 1e-10:
            print("Weights do not add up to 1! Choosing default weight vector...")
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        else:
            self.weights = weights

    def _project(self, x):
        # simultaneous projection
        p = self.map(x)
        d = p - self.Bounds.center
        psi = self.Bounds.half_distance
        d_idx = abs(d) > psi
        x -= (
            self.algorithmic_relaxation**self._k
            / 2
            * (
                self.weights[d_idx]
                * self.inverse_row_norm[d_idx]
                * (d[d_idx] - (psi[d_idx] ** 2) / d[d_idx])
            )
            @ self.A[d_idx, :]
        )

        self._k += 1
        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List[str]) -> float:
        p = self.map(x)
        # residuals are positive if constraints are met
        (res_l, res_u) = self.Bounds.residual(p)
        res_u[res_u > 0] = 0
        res_l[res_l > 0] = 0
        res = -res_u - res_l
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(1 / len(res) * self.total_weights @ (res ** measure[1]))
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure)")
        return measures


class StringAveragedARM(ARMAlgorithm):
    """
    String Averaged ARM Algorithm.
    This class implements the String Averaged ARM (Adaptive Relaxation Method)
    algorithm,
    which is used for feasibility problems involving strings of indices.

    Parameters
    ----------
    A : npt.NDArray
        The matrix A involved in the feasibility problem.
    lb : npt.NDArray
        The lower bounds for the feasibility problem.
    ub : npt.NDArray
        The upper bounds for the feasibility problem.
    strings : List[List[int]]
        A list of lists, where each inner list represents a string of indices.
    algorithmic_relaxation : npt.NDArray or float, optional
        The algorithmic relaxation parameter, by default 1.
    relaxation : float, optional
        The relaxation parameter, by default 1.
    weights : None or List[float], optional
        The weights for each string, by default None. If None, equal weights are assigned.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity, by default True.

    Methods
    -------
    _project(x)
        Projects the input vector x using the string averaged projection method.

    Raises
    ------
    ValueError
        If the number of weights does not match the number of strings.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        strings: List[List[int]],
        algorithmic_relaxation: npt.NDArray | float = 1,
        relaxation: float = 1,
        weights: None | List[float] = None,
        proximity_flag=True,
    ):

        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        self._k = 0
        self.strings = strings

        if weights is None:
            self.weights = xp.ones(len(strings)) / len(strings)

        # if check_weight_validity(weights):
        #    self.weights = weights
        else:
            if len(weights) != len(self.strings):
                raise ValueError("The number of weights must be equal to the number of strings.")

            self.weights = weights

    def _project(self, x):
        xp = cp if self._use_gpu else np
        # string averaged projection
        x_c = x.copy()  # create a general copy of x
        x -= x  # reset x is this viable?
        for string, weight in zip(self.strings, self.weights):
            x_s = x_c.copy()  # generate a copy for individual strings
            for i in string:
                p_i = self.single_map(x_s, i)
                d = p_i - self.Bounds.center[i]
                psi = (self.Bounds.u[i] - self.Bounds.l[i]) / 2
                if xp.abs(d) > psi:
                    self.A.update_step(
                        x_s,
                        -1
                        * self.algorithmic_relaxation**self._k
                        / 2
                        * ((d**2 - psi**2) / d)
                        * self.inverse_row_norm[i],
                        i,
                    )
            x += weight * x_s
        return x
