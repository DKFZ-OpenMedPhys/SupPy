import warnings
from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt
from suppy.feasibility._linear_algorithms import HyperslabFeasibility

try:
    import cupy as cp

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = np


class ARMAlgorithm(HyperslabFeasibility, ABC):
    """
    ARMAlgorithm class for handling feasibility problems with additional
    algorithmic relaxation.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate if proximity constraints should be considered, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)


class SequentialARM(ARMAlgorithm):
    """
    SequentialARM is a class that implements a sequential algorithm for
    Adaptive Relaxation Method (ARM).

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
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
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        cs: None | List[int] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        if cs is None:
            self.cs = xp.arange(self.A.shape[0])
        else:
            self.cs = cs

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np

        for i in self.cs:
            p_i = self.single_map(x, i)
            d = p_i - self.bounds.center[i]
            psi = (self.bounds.u[i] - self.bounds.l[i]) / 2
            if xp.abs(d) > psi:
                self.A.update_step(
                    x,
                    -1
                    * self.algorithmic_relaxation
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
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
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
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np

        if weights is None:
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        elif xp.abs((weights.sum() - 1)) > 1e-10:
            warnings.warn("Weights do not add up to 1! Renormalizing to 1...", stacklevel=2)
            self.weights = weights / weights.sum()
        else:
            self.weights = weights

    def _project(self, x):
        xp = cp if self._use_gpu else np
        # simultaneous projection
        p = self.map(x)
        d = p - self.bounds.center
        psi = self.bounds.half_distance
        d_idx = xp.abs(d) > psi
        x -= (
            self.algorithmic_relaxation
            / 2
            * (
                self.weights[d_idx]
                * self.inverse_row_norm[d_idx]
                * (d[d_idx] - (psi[d_idx] ** 2) / d[d_idx])
            )
            @ self.A[d_idx, :]
        )

        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> list[float]:
        p = self.map(x)
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
                raise ValueError("Invalid proximity measure")
        return measures


class BIPARM(ARMAlgorithm):
    """
    BIPARM Algorithm for feasibility problems.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
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
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        if weights is None:
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        elif xp.abs((weights.sum() - 1)) > 1e-10:
            warnings.warn("Weights do not add up to 1! Renormalizing to 1...", stacklevel=2)
            self.weights = weights / weights.sum()
        else:
            self.weights = weights

    def _project(self, x):
        xp = cp if self._use_gpu else np
        p = self.map(x)
        d = p - self.bounds.center
        psi = self.bounds.half_distance
        d_idx = xp.abs(d) > psi
        x -= (
            self.algorithmic_relaxation
            / 2
            * (
                self.weights[d_idx]
                * self.inverse_row_norm[d_idx]
                * (d[d_idx] - (psi[d_idx] ** 2) / d[d_idx])
            )
            @ self.A[d_idx, :]
        )

        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> list[float]:
        p = self.map(x)
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
                raise ValueError("Invalid proximity measure")
        return measures


class StringAveragedARM(ARMAlgorithm):
    """
    String Averaged ARM Algorithm.
    This class implements the String Averaged ARM (Adaptive Relaxation Method)
    algorithm,
    which is used for feasibility problems involving strings of indices.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    strings : List[List[int]]
        A list of lists, where each inner list represents a string of indices.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
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
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        weights: None | List[float] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        self.strings = strings

        if weights is None:
            self.weights = xp.ones(len(strings)) / len(strings)
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
                d = p_i - self.bounds.center[i]
                psi = (self.bounds.u[i] - self.bounds.l[i]) / 2
                if xp.abs(d) > psi:
                    self.A.update_step(
                        x_s,
                        -1
                        * self.algorithmic_relaxation
                        / 2
                        * ((d**2 - psi**2) / d)
                        * self.inverse_row_norm[i],
                        i,
                    )
            x += weight * x_s
        return x
