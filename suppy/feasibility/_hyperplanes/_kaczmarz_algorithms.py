import warnings
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

from suppy.feasibility._linear_algorithms import HyperplaneFeasibility


class HyperplaneAlgorithm(HyperplaneFeasibility, ABC):
    """
    The HyperplaneAlgorithm class is used to find a feasible solution to
    a set of linear equalities.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity in the algorithm, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag)


class KaczmarzMethod(HyperplaneAlgorithm):
    """
    Kaczmarz method for sequentially solving linear equality constraints.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
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
        b: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        cs: None | List[int] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        if cs is None:
            self.cs = xp.arange(self.A.shape[0])
        else:
            self.cs = cs

    def _project(self, x: npt.NDArray) -> np.ndarray:
        """
        Projects the input array `x` onto the feasible region defined by the
        constraints.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The projected array.
        """
        for i in self.cs:
            p_i = self.single_map(x, i)
            res = self.b[i] - p_i
            self.A.update_step(x, self.algorithmic_relaxation * self.inverse_row_norm[i] * res, i)
        return x


class SequentialWeightedKaczmarz(KaczmarzMethod):
    """
    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
    weights : None, list of float, or npt.NDArray, optional
        The weights assigned to each constraint. If None, default weights are
    used.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the
    iterate by default 1.0.
    weight_decay : float, optional
        Parameter that determines the rate at which the weights are reduced
        after each phase (weights * weight_decay). Default is 1.0.
    cs : None or list of int, optional
        The indices of the constraints to be considered. Default is None.
    proximity_flag : bool, optional
        Flag to indicate if proximity should be considered. Default is True.

    Attributes
    ----------
    weights : npt.NDArray
        The weights assigned to each constraint.
    weight_decay : float
        Decay rate for the weights.
    temp_weight_decay : float
        Initial value for weight decay.
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        weights: None | List[float] | npt.NDArray = None,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        weight_decay: float = 1.0,
        cs: None | List[int] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, algorithmic_relaxation, relaxation, cs, proximity_flag)
        xp = cp if self._use_gpu else np
        self.weight_decay = weight_decay
        self.temp_weight_decay = 1.0

        if weights is None:
            self.weights = xp.ones(self.A.shape[0])
        else:
            self.weights = weights

    def _project(self, x: npt.NDArray) -> np.ndarray:
        """
        Projects the input array `x` onto a feasible region defined by the
        constraints.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The projected array.

        Notes
        -----
        This method iteratively adjusts the input array `x` based on the constraints
        defined in `self.cs`. For each constraint, it computes the projection and
        checks if the constraints are violated. If a constraint is violated, it updates
        the array `x` using a weighted relaxation factor. The weight decay is applied
        to the temporary weight decay after each iteration.
        """
        weighted_relaxation = self.algorithmic_relaxation * self.temp_weight_decay

        for i in self.cs:
            p_i = self.single_map(x, i)
            res = self.b[i] - p_i
            self.A.update_step(
                x, weighted_relaxation * self.weights[i] * self.inverse_row_norm[i] * res, i
            )

        self.temp_weight_decay *= self.weight_decay
        return x


class SimultaneousKaczmarzMethod(HyperplaneAlgorithm):
    """
    SimultaneousKaczmarzMethod is an implementation of the Kaczmarz
    algorithm
    that performs simultaneous projections and proximity calculations.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    proximity_flag : bool, optional
        Flag to indicate if proximity calculations should be performed, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        weights: None | List[float] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag)

        xp = cp if self._use_gpu else np

        if weights is None:
            self.weights = xp.ones(self.A.shape[0]) / self.A.shape[0]
        elif xp.abs((weights.sum() - 1)) > 1e-10:
            warnings.warn("Weights do not add up to 1! Renormalizing to 1...", stacklevel=2)
            self.weights = weights / weights.sum()
        else:
            self.weights = weights

    def _project(self, x: npt.NDArray) -> np.ndarray:
        # simultaneous projection
        p = self.map(x)
        res = self.b - p
        x += self.algorithmic_relaxation * (self.weights * self.inverse_row_norm * res @ self.A)
        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> list[float]:
        p = self.map(x)
        res = abs(self.b - p)
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


class BlockIterativeKaczmarz(HyperplaneAlgorithm):
    """
    Block iterative Kaczmarz algorithm for solving linear equality
    constraints
    in a block-wise manner.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
    weights : List[List[float]] or List[npt.NDArray]
        A list of lists or arrays representing the weights for each block. Each list/array should sum to 1.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    Raises
    ------
    ValueError
        If any of the weight lists do not sum to 1.
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        weights: List[List[float]] | List[npt.NDArray],
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag)

        xp = cp if self._use_gpu else np

        # check that weights is a list of lists that add up to 1 each
        for el in weights:
            if xp.abs((xp.sum(el) - 1)) > 1e-10:
                raise ValueError("Weights do not add up to 1!")

        self.weights = []
        self.block_idxs = [
            xp.where(xp.array(el) > 0)[0] for el in weights
        ]  # get idxs that meet requirements

        # assemble a list of general weights
        self.total_weights = xp.zeros_like(weights[0])
        for el in weights:
            el = xp.asarray(el)
            self.weights.append(el[xp.array(el) > 0])  # remove non zero weights
            self.total_weights += el / len(weights)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        # simultaneous projection
        for el, block_idx in zip(self.weights, self.block_idxs):
            p = self.indexed_map(x, block_idx)
            res = self.b[block_idx] - p

            x += self.algorithmic_relaxation * (
                el * self.inverse_row_norm[block_idx] * res @ self.A[block_idx, :]
            )
        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> list[float]:
        p = self.map(x)
        res = abs(self.b - p)
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(self.total_weights @ (res ** measure[1]))
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure")
        return measures


class StringAveragedKaczmarz(HyperplaneAlgorithm):
    """
    StringAveragedKaczmarz is an implementation of the HyperplaneAlgorithm
    that performs string averaged projections.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
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
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        strings: List[List[int]],
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        weights: None | List[float] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        self.strings = strings
        if weights is None:
            self.weights = xp.ones(len(strings)) / len(strings)
        else:
            if len(weights) != len(self.strings):
                raise ValueError("The number of weights must be equal to the number of strings.")
            self.weights = weights

    def _project(self, x: npt.NDArray) -> np.ndarray:
        # string averaged projection
        x_c = x.copy()  # create a general copy of x
        x -= x  # reset x is this viable?
        for string, weight in zip(self.strings, self.weights):
            x_s = x_c.copy()  # generate a copy for individual strings
            for i in string:
                p_i = self.single_map(x_s, i)
                res_i = self.b[i] - p_i
                self.A.update_step(
                    x_s, self.algorithmic_relaxation * self.inverse_row_norm[i] * res_i, i
                )
            x += weight * x_s
        return x
