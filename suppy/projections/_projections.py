from abc import ABC, abstractmethod
from typing import List, Callable
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    no_gpu = False
except ImportError:
    no_gpu = True
    cp = np


class Projection(ABC):
    """
    Abstract base class for projections used in feasibility algorithms.

    Parameters
    ----------
    relaxation : float, optional
        Relaxation parameter for the projection, by default 1.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity, by default True.

    Attributes
    ----------
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    """

    def __init__(self, relaxation=1, proximity_flag=True, _use_gpu=False):
        self.relaxation = relaxation
        self.proximity_flag = proximity_flag
        self._use_gpu = _use_gpu

    #    @ensure_float_array
    # removed decorator since it leads to unwanted behavior

    def step(self, x: npt.NDArray) -> npt.NDArray:
        """
        Perform the (possibly relaxed) projection of input array 'x' onto
        the constraint.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The (possibly relaxed) projection of 'x' onto the constraint.
        """
        return self.project(x)

    def project(self, x: npt.NDArray) -> npt.NDArray:
        """
        Perform the (possibly relaxed) projection of input array 'x' onto
        the constraint.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The (possibly relaxed) projection of 'x' onto the constraint.
        """
        if self.relaxation == 1:
            return self._project(x)
        else:
            return x.copy() * (1 - self.relaxation) + self.relaxation * (self._project(x))

    @abstractmethod
    def _project(self, x: npt.NDArray) -> npt.NDArray:
        """Internal method to project the point x onto the set."""

    def proximity(self, x: npt.NDArray, proximity_measures: List) -> float:
        """
        Calculate proximity measures of point `x` to the set.

        Parameters
        ----------
        x : npt.NDArray
            Input array for which the proximity measure is to be calculated.

        Returns
        -------
        List[float]
            The proximity measures of the input array `x`.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        if self.proximity_flag:
            return xp.array(self._proximity(x, proximity_measures))
        else:
            return xp.zeros(len(proximity_measures))

    @abstractmethod
    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> float:
        """
        Calculate proximity measures of point `x` to set.

        Parameters
        ----------
        x : npt.NDArray
            Input array for which the proximity measures are to be calculated.
        proximity_measures : List
            List of proximity measures to calculate.

        Returns
        -------
        List[float]
            The proximity measures of the input array `x`.
        """


class BasicProjection(Projection, ABC):
    """
    BasicProjection is an abstract base class that extends the Projection
    class.
    It allows for projecting onto a subset of the input array based on provided
    indices.

    Parameters
    ----------
    idx : npt.NDArray or None, optional
        Indices to apply the projection, by default None.
    relaxation : float, optional
        Relaxation parameter for the projection, by default 1.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity, by default True.

    Attributes
    ----------
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    idx : npt.NDArray
        Subset of the input vector to apply the projection on.
    """

    def __init__(
        self, relaxation=1, idx: npt.NDArray | None = None, proximity_flag=True, _use_gpu=False
    ):
        super().__init__(relaxation, proximity_flag, _use_gpu)
        self.idx = idx if idx is not None else np.s_[:]

    # NOTE: This method should not be required since the base class implementation is sufficient
    # def project(self, x: npt.NDArray) -> npt.NDArray:
    #     """
    #     Perform the (possibly relaxed) projection of input array 'x' onto the constraint.

    #     Parameters
    #     ----------
    #     x : npt.NDArray
    #         The input array to be projected.

    #     Returns
    #     -------
    #     npt.NDArray
    #         The (possibly relaxed) projection of 'x' onto the constraint.
    #     """

    #     if self.relaxation == 1:
    #         return self._project(x)
    #     else:
    #         x[self.idx] = x[self.idx] * (1 - self.relaxation) + self.relaxation * (
    #             self._project(x)[self.idx]
    #         )
    #         return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> List[float]:
        # probably should have some option to choose the distance
        res = x[self.idx] - self._project(x.copy())[self.idx]
        dist = (res**2).sum() ** (1 / 2)
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(dist ** measure[1])
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(dist)
            else:
                raise ValueError("Invalid proximity measure")
        return measures
