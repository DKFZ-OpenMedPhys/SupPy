"""
General implementation for sequential, simultaneous, block iterative and
string averaged projection methods.
"""
from abc import ABC
from typing import List, Callable
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    cp = np
    NO_GPU = True

from suppy.projections._projections import Projection, BasicProjection
from suppy.utils import ensure_float_array


class ProjectionMethod(Projection, ABC):
    """
    A class used to represent methods for projecting a point onto multiple
    sets.

    Parameters
    ----------
    projections : List[Projection]
        A list of Projection objects to be used in the projection method.
    relaxation : float, optional
        A relaxation parameter for the projection method (default is 1).
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity, by default True.

    Attributes
    ----------
    projections : List[Projection]
        The list of Projection objects used in the projection method.
    all_x : array-like or None
        Storage for all x values if storage is enabled during solve.
    proximities : list
        A list to store proximity values during the solve process.
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    """

    def __init__(
        self, projections: List[Projection], relaxation: float = 1, proximity_flag: bool = True
    ):
        super().__init__(relaxation, proximity_flag)
        self.projections = projections
        self.all_x = None
        self.proximities = []

    def visualize(self, ax):
        """
        Visualizes all projection objects (if applicable) on the given
        matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axis on which to visualize the projections.
        """
        for proj in self.projections:
            proj.visualize(ax)

    @ensure_float_array
    def solve(
        self,
        x: npt.NDArray,
        max_iter: int = 500,
        prox_tol: float = 1e-6,
        del_prox_tol: float = 1e-8,
        del_prox_n: int = 5,
        proximity_measures: List | None = None,
        storage: bool = False,
        storage_iters: List[int] | int | None = None,
        alternative_stopping_criterion: Callable | None = None,
        alternative_stopping_criterion_initial_call: Callable | None = None,
    ) -> np.ndarray:
        """
        Solves the optimization problem using an iterative approach.

        Parameters
        ----------
        x : npt.NDArray
            Starting point for the algorithm.
        max_iter : int, optional
            Maximum number of iterations to perform, by default 500.
        prox_tol : float, optional
            The tolerance for the proximity on the constraints, by default 1e-6.
        del_prox_tol : float, optional
            The tolerance for the change in proximity over the last del_prox_n iterations, by default 1e-8.
        del_prox_n : int, optional
            The number of iterations that del_prox_tol needs to be met in a row, by default 5.
        proximity_measures : List, optional
            The proximity measures to calculate, by default a l2 norm measure is used. Right now only the first in the list is used to check the feasibility.
        storage : bool, optional
            Flag indicating whether to store intermediate solutions, by default False.
        storage_iters : List[int] or int, optional
            Controls which iterations are stored (when storage=True). If None, all iterations are stored.
            If a list of ints, only those iteration indices are stored (0 = initial point).
            If an int, storage occurs every that many iterations.
        alternative_stopping_criterion : callable, optional
            Alternative stopping criterion
        alternative_stopping_criterion_initial_call : callable, optional
            Initial call for an alternative stopping criterion

        Returns
        -------
        npt.NDArray
            The solution after the iterative process.
        """
        xp = cp if isinstance(x, cp.ndarray) else np

        if proximity_measures is None:
            proximity_measures = [("p_norm", 2)]
        else:
            # TODO: Check if the proximity measures are valid
            _ = None

        self.proximities = [self.proximity(x, proximity_measures)]
        i = 0

        def _should_store(idx):
            if storage_iters is None:
                return True
            if isinstance(storage_iters, int):
                return idx % storage_iters == 0
            return idx in storage_iters

        if storage is True:
            self.all_x = []
            if _should_store(0):
                if isinstance(x, np.ndarray):
                    self.all_x.append(np.array(x.copy()))
                else:
                    self.all_x.append((x.get()))

        if alternative_stopping_criterion_initial_call is not None:
            stop = alternative_stopping_criterion_initial_call(x, self)
        else:
            stop = False  # criterion for stopping the algorithm

        self._n_tol = 0

        while i < max_iter and not stop:
            x = self.project(x)
            if storage is True and _should_store(i + 1):
                if isinstance(x, np.ndarray):  # convert to np array if cp
                    self.all_x.append(np.array(x.copy()))
                else:
                    self.all_x.append((x.get()))

            self.proximities.append(self.proximity(x, proximity_measures))

            # TODO: If proximity changes x some potential issues!
            if alternative_stopping_criterion is not None:
                stop = alternative_stopping_criterion(x, self)
            else:
                stop = self._stopping_criterion(prox_tol, del_prox_tol, del_prox_n)

            i += 1

        if self.all_x is not None:
            self.all_x = np.array(self.all_x)

        self.proximities = xp.array(self.proximities)
        return x

    def _stopping_criterion(self, prox_tol: float, del_prox_tol: float, del_prox_n: int) -> bool:
        """Returns True when convergence is detected, False otherwise."""
        if self.proximities[-1][0] < prox_tol:
            return True
        else:  # check that last n proximity changes are below a threshold
            if self.proximities[-2][0] - self.proximities[-1][0] < del_prox_tol:
                self._n_tol += 1
            else:
                self._n_tol = 0
            if self._n_tol >= del_prox_n:
                return True
        return False

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> List[float]:
        xp = cp if isinstance(x, cp.ndarray) else np
        proxs = xp.array(
            [xp.array(proj.proximity(x, proximity_measures)) for proj in self.projections]
        )
        measures = []
        for i, measure in enumerate(proximity_measures):
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append((proxs[:, i]).mean())
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(proxs[:, i].max())
            else:
                raise ValueError("Invalid proximity measure")
        return measures


class SequentialProjection(ProjectionMethod):
    """
    Class to represent a sequential projection.

    Parameters
    ----------
    projections : List[Projection]
        A list of projection methods to be applied sequentially.
    relaxation : float, optional
        A relaxation parameter for the projection methods, by default 1.
    control_seq : None, numpy.typing.ArrayLike, or List[int], optional
        An optional sequence that determines the order in which the projections are applied.
        If None, the projections are applied in the order they are provided, by default None.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity, by default True.

    Attributes
    ----------
    projections : List[Projection]
        The list of Projection objects used in the projection method.
    all_x : array-like or None
        Storage for all x values if storage is enabled during solve.
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    control_seq : npt.NDArray or List[int]
        The sequence in which the projections are applied.
    """

    def __init__(
        self,
        projections: List[Projection],
        relaxation: float = 1,
        control_seq: None | npt.NDArray | List[int] = None,
        proximity_flag: bool = True,
    ):
        super().__init__(projections, relaxation, proximity_flag)
        if control_seq is None:
            self.control_seq = np.arange(len(projections))
        else:
            self.control_seq = control_seq

    def _project(self, x: npt.NDArray) -> np.ndarray:
        """
        Sequentially projects the input array `x` using the control
        sequence.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The projected array after applying all projection methods in the control sequence.
        """

        for i in self.control_seq:
            x = self.projections[i].project(x)
        return x


class SimultaneousProjection(ProjectionMethod):
    """
    Class to represent a simultaneous projection.

    Parameters
    ----------
    projections : List[Projection]
        A list of projection methods to be applied.
    weights : npt.NDArray or None, optional
        An array of weights for each projection method. If None, equal weights
        are assigned to each projection. Weights are normalized to sum up to 1. Default is None.
    relaxation : float, optional
        A relaxation parameter for the projection methods. Default is 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity in the projection methods.
        Default is True.

    Attributes
    ----------
    projections : List[Projection]
        The list of Projection objects used in the projection method.
    all_x : array-like or None
        Storage for all x values if storage is enabled during solve.
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    weights : npt.NDArray
        The weights assigned to each projection method.

    Notes
    -----
    While the simultaneous projection is performed simultaneously mathematically, the actual computation right now is sequential.
    """

    def __init__(
        self,
        projections: List[Projection],
        weights: npt.NDArray | None = None,
        relaxation: float = 1,
        proximity_flag: bool = True,
    ):
        super().__init__(projections, relaxation, proximity_flag)
        if weights is None:
            weights = np.ones(len(projections)) / len(projections)
        self.weights = weights / weights.sum()

    def _project(self, x: npt.NDArray) -> np.ndarray:
        """
        Simultaneously projects the input array `x`.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The projected array.
        """
        x_new = 0
        for proj, weight in zip(self.projections, self.weights):
            x_new = x_new + weight * proj.project(x.copy())
        return x_new

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> List[float]:
        xp = cp if isinstance(x, cp.ndarray) else np
        proxs = xp.array(
            [xp.array(proj.proximity(x, proximity_measures)) for proj in self.projections]
        )
        measures = []
        for i, measure in enumerate(proximity_measures):
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(self.weights @ (proxs[:, i]))
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(proxs[:, i].max())
            else:
                raise ValueError("Invalid proximity measure")
        return measures


class StringAveragedProjection(ProjectionMethod):
    """
    Class to represent a string averaged projection.

    Parameters
    ----------
    projections : List[Projection]
        A list of projection methods to be applied.
    strings : List[List]
        A list of strings, where each string is a list of indices of the projection methods to be applied.
    weights : npt.NDArray or None, optional
        An array of weights for each strings. If None, equal weights
        are assigned to each string. Weights are normalized to sum up to 1. Default is None.
    relaxation : float, optional
        A relaxation parameter for the projection methods. Default is 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity in the projection methods.
        Default is True.

    Attributes
    ----------
    projections : List[Projection]
        The list of Projection objects used in the projection method.
    all_x : array-like or None
        Storage for all x values if storage is enabled during solve.
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    strings : List[List]
        A list of strings, where each string is a list of indices of the projection methods to be applied.
    weights : npt.NDArray
        The weights assigned to each projection method.

    Notes
    -----
    While the string projections are performed simultaneously mathematically, the actual computation right now is sequential.
    """

    def __init__(
        self,
        projections: List[Projection],
        strings: List[List],
        weights: npt.NDArray | None = None,
        relaxation: float = 1,
        proximity_flag: bool = True,
    ):
        super().__init__(projections, relaxation, proximity_flag)
        if weights is None:
            self.weights = np.ones(len(strings)) / len(strings)
        else:
            self.weights = weights / weights.sum()
        self.strings = strings

    def _project(self, x: npt.NDArray) -> np.ndarray:
        """
        String averaged projection of the input array `x`.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be projected.

        Returns
        -------
        npt.NDArray
            The projected array after applying all projection methods in the control sequence.
        """
        x_new = 0
        # TODO: Can this be parallelized?
        for weight, string in zip(self.weights, self.strings):
            # run over all individual strings
            x_s = x.copy()  # create a copy for
            for el in string:  # run over all elements in the string sequentially
                x_s = self.projections[el].project(x_s)
            x_new += weight * x_s
        return x_new


class BlockIterativeProjection(ProjectionMethod):
    """
    Class to represent a block iterative projection.

    Parameters
    ----------
    projections : List[Projection]
        A list of projection methods to be applied.
    weights : List[List[float]] | List[npt.NDArray]
        A List of weights for each block of projection methods.
    relaxation : float, optional
        A relaxation parameter for the projection methods. Default is 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity in the projection methods.
        Default is True.

    Attributes
    ----------
    projections : List[Projection]
        The list of Projection objects used in the projection method.
    all_x : array-like or None
        Storage for all x values if storage is enabled during solve.
    relaxation : float
        Relaxation parameter for the projection.
    proximity_flag : bool
        Flag to indicate whether to take this object into account when calculating proximity.
    weights : List[npt.NDArray]
        The weights assigned to each block of projection methods.

    Notes
    -----
    While the individual block projections are performed simultaneously mathematically, the actual computation right now is sequential.
    """

    def __init__(
        self,
        projections: List[Projection],
        weights: List[List[float]] | List[npt.NDArray],
        relaxation: float = 1,
        proximity_flag: bool = True,
    ):
        super().__init__(projections, relaxation, proximity_flag)
        xp = cp if self._use_gpu else np
        # check if weights has the correct format
        for el in weights:
            if len(el) != len(projections):
                raise ValueError("Weights do not match the number of projections!")

            if abs((el.sum() - 1)) > 1e-10:
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
        # TODO: Can this be parallelized?
        for weight, block_idx in zip(self.weights, self.block_idxs):
            x_new = 0
            for i, el in enumerate(block_idx):
                x_new += weight[i] * self.projections[el].project(x.copy())
            x = x_new
        return x

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> List[float]:
        xp = cp if isinstance(x, cp.ndarray) else np
        proxs = xp.array(
            [xp.array(proj.proximity(x, proximity_measures)) for proj in self.projections]
        )
        measures = []
        for i, measure in enumerate(proximity_measures):
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(self.total_weights @ (proxs[:, i]))
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(proxs[:, i].max())
            else:
                raise ValueError("Invalid proximity measure")
        return measures


class MultiBallProjection(BasicProjection, ABC):
    """Projection onto multiple balls."""

    def __init__(
        self,
        centers: npt.NDArray,
        radii: npt.NDArray,
        relaxation: float = 1,
        idx: npt.NDArray | None = None,
        proximity_flag=True,
    ):
        try:
            if isinstance(centers, cp.ndarray) and isinstance(radii, cp.ndarray):
                _use_gpu = True
            elif (isinstance(centers, cp.ndarray)) != (isinstance(radii, cp.ndarray)):
                raise ValueError("Mismatch between input types of centers and radii")
            else:
                _use_gpu = False
        except ModuleNotFoundError:
            _use_gpu = False

        super().__init__(relaxation, idx, proximity_flag, _use_gpu)
        self.centers = centers
        self.radii = radii


class SequentialMultiBallProjection(MultiBallProjection):
    """Sequential projection onto multiple balls."""

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        for i in range(len(self.centers)):
            diff = x[self.idx] - self.centers[i]
            dist = xp.linalg.norm(diff)
            if dist > self.radii[i]:
                x[self.idx] = self.centers[i] + self.radii[i] * diff / dist
        return x


class SimultaneousMultiBallProjection(MultiBallProjection):
    """Simultaneous projection onto multiple balls."""

    def __init__(
        self,
        centers: npt.NDArray,
        radii: npt.NDArray,
        weights: npt.NDArray,
        relaxation: float = 1,
        idx: npt.NDArray | None = None,
        proximity_flag=True,
    ):

        super().__init__(centers, radii, relaxation, idx, proximity_flag)
        self.weights = weights

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        dists = xp.linalg.norm(x[self.idx] - self.centers, axis=1)
        idx = (dists - self.radii) > 0
        x[self.idx] = x[self.idx] - (self.weights[idx] * (1 - self.radii[idx] / dists[idx])) @ (
            x[self.idx] - self.centers[idx]
        )
        return x
