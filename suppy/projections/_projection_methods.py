from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    no_gpu = False
except ImportError:
    cp = np
    no_gpu = True

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
    relaxation : int, optional
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

    def __init__(self, projections: List[Projection], relaxation=1, proximity_flag=True):
        # if all([proj._use_gpu == projections[0]._use_gpu for proj in projections]):
        #    self._use_gpu = projections[0]._use_gpu
        # else:
        #    raise ValueError("Projections do not have the same gpu flag!")
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
        storage: bool = False,
        constr_tol: float = 1e-6,
        proximity_measure: List | None = None,
    ) -> npt.NDArray:
        """
        Solves the optimization problem using an iterative approach.

        Parameters
        ----------
        x : npt.NDArray
            Initial guess for the solution.
        max_iter : int
            Maximum number of iterations to perform.
        storage : bool, optional
            Flag indicating whether to store the intermediate solutions, by default False.
        constr_tol : float, optional
            The tolerance for the constraints, by default 1e-6.
        proximity_measure : List, optional
            The proximity measures to calculate, by default None. Right now only the first in the list is used to check the feasibility.

        Returns
        -------
        npt.NDArray
            The solution after the iterative process.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        if proximity_measure is None:
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
        proximity_flag=True,
    ):

        # TODO: optional: assign order in which projections are applied
        super().__init__(projections, relaxation, proximity_flag)
        if control_seq is None:
            self.control_seq = np.arange(len(projections))
        else:
            self.control_seq = control_seq

    def _project(self, x: npt.NDArray) -> npt.NDArray:
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
        proximity_flag=True,
    ):

        super().__init__(projections, relaxation, proximity_flag)
        if weights is None:
            weights = np.ones(len(projections)) / len(projections)
        self.weights = weights / weights.sum()

    def _project(self, x: float) -> float:
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
        proximity_flag=True,
    ):

        super().__init__(projections, relaxation, proximity_flag)
        if weights is None:
            weights = np.ones(len(strings)) / len(strings)  # assign uniform weights
        else:
            self.weights = weights / weights.sum()
        self.strings = strings

    def _project(self, x: npt.NDArray) -> npt.NDArray:
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
        proximity_flag=True,
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

    def _project(self, x: npt.NDArray) -> npt.NDArray:
        # TODO: Can this be parallelized?
        for weight, block_idx in zip(self.weights, self.block_idxs):
            x_new = 0  # for simultaneous projection, later replaces x

            i = 0
            for el in block_idx:
                x_new += weight[i] * self.projections[el].project(x.copy())
                i += 1
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

    # def __init__(self,
    #             centers: npt.NDArray,
    #             radii: npt.NDArray,
    #             relaxation:float = 1,
    #             idx: npt.NDArray | None = None):

    #     super().__init__(centers, radii, relaxation,idx)

    def _project(self, x: npt.NDArray) -> npt.NDArray:

        for i in range(len(self.centers)):
            if np.linalg.norm(x[self.idx] - self.centers[i]) > self.radii[i]:
                x[self.idx] = self.centers[i] + self.radii[i] * (
                    x[self.idx] - self.centers[i]
                ) / np.linalg.norm(x[self.idx] - self.centers[i])
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

    def _project(self, x: npt.NDArray) -> npt.NDArray:
        # get all indices
        dists = np.linalg.norm(x[self.idx] - self.centers, axis=1)
        idx = (dists - self.radii) > 0
        # project onto halfspaces
        x[self.idx] = x[self.idx] - (self.weights[idx] * (1 - self.radii[idx] / dists[idx])) @ (
            x[self.idx] - self.centers[idx]
        )
        return x
