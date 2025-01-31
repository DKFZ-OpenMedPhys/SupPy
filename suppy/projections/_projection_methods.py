from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp
except ImportError:
    cp = np

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
    def solve(self, x: npt.ArrayLike, max_iter=1000, constr_tol=1e-6, storage=False):
        """
        Solves the feasibility seeking problem by projecting onto the
        different sets.

        Parameters
        ----------
        x : npt.ArrayLike
            Initial input array.
        max_iter : int, optional
            Maximum number of iterations (default is 1000).
        constr_tol : float, optional
            Convergence tolerance for the proximity measure (default is 1e-6).
        storage : bool, optional
            If True, stores all intermediate solutions (default is False).

        Returns
        -------
        x : npt.ArrayLike
            The solution array after the projection process.

        Notes
        -----
        This method requires `cupy` to be installed right now!
        """
        # TODO: This requires cupy to be installed!
        xp = cp if isinstance(x, cp.ndarray) else np
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
            self.proximities.append(self.proximity(x))

            # TODO: If proximity changes x some potential issues!
            if self.proximities[-1] < constr_tol:

                feasible = True
            i += 1
        if self.all_x is not None:
            self.all_x = xp.array(self.all_x)
        return x

    def _proximity(self, x):
        """
        Calculate the average proximity of a given input `x` across all
        projections.

        Parameters
        ----------
        x : array-like
            The input data for which the proximity is to be calculated.

        Returns
        -------
        float
            The average proximity value of the input `x` across all projections.
        """
        return (
            1
            / len(self.projections)
            * sum([proj.proximity(x.copy()) for proj in self.projections])
        )


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
    control_seq : npt.ArrayLike or List[int]
        The sequence in which the projections are applied.
    """

    def __init__(
        self,
        projections: List[Projection],
        relaxation: float = 1,
        control_seq: None | npt.ArrayLike | List[int] = None,
        proximity_flag=True,
    ):

        # TODO: optional: assign order in which projections are applied
        super().__init__(projections, relaxation, proximity_flag)
        if control_seq is None:
            self.control_seq = np.arange(len(projections))
        else:
            self.control_seq = control_seq

    def _project(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Sequentially projects the input array `x` using the control
        sequence.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to be projected.

        Returns
        -------
        npt.ArrayLike
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
    weights : npt.ArrayLike or None, optional
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
    weights : npt.ArrayLike
        The weights assigned to each projection method.

    Notes
    -----
    While the simultaneous projection is performed simultaneously mathematically, the actual computation right now is sequential.
    """

    def __init__(
        self,
        projections: List[Projection],
        weights: npt.ArrayLike | None = None,
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
        x : npt.ArrayLike
            The input array to be projected.

        Returns
        -------
        npt.ArrayLike
            The projected array.
        """
        x_new = 0
        for proj, weight in zip(self.projections, self.weights):
            x_new = x_new + weight * proj.project(x.copy())
        return x_new

    def _proximity(self, x):
        """
        Calculate the proximity of point `x` to set.

        Parameters
        ----------
        x : npt.ArrayLike
            Input array for which the proximity measure is to be calculated.

        Returns
        -------
        float
            The proximity measure of the input array `x`.
        """
        return sum(
            [
                weight * proj.proximity(x.copy())
                for (weight, proj) in zip(self.weights, self.projections)
            ]
        )


class StringAveragedProjection(ProjectionMethod):
    """
    Class to represent a string averaged projection.

    Parameters
    ----------
    projections : List[Projection]
        A list of projection methods to be applied.
    strings : List[List]
        A list of strings, where each string is a list of indices of the projection methods to be applied.
    weights : npt.ArrayLike or None, optional
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
    weights : npt.ArrayLike
        The weights assigned to each projection method.

    Notes
    -----
    While the string projections are performed simultaneously mathematically, the actual computation right now is sequential.
    """

    def __init__(
        self,
        projections: List[Projection],
        strings: List[List],
        weights: npt.ArrayLike | None = None,
        relaxation: float = 1,
        proximity_flag=True,
    ):

        super().__init__(projections, relaxation, proximity_flag)
        if weights is None:
            weights = np.ones(len(strings)) / len(strings)  # assign uniform weights
        else:
            self.weights = weights / weights.sum()
        self.strings = strings

    def _project(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        String averaged projection of the input array `x`.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to be projected.

        Returns
        -------
        npt.ArrayLike
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
    weights : List[List[float]] | List[npt.ArrayLike]
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
    weights : List[npt.ArrayLike]
        The weights assigned to each block of projection methods.

    Notes
    -----
    While the individual block projections are performed simultaneously mathematically, the actual computation right now is sequential.
    """

    def __init__(
        self,
        projections: List[Projection],
        weights: List[List[float]] | List[npt.ArrayLike],
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

    def _project(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # TODO: Can this be parallelized?
        for weight, block_idx in zip(self.weights, self.block_idxs):
            x_new = 0  # for simultaneous projection, later replaces x

            i = 0
            for el in block_idx:
                x_new += weight[i] * self.projections[el].project(x.copy())
                i += 1
            x = x_new
        return x

    def _proximity(self, x):
        xp = cp if isinstance(x, cp.ndarray) else np
        return xp.sum(
            [
                weight * proj.proximity(x.copy())
                for (weight, proj) in zip(self.total_weights, self.projections)
            ]
        )
