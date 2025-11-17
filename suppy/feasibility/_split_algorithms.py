"""Algorithms for split feasibility problem."""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numpy.typing as npt
from scipy import sparse

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    cp = np
    NO_GPU = True

from suppy.utils import LinearMapping
from suppy.utils import ensure_float_array
from suppy.projections._projections import Projection

# from ._algorithms import Feasibility
from suppy.feasibility._linear_algorithms import Feasibility


class SplitFeasibility(Feasibility, ABC):
    """
    Abstract base class used to represent split feasibility problems.

    Parameters
    ----------
    A : npt.NDArray
        Matrix connecting input and target space.
    algorithmic_relaxation : npt.NDArray or float, optional
        Relaxation applied to the entire solution of the projection step, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations, by default True.

    Attributes
    ----------
    A : LinearMapping
        Linear mapping between input and target space.
    proximities : list
        A list to store proximity values during the solve process.
    algorithmic_relaxation : float
        Relaxation applied to the entire solution of the projection step.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        proximity_flag: bool = True,
        _use_gpu: bool = False,
    ):

        _, _use_gpu = LinearMapping.get_flags(A)
        super().__init__(algorithmic_relaxation, 1, proximity_flag, _use_gpu=_use_gpu)
        self.A = LinearMapping(A)
        self.proximities = []
        self.all_x = None

    @ensure_float_array
    def solve(
        self,
        x: npt.NDArray,
        max_iter: int = 10,
        prox_tol: float = 1e-6,
        del_prox_tol: float = 1e-8,
        del_prox_n: int = 5,
        storage: bool = False,
        proximity_measures: List | None = None,
    ) -> npt.NDArray:
        """
        Solves the split feasibility problem for a given input array.

        Parameters
        ----------
        x : npt.NDArray
            Starting point for the algorithm.
        max_iter : int, optional
            The maximum number of iterations (default is 10).
        prox_tol : float, optional
            Stopping criterium for the feasibility seeking algorithm.
            Solution deemed feasible if the proximity drops below this value (default is 1e-6).
        del_prox_tol : float, optional
            The tolerance for the change in proximity over the last del_prox_n iterations, by default 1e-8.
        del_prox_n : int, optional
            The number of iterations to check for the change in proximity, by default 5.
        storage : bool, optional
            A flag indicating whether to store all intermediate solutions (default is False).
        proximity_measures : List, optional
            The proximity measures to calculate, by default None.
            Right now only the first in the list is used to check the feasibility.

        Returns
        -------
        npt.NDArray
            The solution after applying the feasibility seeking algorithm.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        self._n_tol = 0

        if proximity_measures is None:
            proximity_measures = [("p_norm", 2)]
        else:
            # TODO: Check if the proximity measures are valid
            _ = None

        self.proximities = [self.proximity(x, proximity_measures)]
        i = 0

        if storage is True:
            self.all_x = []
            if isinstance(x, cp.ndarray) and not NO_GPU:
                self.all_x.append((x.get()))
            else:
                self.all_x.append(np.array(x.copy()))

        stop = False  # criterion for stopping the algorithm

        while i < max_iter and not stop:
            x, _ = self.step(x)
            if storage is True:
                if isinstance(x, np.ndarray):  # convert to np array if cp
                    self.all_x.append(np.array(x.copy()))
                else:
                    self.all_x.append((x.get()))
            self.proximities.append(self.proximity(x, proximity_measures))

            # TODO: If proximity changes x some potential issues!
            stop = self._stopping_criterion(prox_tol, del_prox_tol, del_prox_n)
            i += 1

        if self.all_x is not None:
            self.all_x = np.array(self.all_x)

        self.proximities = xp.array(self.proximities)

        return x

    def _stopping_criterion(self, prox_tol: float, del_prox_tol: float, del_prox_n: float):
        """"""
        if self.proximities[-1][0] < prox_tol:  # proximity below goal/tolerance
            return True
        else:  # check that last n proximity changes are below a threshold
            if self.proximities[-2][0] - self.proximities[-1][0] < del_prox_tol:
                self._n_tol += 1
            else:
                self._n_tol = 0
            if self._n_tol >= del_prox_n:  # n proximity changes below threshold
                return True
        return False

    def project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
        """
        Projects the input array onto the feasible set.

        Parameters
        ----------
        x : npt.NDArray
            The input array to project.
        y : npt.NDArray, optional
            An optional array for projection (default is None).

        Returns
        -------
        npt.NDArray
            The projected array.
        """

        return self._project(x, y)

    def step(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
        return self.project(x, y)

    @abstractmethod
    def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
        pass

    def map(self, x: npt.NDArray) -> npt.NDArray:
        """
        Maps the input space array to the target space via matrix
        multiplication.

        Parameters
        ----------
        x : npt.NDArray
            The input space array to be map.

        Returns
        -------
        npt.NDArray
            The corresponding target space array.
        """

        return self.A @ x

    def map_back(self, y: npt.NDArray) -> npt.NDArray:
        """
        Transposed map of the target space array to the input space.

        Parameters
        ----------
        y : npt.NDArray
            The target space array to map.

        Returns
        -------
        npt.NDArray
            The corresponding array in input space.
        """

        return self.A.T @ y


class CQAlgorithm(SplitFeasibility):
    """
    Implementation for the CQ algorithm to solve split feasibility problems.

    Parameters
    ----------
    A : npt.NDArray
        Matrix connecting input and target space.
    C_projection : Projection
        The projection operator onto the set C.
    Q_projection : Projection
        The projection operator onto the set Q.
    algorithmic_relaxation : npt.NDArray or float, optional
        Relaxation applied to the entire solution of the projection step, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations, by default True.
    use_gpu : bool, optional
        A flag indicating whether to use GPU for computations, by default False.

    Attributes
    ----------
    A : LinearMapping
        Linear mapping between input and target space.
    C_projection : Projection
        The projection operator onto the set C.
    Q_projection : Projection
        The projection operator onto the set Q.
    proximities : list
        A list to store proximity values during the solve process.
    algorithmic_relaxation : float
        Relaxation applied to the entire solution of the projection step.
    proximity_flag : bool
        A flag indicating whether to use this object for proximity calculations.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        C_projection: Projection,
        Q_projection: Projection,
        algorithmic_relaxation: float = 1,
        proximity_flag=True,
        use_gpu=False,
    ):

        super().__init__(A, algorithmic_relaxation, proximity_flag, use_gpu)
        self.c_projection = C_projection
        self.q_projection = Q_projection

    def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
        """
        Perform one step of the CQ algorithm.

        Parameters
        ----------
        x : npt.NDArray
            The point in the input space to be projected.
        y : npt.NDArray or None, optional
            The point in the target space to be projected,
            obtained through e.g. a perturbation step.
            If None, it is calculated from x.

        Returns
        -------
        npt.NDArray
        """
        if y is None:
            y = self.map(x)

        y_p = self.q_projection.project(y.copy())
        x = x - self.algorithmic_relaxation * self.map_back(y - y_p)

        return self.c_projection.project(x), None

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> float:
        # p = self.map(x)
        return [
            self.c_projection.proximity(x, proximity_measures),
            self.q_projection.proximity(self.map(x), proximity_measures),
        ]
        # return self.q_projection.proximity(p, proximity_measures)
        # TODO: correct?

    def _stopping_criterion(self, prox_tol: float, del_prox_tol: float, del_prox_n: float):

        if self.proximities[-1][1][0] < prox_tol:  # proximity below goal/tolerance
            return True
        else:  # check that last n proximity changes of are below a threshold
            if self.proximities[-2][1][0] - self.proximities[-1][1][0] < del_prox_tol:
                self._n_tol += 1
            else:
                self._n_tol = 0
            if self._n_tol >= del_prox_n:  # n proximity changes below threshold
                return True
        return False


# class LinearExtrapolatedLandweber(SplitFeasibility):
#     """
#     Implementation for a linear extrapolated Landweber algorithm to solve split feasibility problems.

#     Parameters
#     ----------
#     A : npt.NDArray
#         Matrix connecting input and target space.
#     lb: npt.NDArray
#         Lower bounds for the target space.
#     ub: npt.NDArray
#         Upper bounds for the target space.
#     algorithmic_relaxation : npt.NDArray or float, optional
#         Relaxation applied to the entire solution of the projection step, by default 1.
#     proximity_flag : bool, optional
#         A flag indicating whether to use this object for proximity calculations, by default True.
#     """

#     def __init__(
#         self,
#         A: npt.NDArray | sparse.sparray,
#         lb: npt.NDArray,
#         ub: npt.NDArray,
#         algorithmic_relaxation: npt.NDArray | float = 1,
#         proximity_flag=True,
#     ):

#         super().__init__(A, algorithmic_relaxation, proximity_flag)
#         self.bounds = Bounds(lb, ub)

#     def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
#         """
#         Perform one step of the linear extrapolated Landweber algorithm.

#         Parameters
#         ----------
#         x : npt.NDArray
#             The point in the input space to be projected.
#         Returns
#         -------
#         npt.NDArray
#         """
#         p = self.map(x)
#         (res_u, res_l) = self.bounds.residual(p)

#         x -= self.algorithmic_relaxation *


#     def _proximity(self, x: npt.NDArray) -> float:
#         """
#         Calculate the proximity of a point to the set Q.

#         Parameters
#         ----------
#         x : npt.NDArray
#             The point in the input space.

#         Returns
#         -------
#         float
#             The proximity measure.
#         """
#         p = self.map(x)
#         return self.q_projection.proximity(p)


class ProductSpaceAlgorithm(SplitFeasibility):

    """
    Implementation for a product space algorithm to solve split feasibility
    problems.

    Parameters
    ----------
    A : npt.NDArray
        Matrix connecting input and target space.
    C_projection : Projection
        The projection operator onto the set C.
    Q_projection : Projection
        The projection operator onto the set Q.
    algorithmic_relaxation : npt.NDArray or float, optional
        Relaxation applied to the entire solution of the projection step, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        C_projections: List[Projection],
        Q_projections: List[Projection],
        algorithmic_relaxation: npt.NDArray | float = 1,
        proximity_flag=True,
    ):

        super().__init__(A, algorithmic_relaxation, proximity_flag)
        self.c_projections = C_projections
        self.q_projections = Q_projections

        # calculate projection back into Ax=b space
        Z = np.concatenate([A, -1 * np.eye(A.shape[0])], axis=1)
        self.Pv = np.eye(Z.shape[1]) - LinearMapping(Z.T @ (np.linalg.inv(Z @ Z.T)) @ Z)

        print(
            "Warning! This algorithm is only suitable for small scale problems. Use the CQAlgorithm for larger problems."
        )
        self.xs = []
        self.ys = []

    def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
        """
        Perform one step of the product space algorithm.

        Parameters
        ----------
        x : npt.NDArray
            The point in the input space to be projected.
        y : npt.NDArray or None, optional
            The point in the target space to be projected, obtained through e.g. a perturbation step.
            If None, it is calculated from x.

        Returns
        -------
        npt.NDArray
        """
        if y is None:
            y = self.map(x)
        for el in self.c_projections:
            x = el.project(x)
        for el in self.q_projections:
            y = el.project(y)
        xy = self.Pv @ np.concatenate([x, y])
        self.xs.append(xy[: len(x)].copy())
        self.ys.append(xy[len(x) :].copy())
        return xy[: len(x)]  # ,xy[len(x):]

    def _proximity(self, x):
        raise NotImplementedError("Proximity not implemented for ProductSpaceAlgorithm.")
