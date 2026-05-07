"""Algorithms for split feasibility problem."""
import warnings
from abc import ABC, abstractmethod
from typing import List, Callable
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

from suppy.feasibility._linear_algorithms import Feasibility


class SplitFeasibility(Feasibility, ABC):
    """
    Abstract base class used to represent split feasibility problems.

    Parameters
    ----------
    A : npt.NDArray
        Matrix connecting input and target space.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations, by default True.

    Attributes
    ----------
    A : LinearMapping
        Linear mapping between input and target space.
    proximities : list
        A list to store proximity values during the solve process.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
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
        storage_iters: List[int] | int | None = None,
        proximity_measures: List | None = None,
        alternative_stopping_criterion: Callable | None = None,
        alternative_stopping_criterion_initial_call: Callable | None = None,
    ) -> np.ndarray:
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
        storage_iters : List[int] or int, optional
            Controls which iterations are stored (when storage=True). If None, all iterations are stored.
            If a list of ints, only those iteration indices are stored (0 = initial point).
            If an int, storage occurs every that many iterations.
        proximity_measures : List, optional
            The proximity measures to calculate, by default None.
            Right now only the first in the list is used to check the feasibility.
        alternative_stopping_criterion : callable, optional
            Alternative stopping criterion.
        alternative_stopping_criterion_initial_call : callable, optional
            Initial call for an alternative stopping criterion.

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

        def _should_store(idx):
            if storage_iters is None:
                return True
            if isinstance(storage_iters, int):
                return idx % storage_iters == 0
            return idx in storage_iters

        if storage is True:
            self.all_x = []
            if _should_store(0):
                if isinstance(x, cp.ndarray) and not NO_GPU:
                    self.all_x.append((x.get()))
                else:
                    self.all_x.append(np.array(x.copy()))

        if alternative_stopping_criterion_initial_call is not None:
            stop = alternative_stopping_criterion_initial_call(x, self)
        else:
            stop = False

        while i < max_iter and not stop:
            x, _ = self.step(x)
            if storage is True and _should_store(i + 1):
                if isinstance(x, np.ndarray):  # convert to np array if cp
                    self.all_x.append(np.array(x.copy()))
                else:
                    self.all_x.append((x.get()))
            self.proximities.append(self.proximity(x, proximity_measures))

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

    def project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> tuple:
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
        tuple
            A (x, y) pair of projected arrays; y may be None.
        """
        return self._project(x, y)

    def step(self, x: npt.NDArray, y: npt.NDArray | None = None) -> tuple:
        return self.project(x, y)

    @abstractmethod
    def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> tuple:
        pass

    def map(self, x: npt.NDArray) -> np.ndarray:
        """
        Maps the input space array to the target space via matrix
        multiplication.

        Parameters
        ----------
        x : npt.NDArray
            The input space array to be mapped.

        Returns
        -------
        npt.NDArray
            The corresponding target space array.
        """
        return self.A @ x

    def map_back(self, y: npt.NDArray) -> np.ndarray:
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
        The relaxation parameter used by the algorithm, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations, by default True.

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
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    proximity_flag : bool
        A flag indicating whether to use this object for proximity calculations.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        C_projection: Projection,
        Q_projection: Projection,
        algorithmic_relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, algorithmic_relaxation, proximity_flag)
        self.c_projection = C_projection
        self.q_projection = Q_projection

    def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> tuple:
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
        tuple
            A (x, None) pair with the updated input-space point.
        """
        if y is None:
            y = self.map(x)

        y_p = self.q_projection.project(y.copy())
        x = x - self.algorithmic_relaxation * self.map_back(y - y_p)

        return self.c_projection.project(x), None

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> list[float]:
        return [
            self.c_projection.proximity(x, proximity_measures),
            self.q_projection.proximity(self.map(x), proximity_measures),
        ]

    def _stopping_criterion(self, prox_tol: float, del_prox_tol: float, del_prox_n: int) -> bool:
        """Returns True when convergence is detected, False otherwise."""
        if self.proximities[-1][1][0] < prox_tol:
            return True
        else:  # check that last n proximity changes of are below a threshold
            if self.proximities[-2][1][0] - self.proximities[-1][1][0] < del_prox_tol:
                self._n_tol += 1
            else:
                self._n_tol = 0
            if self._n_tol >= del_prox_n:
                return True
        return False


class ProductSpaceAlgorithm(SplitFeasibility):
    """
    Implementation for a product space algorithm to solve split feasibility
    problems.

    Parameters
    ----------
    A : npt.NDArray
        Matrix connecting input and target space.
    C_projections : List[Projection]
        The projection operators onto the sets C.
    Q_projections : List[Projection]
        The projection operators onto the sets Q.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity calculations, by default True.

    Attributes
    ----------
    Pv : npt.NDArray
        Projection matrix onto the constraint manifold.
    xs : list
        Input-space iterates accumulated during solve.
    ys : list
        Target-space iterates accumulated during solve.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        C_projections: List[Projection],
        Q_projections: List[Projection],
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, algorithmic_relaxation, proximity_flag)
        self.c_projections = C_projections
        self.q_projections = Q_projections

        # calculate projection back into Ax=b space
        Z = np.concatenate([A, -1 * np.eye(A.shape[0])], axis=1)
        self.Pv = np.eye(Z.shape[1]) - LinearMapping(Z.T @ (np.linalg.inv(Z @ Z.T)) @ Z)

        warnings.warn(
            "ProductSpaceAlgorithm is only suitable for small scale problems. "
            "Use CQAlgorithm for larger problems.",
            stacklevel=2,
        )
        self.xs = []
        self.ys = []

    def _project(self, x: npt.NDArray, y: npt.NDArray | None = None) -> tuple:
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
        tuple
            A (x, None) pair with the updated input-space point.
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
        return xy[: len(x)], None

    def _proximity(self, x: npt.NDArray, _proximity_measures: List) -> list[float]:
        raise NotImplementedError("Proximity not implemented for ProductSpaceAlgorithm.")
