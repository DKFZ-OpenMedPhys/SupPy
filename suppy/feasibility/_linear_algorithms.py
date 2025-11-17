"""Base classes for linear feasibility problems."""
from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt

from scipy import sparse

from suppy.utils import Bounds
from suppy.utils import LinearMapping
from suppy.utils import ensure_float_array
from suppy.projections._projections import Projection

try:
    import cupy as cp

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = np


class Feasibility(Projection, ABC):
    """
    Parameters
    ----------
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity
    calculations, by default True.

    Attributes
    ----------
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate whether to calculate proximity, by default True.
    _use_gpu : bool, optional
        Flag to indicate whether to use GPU for computations, by default False.
    """

    def __init__(
        self,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
        _use_gpu: bool = False,
    ):
        super().__init__(relaxation, proximity_flag, _use_gpu)
        self.algorithmic_relaxation = algorithmic_relaxation
        self.all_x = None
        self.proximities = None
        # self._n_tol = 0  # counter for stopping criterion

    @ensure_float_array
    def solve(
        self,
        x: npt.NDArray,
        max_iter: int = 500,
        prox_tol: float = 1e-6,
        del_prox_tol: float = 1e-8,
        del_prox_n: int = 5,
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
        prox_tol : float, optional
            The tolerance for the proximity on the constraints, by default 1e-6.
        del_prox_tol : float, optional
            The tolerance for the change in proximity over the last del_prox_n iterations, by default 1e-89.
        del_prox_n : int, optional
            The number of iterations to check for the change in proximity, by default 5.
        proximity_measures : List, optional
            The proximity measures to calculate, by default None. Right now only the first in the list is used to check the feasibility.

        Returns
        -------
        npt.NDArray
            The solution after the iterative process.
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
            if isinstance(x, np.ndarray):
                self.all_x.append(np.array(x.copy()))
            else:
                self.all_x.append((x.get()))

        stop = False  # criterion for stopping the algorithm

        while i < max_iter and not stop:
            x = self.project(x)
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


class LinearFeasibility(Feasibility, ABC):
    """
    LinearFeasibility class for handling linear feasibility problems.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear inequalities
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        Flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate whether to calculate proximity, by default True.
    _use_gpu : bool, optional
        Flag to indicate whether to use GPU for computations, by default False.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        _, _use_gpu = LinearMapping.get_flags(A)
        super().__init__(algorithmic_relaxation, relaxation, proximity_flag, _use_gpu)
        self.A = LinearMapping(A)
        self.inverse_row_norm = 1 / self.A.row_norm(2, 2)

    def map(self, x: npt.NDArray) -> npt.NDArray:
        """
        Applies the linear mapping to the input array x.

        Parameters
        ----------
        x : npt.NDArray
            The input array to which the linear mapping is applied.

        Returns
        -------
        npt.NDArray
            The result of applying the linear mapping to the input array.
        """
        return self.A @ x

    def single_map(self, x: npt.NDArray, i: int) -> npt.NDArray:
        """
        Applies the linear mapping to the input array x at a specific index
        i.

        Parameters
        ----------
        x : npt.NDArray
            The input array to which the linear mapping is applied.
        i : int
            The specific index at which the linear mapping is applied.

        Returns
        -------
        npt.NDArray
            The result of applying the linear mapping to the input array at the specified index.
        """
        return self.A.single_map(x, i)

    def indexed_map(self, x: npt.NDArray, idx: List[int] | npt.NDArray) -> npt.NDArray:
        """
        Applies the linear mapping to the input array x at multiple
        specified
        indices.

        Parameters
        ----------
        x : npt.NDArray
            The input array to which the linear mapping is applied.
        idx : List[int] or npt.NDArray
            The indices at which the linear mapping is applied.

        Returns
        -------
        npt.NDArray
            The result of applying the linear mapping to the input array at the specified indices.
        """
        return self.A.index_map(x, idx)

    # @abstractmethodpass
    # def project(self, x: npt.NDArray) -> npt.NDArray:
    #


class HyperplaneFeasibility(LinearFeasibility, ABC):
    """
    HyperplaneFeasibility class for solving halfspace feasibility problems.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear inequalities
    b : npt.NDArray
        Bound for linear inequalities
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        Flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    b : npt.NDArray
        Bound for linear inequalities
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate whether to calculate proximity, by default True.
    _use_gpu : bool, optional
        Flag to indicate whether to use GPU for computations, by default False.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        b: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, algorithmic_relaxation, relaxation, proximity_flag)
        try:
            len(b)
            if A.shape[0] != len(b):
                raise ValueError("Matrix A and vector b must have the same number of rows.")
        except TypeError:
            # create an array for b if it is a scalar
            if not self.A.gpu:
                b = np.ones(A.shape[0]) * b
            else:
                b = cp.ones(A.shape[0]) * b
        self.b = b

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> float:
        p = self.map(x)
        # residuals are positive  if constraints are met
        res = abs(self.b - p)
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(1 / len(res) * (res ** measure[1]).sum())
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure")
        return measures


class HalfspaceFeasibility(LinearFeasibility, ABC):
    """
    HalfspaceFeasibility class for solving halfspace feasibility problems.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear inequalities
    b : npt.NDArray
        Bound for linear inequalities
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        Flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    b : npt.NDArray
        Bound for linear inequalities
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate whether to calculate proximity, by default True.
    _use_gpu : bool, optional
        Flag to indicate whether to use GPU for computations, by default False.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        b: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, algorithmic_relaxation, relaxation, proximity_flag)
        try:
            len(b)
            if A.shape[0] != len(b):
                raise ValueError("Matrix A and vector b must have the same number of rows.")
        except TypeError:
            # create an array for b if it is a scalar
            if not self.A.gpu:
                b = np.ones(A.shape[0]) * b
            else:
                b = cp.ones(A.shape[0]) * b
        self.b = b

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> float:

        p = self.map(x)
        # residuals are positive  if constraints are met
        res = self.b - p
        res[res > 0] = 0
        res = -res

        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(1 / len(res) * (res ** measure[1]).sum())
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure)")
        return measures


class HyperslabFeasibility(LinearFeasibility, ABC):
    """
    A class used to for solving feasibility problems for hyperslabs.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the linear system.
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : int, optional
        The relaxation parameter, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    bounds : bounds
        Objective for handling the upper and lower bounds of the hyperslab.
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate whether to calculate proximity, by default True.
    _use_gpu : bool, optional
        Flag to indicate whether to use GPU for computations, by default False.
    """

    def __init__(
        self,
        A: npt.NDArray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation=1,
        proximity_flag=True,
    ):
        super().__init__(A, algorithmic_relaxation, relaxation, proximity_flag)
        self.bounds = Bounds(lb, ub)
        if self.A.shape[0] != len(self.bounds.l):
            raise ValueError("Matrix A and bound vector must have the same number of rows.")

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> float:

        p = self.map(x)

        # residuals are positive if constraints are met
        (res_l, res_u) = self.bounds.residual(p)
        res_l[res_l > 0] = 0
        res_u[res_u > 0] = 0
        res = -res_l - res_u

        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(1 / len(res) * (res ** measure[1]).sum())
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(res.max())
            else:
                raise ValueError("Invalid proximity measure)")
        return measures
