from abc import ABC, abstractmethod
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

    no_gpu = False

except ImportError:
    no_gpu = True
    cp = None


class Feasibility(Projection, ABC):
    """
    Parameters
    ----------
    algorithmic_relaxation : npt.ArrayLike or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter for the projection, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use this object for proximity
    calculations, by default True.

    Attributes
    ----------
    algorithmic_relaxation : npt.ArrayLike or float, optional
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
        algorithmic_relaxation: npt.ArrayLike | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
        _use_gpu: bool = False,
    ):
        super().__init__(relaxation, proximity_flag, _use_gpu)
        self.algorithmic_relaxation = algorithmic_relaxation
        self.all_x = None

    @ensure_float_array
    def solve(
        self,
        x: npt.ArrayLike,
        max_iter: int = 500,
        storage: bool = False,
        constr_tol: float = 1e-6,
        proximity_measure: List | None = None,
    ) -> npt.ArrayLike:
        """
        Solves the optimization problem using an iterative approach.

        Parameters
        ----------
        x : npt.ArrayLike
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
        npt.ArrayLike
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


class LinearFeasibility(Feasibility, ABC):
    """
    LinearFeasibility class for handling linear feasibility problems.

    Parameters
    ----------
    A : npt.ArrayLike or sparse.sparray
        Matrix for linear inequalities
    algorithmic_relaxation : npt.ArrayLike or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        Flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    algorithmic_relaxation : npt.ArrayLike or float, optional
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
        A: npt.ArrayLike | sparse.sparray,
        algorithmic_relaxation: npt.ArrayLike | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        _, _use_gpu = LinearMapping.get_flags(A)
        super().__init__(algorithmic_relaxation, relaxation, proximity_flag, _use_gpu)
        self.A = LinearMapping(A)
        self.inverse_row_norm = 1 / self.A.row_norm(2, 2)

    def map(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Applies the linear mapping to the input array x.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to which the linear mapping is applied.

        Returns
        -------
        npt.ArrayLike
            The result of applying the linear mapping to the input array.
        """
        return self.A @ x

    def single_map(self, x: npt.ArrayLike, i: int) -> npt.ArrayLike:
        """
        Applies the linear mapping to the input array x at a specific index
        i.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to which the linear mapping is applied.
        i : int
            The specific index at which the linear mapping is applied.

        Returns
        -------
        npt.ArrayLike
            The result of applying the linear mapping to the input array at the specified index.
        """
        return self.A.single_map(x, i)

    def indexed_map(self, x: npt.ArrayLike, idx: List[int] | npt.ArrayLike) -> npt.ArrayLike:
        """
        Applies the linear mapping to the input array x at multiple
        specified
        indices.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to which the linear mapping is applied.
        idx : List[int] or npt.ArrayLike
            The indices at which the linear mapping is applied.

        Returns
        -------
        npt.ArrayLike
            The result of applying the linear mapping to the input array at the specified indices.
        """
        return self.A.index_map(x, idx)

    # @abstractmethodpass
    # def project(self, x: npt.ArrayLike) -> npt.ArrayLike:
    #


class HyperplaneFeasibility(LinearFeasibility, ABC):
    """
    HyperplaneFeasibility class for solving halfspace feasibility problems.

    Parameters
    ----------
    A : npt.ArrayLike or sparse.sparray
        Matrix for linear inequalities
    b : npt.ArrayLike
        Bound for linear inequalities
    algorithmic_relaxation : npt.ArrayLike or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        Flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    b : npt.ArrayLike
        Bound for linear inequalities
    algorithmic_relaxation : npt.ArrayLike or float, optional
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
        A: npt.ArrayLike | sparse.sparray,
        b: npt.ArrayLike,
        algorithmic_relaxation: npt.ArrayLike | float = 1.0,
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
            if self.A.flag == "numpy" or self.A.flag == "scipy_sparse":
                b = np.ones(A.shape[0]) * b
            elif self.A.flag == "cupy_full" or self.A.flag == "cupy_sparse":
                b = cp.ones(A.shape[0]) * b
        self.b = b

    def _proximity(self, x: npt.ArrayLike, proximity_measures: List) -> float:
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
    A : npt.ArrayLike or sparse.sparray
        Matrix for linear inequalities
    b : npt.ArrayLike
        Bound for linear inequalities
    algorithmic_relaxation : npt.ArrayLike or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : float, optional
        The relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        Flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    b : npt.ArrayLike
        Bound for linear inequalities
    algorithmic_relaxation : npt.ArrayLike or float, optional
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
        A: npt.ArrayLike | sparse.sparray,
        b: npt.ArrayLike,
        algorithmic_relaxation: npt.ArrayLike | float = 1.0,
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
            if self.A.flag == "numpy" or self.A.flag == "scipy_sparse":
                b = np.ones(A.shape[0]) * b
            elif self.A.flag == "cupy_full" or self.A.flag == "cupy_sparse":
                b = cp.ones(A.shape[0]) * b
        self.b = b

    def _proximity(self, x: npt.ArrayLike, proximity_measures: List) -> float:

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
    A : npt.ArrayLike
        The matrix representing the linear system.
    lb : npt.ArrayLike
        The lower bounds for the hyperslab.
    ub : npt.ArrayLike
        The upper bounds for the hyperslab.
    algorithmic_relaxation : npt.ArrayLike or float, optional
        The relaxation parameter for the algorithm, by default 1.0.
    relaxation : int, optional
        The relaxation parameter, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity, by default True.

    Attributes
    ----------
    Bounds : Bounds
        Objective for handling the upper and lower bounds of the hyperslab.
    A : LinearMapping
        Matrix for linear system (stored in internal LinearMapping object).
    algorithmic_relaxation : npt.ArrayLike or float, optional
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
        A: npt.ArrayLike,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
        algorithmic_relaxation: npt.ArrayLike | float = 1.0,
        relaxation=1,
        proximity_flag=True,
    ):
        super().__init__(A, algorithmic_relaxation, relaxation, proximity_flag)
        self.Bounds = Bounds(lb, ub)
        if self.A.shape[0] != len(self.Bounds.l):
            raise ValueError("Matrix A and bound vector must have the same number of rows.")

    def _proximity(self, x: npt.ArrayLike, proximity_measures: List) -> float:

        p = self.map(x)

        # residuals are positive if constraints are met
        (res_l, res_u) = self.Bounds.residual(p)
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
