from typing import List
import numpy as np
import numpy.typing as npt
from scipy import sparse

try:
    import cupy as cp

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = np


from suppy.feasibility._hyperplanes._kaczmarz_algorithms import SimultaneousKaczmarzMethod


class DROPHyperplane(SimultaneousKaczmarzMethod):
    """
    Diagonally Relaxed Orthogonal Projections (DROP) algorithm for solving
    linear equalities of the form Ax = b.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Sparse matrix for linear systems. Must support ``getnnz(axis=0)``.
    b : npt.NDArray
        Bound for linear systems
    algorithmic_relaxation : npt.NDArray or float, optional
        The relaxation parameter used by the algorithm, by default 1.0.
    relaxation : float, optional
        Outer relaxation parameter, applied to the entire solution of the iterate by default 1.0.
    proximity_flag : bool, optional
        Flag to indicate if proximity calculations should be performed, by default True.

    References
    ----------
    - [1] https://epubs.siam.org/doi/10.1137/050639399
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        b: npt.NDArray,
        algorithmic_relaxation: npt.NDArray | float = 1.0,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        # not NO_GPU guard ensures cp.sparse is never accessed when CuPy is unavailable
        A_cpu = A.get() if (not NO_GPU and isinstance(A, cp.sparse.csr_matrix)) else A
        xp = np if isinstance(A_cpu, np.ndarray) else cp
        nnz_counts = xp.array(
            A_cpu.getnnz(axis=0), dtype=xp.float32
        )  # number of non-zeros per row
        del A_cpu

        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag=proximity_flag)

        self.inv_nnz_counts = 1 / nnz_counts

    def _project(self, x: npt.NDArray) -> np.ndarray:
        # simultaneous projection
        p = self.map(x)
        res = self.b - p
        x += (
            self.inv_nnz_counts
            * self.algorithmic_relaxation
            * (self.inverse_row_norm * res @ self.A)
        )
        return x
