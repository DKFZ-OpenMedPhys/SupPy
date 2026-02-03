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


from suppy.feasibility._hyperplanes._kaczmarz_algorithms import SimultaneousKaczmarzMethod


class DROPHyperplane(SimultaneousKaczmarzMethod):
    """
    Diagonally Relaxed Orthogonal Projections (DROP) algorithm for solving
    linear inequalities of the
    form Ax = b.

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
        Flag to indicate if proximity calculations should be performed, by default True.
    """

    def __init__(self, A, b, algorithmic_relaxation=1, relaxation=1, proximity_flag=True):
        A_cpu = A.get() if (isinstance(A, cp.sparse.csr_matrix) and not NO_GPU) else A
        xp = np if isinstance(A_cpu, np.ndarray) else cp
        nnz_counts = xp.array(
            A_cpu.getnnz(axis=0), dtype=xp.float32
        )  # number of non-zeros per row
        del A_cpu

        super().__init__(A, b, algorithmic_relaxation, relaxation, proximity_flag=proximity_flag)

        self.inv_nnz_counts = 1 / nnz_counts

    def _project(self, x):
        # simultaneous projection
        p = self.map(x)
        res = self.b - p
        x += (
            self.inv_nnz_counts
            * self.algorithmic_relaxation
            * (self.inverse_row_norm * res @ self.A)
        )
        return x
