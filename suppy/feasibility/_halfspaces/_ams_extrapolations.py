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

from suppy.feasibility._halfspaces._ams_algorithms import SimultaneousAMSHalfspace


class ExtrapolatedLandweberHalfspace(SimultaneousAMSHalfspace):
    """
    Extrapolated Landweber method for solving linear inequalities of the
    form Ax <= b.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the linear constraints.
    b : npt.NDArray
        Bound for linear inequalities
    weights : List[List[float]] or List[npt.NDArray]
        A list of lists or arrays representing the weights for each block. Each list/array should sum to 1.
    relaxation : float, optional
        The relaxation parameter for the constraints, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.
    """

    def __init__(self, A, b, relaxation=1, weights=None, proximity_flag=True):
        super().__init__(A, b, 1, relaxation, weights, proximity_flag)
        self.a_i = self.A.row_norm(2, 2)
        self.weight_norm = self.weights / self.a_i
        self.sigmas = []

    def _project(self, x):
        p = self.map(x)
        res = self.b - p
        res_idx = res < 0
        if not (np.any(res_idx)):
            self.sigmas.append(0)
            return x
        t = self.weight_norm[res_idx] * res[res_idx]
        t_2 = t @ self.A[res_idx, :]
        sig = (res[res_idx] @ t) / (t_2 @ t_2)
        self.sigmas.append(sig)
        x += sig * t_2

        return x


class AdaptiveStepLandweberHalfspace(SimultaneousAMSHalfspace):
    """
    Extrapolated Landweber algorithm for solving linear inequalities of the
    form Ax <= b.

    Parameters
    ----------
    A : npt.NDArray
        The matrix representing the linear constraints.
    b : npt.NDArray
        Bound for linear inequalities
    weights : List[List[float]] or List[npt.NDArray]
        A list of lists or arrays representing the weights for each block. Each list/array should sum to 1.
    relaxation : float, optional
        The relaxation parameter for the constraints, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.
    """

    def __init__(self, A, b, relaxation=1, weights=None, proximity_flag=True):
        super().__init__(A, b, 1, relaxation, weights, proximity_flag)

    def _project(self, x):
        xp = cp if self._use_gpu else np
        p = self.map(x)
        res = self.b - p
        res_idx = res < 0

        if not (np.any(res_idx)):
            # self.sigmas.append(0)
            return x

        u_t = (self.weights[res_idx] * res[res_idx]) @ self.A[res_idx, :]
        Au_t = self.A @ u_t  # A*AT*res
        sig = (u_t @ u_t) / (Au_t @ (self.weights * Au_t))
        x += sig * u_t

        return x
