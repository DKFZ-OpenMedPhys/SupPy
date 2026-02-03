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


from suppy.feasibility._bands._ams_algorithms import SimultaneousAMSHyperslab


class ExtrapolatedLandweberHyperslab(SimultaneousAMSHyperslab):
    """
    Extrapolated Landweber algorithm for solving linear inequalities of the
    form lb <= Ax <= ub.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    weights : List[List[float]] or List[npt.NDArray]
        A list of lists or arrays representing the weights for each block. Each list/array should sum to 1.
    relaxation : float, optional
        The relaxation parameter for the constraints, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.
    """

    def __init__(self, A, lb, ub, relaxation=1, weights=None, proximity_flag=True):
        super().__init__(A, lb, ub, 1, relaxation, weights, proximity_flag)
        self.a_i = self.A.row_norm(2, 2)
        # To avoid division by zero
        self.weight_norm = self.weights / self.a_i
        self.weight_norm[self.a_i == 0] = 0
        # TODO: Supposed to help with division by zero, but algorithm produces nans once all other criteria are met

    def _project(self, x):

        xp = cp if self._use_gpu else np
        p = self.map(x)
        (res_l, res_u) = self.bounds.residual(p)
        d_idx = res_u < 0
        c_idx = res_l < 0
        if not (xp.any(d_idx)) and not (xp.any(c_idx)):
            return x
        t_u = self.weight_norm[d_idx] * res_u[d_idx]  # D*(Ax-b)+
        t_l = self.weight_norm[c_idx] * res_l[c_idx]
        t_u_2 = t_u @ self.A[d_idx, :]
        t_l_2 = t_l @ self.A[c_idx, :]

        sig = ((res_l[c_idx] @ (t_l)) + (res_u[d_idx] @ (t_u))) / (
            (t_u_2 - t_l_2) @ (t_u_2 - t_l_2)
        )
        x += sig * (t_u_2 - t_l_2)

        if xp.isnan(x).any():
            raise ValueError("NaN encountered in Extrapolated Landweber Hyperslab projection.")

        return x


class BlockIterativeExtrapolatedLandweberHyperslab(ExtrapolatedLandweberHyperslab):
    def __init__(
        self,
        A,
        lb,
        ub,
        length_blocks,
        algorithmic_relaxation=1,
        relaxation=1,
        weights=None,
        proximity_flag=True,
    ):
        super().__init__(A, lb, ub, algorithmic_relaxation, relaxation, weights, proximity_flag)
        self.length_blocks = length_blocks

        self.As = []
        self.ls = []
        self.us = []
        self.block_bounds = []
        self.weight_norms = []
        _lower_bound = 0
        for i in range(len(self.length_blocks)):
            self.As.append(A[_lower_bound : _lower_bound + self.length_blocks[i]])
            self.ls.append(lb[_lower_bound : _lower_bound + self.length_blocks[i]])
            self.us.append(ub[_lower_bound : _lower_bound + self.length_blocks[i]])
            self.weight_norms.append(
                self.weight_norm[_lower_bound : _lower_bound + self.length_blocks[i]]
            )
            _lower_bound += self.length_blocks[i]

    def _project(self, x):
        xp = cp if self._use_gpu else np
        for (i, A) in enumerate(self.As):
            p = A @ x
            res_l, res_u = p - self.ls[i], self.us[i] - p
            d_idx = res_u < 0
            c_idx = res_l < 0
            if not (xp.any(d_idx)) and not (xp.any(c_idx)):
                continue
            t_u = self.weight_norms[i][d_idx] * res_u[d_idx]  # D*(Ax-b)+
            t_l = self.weight_norms[i][c_idx] * res_l[c_idx]
            t_u_2 = t_u @ A[d_idx, :]
            t_l_2 = t_l @ A[c_idx, :]

            sig = ((res_l[c_idx] @ (t_l)) + (res_u[d_idx] @ (t_u))) / (
                (t_u_2 - t_l_2) @ (t_u_2 - t_l_2)
            )
            x += sig * (t_u_2 - t_l_2)
        return x


class AdaptiveStepLandweberHyperslab(SimultaneousAMSHyperslab):
    """
    Extrapolated Landweber algorithm for solving linear inequalities of the
    form lb <= Ax <= ub.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    weights : List[List[float]] or List[npt.NDArray]
        A list of lists or arrays representing the weights for each block. Each list/array should sum to 1.
    relaxation : float, optional
        The relaxation parameter for the constraints, by default 1.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.
    """

    def __init__(self, A, lb, ub, relaxation=1, weights=None, proximity_flag=True):
        super().__init__(A, lb, ub, 1, relaxation, weights, proximity_flag)

    def _project(self, x):
        xp = cp if self._use_gpu else np
        p = self.map(x)
        (res_l, res_u) = self.bounds.residual(p)
        d_idx = res_u < 0
        c_idx = res_l < 0

        if not (xp.any(d_idx)) and not (xp.any(c_idx)):
            # self.sigmas.append(0)
            return x

        u_t_u = (self.weights[d_idx] * res_u[d_idx]) @ self.A[d_idx, :]
        u_t_l = (self.weights[c_idx] * res_l[c_idx]) @ self.A[c_idx, :]

        u_diff = u_t_u - u_t_l  # AT*(res_ub+res_lb)
        Au_t = self.A @ u_diff  # A*AT*(res_ub+res_lb)
        sig = (u_diff @ u_diff) / (Au_t @ (self.weights * Au_t))
        x += sig * (u_t_u - u_t_l)

        return x
