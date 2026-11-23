import math
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


from suppy.feasibility._bands._ams_algorithms import SimultaneousAMSHyperslab


class FISTAHyperslab(SimultaneousAMSHyperslab):
    """
    FISTA for solving linear inequalities of the form lb <= Ax <= ub.

    Each call of `project` performs one accelerated gradient step with step
    size 1/L on f(x) = 1/2 sum_i w_i d_i(x)^2, where d_i(x) is the distance
    of (Ax)_i to [lb_i, ub_i]. f is half the ("p_norm", 2) proximity of this
    object and L = lambda_max(A^T W A) with W = diag(w). The momentum is the
    one of Beck and Teboulle [1]: z_k = x_k + (t_{k-1} - 1) / t_k
    (x_k - x_{k-1}) with t_1 = 1 and t_k = (1 + sqrt(1 + 4 t_{k-1}^2)) / 2,
    where x_k and x_{k-1} are the inputs of the current and the previous
    call; in superiorization these are the perturbed iterates. It is cleared
    by `reset`. Bounds on x (e.g. x >= 0) are added by combining with a
    `BoxProjection` in a `SequentialProjection`.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems.
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    weights : None or List[float], optional
        The weights w of the constraints, renormalized to sum to 1, by default
        None (equal weights).
    lipschitz : float or None, optional
        The Lipschitz constant L of the gradient of f. If None, L is estimated
        by power iteration and increased by 1 %, since the estimate approaches
        lambda_max from below. By default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    Attributes
    ----------
    lipschitz : float
        The Lipschitz constant L used for the step size 1/L.

    References
    ----------
    - [1] https://doi.org/10.1137/080716542
    - [2] https://arxiv.org/abs/1109.2415
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        weights: None | List[float] | npt.NDArray = None,
        lipschitz: float | None = None,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, weights, proximity_flag)
        self.lipschitz = self._estimate_lipschitz() if lipschitz is None else lipschitz
        if not self.lipschitz > 0:
            raise ValueError("The Lipschitz constant must be positive.")
        self._k = 0
        self._t = 1.0
        self._x_prev = None

    def reset(self) -> None:
        """Clear the momentum."""
        self._k = 0
        self._t = 1.0
        self._x_prev = None

    def _weighted_residual_map(self, r: npt.NDArray) -> np.ndarray:
        # (w * r) @ A instead of A.T @ (w * r): avoids building the transpose of a sparse A
        return (self.weights * r) @ self.A.A

    def _estimate_lipschitz(self, max_iter: int = 1000, tol: float = 1e-6) -> float:
        xp = cp if self._use_gpu else np
        dtype = np.result_type(self.A.dtype, np.float32)
        v = xp.asarray(np.random.default_rng(0).standard_normal(self.A.shape[1]), dtype=dtype)
        v /= xp.linalg.norm(v)
        lam = 0.0
        for _ in range(max_iter):
            u = self._weighted_residual_map(self.map(v))
            lam_new = float(v @ u)
            norm_u = float(xp.linalg.norm(u))
            if norm_u == 0:
                return 0.0
            v = u / norm_u
            if abs(lam_new - lam) <= tol * lam_new:
                break
            lam = lam_new
        return 1.01 * lam_new

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        if self._x_prev is None:
            z = x
        else:
            t_next = (1 + math.sqrt(1 + 4 * self._t**2)) / 2
            z = x + (self._t - 1) / t_next * (x - self._x_prev)
            self._t = t_next
        self._x_prev = x.copy()
        self._k += 1

        (res_l, res_u) = self.bounds.residual(self.map(z))
        # signed violation: negative below lb, positive above ub
        r = xp.minimum(res_l, 0) - xp.minimum(res_u, 0)
        step = self._weighted_residual_map(r) / self.lipschitz
        return z - step.astype(z.dtype, copy=False)
