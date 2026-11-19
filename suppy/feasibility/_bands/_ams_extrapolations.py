from abc import ABC
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
from suppy.feasibility._linear_algorithms import _normalized_weights


class LandweberHyperslab(SimultaneousAMSHyperslab):
    """
    Projected Landweber method with a fixed step size for solving linear
    inequalities of the form lb <= Ax <= ub.

    Each call of `project` performs one iteration

        x <- x + step_size * A^T W (P(Ax) - Ax),

    where P is the projection onto the box [lb, ub], i.e. one gradient step
    on f(x) = 1/2 ||W^(1/2) (P(Ax) - Ax)||^2. The iteration converges for
    0 < step_size < 2 / lambda_max(A^T W A).

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems.
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    step_size : float or None, optional
        The step size of the iteration. If None, the step size is set to
        1 / lambda_max(A^T W A) (= 1 / ||A||_2^2 for W = I), estimated by
        power iteration and increased by 1 %, since the estimate approaches
        it from below. By default None.
    weights : None, List[float] or npt.NDArray, optional
        Nonnegative weights w of the constraints, W = diag(w). They are used
        as given in the update and renormalized to sum to 1 for the
        proximity measures. If None, W = I. By default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    Attributes
    ----------
    step_size : float
        The step size used by the iteration.
    landweber_weights : npt.NDArray or None
        The weights w used in the update, None for W = I.

    References
    ----------
    - [1] https://doi.org/10.2307/2372313
    - [2] https://doi.org/10.1007/978-3-642-30901-4
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        step_size: float | None = None,
        weights: None | List[float] | npt.NDArray = None,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, _normalized_weights(weights), proximity_flag)
        self._setup_landweber(weights, step_size)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        (res_l, res_u) = self.bounds.residual(self.map(x))
        # P(Ax) - Ax: positive below lb, negative above ub
        res = xp.minimum(res_u, 0) - xp.minimum(res_l, 0)
        if self.landweber_weights is not None:
            res = self.landweber_weights * res
        x += self.step_size * (res @ self.A)
        return x


class ExtrapolatedSimultaneousProjectionHyperslab(SimultaneousAMSHyperslab):
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
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    References
    ----------
    - [1] https://doi.org/10.1007/s11075-025-02163-0
    - [2] https://doi.org/10.1007/978-3-642-30901-4
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, weights, proximity_flag)
        self.a_i = self.A.row_norm(2, 2)
        # To avoid division by zero
        self.weight_norm = self.weights / self.a_i
        self.weight_norm[self.a_i == 0] = 0
        # if only all-zero rows are violated the step is 0/0: _project then returns x unchanged

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        p = self.map(x)
        (res_l, res_u) = self.bounds.residual(p)
        d_idx = res_u < 0
        c_idx = res_l < 0
        if not (xp.any(d_idx)) and not (xp.any(c_idx)):
            return x
        t_u = self.weight_norm[d_idx] * res_u[d_idx]
        t_l = self.weight_norm[c_idx] * res_l[c_idx]
        t_u_2 = t_u @ self.A[d_idx, :]
        t_l_2 = t_l @ self.A[c_idx, :]

        v = t_u_2 - t_l_2
        denom = v @ v
        if (
            denom == 0
        ):  # no usable step, e.g. v @ v underflows (float32) once the violations are tiny
            return x
        sig = ((res_l[c_idx] @ t_l) + (res_u[d_idx] @ t_u)) / denom
        x += sig * v

        if xp.isnan(x).any():
            raise ValueError("NaN encountered in Extrapolated Landweber Hyperslab projection.")

        return x


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
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    References
    ----------
    - [1] https://doi.org/10.1007/s11075-025-02163-0
    - [2] https://doi.org/10.1007/978-3-642-30901-4
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, weights, proximity_flag)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        p = self.map(x)
        (res_l, res_u) = self.bounds.residual(p)
        d_idx = res_u < 0
        c_idx = res_l < 0
        if not (xp.any(d_idx)) and not (xp.any(c_idx)):
            return x
        t_u = self.weights[d_idx] * res_u[d_idx]
        t_l = self.weights[c_idx] * res_l[c_idx]
        t_u_2 = t_u @ self.A[d_idx, :]
        t_l_2 = t_l @ self.A[c_idx, :]

        v = t_u_2 - t_l_2
        denom = v @ v
        if (
            denom == 0
        ):  # no usable step, e.g. v @ v underflows (float32) once the violations are tiny
            return x
        sig = ((res_l[c_idx] @ t_l) + (res_u[d_idx] @ t_u)) / denom
        x += sig * v

        if xp.isnan(x).any():
            raise ValueError("NaN encountered in Extrapolated Landweber Hyperslab projection.")

        return x


class BlockIterativeExtrapolatedLandweberHyperslab(ExtrapolatedLandweberHyperslab):
    """
    Block-iterative variant of the Extrapolated Landweber algorithm for
    hyperslabs, solving lb <= Ax <= ub block by block.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems.
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    length_blocks : List[int]
        Number of rows in each block. Must sum to the total number of rows in A.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        length_blocks: List[int],
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, relaxation, weights, proximity_flag)
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

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        for i, A_block in enumerate(self.As):
            p = A_block @ x
            res_l, res_u = p - self.ls[i], self.us[i] - p
            d_idx = res_u < 0
            c_idx = res_l < 0
            if not (xp.any(d_idx)) and not (xp.any(c_idx)):
                continue
            t_u = self.weight_norms[i][d_idx] * res_u[d_idx]
            t_l = self.weight_norms[i][c_idx] * res_l[c_idx]
            t_u_2 = t_u @ A_block[d_idx, :]
            t_l_2 = t_l @ A_block[c_idx, :]

            v = t_u_2 - t_l_2
            denom = v @ v
            if denom == 0:  # no usable step for this block
                continue
            sig = ((res_l[c_idx] @ t_l) + (res_u[d_idx] @ t_u)) / denom
            x += sig * v
        return x


class AdaptiveStepLandweberHyperslab(SimultaneousAMSHyperslab):
    """
    Adaptive step-size Landweber algorithm for solving linear inequalities
    of
    the form lb <= Ax <= ub.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    References
    ----------
    - [1] https://doi.org/10.1515/jiip-2015-0082
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        weights: None | List[float] | npt.NDArray = None,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, weights, proximity_flag)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        p = self.map(x)
        (res_l, res_u) = self.bounds.residual(p)
        d_idx = res_u < 0
        c_idx = res_l < 0

        if not (xp.any(d_idx)) and not (xp.any(c_idx)):
            return x

        u_t_u = (self.weights[d_idx] * res_u[d_idx]) @ self.A[d_idx, :]
        u_t_l = (self.weights[c_idx] * res_l[c_idx]) @ self.A[c_idx, :]

        u_diff = u_t_u - u_t_l
        Au_t = self.A @ u_diff
        denom = Au_t @ (self.weights * Au_t)
        if denom == 0:  # no usable step, e.g. underflow (float32) once the violations are tiny
            return x
        sig = (u_diff @ u_diff) / denom
        x += sig * u_diff

        return x


class AdaptiveStepLandweberHyperslab_new(SimultaneousAMSHyperslab):
    """
    Adaptive step-size Landweber algorithm for solving linear inequalities
    of
    the form lb <= Ax <= ub.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    References
    ----------
    - [1] https://doi.org/10.1515/jiip-2015-0082
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        weights: None | List[float] | npt.NDArray = None,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, weights, proximity_flag)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        p = self.map(x)
        res_l, res_u = self.bounds.residual(p)

        # Weighted violation per row; zero where the constraint is satisfied.
        r = self.weights * (xp.minimum(res_u, 0) - xp.minimum(res_l, 0))

        if not r.any():
            return x

        g = self.A.T @ r  # negative gradient (same as u_diff)
        Ag = self.A @ g
        denom = Ag @ (self.weights * Ag)

        if denom <= xp.finfo(x.dtype).tiny:
            return x  # no usable descent direction

        sig = (g @ g) / denom
        x += sig * g  # or: return x + sig * g, to avoid mutation
        return x


class BlockIterativeAdaptiveStepLandweberHyperslab(AdaptiveStepLandweberHyperslab):
    """
    Sequential block-iterative variant of the adaptive step-size Landweber
    algorithm for solving linear inequalities of the form lb <= Ax <= ub
    block by block.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems.
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    length_blocks : List[int]
        Number of rows in each block. Must sum to the total number of rows in A.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    References
    ----------
    - [1] https://doi.org/10.1515/jiip-2015-0082
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        length_blocks: List[int],
        weights: None | List[float] | npt.NDArray = None,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, weights, relaxation, proximity_flag)
        self.length_blocks = length_blocks

        self.As = []
        self.ls = []
        self.us = []
        self.block_weights = []
        _lower_bound = 0
        for i in range(len(self.length_blocks)):
            self.As.append(A[_lower_bound : _lower_bound + self.length_blocks[i]])
            self.ls.append(lb[_lower_bound : _lower_bound + self.length_blocks[i]])
            self.us.append(ub[_lower_bound : _lower_bound + self.length_blocks[i]])
            self.block_weights.append(
                self.weights[_lower_bound : _lower_bound + self.length_blocks[i]]
            )
            _lower_bound += self.length_blocks[i]

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        for i, A_block in enumerate(self.As):
            p = A_block @ x
            res_l, res_u = p - self.ls[i], self.us[i] - p
            d_idx = res_u < 0
            c_idx = res_l < 0
            if not (xp.any(d_idx)) and not (xp.any(c_idx)):
                continue

            weights = self.block_weights[i]
            u_t_u = (weights[d_idx] * res_u[d_idx]) @ A_block[d_idx, :]
            u_t_l = (weights[c_idx] * res_l[c_idx]) @ A_block[c_idx, :]

            u_diff = u_t_u - u_t_l
            Au_t = A_block @ u_diff
            denom = Au_t @ (weights * Au_t)
            if denom == 0:
                continue
            sig = (u_diff @ u_diff) / denom
            x += sig * u_diff

        return x


class AdaptiveStepLandweberHyperslab2(SimultaneousAMSHyperslab):
    """
    Adaptive step-size Landweber algorithm (variant 2) for solving linear
    inequalities of the form lb <= Ax <= ub.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    lb : npt.NDArray
        The lower bounds for the hyperslab.
    ub : npt.NDArray
        The upper bounds for the hyperslab.
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    proximity_flag : bool, optional
        A flag indicating whether to use proximity measures, by default True.

    References
    ----------
    - [1] https://doi.org/10.1515/jiip-2015-0082
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        lb: npt.NDArray,
        ub: npt.NDArray,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, lb, ub, 1.0, relaxation, weights, proximity_flag)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        p = self.map(x)
        (res_l, res_u) = self.bounds.residual(p)
        d_idx = res_u < 0
        c_idx = res_l < 0

        res_u[~d_idx] = 0
        res_l[~c_idx] = 0

        if not (xp.any(d_idx)) and not (xp.any(c_idx)):
            return x

        idx = d_idx | c_idx
        # Note: weights[idx] has shape (k,); result of @ has shape (n,).
        # This formula is designed for cases where the weighted combination is applied element-wise.
        u_diff = self.weights[idx] * (res_u[idx] - res_l[idx]) @ self.A[idx, :]

        Au_t = self.A @ u_diff
        denom = Au_t @ (self.weights * Au_t)
        if denom == 0:  # no usable step, e.g. underflow (float32) once the violations are tiny
            return x
        sig = (u_diff @ u_diff) / denom
        x += sig * u_diff

        return x
