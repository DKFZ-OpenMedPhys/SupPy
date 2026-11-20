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
from suppy.feasibility._linear_algorithms import _normalized_weights, _safe_step_size


class LandweberHyperplane(SimultaneousKaczmarzMethod):
    """
    Landweber method with a fixed step size for solving linear equalities of
    the form Ax = b.

    Each call of `project` performs one iteration

        x <- x + step_size * A^T W (b - Ax),

    i.e. one gradient step on f(x) = 1/2 ||W^(1/2) (Ax - b)||^2. The
    iteration converges for 0 < step_size < 2 / lambda_max(A^T W A).

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems.
    b : npt.NDArray
        Bound for linear systems.
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
        Flag to indicate if proximity calculations should be performed, by
        default True.

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
        b: npt.NDArray,
        step_size: float | None = None,
        weights: None | List[float] | npt.NDArray = None,
        relaxation: float = 1.0,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, 1.0, relaxation, _normalized_weights(weights), proximity_flag)
        self._setup_landweber(weights, step_size)

    def _project(self, x: npt.NDArray) -> np.ndarray:
        res = self.b - self.map(x)
        if self.landweber_weights is not None:
            res = self.landweber_weights * res
        x += self.step_size * (res @ self.A.A)
        return x


class ExtrapolatedLandweberHyperplane(SimultaneousKaczmarzMethod):
    """
    Extrapolated Landweber algorithm for solving linear equalities of the
    form Ax = b.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    proximity_flag : bool, optional
        Flag to indicate if proximity calculations should be performed, by default True.

    References
    ----------
    - [1] https://doi.org/10.1007/s11075-025-02163-0
    - [2] https://doi.org/10.1007/978-3-642-30901-4
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        b: npt.NDArray,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, 1.0, relaxation, weights, proximity_flag)
        self.a_i = self.A.row_norm(2, 2)
        self.weight_norm = self.weights / self.a_i
        self.sigmas = []

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        res = self.b - self.map(x)
        t = self.weight_norm * res
        t_2 = t @ self.A
        # step size 0 if all constraints are met (t_2 = 0) or t_2 @ t_2 underflows
        sig = _safe_step_size(xp, res @ t, t_2 @ t_2)
        self.sigmas.append(sig)
        x += sig * t_2

        return x


class AdaptiveStepLandweberHyperplane(SimultaneousKaczmarzMethod):
    """
    Landweber algorithm with adaptive step size for solving linear
    equalities
    of the form Ax = b.

    Parameters
    ----------
    A : npt.NDArray or sparse.sparray
        Matrix for linear systems
    b : npt.NDArray
        Bound for linear systems
    relaxation : float, optional
        The outer relaxation parameter, by default 1.0.
    weights : None or List[float], optional
        The weights for the constraints, by default None.
    proximity_flag : bool, optional
        Flag to indicate if proximity calculations should be performed, by default True.

    References
    ----------
    - [1] https://doi.org/10.1515/jiip-2015-0082
    """

    def __init__(
        self,
        A: npt.NDArray | sparse.sparray,
        b: npt.NDArray,
        relaxation: float = 1.0,
        weights: None | List[float] | npt.NDArray = None,
        proximity_flag: bool = True,
    ):
        super().__init__(A, b, 1.0, relaxation, weights, proximity_flag)
        self.sigmas = []

    def _project(self, x: npt.NDArray) -> np.ndarray:
        xp = cp if self._use_gpu else np
        res = self.b - self.map(x)
        u_t = (self.weights * res) @ self.A
        Au_t = self.A @ u_t
        # step size 0 if all constraints are met (u_t = 0) or there is no usable step
        sig = _safe_step_size(xp, u_t @ u_t, Au_t @ (self.weights * Au_t))
        self.sigmas.append(sig)
        x += sig * u_t

        return x
