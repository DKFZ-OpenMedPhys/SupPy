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
        p = self.map(x)
        res = self.b - p
        res_idx = res != 0
        if not xp.any(res_idx):
            self.sigmas.append(0)
            return x
        t = self.weight_norm * res
        t_2 = t @ self.A
        sig = (res @ t) / (t_2 @ t_2)
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
        p = self.map(x)
        res = self.b - p
        res_idx = res != 0
        if not xp.any(res_idx):
            self.sigmas.append(0)
            return x
        u_t = (self.weights * res) @ self.A
        Au_t = self.A @ u_t
        sig = (u_t @ u_t) / (Au_t @ (self.weights * Au_t))
        self.sigmas.append(sig)
        x += sig * u_t

        return x
