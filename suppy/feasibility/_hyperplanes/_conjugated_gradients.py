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

from suppy.feasibility._linear_algorithms import HyperplaneFeasibility


class ConjugatedGradients(HyperplaneFeasibility):
    """Conjugated Gradients algorithm for solving linear equalities Ax = b.

    The update step is chosen to be perturbation-resilient. CG state
    (``last_g``, ``last_p``) is NOT reset between successive ``solve()``
    calls — this enables warm-starting but means a second call continues
    from the final direction of the first.

    Parameters
    ----------
    A : npt.NDArray
        Matrix for the linear system.
    b : npt.NDArray
        Right-hand side vector.
    proximity_flag : bool, optional
        Flag to indicate whether to use proximity, by default True.
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        proximity_flag: bool = True,
    ):
        super().__init__(
            A, b, algorithmic_relaxation=1.0, relaxation=1.0, proximity_flag=proximity_flag
        )
        self.last_g = None
        self.last_p = None

    def _project(self, x: npt.NDArray) -> np.ndarray:
        if self.last_p is None:  # no previous run
            self.last_g = self.A.T @ (self.A @ x - self.b)
            self.last_p = -self.last_g

        else:
            g = self.A.T @ (self.A @ x - self.b)
            beta = -(g @ g) / (self.last_g @ self.last_p)
            self.last_p = -g + beta * self.last_p
            self.last_g = g

        Ap = self.A @ self.last_p
        alpha = -1 * (self.last_g @ self.last_p) / (Ap @ Ap)
        x += alpha * self.last_p
        return x

    def precondition(self, x: npt.NDArray) -> np.ndarray:
        """Perform a single steepest-descent step to initialise CG from a good
        direction.
        """
        g = self.A.T @ (self.A @ x - self.b)
        self.last_g = g
        self.last_p = -g
        x -= (g @ g) / (g @ (self.A.T @ (self.A @ g))) * g

        return x
