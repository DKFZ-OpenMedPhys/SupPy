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
from suppy.utils import LinearMapping


class ConjugatedGradients(HyperplaneFeasibility):
    """Conjugated Gradients algorithm for solving linear systems. Update step
    is chosen to be perturbation resilient.
    """

    def __init__(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        proximity_flag: bool = True,
    ):
        super().__init__(
            A, b, algorithmic_relaxation=1, relaxation=1, proximity_flag=proximity_flag
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
        g = self.A.T @ (self.A @ x - self.b)
        self.last_g = g
        self.last_p = -g
        x -= (g @ g) / (g @ (self.A.T @ (self.A @ g))) * g

        return x
