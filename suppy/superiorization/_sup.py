from abc import ABC, abstractmethod
from typing import Callable
import numpy.typing as npt
from suppy.perturbations import Perturbation, PowerSeriesGradientPerturbation


class FeasibilityPerturbation(ABC):
    """
    Abstract base class for perturbation approaches of feasibility seeking
    algorithms.

    Parameters
    ----------
    basic : Callable
        The underlying feasibility seeking algorithm.

    Attributes
    ----------
    basic : Callable
        The underlying feasibility seeking algorithm.
    """

    def __init__(self, basic: Callable):
        self.basic = basic

    @abstractmethod
    # @ensure_float_array
    def solve(self, x_0: npt.NDArray):
        """
        Solve the perturbed feasibility seeking problem.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial guess for the solution.

        Returns
        -------
        None
            This method should be overridden by subclasses to provide the actual implementation.
        """
        pass
